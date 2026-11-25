"""
models/fast_recurrent.py -- drop-in replacement for RecurrentNetwork.recurrent()

Fast recurrent inference:
  * one compiled single step
  * SPE applied from pre-built per-position tables
  * optional CUDA-graph capture of the whole T-step rollout
  * fresh SPE noise on every rollout
  * padded batch sizes to keep compiled shapes static

CUDA graphs are disabled automatically for POINTWISE_KIND=conv because the
conv implementation uses indexing operations that are not safe inside the
outer CUDA graph in this implementation.

Environment:
    FAST_RECURRENT=1

Optional:
    FAST_RECURRENT_CUDAGRAPH=0
        Disable CUDA graphs completely.

    FAST_PREWARM_T=4,8,16,32,64
        Prewarm selected T values.

Do not combine with COMPILE_PREDICTOR=1.
"""

import math
import os

import torch
import torch._dynamo

import models.speculative as sp


def pad_size(B, main=None, min_size=1024):
    if B <= min_size:
        q = min_size
    else:
        e = math.ceil(math.log2(B) * 4 - 1e-9) / 4
        q = max(int(math.ceil(2 ** e / 256.0)) * 256, B)

    if main is not None and B <= main < q:
        return main

    return q


class FastRecurrent:
    def __init__(
        self,
        net,
        main_batch=None,
        t_max=64,
        compile_mode=None,
        use_cuda_graph=None,
    ):
        assert (
            getattr(net, "n_condition", None) is None
        ), "n_condition not supported"

        self.net = net
        self.main = main_batch
        self.t_max = t_max

        self.orig = net.recurrent
        self.cache = {}

        # The conv implementation currently contains indexing operations
        # that are not safe to put inside the outer CUDA graph.
        pointwise_kind = os.environ.get("POINTWISE_KIND", "linear").lower()

        if use_cuda_graph is None:
            use_cuda_graph = (
                os.environ.get("FAST_RECURRENT_CUDAGRAPH", "1") == "1"
            )

        if pointwise_kind == "conv":
            use_cuda_graph = False

        self.use_cuda_graph = use_cuda_graph

        if self.use_cuda_graph:
            self.pool = torch.cuda.graph_pool_handle()
        else:
            self.pool = None

        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit,
            128,
        )

        net_ = net

        def step(y, pos, states, tables, ones_w):
            yv, caches = sp.net_verify(
                net_,
                None,
                y,
                pos,
                states,
                tables,
                skip_token=True,
            )
            return yv, sp.advance(states, caches, ones_w)

        # Compile exactly one recurrent step.
        #
        # This is intentionally kept separate from the outer CUDA graph.
        # The compiled step has already been verified for numerical parity.
        self.step_c = torch.compile(
            step,
            mode=compile_mode,
        )

    def _make_run(
        self,
        Bp,
        T,
        token_buf,
        pos_all,
        zero_y,
        ones_w,
    ):
        net = self.net
        step_c = self.step_c

        def run():
            # Fresh noise on every execution/replay.
            tables = sp.pe_tables(
                net,
                sp.fast_noise(net, self.t_max),
            )

            # Token embedding is constant for the whole T-step rollout.
            lat_tok = sp._pw(
                net.token_layer,
                token_buf.transpose(1, 2),
            )

            states = sp.init_states(net, Bp)

            # First prediction.
            y, caches = sp.net_verify(
                net,
                token_buf,
                zero_y,
                pos_all[0:1],
                states,
                tables,
                lat_tok=lat_tok,
            )

            states = sp.advance(
                states,
                caches,
                ones_w,
            )

            outs = [y]

            # Remaining predictions.
            for t in range(1, T):
                y, states = step_c(
                    y,
                    pos_all[t:t + 1],
                    states,
                    tables,
                    ones_w,
                )
                outs.append(y)

            return torch.cat(outs, dim=1)

        return run

    def _build(self, Bp, T):
        net = self.net

        w = net.out_layer.weight
        dev = w.device
        dt = w.dtype
        C = w.shape[0]

        in_dim = net.token_layer.weight.shape[1]

        # Long-lived static input/output tensors.
        token_buf = torch.zeros(
            Bp,
            in_dim,
            1,
            device=dev,
            dtype=dt,
        )

        pos_all = torch.arange(
            self.t_max,
            device=dev,
        )

        zero_y = torch.zeros(
            Bp,
            1,
            C,
            device=dev,
            dtype=dt,
        )

        ones_w = torch.ones(
            Bp,
            1,
            device=dev,
            dtype=dt,
        )

        run = self._make_run(
            Bp,
            T,
            token_buf,
            pos_all,
            zero_y,
            ones_w,
        )

        # ------------------------------------------------------------
        # Eager/compiled path
        # ------------------------------------------------------------
        #
        # This is the path used for POINTWISE_KIND=conv.
        #
        # We still run the compiled recurrent step, but do not capture the
        # entire rollout in a raw CUDA graph.
        if not self.use_cuda_graph:
            return {
                "graph": None,
                "token": token_buf,
                "run": run,
                "out": None,
            }

        # ------------------------------------------------------------
        # CUDA graph path
        # ------------------------------------------------------------

        with torch.no_grad():
            s = torch.cuda.Stream()

            s.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(s):
                for _ in range(3):
                    run()

            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()

            with torch.cuda.graph(
                g,
                pool=self.pool,
                capture_error_mode="thread_local"
            ):
                out = run()

        return {
            "graph": g,
            "token": token_buf,
            "run": run,
            "out": out,
        }

    @torch.no_grad()
    def __call__(self, token, num_pred, **kw):
        if (
            kw
            or num_pred < 1
            or num_pred > self.t_max
        ):
            return self.orig(
                token=token,
                num_pred=num_pred,
                **kw,
            )

        B = token.shape[0]
        Bp = pad_size(
            B,
            self.main,
        )

        key = (Bp, num_pred)

        if key not in self.cache:
            self.cache[key] = self._build(
                Bp,
                num_pred,
            )

        entry = self.cache[key]

        tok = entry["token"]

        # Copy real input into the static buffer.
        tok[:B].copy_(token)

        # Clear padded rows.
        if Bp > B:
            tok[B:].zero_()

        # ------------------------------------------------------------
        # CUDA graph path
        # ------------------------------------------------------------
        if entry["graph"] is not None:
            entry["graph"].replay()

            return (
                entry["out"][:B]
                .transpose(1, 2)
                .contiguous()
            )

        # ------------------------------------------------------------
        # Compiled eager path
        # ------------------------------------------------------------
        #
        # Used for POINTWISE_KIND=conv.
        #
        # run() creates the required state internally and executes the
        # compiled single-step kernel without the problematic outer graph.
        out = entry["run"]()

        return (
            out[:B]
            .transpose(1, 2)
            .contiguous()
        )

    def prewarm(self, sizes, Ts):
        for B in sizes:
            Bp = pad_size(
                B,
                self.main,
            )

            for T in Ts:
                key = (Bp, T)

                if key not in self.cache:
                    self.cache[key] = self._build(
                        Bp,
                        T,
                    )


def install(
    model,
    batch_size=None,
    prewarm_Ts=None,
):
    net = model.predictor.layer

    fr = FastRecurrent(
        net,
        main_batch=batch_size,
    )

    net._fast = fr

    net.recurrent = lambda token, num_pred, **kw: fr(
        token,
        num_pred,
        **kw,
    )

    if prewarm_Ts and batch_size:
        fr.prewarm(
            [batch_size],
            prewarm_Ts,
        )

    return model


def install_if_requested(
    model,
    batch_size=None,
):
    if os.environ.get("FAST_RECURRENT") == "1":
        Ts = [
            int(x)
            for x in os.environ.get(
                "FAST_PREWARM_T",
                "",
            ).split(",")
            if x
        ]

        Ts = Ts or None

        return install(
            model,
            batch_size,
            Ts,
        )

    return model