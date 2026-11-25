"""
Can torch.compile fuse the memory-bound pieces (LayerNorm, elementwise, state
update) if we compile ONE step and call it T-1 times (instead of compiling the
fully unrolled 64-step loop, where Inductor produced wrong output before)?

Step 0 (token embedding) runs eagerly; steps 1..T-1 use the compiled step.
Parity vs recurrent() with replayed noise is checked FIRST -- if it fails the
timing is meaningless and the script says so.

    POINTWISE_KIND=conv CUDA_VISIBLE_DEVICES=<free> python bench_compiled_step.py
    BATCHES="5000,10000"   COMPILE_MODE=default|max-autotune-no-cudagraphs
"""
import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import gc
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

import parity_verify as pv
import models.speculative as sp
from bench_baseline import graphed
from bench_seqfast import seq_fast

T = 64
BATCHES = [int(x) for x in os.environ.get("BATCHES", "5000,10000,50000,100000").split(",")]
MODE = os.environ.get("COMPILE_MODE", "default")


def main():
    model, inputs = pv.load()
    print(f"POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')}, GPU={os.environ['CUDA_VISIBLE_DEVICES']}, "
          f"T={T}, compile mode={MODE}")
    for B in BATCHES:
        torch._dynamo.reset()
        pv.N_POINTS = B
        torch.cuda.reset_peak_memory_stats()
        net, token = pv.capture_token(model, inputs)
        dev, C = token.device, sp._out_dim(net)
        print(f"\n=== B={B} ===")
        with torch.no_grad():
            torch.manual_seed(0)
            ref = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()
            torch.manual_seed(0)
            tables = sp.pe_tables(net, sp.replay_noise(net, T))
            lat_tok = sp._pw(net.token_layer, token.transpose(1, 2))
            pos_all = torch.arange(T, device=dev)
            zero_y = torch.zeros(B, 1, C, device=dev)
            ones_w = torch.ones(B, 1, device=dev)

            def step(y, pos, states):
                yv, caches = sp.net_verify(net, token, y, pos, states, tables, skip_token=True)
                return yv, sp.advance(states, caches, ones_w)

            step_c = torch.compile(step, mode=None if MODE == "default" else MODE)

            def run_compiled():
                states = sp.init_states(net, B)
                y, caches = sp.net_verify(net, token, zero_y, pos_all[0:1], states, tables, lat_tok=lat_tok)
                states = sp.advance(states, caches, ones_w)
                outs = [y]
                for t in range(1, T):
                    y, states = step_c(y, pos_all[t:t + 1], states)
                    outs.append(y)
                return torch.cat(outs, 1)

            import time
            t0 = time.time()
            out = run_compiled()
            torch.cuda.synchronize()
            print(f"   first call (incl. compile): {time.time() - t0:.1f} s")
            d = (out - ref).abs().max().item()
            ok = d < 1e-4
            print(f"   parity compiled-step vs recurrent (same noise): max abs {d:.2e}  {'OK' if ok else 'WRONG OUTPUT -- do not trust timings'}")

            run_plain = lambda: seq_fast(net, token, T, tables, lat_tok, pos_all, zero_y, ones_w)
            rep_p, _ = graphed(run_plain)
            t_plain_g = pv.timeit(rep_p) if rep_p else float("nan")
            rep_r, _ = graphed(lambda: net.recurrent(token=token, num_pred=T))
            t_rec_g = pv.timeit(rep_r) if rep_r else float("nan")
            t_comp = pv.timeit(run_compiled)
            rep_c, _ = graphed(run_compiled)
            t_comp_g = pv.timeit(rep_c) if rep_c else float("nan")
            us = lambda ms: ms * 1000 / B
            print(f"   recurrent graph       {t_rec_g:8.1f} ms ({us(t_rec_g):5.2f} us/pt)")
            print(f"   seq_fast graph        {t_plain_g:8.1f} ms ({us(t_plain_g):5.2f} us/pt)")
            print(f"   compiled step eager   {t_comp:8.1f} ms ({us(t_comp):5.2f} us/pt)")
            print(f"   compiled step graph   {t_comp_g:8.1f} ms ({us(t_comp_g):5.2f} us/pt)  "
                  f"-> {t_rec_g / t_comp_g:.2f}x vs recurrent graph, {t_plain_g / t_comp_g:.2f}x vs seq_fast graph")
        print(f"   peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        del rep_p, rep_r, rep_c
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
