"""
Does a cheaper SEQUENTIAL step beat net.recurrent()? (no speculation involved)

seq_fast = the verify code path run one step at a time (K=1) with:
  * SPE noise pre-drawn once (tables), so q/k position codes are applied as a
    small GEMM  q[B,F] @ table[t][F,R]  instead of materialising a [B,F,R]
    tensor per layer per step (the original SPEFilter does
    (qbar * q[...,None]).sum(-2)),
  * the token embedding computed once, not every step,
  * no per-step randn launches.
Numerically it must match recurrent() given the same noise (parity printed).

    POINTWISE_KIND=conv CUDA_VISIBLE_DEVICES=<free> python bench_seqfast.py
    PROFILE=1 ...  also prints the top CUDA kernels of the ORIGINAL recurrent().
    BATCHES="5000,10000" to restrict.
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

T = 64
BATCHES = [int(x) for x in os.environ.get("BATCHES", "5000,10000,50000,100000").split(",")]


def seq_fast(net, token, T, tables, lat_tok, pos_all, zero_y, ones_w):
    states = sp.init_states(net, token.shape[0])
    y, outs = zero_y, []
    for t in range(T):
        pos = pos_all[t:t + 1]
        y, caches = sp.net_verify(net, token, y, pos, states, tables, lat_tok=lat_tok, skip_token=(t > 0))
        states = sp.advance(states, caches, ones_w)
        outs.append(y)
    return torch.cat(outs, 1)


def profile_recurrent(net, token):
    from torch.profiler import profile, ProfilerActivity
    net.recurrent(token=token, num_pred=8)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        net.recurrent(token=token, num_pred=8)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12, max_name_column_width=55))


def main():
    model, inputs = pv.load()
    print(f"POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')}, GPU={os.environ['CUDA_VISIBLE_DEVICES']}, T={T}")
    for B in BATCHES:
        pv.N_POINTS = B
        torch.cuda.reset_peak_memory_stats()
        net, token = pv.capture_token(model, inputs)
        dev, C = token.device, sp._out_dim(net)
        print(f"\n=== B={B} ===")
        with torch.no_grad():
            if os.environ.get("PROFILE"):
                profile_recurrent(net, token)

            # parity vs recurrent with identical noise
            torch.manual_seed(0)
            ref = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()
            torch.manual_seed(0)
            tables = sp.pe_tables(net, sp.replay_noise(net, T))
            lat_tok = sp._pw(net.token_layer, token.transpose(1, 2))
            pos_all = torch.arange(T, device=dev)
            zero_y = torch.zeros(B, 1, C, device=dev)
            ones_w = torch.ones(B, 1, device=dev)
            run = lambda: seq_fast(net, token, T, tables, lat_tok, pos_all, zero_y, ones_w)
            d = (run() - ref).abs().max().item()
            print(f"   parity seq_fast vs recurrent (same noise): max abs {d:.2e}")

            t_rec = pv.timeit(lambda: net.recurrent(token=token, num_pred=T))
            rep_r, _ = graphed(lambda: net.recurrent(token=token, num_pred=T))
            t_rec_g = pv.timeit(rep_r) if rep_r else float("nan")
            t_fast = pv.timeit(run)
            rep_f, _ = graphed(run)
            t_fast_g = pv.timeit(rep_f) if rep_f else float("nan")
            us = lambda ms: ms * 1000 / B
            print(f"   recurrent eager {t_rec:8.1f} ms ({us(t_rec):5.2f} us/pt) | graph {t_rec_g:8.1f} ms ({us(t_rec_g):5.2f} us/pt)")
            print(f"   seq_fast  eager {t_fast:8.1f} ms ({us(t_fast):5.2f} us/pt) | graph {t_fast_g:8.1f} ms ({us(t_fast_g):5.2f} us/pt)")
            print(f"   speedup vs recurrent-graph: {t_rec_g / t_fast_g:.2f}x   vs recurrent-eager: {t_rec / t_fast_g:.2f}x")
        print(f"   peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        del rep_r, rep_f
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
