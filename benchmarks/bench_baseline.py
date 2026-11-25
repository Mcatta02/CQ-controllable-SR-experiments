"""
Decisive baseline benchmark for the speculative-decoding direction.

For each batch size B (and the POINTWISE_KIND set in the environment):
  * eager net.recurrent(T=64)
  * CUDA-graphed net.recurrent(T=64)       <- the real sequential floor
  * isolated verify+commit round for K in KS (eager and graphed)
  * break-even: steps a round must commit on average to tie the sequential
    baseline (eager and graphed). Needed commit for speedup s is s x this.

Run once per layer kind (kind is fixed at model construction):
    POINTWISE_KIND=linear CUDA_VISIBLE_DEVICES=<free gpu> python bench_baseline.py
    POINTWISE_KIND=conv   CUDA_VISIBLE_DEVICES=<free gpu> python bench_baseline.py
BATCHES="5000,10000" to restrict. Large B + large K needs several GiB extra.
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

T = 64
KS = (2, 4, 8, 16)
BATCHES = [int(x) for x in os.environ.get("BATCHES", "5000,10000,50000,100000").split(",")]
NAN = float("nan")


def graphed(fn):
    """Capture fn() (static tensors only) in a CUDA graph. Returns (replay, out) or (None, None)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        torch.cuda.synchronize()
        return g.replay, out
    except Exception as e:  # capture can fail on RNG / hidden syncs
        print(f"   [graph capture failed: {repr(e)[:160]}]")
        torch.cuda.synchronize()
        return None, None


def main():
    model, inputs = pv.load()
    kind = os.environ.get("POINTWISE_KIND", "linear")
    print(f"POINTWISE_KIND={kind}, GPU={os.environ['CUDA_VISIBLE_DEVICES']}, T={T}")
    for B in BATCHES:
        pv.N_POINTS = B
        torch.cuda.reset_peak_memory_stats()
        net, token = pv.capture_token(model, inputs)
        dev = token.device
        print(f"\n=== B={B} ===")
        with torch.no_grad():
            t_eager = pv.timeit(lambda: net.recurrent(token=token, num_pred=T))
            replay, g_out = graphed(lambda: net.recurrent(token=token, num_pred=T))
            if replay is not None:
                t_graph = pv.timeit(replay)
                ref = net.recurrent(token=token, num_pred=T)
                print(f"   graph output sanity: max|graph-eager| = {(g_out - ref).abs().max().item():.2e} "
                      f"(different noise draws, expect ~1e-2 scale)")
            else:
                t_graph = NAN
            print(f"   sequential T={T}: eager {t_eager:8.1f} ms ({t_eager / T:5.2f} ms/step) | "
                  f"graph {t_graph:8.1f} ms ({t_graph / T:5.2f} ms/step)")

            tables = sp.pe_tables(net, sp.fast_noise(net, T))
            C = sp._out_dim(net)
            print("   K | verify+commit round: eager ms / graph ms | break-even commit: vs eager / vs graph")
            for K in KS:
                y_prev = torch.randn(B, K, C, device=dev) * 0.3
                pos = (10 + torch.arange(K, device=dev)).expand(B, K).contiguous()
                w = torch.ones(B, K, device=dev)
                states = sp.init_states(net, B)

                def rnd():
                    y, caches = sp.net_verify(net, token, y_prev, pos, states, tables)
                    return y, sp.advance(states, caches, w)

                t_e = pv.timeit(rnd)
                rep, _ = graphed(rnd)
                t_g = pv.timeit(rep) if rep is not None else NAN
                be_e = t_e / (t_eager / T)
                be_g = (t_g / (t_graph / T)) if (rep is not None and replay is not None) else NAN
                print(f"  {K:2d} | {t_e:8.2f} / {t_g:8.2f}                    | {be_e:5.2f} / {be_g:5.2f}")
                del rep, y_prev, pos, w, states
        print(f"   peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        del replay, g_out, token
        gc.collect()
        torch.cuda.empty_cache()
    print("\nNeeded mean commit per round for speedup s over the baseline = s x break-even "
          "(the driver's own host syncs/gathers make real rounds cost more than shown).")


if __name__ == "__main__":
    main()
