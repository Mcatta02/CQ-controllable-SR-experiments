"""
Memory / time trade-off. One process per chunk size, so graph buffers from one
setting do not inflate the next.

  for BS in 1000 5000 25000 100000; do FAST=0 BS=$BS python bench_memory.py; done
  for BS in 1000 5000 25000 100000; do FAST=1 BS=$BS python bench_memory.py; done

env: FAST=0|1 (1 = fast_recurrent + fast_post), BS (queries per call),
     TS=16,64, N=200000 (points of the first val image used), REPS=2,
     POINTWISE_KIND, CUDA_VISIBLE_DEVICES.
Prints, per T: time per 100k points (warm) and peak GPU memory (MB), total and
above the baseline (model weights + encoder features + inputs, measured before
any query). Peak includes CUDA-graph pools and compiled-step buffers (FAST=1).
"""
import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

import parity_verify as pv

FAST = os.environ.get("FAST", "1") == "1"
BS = int(os.environ.get("BS", 10000))
TS = [int(x) for x in os.environ.get("TS", "16,64").split(",")]
N = int(os.environ.get("N", 200000))
REPS = int(os.environ.get("REPS", 2))


def run(model, coord, cell, n):
    for ql in range(0, n, BS):
        qr = min(ql + BS, n)
        model.query(coord[:, ql:qr], cell[:, ql:qr])


def main():
    model, inputs = pv.load()
    inp, coord, cell = (inputs[k].cuda() for k in ("inp", "coord", "cell"))
    n = min(N, coord.shape[1])
    with torch.no_grad():
        model.gen_feat(inp)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        if FAST:
            from models.fast_recurrent import install
            from models.fast_post import install_fast_post
            install(model, batch_size=BS)
            install_fast_post(model)
        print(f"FAST={int(FAST)} POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')} BS={BS} "
              f"N={n} baseline {base / 2**20:.0f} MB")
        for T in TS:
            model.predictor.num_pred = model.num_pred = model.num_preds = T
            run(model, coord, cell, n)            # warm: compile / graph capture
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(REPS):
                run(model, coord, cell, n)
            torch.cuda.synchronize()
            ms_per_100k = (time.time() - t0) / REPS * 1000 / n * 100000
            peak = torch.cuda.max_memory_allocated()
            print(f"   T={T:3d}: {ms_per_100k:9.1f} ms / 100k pts | peak {peak / 2**20:8.0f} MB "
                  f"(above baseline {(peak - base) / 2**20:8.0f} MB)")


if __name__ == "__main__":
    main()