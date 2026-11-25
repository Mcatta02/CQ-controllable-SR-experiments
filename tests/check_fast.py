"""
Equivalence + speed of the fast recurrent inside the REAL model.query() path.

    POINTWISE_KIND=conv CUDA_VISIBLE_DEVICES=<free> python check_fast.py
    N=100000 BS=10000 TS=64,16 python check_fast.py

Reports, per T:
  * seed noise scale: |orig(seed0) - orig(seed1)| on the reconstructed RGB
  * |orig - fast| (should be the same order as the seed noise, not larger)
  * fast called twice gives different outputs (fresh noise per replay)
  * time per pass over N points in chunks of BS (model.query end to end)
"""
import os as _os, sys as _sys  # run from repo root: python tests/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

import parity_verify as pv
from models.fast_recurrent import install
from models.fast_post import install_fast_post

N = int(os.environ.get("N", 100000))
BS = int(os.environ.get("BS", 10000))
TS = [int(x) for x in os.environ.get("TS", "64,16").split(",")]


def set_T(model, T):
    model.predictor.num_pred = model.num_pred = model.num_preds = T


def recon(model, coord, cell, n):
    outs = []
    for ql in range(0, n, BS):
        qr = min(ql + BS, n)
        outs.append(model.query(coord[:, ql:qr], cell[:, ql:qr])["recon"])
    return torch.cat(outs, 1)


def timed(fn, reps=2):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main():
    model, inputs = pv.load()
    inp, coord, cell = (inputs[k].cuda() for k in ("inp", "coord", "cell"))
    n = min(N, coord.shape[1])
    print(f"POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')}, N={n}, BS={BS}, Ts={TS}")
    ref0, ref1, t_orig = {}, {}, {}
    with torch.no_grad():
        model.gen_feat(inp)
        for T in TS:
            set_T(model, T)
            torch.manual_seed(0)
            ref0[T] = recon(model, coord, cell, n)
            torch.manual_seed(1)
            ref1[T] = recon(model, coord, cell, n)
            t_orig[T] = timed(lambda: recon(model, coord, cell, n))

        install(model, batch_size=BS)
        install_fast_post(model)
        for T in TS:
            set_T(model, T)
            t0 = time.time()
            b1 = recon(model, coord, cell, n)
            torch.cuda.synchronize()
            build = time.time() - t0
            b2 = recon(model, coord, cell, n)
            t_fast = timed(lambda: recon(model, coord, cell, n))
            seed = (ref0[T] - ref1[T]).abs()
            d = (ref0[T] - b1).abs()
            fresh = (b1 - b2).abs()
            print(f"\n[T={T}]")
            print(f"   seed noise  |orig0-orig1|: mean {seed.mean().item():.3e} max {seed.max().item():.3e}")
            print(f"   |orig - fast|            : mean {d.mean().item():.3e} max {d.max().item():.3e}")
            print(f"   |fast call1 - call2|     : mean {fresh.mean().item():.3e} (must be > 0: fresh noise)")
            print(f"   first-call build time    : {build:.1f} s (compile+capture, one-off)")
            print(f"   time over {n} pts: orig {t_orig[T]:8.1f} ms | fast {t_fast:8.1f} ms | speedup {t_orig[T] / t_fast:.2f}x")
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
