"""
Where does the time go on the fast path (fixed T), per image, WARM?

    POINTWISE_KIND=linear CUDA_VISIBLE_DEVICES=<gpu number> python profile_stages.py
    BS=100000  TS=64,16  REPS=3  MODES=query,query_t

Needs models/fast_recurrent.py and models/fast_post.py (+ your liif_base.py with fid).
For each mode (model.query = local-ensemble path? / model.query_t) and T it prints, per image:
  wall ms, and GPU-event ms for gen_feat, _fid, predictor.forward, recurrent,
  the remainder ("other" = grid_sample, rel coords, decoder, accumulate, ...),
  number of predictor calls, rows entering predictor vs rows entering recurrent
  (after dedup), and model.local_ensemble.
Reference floors (this model, from earlier benchmarks) for ~170k LR pixels / 2.7M HR px:
  rollout ~4 us per LR pixel at T=64  -> ~0.7 s ;  decode traffic ~0.02-0.3 s.
"""
import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch

import parity_verify as pv
from models.fast_recurrent import install
from models.fast_post import install_fast_post

BS = int(os.environ.get("BS", 100000))
TS = [int(x) for x in os.environ.get("TS", "64,16").split(",")]
REPS = int(os.environ.get("REPS", 3))
MODES = os.environ.get("MODES", "query,query_t").split(",")


class Probe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.ev, self.cnt, self.rows = {}, {}, {}

    def wrap(self, obj, name, label, rows_of=None):
        f = getattr(obj, name)

        def w(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            r = f(*a, **k)
            e.record()
            self.ev.setdefault(label, []).append((s, e))
            self.cnt[label] = self.cnt.get(label, 0) + 1
            if rows_of is not None:
                self.rows[label] = self.rows.get(label, 0) + rows_of(a, k)
            return r

        setattr(obj, name, w)

    def ms(self, label):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.ev.get(label, []))


def run(model, coord, cell, n, mode, T):
    for ql in range(0, n, BS):
        qr = min(ql + BS, n)
        if mode == "query":
            model.query(coord[:, ql:qr], cell[:, ql:qr])
        else:
            model.query_t(coord[:, ql:qr], cell[:, ql:qr], T)


def main():
    model, inputs = pv.load()
    inp, coord, cell = (inputs[k].cuda() for k in ("inp", "coord", "cell"))
    n = coord.shape[1]
    install(model, batch_size=BS)
    install_fast_post(model)
    net, pred = model.predictor.layer, model.predictor
    p = Probe()
    p.wrap(model, "gen_feat", "gen_feat")
    p.wrap(model, "_fid", "fid")
    p.wrap(pred, "forward", "predictor", lambda a, k: a[0].shape[0])
    p.wrap(net, "recurrent", "recurrent",
           lambda a, k: (k["token"] if "token" in k else a[0]).shape[0])
    print(f"POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')}, HR points {n}, BS {BS}, "
          f"chunks {-(-n // BS)}, local_ensemble={getattr(model, 'local_ensemble', 'n/a')}")
    with torch.no_grad():
        for mode in MODES:
            for T in TS:
                model.predictor.num_pred = model.num_pred = model.num_preds = T
                model.gen_feat(inp)
                run(model, coord, cell, n, mode, T)       # warm: builds/compiles
                run(model, coord, cell, n, mode, T)
                p.reset()
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(REPS):
                    model.gen_feat(inp)
                    run(model, coord, cell, n, mode, T)
                torch.cuda.synchronize()
                wall = (time.time() - t0) / REPS * 1000
                g = {k: p.ms(k) / REPS for k in ("gen_feat", "fid", "predictor", "recurrent")}
                other = wall - g["gen_feat"] - g["fid"] - g["predictor"]
                calls = p.cnt.get("predictor", 0) / REPS
                rows_p = p.rows.get("predictor", 0) / REPS
                rows_r = p.rows.get("recurrent", 0) / REPS
                print(f"\n[{mode}, T={T}] per image (warm, mean of {REPS})")
                print(f"   wall {wall:8.1f} ms | gen_feat {g['gen_feat']:7.1f} | _fid {g['fid']:7.1f} | "
                      f"predictor {g['predictor']:7.1f} (of which recurrent {g['recurrent']:7.1f}) | other {other:7.1f}")
                print(f"   predictor calls {calls:.0f}, rows into predictor {rows_p:.0f}, rows into recurrent {rows_r:.0f} "
                      f"(dedup x{rows_p / max(rows_r, 1):.1f}); recurrent calls {p.cnt.get('recurrent', 0) / REPS:.0f}")
    print(f"\npeak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
