"""
bench_ladder.py -- one measurement per process (so graph/compile memory never leaks
between settings). Works in the ORIGINAL repo (MODE=base) and in your optimized repo.

MODE (what is switched on):
  base   nothing installed                         (original repo, or your repo without the fast path)
  graph  fast_recurrent only (compiled step + CUDA graphs)
  post   fast_post only (fused phase + dedup + compiled decode, eager rollout)
  full   fast_recurrent + fast_post

Model source: CKPT=<checkpoint>   (trained weights, as in test.py)
          or  MODEL_YAML=<train config>  (random weights -- fine for SPEED and MEMORY of e.g. LTE)
Data: CONFIG=configs/test_one.yaml (same yaml in both repos!), IMG=0 (image index), N=200000 points
Knobs: BS (queries per call), TS=16,64, REPS=2, POINTWISE_KIND=linear|conv (your repo only),
       CUDA_VISIBLE_DEVICES, TAG (label for the CSV), OUT=ladder_results.csv

Example (your repo):
  POINTWISE_KIND=conv MODE=full BS=100000 TAG=cur CKPT=save/recurrent_lte_paper_repro/epoch-best.pth python bench_ladder.py
"""
import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import csv
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import yaml

import models
import datasets
import utils

MODE = os.environ.get("MODE", "base")
TAG = os.environ.get("TAG", "run")
CONFIG = os.environ.get("CONFIG", "configs/test_one.yaml")
CKPT = os.environ.get("CKPT", "")
MODEL_YAML = os.environ.get("MODEL_YAML", "")
IMG = int(os.environ.get("IMG", 0))
BS = int(os.environ.get("BS", 100000))
TS = [int(x) for x in os.environ.get("TS", "16,64").split(",")]
N = int(os.environ.get("N", 200000))
REPS = int(os.environ.get("REPS", 2))
OUT = os.environ.get("OUT", "ladder_results.csv")


def load():
    with open(CONFIG) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if cfg.get("data_norm"):
        utils.set_normalizer(**cfg["data_norm"])
    cfg["test_dataset"]["dataset"]["args"]["first_k"] = IMG + 1       # don't cache 100 images
    ds = datasets.make(cfg["test_dataset"]["dataset"])
    ds = datasets.make(cfg["test_dataset"]["wrapper"], args={"dataset": ds})
    inputs, _ = ds[IMG]
    inp, coord, cell = (inputs[k].unsqueeze(0).cuda() for k in ("inp", "coord", "cell"))
    if MODEL_YAML:
        with open(MODEL_YAML) as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)["model"]
        model = models.make(spec).cuda().eval()
    else:
        model = models.make(torch.load(CKPT, map_location="cpu")["model"], load_sd=True).cuda().eval()
    return model, inp, coord, cell


def run(model, coord, cell, n):
    for ql in range(0, n, BS):
        qr = min(ql + BS, n)
        model.query(coord[:, ql:qr], cell[:, ql:qr])


def main():
    model, inp, coord, cell = load()
    n = min(N, coord.shape[1])
    with torch.no_grad():
        model.gen_feat(inp)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        if MODE in ("graph", "full"):
            from models.fast_recurrent import install
            install(model, batch_size=BS)
        if MODE in ("post", "full"):
            from models.fast_post import install_fast_post
            install_fast_post(model)
        kind = os.environ.get("POINTWISE_KIND", "linear")
        print(f"[{TAG}] MODE={MODE} kind={kind} BS={BS} N={n} LRpx={model.feat.shape[-2] * model.feat.shape[-1]} "
              f"HRpts={coord.shape[1]} baseline {base / 2**20:.0f} MB", flush=True)
        new_file = not os.path.exists(OUT)
        for T in TS:
            model.predictor.num_pred = model.num_pred = model.num_preds = T
            run(model, coord, cell, n)                       # warm (compile / capture / allocator)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(REPS):
                run(model, coord, cell, n)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / REPS * 1000 / n * 100000
            peak = torch.cuda.max_memory_allocated() / 2**20
            print(f"   T={T:3d}: {ms:9.1f} ms / 100k pts | peak {peak:8.0f} MB", flush=True)
            with open(OUT, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["tag", "mode", "kind", "bs", "T", "ms_per_100k", "peak_mb", "baseline_mb", "n_pts"])
                    new_file = False
                w.writerow([TAG, MODE, kind, BS, T, f"{ms:.2f}", f"{peak:.0f}", f"{base / 2**20:.0f}", n])


if __name__ == "__main__":
    main()
