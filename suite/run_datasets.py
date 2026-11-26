#!/usr/bin/env python
"""Multi-dataset evaluation:original vs accelerated, same checkpoint,
T in {4,16,64}, PSNR / LPIPS / time / peak memory, per dataset and scale. Runs the existing "main"
job of run_suite.py once per (dataset, scale); nothing else changes.

  python suite/run_datasets.py --dry-run
  python suite/run_datasets.py --datasets div2k_x2 div2k_x3 div2k_x4 bsd100 urban100 manga109
  python suite/run_datasets.py --report-only suite/results/datasets_<run>

Put the HR images of each dataset in a folder of .png files and list them in the DATASETS table
(or pass --root NAME=PATH). Manga109 needs a licence form from the authors; BSD100 / Urban100 are
available in the usual SR benchmark packs (convert .jpg to .png if needed).
"""
import argparse, json, os, subprocess, sys, time, glob
import numpy as np, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = {   # name: (HR folder, scale)
    "div2k_x2": ("load/div2k/DIV2K_valid_HR", 2),
    "div2k_x3": ("load/div2k/DIV2K_valid_HR", 3),
    "div2k_x4": ("load/div2k/DIV2K_valid_HR", 4),
    "bsd100":   ("load/benchmark/B100/HR", 4),
    "urban100": ("load/benchmark/Urban100/HR", 4),
    "manga109": ("load/benchmark/Manga109/HR", 4),
}
TS = [4, 16, 64]


def report(run_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(run_dir, "*"))):
        name = os.path.basename(d)
        rec = {}
        for impl in ("orig", "fast"):
            f = glob.glob(os.path.join(d, "**", "main", f"{impl}.json"), recursive=True)
            if f:
                rec[impl] = json.load(open(f[0]))["records"]
        if len(rec) < 2:
            continue
        for T in TS:
            o = [r for r in rec["orig"] if r["T"] == T and not r.get("oom")]
            f = [r for r in rec["fast"] if r["T"] == T and not r.get("oom")]
            if not o or not f:
                continue
            m = lambda rs, k: float(np.mean([r[k] for r in rs]))
            to, tf = m(o, "time_s"), m(f, "time_s")
            # speed-up per image, and its dependence on resolution (queries per image)
            oi = {r["image"]: r for r in o}; fi = {r["image"]: r for r in f}
            ids = sorted(set(oi) & set(fi))
            sp = np.array([oi[i]["time_s"] / fi[i]["time_s"] for i in ids])
            q = np.array([oi[i]["n_queries"] for i in ids])
            corr = float(np.corrcoef(np.log(q), sp)[0, 1]) if len(ids) > 2 and q.std() > 0 else float("nan")
            rows.append((name, T, len(ids), m(o, "psnr"), m(f, "psnr"), m(o, "lpips"), m(f, "lpips"),
                         to, tf, to / tf, m(o, "peak_mb"), m(f, "peak_mb"), float(np.mean(q)), corr))
    hdr = "dataset,T,n,psnr_orig,psnr_fast,lpips_orig,lpips_fast,time_orig,time_fast,speedup,peakMB_orig,peakMB_fast,mean_queries,corr_logQ_speedup"
    out = os.path.join(run_dir, "datasets_summary.csv")
    with open(out, "w") as fh:
        fh.write(hdr + "\n")
        for r in rows:
            fh.write(",".join(str(x) if isinstance(x, (str, int)) else f"{x:.4f}" for x in r) + "\n")
    print(open(out).read()); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "suite_config.yaml"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--root", nargs="*", default=[], help="NAME=PATH overrides of the HR folders")
    ap.add_argument("--n", type=int, default=None, help="limit images per dataset (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", default=None)
    a = ap.parse_args()
    if a.report_only:
        return report(a.report_only)
    cfg = yaml.safe_load(open(a.config))
    roots = {k: v[0] for k, v in DATASETS.items()}
    roots.update(dict(x.split("=", 1) for x in a.root))
    run_id = "datasets_" + time.strftime("%Y%m%d_%H%M") + f"_gpu{cfg.get('gpu')}"   # unique when several GPUs run at once
    base = os.path.join(HERE, "results", run_id)
    for name in a.datasets:
        scale = DATASETS[name][1]
        path = os.path.expanduser(roots[name])
        path = path if os.path.isabs(path) else os.path.join(os.path.dirname(HERE), path)
        n_img = len([f for f in os.listdir(path) if f.lower().endswith(".png")]) if os.path.isdir(path) else 0
        if n_img == 0:
            print(f"[skip] {name}: no .png files in {path}"); continue
        c = yaml.safe_load(yaml.dump(cfg))
        c["paths"]["data_root"] = path
        P = c["presets"]["full"]
        P.update(Ts_main=TS, n_main=a.n or n_img, scale_main=scale)
        tmp = os.path.join(base, f"{name}.yaml"); os.makedirs(base, exist_ok=True)
        yaml.safe_dump(c, open(tmp, "w"))
        cmd = [sys.executable, os.path.join(HERE, "run_suite.py"), "--config", tmp, "--preset", "full",
               "--only", "main", "--out", base, "--run-id", name] + (["--dry-run"] if a.dry_run else [])
        print(f"== {name}: {n_img} images, x{scale}, T={TS}"); subprocess.run(cmd, check=False)
    if not a.dry_run:
        report(base)


if __name__ == "__main__":
    main()
