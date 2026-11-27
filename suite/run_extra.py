#!/usr/bin/env python
"""
Orchestrator for the extended experiments (never parallel: timings need an idle GPU).

  python suite/run_extra.py --dry-run                       # list the jobs
  python suite/run_extra.py --preset quick --only scale     # tiny smoke test on a small GPU
  python suite/run_extra.py --preset full                   # everything (A100 / V100)
  python suite/run_extra.py --report-only suite/results/extra_<run>

Groups (--only):
  scale     scale factors 2..12, T=4/16/64: time, sharing ratio, PSNR/LPIPS, speed-up vs sharing        (worker.py eval)
  scal      LR sizes up to 1024, queries per call 5k..150k, time, peak memory, OOM                       (worker_extra scal)
  noise     identical-noise comparison: generator outputs per iteration, phase head, RGB                (worker_extra noise_equiv)
  exact     same noise, different query arrangements (shuffle, split, regions)                          (worker_extra arrange_exact)
  stat      full images, non-integer scales x batch sizes x orders: ratio, time, PSNR                   (worker_extra arrange_stat)
  advis     data for the local-quality analysis of adaptive inference                                   (worker_extra adapt_vis)
The report (extra_report.py) is run at the end unless --no-report is given.
"""
import argparse
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suitelib as sl  # noqa: E402

GROUPS = ["scale", "scal", "noise", "exact", "stat", "advis"]

PRESETS = {
    "quick": dict(
        scale=dict(scales=[2, 4, 8], Ts=[4, 16], n=2, bsize=5000, reps=1),
        scal=dict(lr_sizes=[64, 128], bsizes=[2000, 5000], Ts=[16], n=1, reps=1, scale=4, max_queries=2_000_000),
        noise=dict(scales=[4], Ts=[4, 16], n_q=2000),
        exact=dict(scales=[4], Ts=[4, 16], n_q=3000, small=500),
        stat=dict(scales=[3.5], chunks=[2000], Ts=[4], n=1),
        advis=dict(n=1, Ts=[4, 16, 64], budget_sets=[[16, 64]], bsize=5000)),
    "full": dict(
        scale=dict(scales=[2, 3, 4, 6, 8, 12], Ts=[4, 16, 64], n=20, bsize=150000, reps=1),
        scal=dict(lr_sizes=[128, 256, 512, 1024], bsizes=[5000, 10000, 25000, 50000, 100000, 150000], Ts=[16], n=3, reps=2,
                  scale=4, max_queries=17_000_000),
        noise=dict(scales=[2, 3.5, 4], Ts=[4, 16, 64], n_q=20000),
        exact=dict(scales=[3.5, 4], Ts=[4, 16, 64], n_q=20000, small=2000),
        stat=dict(scales=[2.5, 3.5, 4], chunks=[5000, 25000, 150000], Ts=[16], n=5),
        advis=dict(n=4, Ts=[4, 8, 16, 32, 64], budget_sets=[[16, 64], [8, 16, 32, 64], [4, 8, 16, 32, 64]], bsize=150000)),
}


def build_jobs(cfg, P, run_dir, nc):
    paths = {k: sl.expand(v) for k, v in cfg["paths"].items()}
    O, F = paths["orig_root"], paths["fast_root"]
    jobs = []

    def base(root, label, mode="none", flags=None, **kw):
        s = dict(root=root, ckpt=paths["checkpoint"], data_root=paths["data_root"], kind="linear", no_compile=bool(nc),
                 mode=mode, scale=4, bsize=150000, n_images=1, Ts=[16], seeds=[0], reps=1, label=label)
        if flags is not None:
            s["flags"] = flags
            s["no_compile"] = bool(nc) or "C" not in flags
        s.update(kw)
        return s

    def add(group, cmd, s, name, script="worker_extra.py"):
        s["out"] = os.path.join(run_dir, group, name + ".json")
        jobs.append(dict(group=group, cmd=cmd, spec=s, name=name, script=script))

    impls = [("orig", O, "none"), ("fast", F, "full")]

    # ---- scales (worker.py eval: records time, psnr, lpips, peak memory and the sharing statistics)
    p = P["scale"]
    for sc in p["scales"]:
        for label, root, mode in impls:
            add("scale", "eval", base(root, label, mode, scale=sc, bsize=p["bsize"], n_images=p["n"], Ts=p["Ts"], reps=p["reps"],
                                      seeds=[0 if label == "orig" else 1]), f"{label}_x{sc}", script="worker.py")

    # ---- LR sizes x queries per call (one process per chunk size: graph pools must not leak)
    p = P["scal"]
    for bs in p["bsizes"]:
        for label, root, mode in impls:
            add("scal", "scal", base(root, label, mode, bsize=bs, n_images=p["n"], Ts=p["Ts"], reps=p["reps"], scale=p["scale"],
                                     lr_sizes=p["lr_sizes"], max_queries=p["max_queries"]), f"{label}_bs{bs}")

    # ---- identical noise. Reference = unmodified upstream; tests = refactored original engine, sharing only (same seed),
    #      full engine with the original's noise replayed (eager+compile; a CUDA graph cannot take replayed noise) and with its own noise
    p = P["noise"]
    tests = [("orig", O, dict(mode="none"), False),
             ("fast_none", F, dict(mode="none"), False),
             ("fast_SF", F, dict(flags="SF"), False),
             ("fast_SFE_replay", F, dict(flags="SFE"), True),
             ("fast_SFEC_replay", F, dict(flags="SFEC"), True),
             ("fast_SFEC_ownnoise", F, dict(flags="SFEC"), False)]
    for sc in p["scales"]:
        for label, root, kw, replay in tests:
            s = base(root, label, scale=sc, bsize=p["n_q"], Ts=p["Ts"], n_q=p["n_q"], image=0, seed=0, replay=replay,
                     dump_dir=os.path.join(run_dir, "noise", f"dump_x{sc}"), **{k: v for k, v in kw.items() if k != "flags"},
                     **({"flags": kw["flags"]} if "flags" in kw else {}))
            if "flags" in kw:
                s["no_compile"] = bool(nc) or "C" not in kw["flags"]
            add("noise", "noise_equiv", s, f"{label}_x{sc}")

    # ---- arrangements with the same noise in every call
    p = P["exact"]
    arr = [("orig", O, dict(mode="none")), ("fast_SF", F, dict(flags="SF")), ("fast_SFEC", F, dict(flags="SFEC")),
           ("fast_SFECG", F, dict(flags="SFECG"))]
    for sc in p["scales"]:
        for label, root, kw in arr:
            s = base(root, label, scale=sc, bsize=p["n_q"], Ts=p["Ts"], n_q=p["n_q"], small_chunk=p["small"], image=0, seed=0,
                     **{k: v for k, v in kw.items() if k != "flags"})
            if "flags" in kw:
                s["flags"] = kw["flags"]
                s["no_compile"] = bool(nc) or "C" not in kw["flags"]
            add("exact", "arrange_exact", s, f"{label}_x{sc}")

    # ---- arrangements, statistical, full images (one process per chunk size)
    p = P["stat"]
    for ch in p["chunks"]:
        for label, root, mode in impls:
            add("stat", "arrange_stat", base(root, label, mode, bsize=ch, n_images=p["n"], Ts=p["Ts"], scales=p["scales"],
                                             chunks=[ch]), f"{label}_chunk{ch}")

    # ---- local quality of adaptive inference (fast implementation)
    p = P["advis"]
    add("advis", "adapt_vis", base(F, "fast", "full", bsize=p["bsize"], n_images=p["n"], Ts=p["Ts"], budget_sets=p["budget_sets"],
                                   tile=8, metric="dct", dump_dir=os.path.join(run_dir, "advis", "dump")), "advis")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "suite_config.yaml"))
    ap.add_argument("--preset", default="full", choices=list(PRESETS))
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {GROUPS}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--report-only", default=None, help="existing run directory")
    a = ap.parse_args()
    if a.report_only:
        return subprocess.run([sys.executable, os.path.join(HERE, "extra_report.py"), a.report_only]).returncode
    cfg = yaml.safe_load(open(a.config))
    # describe the GPU the workers will use (device 0 of this process is otherwise the first physical GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", "0"))
    env = sl.env_info()
    nc = cfg.get("no_compile", "auto")
    if nc == "auto":
        nc = float(env.get("compute_capability", "9.0")) < 7.0
    gpu = (env.get("gpu", "nogpu")).replace(" ", "").replace("NVIDIA", "").replace("GeForce", "")
    run_id = a.run_id or f"extra_{time.strftime('%Y%m%d_%H%M')}_{a.preset}_{gpu}"
    run_dir = os.path.join(a.out, run_id)
    jobs = [j for j in build_jobs(cfg, PRESETS[a.preset], run_dir, nc)
            if j["group"] in (a.only or GROUPS) and j["group"] not in a.skip]
    print(f"run {run_id}: {len(jobs)} jobs, preset={a.preset}, gpu={env.get('gpu')}, no_compile={nc}")
    if a.dry_run:
        for j in jobs:
            s = j["spec"]
            print(f"  [{j['group']:5s}] {j['name']:26s} {j['cmd']:13s} flags={s.get('flags')} mode={s.get('mode')} "
                  f"scale={s.get('scale')} Ts={s.get('Ts')} bs={s.get('bsize')}")
        return
    paths = {k: sl.expand(v) for k, v in cfg["paths"].items()}
    bad = [f"{k}: {paths[k]}" for k in ("orig_root", "fast_root", "checkpoint", "data_root") if not os.path.exists(paths[k])]
    if bad:
        sys.exit("Wrong paths in suite_config.yaml (nothing was run):\n  " + "\n  ".join(bad))
    os.makedirs(run_dir, exist_ok=True)
    sl.write_json(os.path.join(run_dir, "env.json"), dict(env=env, preset=a.preset, config=cfg, no_compile=bool(nc),
                                                          roots={k: dict(path=sl.expand(v), git=sl.git_info(sl.expand(v)))
                                                                 for k, v in cfg["paths"].items() if k.endswith("_root")}))
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.get("gpu", "0")))
    manifest = []
    for n, j in enumerate(jobs, 1):
        out = j["spec"]["out"]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if a.resume and os.path.exists(out):
            print(f"[{n}/{len(jobs)}] {j['group']}/{j['name']}: skipped (exists)")
            continue
        spec_path = out[:-5] + ".spec.json"
        sl.write_json(spec_path, j["spec"])
        t0 = time.time()
        with open(out[:-5] + ".log", "w") as lf:
            try:
                rc = subprocess.run([cfg.get("python", "python"), os.path.join(HERE, j["script"]), j["cmd"], spec_path],
                                    env=child_env, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=cfg.get("job_timeout_s", 21600)).returncode
                status = "ok" if rc == 0 else f"failed(rc={rc})"
            except subprocess.TimeoutExpired:
                status = "timeout"
        print(f"[{n}/{len(jobs)}] {j['group']}/{j['name']}: {status} in {time.time() - t0:.0f}s"
              + ("" if status == "ok" else f"  -> see {out[:-5]}.log"))
        manifest.append(dict(job=j["name"], group=j["group"], status=status, seconds=time.time() - t0))
        sl.write_json(os.path.join(run_dir, "manifest.json"), manifest)
    bad = [m for m in manifest if m["status"] != "ok"]
    print(f"done: {len(manifest) - len(bad)} ok, {len(bad)} failed. results in {run_dir}")
    if not a.no_report:
        subprocess.run([sys.executable, os.path.join(HERE, "extra_report.py"), run_dir])


if __name__ == "__main__":
    main()
