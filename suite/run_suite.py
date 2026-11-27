#!/usr/bin/env python
"""
Orchestrator. Builds the list of worker jobs for each test group, runs them one after another
(never in parallel: timings need an otherwise idle GPU) and stores one JSON per job.

  python run_suite.py --preset quick                 # mock-up on a small GPU
  python run_suite.py --preset full --report         # final run
  python run_suite.py --preset quick --only equiv speed_T --dry-run
  python run_suite.py --preset quick --resume --run-id <existing>    # continue an interrupted run
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

ABL_FLAGS = ["S", "F", "SF", "SFC", "E", "EC", "EG", "ECG",
             "FECG", "SECG", "SFEG", "SFEC", "SFECG"]      # single, pairs, all, and leave-one-out
ALL_TESTS = ["main", "equiv", "speed_T", "scale", "ablation", "stages", "memory", "kind", "spec", "resolution", "adaptive"]


def build_jobs(cfg, P, run_dir, no_compile):
    paths = {k: sl.expand(v) for k, v in cfg["paths"].items()}
    jobs = []

    def spec(root, label, mode, test, **kw):
        s = dict(root=root, ckpt=paths["checkpoint"], data_root=paths["data_root"], kind="linear",
                 no_compile=no_compile, mode=mode, scale=P.get("scale_main", 4), bsize=P["eval_bsize"], n_images=P["n_main"],
                 Ts=P["Ts_main"], seeds=[0], reps=1, label=label)
        s.update(kw)
        name = kw.pop("_name", label)
        s["out"] = os.path.join(run_dir, test, s.get("_name", label) + ".json")
        s.pop("_name", None)
        return s

    def add(test, cmd, s, name=None):
        if name:
            s["out"] = os.path.join(run_dir, test, name + ".json")
        jobs.append(dict(test=test, cmd=cmd, spec=s, name=os.path.basename(s["out"])[:-5]))

    O, F, X = paths["orig_root"], paths["fast_root"], paths["exp_root"]

    def impls():
        return [("orig", O, "none"), ("fast", F, "full")]

    # ---- main table: all images, one seed each
    for label, root, mode in impls():
        add("main", "eval", spec(root, label, mode, "main", seeds=[0 if label == "orig" else 1]))
    # ---- statistical equivalence: several seeds per implementation, pixel dumps
    for label, root, mode in impls():
        add("equiv", "eval", spec(root, label, mode, "equiv", n_images=P["n_equiv"], Ts=P["Ts_equiv"],
                                  seeds=P["equiv_seeds"],
                                  dump_dir=os.path.join(run_dir, "equiv", "dumps"), n_dump=P["n_dump"]))
    # ---- time versus T
    for label, root, mode in impls():
        add("speed_T", "eval", spec(root, label, mode, "speed_T", n_images=P["n_time"], Ts=P["Ts_sweep"],
                                    reps=P["reps"]))
    # ---- speed-up versus scale
    for s in P["scales"]:
        for label, root, mode in impls():
            add("scale", "eval", spec(root, label, mode, "scale", scale=s, n_images=P["n_scale"],
                                      Ts=[P["T_scale"]], reps=P["reps"]), name=f"{label}_x{s}")
    # ---- ablation of the individual components (mode names: see worker.apply_mode)
    #   flags: S sharing, F fusion, E fast engine (SPE tables etc.), C torch.compile, G CUDA graph
    abl = [("orig", O, "none", None), ("fast_none", F, "none", None)]
    abl += [(f"abl_{fl}", F, "flags", fl) for fl in ABL_FLAGS]
    for label, root, mode, fl in abl:
        kw = dict(n_images=P["n_abl"], Ts=P["Ts_abl"], reps=P["reps"])
        if fl is not None:
            kw.update(flags=fl, no_compile=bool(no_compile) or "C" not in fl)
        add("ablation", "eval", spec(root, label, mode, "ablation", **kw))
    # ---- time / peak memory versus image size (timing only)
    for label, root, mode in impls():
        add("resolution", "resolution", spec(root, label, mode, "resolution", n_images=1, Ts=[P["T_res"]],
                                             lr_sizes=P["res_sizes"], reps=P["reps"]))
    # ---- stage breakdown
    for label, root, mode in impls():
        add("stages", "stages", spec(root, label, mode, "stages", Ts=P["Ts_stages"], reps=2, n_images=1))
    # ---- time/memory versus chunk size (one process per chunk size: graph pools must not leak)
    for bs in P["mem_bsizes"]:
        for label, root, mode in impls():
            add("memory", "memory", spec(root, label, mode, "memory", bsize=bs, Ts=P["mem_Ts"], n_images=1,
                                         n_points=P["mem_points"], reps=2), name=f"{label}_bs{bs}")
    # ---- parallel baseline (LTE, random weights): the paper's comparison model
    for bs in P["mem_bsizes"]:
        add("memory", "memory", spec(O, "lte", "none", "memory", bsize=bs, Ts=[64], n_images=1, n_points=P["mem_points"], reps=2,
                                     model_yaml=os.path.join(O, "configs", "train-div2k", "lte.yaml")), name=f"lte_bs{bs}")
    # ---- linear versus conv pointwise layers (fast implementation only)
    for kind in ("linear", "conv"):
        for bs in P["kind_bsizes"]:
            add("kind", "memory", spec(F, f"fast_{kind}", "full", "kind", kind=kind, bsize=bs, Ts=[P["T_kind"]],
                                       n_images=1, n_points=P["mem_points"], reps=2), name=f"fast_{kind}_bs{bs}")
    # ---- adaptive budgets at matched average T (fast implementation, all components)
    for bset in P["adapt_budgets"]:
        tag = "-".join(str(b) for b in bset)
        add("adaptive", "adaptive", spec(F, f"adaptive_{tag}", "full", "adaptive", n_images=P["n_adapt"], Ts=[64],
                                         budgets=bset, metrics=["sobel", "variance", "dct", "random"], tile=8,
                                         reps=1), name=f"b{tag}")
    # ---- fixed T = 4, 8, 16, 32 (64 is fixed_max) on the same images, for the Pareto comparison
    add("adaptive", "adaptive", spec(F, "adaptive_fixedT", "full", "adaptive", n_images=P["n_adapt"], Ts=[64],
                                     budgets=[64], metrics=[], fixed_Ts=[4, 8, 16, 32], tile=8, reps=1), name="fixedT")
    # ---- negative result (needs models/speculative.py)
    add("spec", "spec", spec(X, "spec", "none", "spec", n_images=1, T=64, n_tokens=P["n_tokens"], Ks=[2, 4, 8, 16]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "suite_config.yaml"))
    ap.add_argument("--preset", default="quick", choices=["quick", "full"])
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {ALL_TESTS}")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    P = cfg["presets"][a.preset]
    chk = {k: sl.expand(v) for k, v in cfg["paths"].items()}
    bad = [f"{k}: {chk[k]}" for k in ("orig_root", "fast_root", "exp_root", "checkpoint", "data_root") if not os.path.exists(chk[k])]
    if not bad and not [f for f in os.listdir(chk["data_root"]) if f.lower().endswith(".png")]:
        bad.append(f"data_root: {chk['data_root']} contains no .png files")
    if bad and not a.dry_run:
        sys.exit("Wrong paths in suite_config.yaml (nothing was run):\n  " + "\n  ".join(bad))
    # describe the GPU the workers will use (device 0 of this process is otherwise the first physical GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", "0"))
    env = sl.env_info()
    nc = cfg.get("no_compile", "auto")
    if nc == "auto":
        cc = env.get("compute_capability", "9.0")
        nc = float(cc) < 7.0
    gpu_slug = (env.get("gpu", "nogpu")).replace(" ", "").replace("NVIDIA", "").replace("GeForce", "")
    run_id = a.run_id or f"{time.strftime('%Y%m%d_%H%M')}_{a.preset}_{gpu_slug}"
    run_dir = os.path.join(a.out, run_id)
    jobs = build_jobs(cfg, P, run_dir, bool(nc))
    sel = a.only or ALL_TESTS
    jobs = [j for j in jobs if j["test"] in sel and j["test"] not in a.skip]
    print(f"run {run_id}: {len(jobs)} jobs, preset={a.preset}, no_compile={nc}, gpu={env.get('gpu')}")
    if a.dry_run:
        for j in jobs:
            s = j["spec"]
            print(f"  [{j['test']:9s}] {j['name']:24s} {j['cmd']:8s} root={os.path.basename(s['root'])} mode={s['mode']} "
                  f"scale={s.get('scale')} Ts={s.get('Ts')} n={s.get('n_images')} bs={s.get('bsize')} seeds={s.get('seeds')}")
        return

    os.makedirs(run_dir, exist_ok=True)
    sl.write_json(os.path.join(run_dir, "env.json"), dict(env=env, config=cfg, preset=a.preset, no_compile=bool(nc),
                                                          roots={k: dict(path=sl.expand(v), git=sl.git_info(sl.expand(v)))
                                                                 for k, v in cfg["paths"].items() if k.endswith("_root")}))
    child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.get("gpu", "0")))
    worker = os.path.join(HERE, "worker.py")
    manifest = []
    for n, j in enumerate(jobs, 1):
        out = j["spec"]["out"]
        spec_path = out[:-5] + ".spec.json"
        if a.resume and os.path.exists(out):
            print(f"[{n}/{len(jobs)}] {j['test']}/{j['name']}: skipped (exists)")
            manifest.append(dict(job=j["name"], test=j["test"], status="skipped"))
            continue
        sl.write_json(spec_path, j["spec"])
        t0 = time.time()
        with open(out[:-5] + ".log", "w") as lf:
            try:
                rc = subprocess.run([cfg.get("python", "python"), worker, j["cmd"], spec_path], env=child_env,
                                    stdout=lf, stderr=subprocess.STDOUT, timeout=cfg.get("job_timeout_s", 21600)).returncode
                status = "ok" if rc == 0 else f"failed(rc={rc})"
            except subprocess.TimeoutExpired:
                status = "timeout"
        dt = time.time() - t0
        print(f"[{n}/{len(jobs)}] {j['test']}/{j['name']}: {status} in {dt:.0f}s" + ("" if status == "ok" else f"  -> see {out[:-5]}.log"))
        manifest.append(dict(job=j["name"], test=j["test"], status=status, seconds=dt))
        sl.write_json(os.path.join(run_dir, "manifest.json"), manifest)
    bad = [m for m in manifest if m["status"] not in ("ok", "skipped")]
    print(f"done: {len(manifest) - len(bad)} ok, {len(bad)} failed. results in {run_dir}")
    if a.report:
        subprocess.run([sys.executable, os.path.join(HERE, "make_report.py"), run_dir])


if __name__ == "__main__":
    main()
