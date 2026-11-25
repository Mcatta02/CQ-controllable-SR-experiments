#!/usr/bin/env python
"""
Checks everything the suite needs BEFORE a long run, and prints PASS / WARN / FAIL per item.
    python env_check.py [--config suite_config.yaml]
"""
import argparse
import glob
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suitelib as sl  # noqa: E402

rows = []


def rec(status, item, detail=""):
    rows.append((status, item, detail))
    print(f"[{status:4s}] {item}" + (f": {detail}" if detail else ""))


def sub(root, code, timeout=240):
    """Run python code with `root` as cwd/first sys.path entry (its own models/utils)."""
    pre = f"import os,sys; os.chdir({root!r}); sys.path.insert(0,{root!r}); "
    r = subprocess.run([sys.executable, "-c", pre + code], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()[-600:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "suite_config.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    P = {k: sl.expand(v) for k, v in cfg["paths"].items()}

    env = sl.env_info()
    rec("PASS" if "gpu" in env else "FAIL", "GPU", f"{env.get('gpu')} (cc {env.get('compute_capability')}, {env.get('gpu_mem_gb')} GB), "
        f"torch {env.get('torch')}, CUDA {env.get('cuda')}, driver {env.get('driver')}")
    try:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        apps = ""
    rec("WARN" if apps else "PASS", "GPU idle", "other processes: " + apps.replace("\n", "; ") if apps else "no other compute processes (timings are only valid on an idle GPU)")
    try:
        import torch
        cc = float(env.get("compute_capability", "0"))
        try:
            f = torch.compile(lambda x: x * 2 + 1)
            f(torch.ones(4, device="cuda"))
            rec("PASS", "torch.compile (Inductor/Triton)", "works")
        except Exception as e:
            rec("WARN" if cc < 7.0 else "FAIL", "torch.compile (Inductor/Triton)",
                f"not available ({str(e).splitlines()[0][:120]}). The suite runs the fast path with compile disabled; "
                "the 'compiled step' contribution can only be measured on a newer GPU.")
        g = torch.cuda.CUDAGraph(); x = torch.ones(8, device="cuda")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = x * 2
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y = x * 2
        g.replay(); torch.cuda.synchronize()
        rec("PASS", "CUDA graphs", "capture and replay work")
    except Exception as e:
        rec("FAIL", "torch/CUDA", str(e)[:200])

    for k in ("orig_root", "fast_root", "exp_root"):
        ok = os.path.isdir(P[k]) and os.path.isfile(os.path.join(P[k], "models", "liif_base.py"))
        gi = sl.git_info(P[k]) if ok else {}
        rec("PASS" if ok else "FAIL", k, f"{P[k]}  commit={str(gi.get('commit'))[:8]} dirty_files={gi.get('dirty_files')}" if ok else f"{P[k]} (missing, or has no models/liif_base.py)")
    if P["orig_root"] == P["fast_root"]:
        rec("FAIL", "orig_root vs fast_root", "they are the same folder: the comparison would be meaningless")

    ck = P["checkpoint"]
    if os.path.isfile(ck):
        rc, out = sub(P["orig_root"], f"import torch; m=torch.load({ck!r},map_location='cpu')['model']; "
                      f"print(m['name'], sorted(m['args'].keys()), 'keys in sd:', len(m['sd']))")
        rec("PASS" if rc == 0 else "FAIL", "checkpoint", f"{ck}  sha256={sl.sha256_file(ck)[:16]}  | {out[-160:]}")
    else:
        rec("FAIL", "checkpoint", f"{ck} not found")

    dr = P["data_root"]
    n = len(glob.glob(os.path.join(dr, "*.png"))) if os.path.isdir(dr) else 0
    rec("PASS" if n >= 100 else ("WARN" if n > 0 else "FAIL"), "DIV2K validation images", f"{n} png files in {dr} (final run needs 100)")

    rc, out = sub(P["orig_root"], f"import models, datasets, utils; print('ok')")
    rec("PASS" if rc == 0 else "FAIL", "orig_root imports (models, datasets, utils)", "" if rc == 0 else out[-250:])
    rc, out = sub(P["fast_root"],
                  "import inspect, models, utils; from models.fast_recurrent import install; from models.fast_post import install_fast_post; "
                  "from models.liif_base import LIIFBase; "
                  "print('install', inspect.signature(install), '| install_fast_post', inspect.signature(install_fast_post), '| _fid:', hasattr(LIIFBase,'_fid'))")
    rec("PASS" if rc == 0 else "FAIL", "fast_root has the fast path", out[-250:])
    rc, out = sub(P["exp_root"], "import models.speculative as sp; print([n for n in ('replay_noise','pe_tables','net_verify','generate_jacobi','generate_speculative') if hasattr(sp,n)])")
    rec("PASS" if rc == 0 else "WARN", "exp_root has models/speculative.py", out[-200:] if rc == 0 else "missing -> the negative-result tests will fail (point exp_root to the experiments folder)")
    rc, out = sub(P["orig_root"], "from lpips import LPIPS; LPIPS(net='alex'); print('ok')")
    rec("PASS" if rc == 0 else "FAIL", "LPIPS weights", "" if rc == 0 else out[-200:])
    free = os.statvfs(os.path.expanduser("~")).f_bavail * os.statvfs(os.path.expanduser("~")).f_frsize / 2 ** 30
    rec("PASS" if free > 5 else "WARN", "free disk space (home)", f"{free:.0f} GB")

    sl.write_json(os.path.join(HERE, "results", "env_check.json"), dict(env=env, paths=P, rows=rows))
    bad = [r for r in rows if r[0] == "FAIL"]
    print(f"\n{len(rows) - len(bad)} checks ok or warned, {len(bad)} FAILED." + ("  Fix the FAIL items before running the suite." if bad else "  Ready."))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
