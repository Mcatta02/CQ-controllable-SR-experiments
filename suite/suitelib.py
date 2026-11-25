"""Shared helpers for the evaluation suite (no PyTorch needed except in env_info)."""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand(p):
    """Expand "~"; relative paths are taken relative to the repository root, not the cwd."""
    return os.path.normpath(os.path.join(REPO_ROOT, os.path.expanduser(p)))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_info(root):
    st = _run(["git", "status", "--porcelain"], root)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], root),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty_files": None if st is None else len(st.splitlines()),
    }


def env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["default_dtype"] = str(torch.get_default_dtype())
        info["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        info["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2 ** 30, 1),
                        compute_capability=f"{p.major}.{p.minor}")
    except Exception as e:  # pragma: no cover
        info["torch_error"] = str(e)
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        info["driver"] = drv.splitlines()[0]
    return info


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(x, fn=np.mean, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    s = np.array([fn(x[i]) for i in idx])
    return (float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2)))


def paired_ratio_ci(a, b, n=2000, alpha=0.05, seed=0):
    """CI of mean(a)/mean(b) resampling the (paired) items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    point = float(a.mean() / b.mean())
    if m < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    r = a[idx].mean(1) / b[idx].mean(1)
    return point, float(np.quantile(r, alpha / 2)), float(np.quantile(r, 1 - alpha / 2))


def linfit(x, y):
    """y = a + b x. Returns (a, b)."""
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(a), float(b)


def linfit_ci(x, Y, n=1000, alpha=0.05, seed=0):
    """Y: array [n_items, n_x] (per-item values at each x). Bootstrap over items.
    Returns dict with a, b and their (lo, hi) CIs."""
    Y = np.asarray(Y, float)
    a, b = linfit(x, Y.mean(0))
    if Y.shape[0] < 2:
        return dict(a=a, b=b, a_ci=(a, a), b_ci=(b, b))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for _ in range(n):
        idx = rng.integers(0, Y.shape[0], size=Y.shape[0])
        aa, bb = linfit(x, Y[idx].mean(0))
        A.append(aa); B.append(bb)
    q = lambda v: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)))
    return dict(a=a, b=b, a_ci=q(A), b_ci=q(B))


def fmt_ci(v, ci, nd=2):
    return f"{v:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
