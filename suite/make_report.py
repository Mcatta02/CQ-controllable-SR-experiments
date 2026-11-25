#!/usr/bin/env python
"""
Turns one run directory (results/<run_id>/) into paper-ready figures and tables.

    python make_report.py results/<run_id>

Writes <run>/figs/*.pdf|png and <run>/tables/*.tex|csv and <run>/summary.json.
Every figure is optional: if the data for it is missing, it is skipped with a warning.
Print-readable sizes (single column 3.3 in, double column 6.9 in), serif fonts, and every series
differs by marker AND line style, so nothing relies on colour alone.
"""
import glob
import json
import os
import sys
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import suitelib as sl  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
# Okabe-Ito colour-blind-safe palette
COL = dict(lte="#009E73", orig="#4D4D4D", fast="#D55E00", fast_none="#0072B2", fast_graph="#009E73", fast_post="#CC79A7",
           fast_full="#D55E00", linear="#0072B2", conv="#E69F00")
MRK = dict(lte="^", orig="o", fast="s", fast_none="^", fast_graph="D", fast_post="v", fast_full="s", linear="o", conv="s")
LST = dict(lte=":", orig="--", fast="-", fast_none="-.", fast_graph=":", fast_post=(0, (5, 1)), fast_full="-", linear="-", conv="--")
NAME = dict(lte="LTE (parallel baseline)", orig="Original", fast="Accelerated", fast_none="Earlier optimisations only",
            fast_graph="+ compiled step / graph only", fast_post="+ sharing & fusion only",
            fast_full="All (accelerated)", linear="linear layers", conv="conv layers")
W1, W2 = 3.3, 6.9


class Run:
    def __init__(self, d):
        self.d = d
        p = os.path.join(d, "env.json")
        self.env = sl.read_json(p) if os.path.exists(p) else {}
        self.synthetic = bool(self.env.get("synthetic"))
        self.gpu = (self.env.get("env", {}) or {}).get("gpu", "unknown GPU")
        self.preset = self.env.get("preset", "?")
        os.makedirs(os.path.join(d, "figs"), exist_ok=True)
        os.makedirs(os.path.join(d, "tables"), exist_ok=True)

    def job(self, test, name):
        p = os.path.join(self.d, test, name + ".json")
        if not os.path.exists(p):
            return None
        j = sl.read_json(p)
        return j if j.get("ok", True) else None

    def recs(self, test, name):
        j = self.job(test, name)
        return [] if j is None else j.get("records", [])

    def names(self, test, prefix=""):
        return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(self.d, test, prefix + "*.json"))
                      if not p.endswith(".spec.json"))

    def save(self, fig, name):
        if self.synthetic:
            fig.text(0.5, 0.5, "SYNTHETIC DATA", ha="center", va="center", fontsize=28, color="red", alpha=0.18, rotation=25)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(self.d, "figs", f"{name}.{ext}"), bbox_inches="tight")
        plt.close(fig)
        print(f"  figure: {name}")

    def tex(self, name, header, rows, caption, label, colspec=None):
        colspec = colspec or ("l" + "c" * (len(header) - 1))
        lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                 f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
        lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        with open(os.path.join(self.d, "tables", name + ".tex"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(self.d, "tables", name + ".csv"), "w") as f:
            f.write(",".join(h.replace(",", ";") for h in header) + "\n")
            for r in rows:
                f.write(",".join(str(c).replace(",", ";") for c in r) + "\n")
        print(f"  table:  {name}")


def mat(records, key, Ts):
    """[n_images, len(Ts)] matrix of `key` averaged over seeds/reps; NaN where missing."""
    recs = [r for r in records if key in r and not r.get("oom") and r.get("seed", 0) >= 0]
    imgs = sorted({r["image"] for r in recs})
    M = np.full((len(imgs), len(Ts)), np.nan)
    for ti, T in enumerate(Ts):
        for ii, im in enumerate(imgs):
            v = [r[key] for r in recs if r["T"] == T and r["image"] == im]
            if v:
                M[ii, ti] = np.mean(v)
    return M, imgs


def Tlist(records):
    return sorted({r["T"] for r in records if "T" in r})


def safe(fn, run, *a):
    try:
        fn(run, *a)
    except Exception as e:  # keep going: one missing test must not stop the report
        print(f"  [skip] {fn.__name__}: {e!r}")
        if os.environ.get("REPORT_DEBUG"):
            traceback.print_exc()


SUMMARY = {}


# ===================================================================== speed
def fig_time_vs_T(run):
    ro, rf = run.recs("speed_T", "orig"), run.recs("speed_T", "fast")
    Ts = sorted(set(Tlist(ro)) & set(Tlist(rf)))
    if not Ts:
        raise ValueError("no speed_T data")
    Mo, _ = mat(ro, "time_s", Ts)
    Mf, _ = mat(rf, "time_s", Ts)
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(W2, 2.4))
    fits = {}
    for lab, M in (("orig", Mo), ("fast", Mf)):
        mean = np.nanmean(M, 0)
        ci = np.array([sl.bootstrap_ci(M[:, k][~np.isnan(M[:, k])]) for k in range(len(Ts))])
        ax[0].fill_between(Ts, ci[:, 0], ci[:, 1], color=COL[lab], alpha=0.2, lw=0)
        f = sl.linfit_ci(Ts, np.nan_to_num(M, nan=np.nanmean(M)))
        fits[lab] = f
        xx = np.array([min(Ts), max(Ts)])
        ax[0].plot(xx, f["a"] + f["b"] * xx, color=COL[lab], ls=LST[lab], lw=0.8, alpha=0.8)
        ax[0].plot(Ts, mean, marker=MRK[lab], color=COL[lab], ls="none", ms=4,
                   label=f"{NAME[lab]}: {f['a']:.2f} s + {f['b'] * 1000:.1f} ms $\\cdot T$")
    ax[0].set_xlabel("Number of components $T$"); ax[0].set_ylabel("Time per image (s)")
    ax[0].set_title("(a) Time vs. $T$ (points: mean, band: 95% CI, line: linear fit)")
    ax[0].legend(frameon=False, loc="upper left")
    sp = [sl.paired_ratio_ci(Mo[:, k], Mf[:, k]) for k in range(len(Ts))]
    ax[1].errorbar(Ts, [s[0] for s in sp], yerr=[[s[0] - s[1] for s in sp], [s[2] - s[0] for s in sp]], marker="s",
                   color=COL["fast"], capsize=2, ms=4)
    ax[1].set_xlabel("Number of components $T$"); ax[1].set_ylabel("Speed-up (original / accelerated)")
    ax[1].set_title("(b) Speed-up (95% CI, paired by image)")
    ax[1].axhline(1, color="k", lw=0.5, ls=":")
    run.save(fig, "fig_time_vs_T")
    SUMMARY["fit"] = {k: dict(a_s=v["a"], b_ms=v["b"] * 1000, a_ci=v["a_ci"], b_ci_ms=[x * 1000 for x in v["b_ci"]]) for k, v in fits.items()}
    SUMMARY["speedup_vs_T"] = {int(T): s[0] for T, s in zip(Ts, sp)}
    rows = [[lab.replace("_", " "), sl.fmt_ci(fits[lab]["a"], fits[lab]["a_ci"], 2), sl.fmt_ci(fits[lab]["b"] * 1000, [x * 1000 for x in fits[lab]["b_ci"]], 1)] for lab in ("orig", "fast")]
    rows.append(["ratio", f"{fits['orig']['a'] / fits['fast']['a']:.1f}$\\times$", f"{fits['orig']['b'] / fits['fast']['b']:.1f}$\\times$"])
    run.tex("tab_fit", ["Implementation", "Fixed cost $a$ (s) [95\\% CI]", "Cost per component $b$ (ms) [95\\% CI]"], rows,
            "Time per image modelled as $a+bT$ (least squares over the sweep; CI by bootstrap over images).", "tab:fit")


def tab_main(run):
    ro, rf = run.recs("main", "orig"), run.recs("main", "fast")
    Ts = sorted(set(Tlist(ro)) & set(Tlist(rf)))
    if not Ts:
        raise ValueError("no main data")
    rows = []
    for k, T in enumerate(Ts):
        d = {}
        for lab, R in (("orig", ro), ("fast", rf)):
            d[lab] = {key: mat(R, key, [T])[0][:, 0] for key in ("psnr", "lpips", "time_s")}
        sp = sl.paired_ratio_ci(d["orig"]["time_s"], d["fast"]["time_s"])
        dp = d["fast"]["psnr"] - d["orig"]["psnr"]
        ci = sl.bootstrap_ci(dp)
        for lab in ("orig", "fast"):
            rows.append([T if lab == "orig" else "", NAME[lab], f"{np.mean(d[lab]['psnr']):.2f}", f"{np.mean(d[lab]['lpips']):.3f}",
                         f"{np.mean(d[lab]['time_s']):.2f}", "1.0$\\times$" if lab == "orig" else f"{sp[0]:.1f}$\\times$ [{sp[1]:.1f}, {sp[2]:.1f}]"])
        SUMMARY.setdefault("main", {})[int(T)] = dict(speedup=sp[0], speedup_ci=sp[1:], dpsnr_mean=float(dp.mean()), dpsnr_ci=ci,
                                                       psnr_orig=float(d["orig"]["psnr"].mean()), psnr_fast=float(d["fast"]["psnr"].mean()))
    run.tex("tab_main", ["$T$", "Implementation", "PSNR (dB)", "LPIPS", "Time (s)", "Speed-up [95\\% CI]"], rows,
            f"Results on the DIV2K validation images ($\\times4$, released checkpoint, {run.gpu}); time per image.", "tab:main", "rlcccc")
    # Bland-Altman style plot of the per-image PSNR differences
    pick = sorted({Ts[0], Ts[len(Ts) // 2], Ts[-1]})
    fig, ax = plt.subplots(1, len(pick), constrained_layout=True, figsize=(W2, 2.2), squeeze=False, sharey=True)
    for a_, T in zip(ax[0], pick):
        po = mat(ro, "psnr", [T])[0][:, 0]; pf = mat(rf, "psnr", [T])[0][:, 0]
        d_, m_ = pf - po, (pf + po) / 2
        a_.scatter(m_, d_, s=8, color=COL["fast"], marker="s")
        s_ = d_.std(ddof=1) if len(d_) > 1 else 0
        for y, ls in ((d_.mean(), "-"), (d_.mean() + 1.96 * s_, "--"), (d_.mean() - 1.96 * s_, "--")):
            a_.axhline(y, color="k", lw=0.6, ls=ls)
        a_.set_title(f"$T={T}$: mean {d_.mean():+.4f} dB"); a_.set_xlabel("Mean PSNR of the two (dB)")
    ax[0][0].set_ylabel("PSNR(accel.) $-$ PSNR(orig.) (dB)")
    run.save(fig, "fig_psnr_agreement")


# ===================================================================== ablation / scale / stages
def fig_ablation(run):
    labs = [l for l in ("orig", "fast_none", "fast_graph", "fast_post", "fast_full") if run.recs("ablation", l)]
    if len(labs) < 2:
        raise ValueError("no ablation data")
    Ts = sorted(set.intersection(*[set(Tlist(run.recs("ablation", l))) for l in labs]))
    fig, axs = plt.subplots(1, len(Ts), constrained_layout=True, figsize=(W2, 2.3), squeeze=False, sharey=True)
    rows = []
    for ax, T in zip(axs[0], Ts):
        times = {l: mat(run.recs("ablation", l), "time_s", [T])[0][:, 0] for l in labs}
        base = times["orig"].mean() if "orig" in times else np.nan
        for i, l in enumerate(labs):
            m = times[l].mean(); ci = sl.bootstrap_ci(times[l])
            ax.barh(i, m, xerr=[[m - ci[0]], [ci[1] - m]], color=COL.get(l, "gray"), edgecolor="k", lw=0.5,
                    hatch="///" if l == "orig" else None, error_kw=dict(lw=0.6, capsize=2))
            ax.text(ci[1], i, f"  {m:.2f} s ({base / m:.1f}$\\times$)", va="center", fontsize=6.5)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels([NAME[l] for l in labs]); ax.invert_yaxis()
        ax.set_xlabel("Time per image (s)"); ax.set_title(f"$T={T}$")
        ax.set_xlim(0, max(times[l].mean() for l in labs) * 1.6)
        ax.tick_params(labelleft=(T == Ts[0]))
        rows += [[T, NAME[l], f"{times[l].mean():.2f}", f"{base / times[l].mean():.1f}$\\times$"] for l in labs]
        SUMMARY.setdefault("ablation", {})[int(T)] = {l: float(times[l].mean()) for l in labs}
    run.save(fig, "fig_ablation")
    run.tex("tab_ablation", ["$T$", "Configuration", "Time (s)", "Speed-up"], rows,
            "Contribution of each component to the speed-up (time per image).", "tab:ablation", "rlcc")


def fig_scale(run):
    names = run.names("scale", "orig_x")
    scales = sorted(float(n.split("_x")[1]) for n in names)
    if not scales:
        raise ValueError("no scale data")
    to, tf, sp, lo, hi = [], [], [], [], []
    for s in scales:
        a = mat(run.recs("scale", f"orig_x{s:g}"), "time_s", [run.recs("scale", f"orig_x{s:g}")[0]["T"]])[0][:, 0] if run.recs("scale", f"orig_x{s:g}") else None
        b = mat(run.recs("scale", f"fast_x{s:g}"), "time_s", [run.recs("scale", f"fast_x{s:g}")[0]["T"]])[0][:, 0] if run.recs("scale", f"fast_x{s:g}") else None
        if a is None or b is None:
            continue
        to.append(a.mean()); tf.append(b.mean()); r = sl.paired_ratio_ci(a, b); sp.append(r[0]); lo.append(r[1]); hi.append(r[2])
    scales = scales[:len(sp)]
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(W2, 2.3))
    ax[0].plot(scales, to, marker="o", ls="--", color=COL["orig"], label=NAME["orig"])
    ax[0].plot(scales, tf, marker="s", color=COL["fast"], label=NAME["fast"])
    ax[0].set_xlabel("Upscaling factor $s$"); ax[0].set_ylabel("Time per image (s)"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) Time vs. scale")
    ax[1].errorbar(scales, sp, yerr=[np.array(sp) - lo, np.array(hi) - sp], marker="s", color=COL["fast"], capsize=2, label="Measured speed-up")
    xs = np.linspace(min(scales), max(scales), 50)
    ax[1].plot(xs, xs ** 2, "k:", label="$s^2$ (fewer rollouts, ideal)")
    ax[1].set_xlabel("Upscaling factor $s$"); ax[1].set_ylabel("Speed-up"); ax[1].legend(frameon=False)
    ax[1].set_title("(b) Speed-up vs. scale")
    run.save(fig, "fig_scale")
    SUMMARY["scale"] = {float(s): float(v) for s, v in zip(scales, sp)}


def fig_stages(run):
    ro, rf = run.recs("stages", "orig"), run.recs("stages", "fast")
    Ts = sorted(set(Tlist(ro)) & set(Tlist(rf)))
    if not Ts:
        raise ValueError("no stages data")
    comps = [("gen_feat_ms", "encoder"), ("fid_ms", "pixel lookup"), ("rest_ms", "predictor (rest)"),
             ("recurrent_ms", "rollout"), ("decoder_ms", "decoder"), ("other_ms", "other (copies, ...)")]
    hatches = [None, "..", "\\\\", "//", "xx", "--"]
    cols = ["#999999", "#E69F00", "#56B4E9", "#D55E00", "#009E73", "#CCCCCC"]
    fig, axs = plt.subplots(1, len(Ts), constrained_layout=True, figsize=(W2, 2.2), squeeze=False, sharey=True)
    for ax, T in zip(axs[0], Ts):
        for yi, (lab, R) in enumerate((("orig", ro), ("fast", rf))):
            r = [x for x in R if x["T"] == T][0]
            vals = dict(r); vals["rest_ms"] = max(r["predictor_ms"] - r["recurrent_ms"], 0)
            left = 0
            for (k, nm), h, c in zip(comps, hatches, cols):
                v = vals.get(k, 0) or 0
                ax.barh(yi, v / 1000, left=left, color=c, hatch=h, edgecolor="k", lw=0.4, label=nm if yi == 0 else None)
                left += v / 1000
            ax.text(left, yi, f" {left:.2f} s", va="center", fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels([NAME["orig"], NAME["fast"]]); ax.invert_yaxis()
        ax.set_xlabel("Time per image (s)"); ax.set_title(f"$T={T}$"); ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
        ax.tick_params(labelleft=(T == Ts[0]))
    axs[0][0].legend(frameon=False, fontsize=6, loc="lower right")
    run.save(fig, "fig_stages")


# ===================================================================== memory / kind
def _mem_rows(run, test, labs):
    out = {}
    for lab in labs:
        for n in run.names(test, lab + "_bs"):
            for r in run.recs(test, n):
                out.setdefault(lab, []).append(r)
    return out


def fig_memory(run):
    data = _mem_rows(run, "memory", ["orig", "fast", "lte"])
    if not data:
        raise ValueError("no memory data")
    Tmax = max(r["T"] for rs in data.values() for r in rs)
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(W2, 2.4))
    cap = (run.env.get("env", {}) or {}).get("gpu_mem_gb")
    for lab, rs in data.items():
        pts = sorted([r for r in rs if r["T"] == Tmax and not r.get("oom")], key=lambda r: r["bsize"])
        oom = sorted([r["bsize"] for r in rs if r["T"] == Tmax and r.get("oom")])
        if pts:
            ax[0].plot([p["peak_mb"] for p in pts], [p["ms_per_100k"] for p in pts], marker=MRK[lab], ls=LST[lab], color=COL[lab], label=NAME[lab], ms=4)
            for p in pts:
                ax[0].annotate(f"{p['bsize'] / 1000:g}k", (p["peak_mb"], p["ms_per_100k"]), fontsize=5.5, xytext=(2, 3), textcoords="offset points")
            ax[1].plot([p["bsize"] for p in pts], [p["peak_mb"] for p in pts], marker=MRK[lab], ls=LST[lab], color=COL[lab], label=NAME[lab], ms=4)
        if oom:
            ax[1].plot(oom, [cap * 1024 if cap else np.nan] * len(oom), "x", color=COL[lab], ms=6, label=f"{NAME[lab]}: out of memory")
    for a in ax[:2]:
        a.set_xscale("log")
    ax[0].set_yscale("log"); ax[1].set_yscale("log"); ax[1].set_xscale("log")
    ax[0].set_xlabel("Peak GPU memory (MB)"); ax[0].set_ylabel(f"ms per 100k queries ($T={Tmax}$)"); ax[0].set_title("(a) Time vs. memory")
    ax[1].set_xlabel("Queries per call"); ax[1].set_ylabel("Peak GPU memory (MB)"); ax[1].set_title("(b) Memory vs. queries per call")
    ax[0].legend(frameon=False, loc="upper right"); ax[1].legend(frameon=False, fontsize=6, loc="upper left")
    # (c) memory vs T at the largest chunk size for which every implementation completed every T
    bad = {r["bsize"] for rs in data.values() for r in rs if r.get("oom")}
    cands = sorted({r["bsize"] for rs in data.values() for r in rs} - bad)
    common_bs = cands[-1] if cands else min(r["bsize"] for rs in data.values() for r in rs)
    for lab, rs in data.items():
        ok = [r for r in rs if not r.get("oom")]
        if not ok:
            continue
        bs = common_bs
        pts = sorted([r for r in ok if r["bsize"] == bs], key=lambda r: r["T"])
        ax[2].plot([p["T"] for p in pts], [p["peak_mb"] for p in pts], marker=MRK[lab], ls=LST[lab], color=COL[lab], ms=4, label=f"{NAME[lab]} ({bs / 1000:g}k/call)")
    ax[2].set_xlabel("Number of components $T$"); ax[2].set_ylabel("Peak GPU memory (MB)"); ax[2].set_title("(c) Memory vs. $T$"); ax[2].legend(frameon=False, fontsize=6)
    run.save(fig, "fig_memory")
    rows = []
    bss = sorted({r["bsize"] for rs in data.values() for r in rs})
    for bs in bss:
        row = [f"{bs:,}"]
        for lab in ("orig", "fast", "lte"):
            r = [x for x in data.get(lab, []) if x["bsize"] == bs and x["T"] == Tmax]
            row.append("OOM" if r and r[0].get("oom") else (f"{r[0]['ms_per_100k']:.0f} / {r[0]['peak_mb']:.0f}" if r else "--"))
        rows.append(row)
    run.tex("tab_memory", ["Queries per call", "Original (ms / MB)", "Accelerated (ms / MB)", "LTE (ms / MB)"], rows,
            f"Time per 100,000 queries and peak GPU memory at $T={Tmax}$.", "tab:memory")


def fig_kind(run):
    data = _mem_rows(run, "kind", ["fast_linear", "fast_conv"])
    if not data:
        raise ValueError("no kind data")
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(W2, 2.2))
    bss = sorted({r["bsize"] for rs in data.values() for r in rs})
    w = 0.38
    for k, (lab, key) in enumerate((("fast_linear", "linear"), ("fast_conv", "conv"))):
        rs = {r["bsize"]: r for r in data.get(lab, [])}
        for a, field, yl in ((ax[0], "ms_per_100k", "ms per 100k queries"), (ax[1], "peak_mb", "Peak GPU memory (MB)")):
            vals = [rs[b][field] if b in rs and not rs[b].get("oom") else np.nan for b in bss]
            a.bar(np.arange(len(bss)) + (k - 0.5) * w, vals, w, color=COL[key], hatch=None if key == "linear" else "//", edgecolor="k", lw=0.5, label=NAME[key])
            a.set_xticks(range(len(bss))); a.set_xticklabels([f"{b / 1000:g}k" for b in bss]); a.set_ylabel(yl); a.set_xlabel("Queries per call")
    ax[0].legend(frameon=False); ax[0].set_title("(a) Time"); ax[1].set_title("(b) Peak memory")
    run.save(fig, "fig_layer_kind")


# ===================================================================== equivalence
def _pair_values(run, key, T, labs_seeds):
    """|delta| of per-image metric between implementation/seed pairs."""
    R = {ls: mat([r for r in run.recs("equiv", ls[0]) if r.get("seed") == ls[1]], key, [T])[0][:, 0] for ls in labs_seeds}
    return R


def _equiv_groups(run, key, T):
    R = _pair_values(run, key, T, [("orig", 0), ("orig", 1), ("fast", 2), ("fast", 3)])
    g = {"orig vs orig (seeds)": [np.abs(R[("orig", 0)] - R[("orig", 1)])],
         "orig vs accelerated": [np.abs(R[("orig", a)] - R[("fast", b)]) for a in (0, 1) for b in (2, 3)],
         "accelerated vs accelerated (seeds)": [np.abs(R[("fast", 2)] - R[("fast", 3)])]}
    return {k: np.concatenate([x[~np.isnan(x)] for x in v]) for k, v in g.items()}


def _dump(run, lab, T, seed, i):
    p = os.path.join(run.d, "equiv", "dumps", f"{lab}_T{T}_s{seed}_i{i}.npz")
    return np.load(p) if os.path.exists(p) else None


def _pixel_groups(run, T):
    out = {"orig vs orig (seeds)": [], "orig vs accelerated": [], "accelerated vs accelerated (seeds)": []}
    files = glob.glob(os.path.join(run.d, "equiv", "dumps", f"orig_T{T}_s0_i*.npz"))
    for p in files:
        i = int(p.split("_i")[-1][:-4])
        A = {(l, s): _dump(run, l, T, s, i) for l, s in (("orig", 0), ("orig", 1), ("fast", 2), ("fast", 3))}
        if any(v is None for v in A.values()):
            continue
        out["orig vs orig (seeds)"].append(np.abs(A[("orig", 0)]["crop"] - A[("orig", 1)]["crop"]).ravel())
        out["accelerated vs accelerated (seeds)"].append(np.abs(A[("fast", 2)]["crop"] - A[("fast", 3)]["crop"]).ravel())
        for a in (0, 1):
            for b in (2, 3):
                out["orig vs accelerated"].append(np.abs(A[("orig", a)]["crop"] - A[("fast", b)]["crop"]).ravel())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def fig_equiv(run):
    Ts = sorted(set(Tlist(run.recs("equiv", "orig"))) & set(Tlist(run.recs("equiv", "fast"))))
    if not Ts:
        raise ValueError("no equiv data")
    Tmax = Ts[-1]
    cols = {"orig vs orig (seeds)": COL["orig"], "orig vs accelerated": COL["fast"], "accelerated vs accelerated (seeds)": COL["fast_none"]}
    hat = {"orig vs orig (seeds)": None, "orig vs accelerated": "///", "accelerated vs accelerated (seeds)": "\\\\"}
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(W2, 2.4))
    for ai, (key, yl) in enumerate((("psnr", "|$\\Delta$PSNR| per image (dB)"), ("lpips", "|$\\Delta$LPIPS| per image"))):
        w = 0.26
        for ti, T in enumerate(Ts):
            g = _equiv_groups(run, key, T)
            for gi, (name, v) in enumerate(g.items()):
                ax[ai].boxplot([v], positions=[ti + (gi - 1) * w], widths=w * 0.9, patch_artist=True, showfliers=False,
                               boxprops=dict(facecolor=cols[name], alpha=0.65, hatch=hat[name], lw=0.5), medianprops=dict(color="k", lw=0.8),
                               whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        ax[ai].set_xticks(range(len(Ts))); ax[ai].set_xticklabels([f"$T={T}$" for T in Ts]); ax[ai].set_ylabel(yl)
        ax[ai].set_title(f"({'ab'[ai]}) Per-image differences")
    for name in cols:
        ax[0].bar([0], [0], color=cols[name], alpha=0.65, hatch=hat[name], edgecolor="k", lw=0.5, label=name)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=6.5)
    pg = _pixel_groups(run, Tmax)
    if pg:
        allv = np.concatenate(list(pg.values()))
        bins = np.logspace(-8, np.log10(max(allv.max(), 1e-6)) + 0.1, 60)
        for name, v in pg.items():
            ax[2].hist(np.clip(v, bins[0], None), bins=bins, histtype="step", color=cols[name], lw=1.1, label=name, ls="-" if "orig vs acc" in name else "--")
        ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("|pixel difference| (output range 0-1)")
        ax[2].set_ylabel("Number of pixels"); ax[2].set_title(f"(c) Pixel differences, $T={Tmax}$")
    run.save(fig, "fig_equivalence")
    rows = []
    for T in Ts:
        gp, gl, gx = _equiv_groups(run, "psnr", T), _equiv_groups(run, "lpips", T), _pixel_groups(run, T)
        for name in gp:
            px = gx.get(name)
            rows.append([T, name, f"{gp[name].mean():.4f}", f"{gp[name].max():.4f}", f"{gl[name].mean():.5f}",
                         "--" if px is None else f"{px.mean():.1e}", "--" if px is None else f"{np.quantile(px, 0.999):.1e}",
                         "--" if px is None else f"{px.max():.1e}"])
        SUMMARY.setdefault("equiv", {})[int(T)] = {n: dict(dpsnr_mean=float(gp[n].mean()), dpsnr_max=float(gp[n].max())) for n in gp}
    run.tex("tab_equiv", ["$T$", "Pair", "mean $|\\Delta$PSNR|", "max $|\\Delta$PSNR|", "mean $|\\Delta$LPIPS|", "mean pixel diff.", "99.9\\% pixel diff.", "max pixel diff."],
            rows, "Agreement between implementations compared with the run-to-run variation of the model itself (different seeds).", "tab:equiv", "llccccc c".replace(" ", ""))


def fig_qualitative(run):
    Ts = sorted(set(Tlist(run.recs("equiv", "orig"))))
    T = Ts[-1]
    files = sorted(glob.glob(os.path.join(run.d, "equiv", "dumps", f"orig_T{T}_s0_i*.npz")))
    if not files:
        raise ValueError("no dumps")
    i = int(files[0].split("_i")[-1][:-4])
    A = {k: _dump(run, *k, i) if False else _dump(run, k[0], T, k[1], i) for k in (("orig", 0), ("orig", 1), ("fast", 2))}
    if any(v is None for v in A.values()):
        raise ValueError("incomplete dumps")
    img = lambda a: np.clip(np.transpose(a, (1, 2, 0)), 0, 1)
    d_of = np.abs(A[("orig", 0)]["crop"] - A[("fast", 2)]["crop"]).mean(0)
    d_oo = np.abs(A[("orig", 0)]["crop"] - A[("orig", 1)]["crop"]).mean(0)
    k = 0.5 / max(np.quantile(d_oo, 0.99), 1e-9)
    fig, ax = plt.subplots(1, 5, constrained_layout=True, figsize=(W2, 1.6))
    panels = [(img(A[("orig", 0)]["gt"]), "Ground truth", None), (img(A[("orig", 0)]["crop"]), "Original", None),
              (img(A[("fast", 2)]["crop"]), "Accelerated", None), (np.clip(d_of * k, 0, 1), f"|orig. $-$ accel.| $\\times${k:.0f}", "gray_r"),
              (np.clip(d_oo * k, 0, 1), f"|orig. $-$ orig.'| $\\times${k:.0f}", "gray_r")]
    for a, (im, t, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm, vmin=0 if cm else None, vmax=1 if cm else None); a.set_title(t, fontsize=6.5); a.axis("off")
    run.save(fig, "fig_qualitative")


def fig_cold(run):
    rows = []
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(W1, 2.2))
    for lab in ("orig", "fast"):
        R = run.recs("speed_T", lab)
        Ts = Tlist(R)
        cold = [np.mean([r["cold_s"] for r in R if r["T"] == T and "cold_s" in r]) for T in Ts]
        steady = [np.nanmean(mat(R, "time_s", [T])[0]) for T in Ts]
        ax.plot(Ts, cold, marker=MRK[lab], ls=LST[lab], color=COL[lab], label=f"{NAME[lab]}: first call", ms=4)
        ax.plot(Ts, steady, marker=MRK[lab], ls=LST[lab], color=COL[lab], alpha=0.45, label=f"{NAME[lab]}: steady state", ms=3)
        rows.append((lab, Ts, cold, steady))
    ax.set_xlabel("Number of components $T$"); ax.set_ylabel("Time per image (s)"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("First-call cost (compilation, graph capture)")
    run.save(fig, "fig_first_call")
    (_, Ts, cf, sf), (_, _, co, so) = rows[1], rows[0]
    be = [(c - s) / max(o - s2, 1e-9) for c, s, o, s2 in zip(cf, sf, so, sf)]
    SUMMARY["break_even_images"] = {int(T): float(b) for T, b in zip(Ts, be)}


# ===================================================================== negative result
def fig_spec(run):
    j = run.job("spec", "spec")
    if j is None:
        raise ValueError("no spec data")
    fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(W2, 2.3))
    it = [c["iter"] for c in j["jacobi"]]
    ax[0].semilogy(it, [max(c["max_err"], 1e-9) for c in j["jacobi"]], color=COL["fast"], label="max error")
    ax[0].semilogy(it, [max(c["mean_err"], 1e-9) for c in j["jacobi"]], color=COL["orig"], ls="--", label="mean error")
    ax[0].axhline(j["seed_to_seed_max_abs"], color="k", ls=":", lw=0.8); ax[0].text(it[-1], j["seed_to_seed_max_abs"], "seed-to-seed max", ha="right", va="bottom", fontsize=6)
    be_it = j["seq_eager_ms"] / j["full_verify_pass_ms"]
    ax[0].axvline(be_it, color=COL["fast_graph"], ls="-.", lw=0.9); ax[0].text(be_it, ax[0].get_ylim()[0] * 3, f" break-even\n ({be_it:.1f} passes)", fontsize=6)
    ax[0].set_xlabel("Jacobi iteration"); ax[0].set_ylabel("Error vs. sequential output"); ax[0].set_title("(a) Jacobi convergence"); ax[0].legend(frameon=False)
    Ks = [r["K"] for r in j["rounds"]]
    per_step_e = j["seq_eager_ms"] / j["T"]
    per_step_g = (j["seq_graph_ms"] / j["T"]) if j.get("seq_graph_ms") else None
    be_e = [r["eager_ms"] / per_step_e for r in j["rounds"]]
    ax[1].plot(Ks, Ks, "k:", label="max. possible ($K$)")
    ax[1].plot(Ks, be_e, marker="o", color=COL["orig"], ls="--", label="vs. eager loop")
    if per_step_g:
        be_g = [r["graph_ms"] / per_step_g if r.get("graph_ms") else np.nan for r in j["rounds"]]
        ax[1].plot(Ks, be_g, marker="s", color=COL["fast"], label="vs. graphed loop")
    ax[1].set_xlabel("Window size $K$"); ax[1].set_ylabel("Steps to commit per round"); ax[1].set_title("(b) Break-even per round"); ax[1].legend(frameon=False)
    ax[2].bar(["eager\nloop", "graphed\nloop", "one full\nverify pass"], [j["seq_eager_ms"], j.get("seq_graph_ms") or np.nan, j["full_verify_pass_ms"]],
              color=[COL["orig"], COL["fast"], COL["fast_graph"]], edgecolor="k", lw=0.5)
    ax[2].set_ylabel("Time (ms)"); ax[2].set_title(f"(c) Cost of one pass ({j['n_tokens']} tokens)")
    run.save(fig, "fig_negative_result")
    rows = [[r["K"], f"{r['eager_ms']:.2f}", "--" if not r.get("graph_ms") else f"{r['graph_ms']:.2f}", f"{r['eager_ms'] / per_step_e:.2f}",
             "--" if not per_step_g or not r.get("graph_ms") else f"{r['graph_ms'] / per_step_g:.2f}"] for r in j["rounds"]]
    run.tex("tab_spec", ["$K$", "Round, eager (ms)", "Round, graph (ms)", "Break-even vs. eager", "Break-even vs. graphed"], rows,
            "Verify-and-commit round for $K$ steps and the average number of steps it must commit to match the sequential loop.", "tab:spec")
    SUMMARY["spec"] = dict(verify_max_abs=j["verify_max_abs"], seed_to_seed=j["seed_to_seed_max_abs"], jacobi_final=j["jacobi"][-1])


# ===================================================================== main
def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    run = Run(sys.argv[1])
    print(f"report for {run.d}  (GPU: {run.gpu}, preset: {run.preset}{', SYNTHETIC' if run.synthetic else ''})")
    for fn in (tab_main, fig_time_vs_T, fig_ablation, fig_scale, fig_stages, fig_memory, fig_kind, fig_equiv,
               fig_qualitative, fig_cold, fig_spec):
        safe(fn, run)
    SUMMARY["gpu"] = run.gpu
    SUMMARY["preset"] = run.preset
    sl.write_json(os.path.join(run.d, "summary.json"), SUMMARY)
    print("done.")


if __name__ == "__main__":
    main()
