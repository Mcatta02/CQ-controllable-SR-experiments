#!/usr/bin/env python
"""
Report for the extra experiments (CPU only: numpy + matplotlib).  python suite/extra_report.py <run_dir>
Writes <run_dir>/report/*.csv, *.tex, *.pdf and report.txt. Every section is skipped (with a message) if its data is missing.

  6  scale     table + figures: speed-up vs scale and vs measured sharing ratio, quality vs scale, cost-model fit
  7  scal      table + figures: time / peak memory / OOM vs LR size and queries per call, cost-model fit, bottleneck share
  3  noise     identical-noise comparison: generator outputs per iteration, phase head, RGB (max / mean / relative error)
  8  exact     arrangement correctness with the same noise; stat: ratio / time / PSNR for orders x batch sizes x scales
  9  advis     local quality of adaptive inference: maps, per-region and per-tile statistics, tile-boundary artefacts
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = None
LOG = []
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False})
CO, CF = "#5B6B76", "#C46A00"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def load_group(run, group):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, group, "*.json"))):
        if f.endswith(".spec.json"):
            continue
        d = json.load(open(f))
        if d.get("ok"):
            out[os.path.basename(f)[:-5]] = d
        else:
            log(f"[warn] {group}/{os.path.basename(f)} failed: {d.get('error')}")
    return out


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{x:.6g}" if isinstance(x, (float, np.floating)) else str(x) for x in r) + "\n")


def nnls(A, y):
    """Non-negative least squares (small problems): scipy if present, else clipped lstsq on active set."""
    try:
        from scipy.optimize import nnls as _n
        return _n(A, y)[0]
    except Exception:
        x = np.linalg.lstsq(A, y, rcond=None)[0]
        act = np.ones(A.shape[1], bool)
        for _ in range(A.shape[1]):
            if (x >= 0).all():
                break
            act &= x >= 0
            x = np.zeros(A.shape[1])
            if act.any():
                x[act] = np.linalg.lstsq(A[:, act], y, rcond=None)[0]
        return np.clip(x, 0, None)


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else float("nan")


def fit_models(Q, R, T, t_o, t_f):
    """time_orig ~ a_o*Q*T + c_o            (one rollout per query)
       time_fast ~ a*R*T + b*Q*T + c         (rollouts of the R distinct pixels + per-query work, e.g. decoding)"""
    Q, R, T, t_o, t_f = map(np.asarray, (Q, R, T, t_o, t_f))
    Ao = np.stack([Q * T, np.ones_like(Q, dtype=float)], 1)
    co = nnls(Ao, t_o)
    Af = np.stack([R * T, Q * T, np.ones_like(Q, dtype=float)], 1)
    cf = nnls(Af, t_f)
    return co, cf, Ao @ co, Af @ cf


# ===================================================================== scales
def sec_scale(run):
    G = load_group(run, "scale")
    if not G:
        return log("[skip] scale: no data")
    recs = {}
    for name, d in G.items():
        impl, sc = name.split("_x")
        for r in d["records"]:
            if r.get("seed") == -1:
                continue
            recs.setdefault((impl, float(sc)), []).append(r)
    scales = sorted({k[1] for k in recs})
    Ts = sorted({r["T"] for v in recs.values() for r in v})
    rows, pts = [], []
    for sc in scales:
        for T in Ts:
            o = {r["image"]: r for r in recs.get(("orig", sc), []) if r["T"] == T and not r.get("oom")}
            f = {r["image"]: r for r in recs.get(("fast", sc), []) if r["T"] == T and not r.get("oom")}
            n_oom_o = sum(1 for r in recs.get(("orig", sc), []) if r["T"] == T and r.get("oom"))
            ids = sorted(set(o) & set(f))
            if not ids:
                rows.append((sc, T, 0, "", "", "", "", "", "", "", "", "", n_oom_o))
                continue
            m = lambda d, k: float(np.mean([d[i][k] for i in ids]))
            sp = np.array([o[i]["time_s"] / f[i]["time_s"] for i in ids])
            rho = m(f, "share_ratio_chunked")
            rows.append((sc, T, len(ids), m(o, "time_s"), m(f, "time_s"), float(sp.mean()), rho, sc ** 2, m(o, "psnr"), m(f, "psnr"),
                         m(o, "lpips"), m(f, "lpips"), n_oom_o))
            for i in ids:
                pts.append((sc, T, f[i]["n_queries"], f[i]["n_rollouts_chunked"], o[i]["time_s"], f[i]["time_s"]))
    hdr = ["scale", "T", "n_img", "time_orig_s", "time_fast_s", "speedup", "sharing_measured", "sharing_theory_s2", "psnr_orig",
           "psnr_fast", "lpips_orig", "lpips_fast", "orig_oom_images"]
    write_csv("scale_table.csv", hdr, rows)
    # LaTeX table (T=4,16,64 blocks)
    with open(os.path.join(OUT, "tab_scale_ext.tex"), "w") as f:
        f.write("\\begin{tabular}{rrrrrrrrr}\n\\toprule\n$s$ & $T$ & time orig (s) & time ours (s) & speed-up & $\\rho$ meas. & $s^2$ & PSNR orig & PSNR ours \\\\\n\\midrule\n")
        for r in rows:
            if r[2] == 0:
                f.write(f"{r[0]:g} & {r[1]} & \\multicolumn{{7}}{{c}}{{original out of memory}} \\\\\n")
            else:
                f.write(f"{r[0]:g} & {r[1]} & {r[3]:.2f} & {r[4]:.2f} & {r[5]:.1f}$\\times$ & {r[6]:.1f} & {r[7]:.0f} & {r[8]:.2f} & {r[9]:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    if not pts:
        return log("[scale] no paired images")
    P = np.array(pts, float)
    co, cf, po, pf = fit_models(P[:, 2], P[:, 3], P[:, 1], P[:, 4], P[:, 5])
    log(f"[scale] cost model orig: t = {co[0]:.3e}*Q*T + {co[1]:.3f}  (R2 {r2(P[:, 4], po):.4f})")
    log(f"[scale] cost model ours: t = {cf[0]:.3e}*R*T + {cf[1]:.3e}*Q*T + {cf[2]:.3f}  (R2 {r2(P[:, 5], pf):.4f})")
    if cf[1] > 0:
        log(f"[scale] speed-up ceiling as sharing -> infinity (per-query decoding bound): {co[0] / cf[1]:.1f}x")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for T, mk in zip(Ts, "osd^v"):
        rr = [r for r in rows if r[1] == T and r[2] > 0]
        if not rr:
            continue
        ax[0].plot([r[0] for r in rr], [r[5] for r in rr], mk + "-", ms=4, label=f"$T$={T}")
        ax[1].plot([r[6] for r in rr], [r[5] for r in rr], mk + "-", ms=4, label=f"$T$={T}")
        ax[2].plot([r[0] for r in rr], [r[9] - r[8] for r in rr], mk + "-", ms=4, label=f"$T$={T}")
    ax[0].set_xlabel("scale $s$"); ax[0].set_ylabel("speed-up"); ax[0].set_xscale("log", base=2); ax[0].set_xticks(scales); ax[0].set_xticklabels([f"{s:g}" for s in scales])
    ax[1].set_xlabel("measured sharing ratio $\\rho$"); ax[1].set_ylabel("speed-up"); ax[1].set_xscale("log"); ax[1].minorticks_off(); ax[1].set_xticks([4, 16, 64, 144]); ax[1].set_xticklabels(["4", "16", "64", "144"])
    if cf[1] > 0:
        ax[1].axhline(co[0] / cf[1], color="k", ls=":", lw=0.8)
        ax[1].text(ax[1].get_xlim()[0] * 1.1, co[0] / cf[1], " decoding-bound ceiling", va="bottom", fontsize=6.5)
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlabel("scale $s$"); ax[2].set_ylabel("PSNR ours - orig (dB)")
    ax[2].set_xscale("log", base=2); ax[2].set_xticks(scales); ax[2].set_xticklabels([f"{s:g}" for s in scales])
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scale_ext.pdf")); plt.close(fig)
    # quality against scale (orig), training range marked
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for T, mk in zip(Ts, "osd^v"):
        rr = [r for r in rows if r[1] == T and r[2] > 0]
        ax.plot([r[0] for r in rr], [r[8] for r in rr], mk + "-", ms=4, color=CO, alpha=0.5 + 0.5 * T / max(Ts), label=f"orig $T$={T}")
        ax.plot([r[0] for r in rr], [r[9] for r in rr], mk + "--", ms=4, color=CF, alpha=0.5 + 0.5 * T / max(Ts), label=f"ours $T$={T}")
    ax.axvspan(4, max(scales), color="0.92", zorder=0); ax.text(4.1, ax.get_ylim()[1], "outside training range", va="top", fontsize=6.5)
    ax.set_xscale("log", base=2); ax.set_xticks(scales); ax.set_xticklabels([f"{s:g}" for s in scales])
    ax.set_xlabel("scale $s$"); ax.set_ylabel("PSNR (dB)"); ax.legend(frameon=False, fontsize=5.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scale_quality.pdf")); plt.close(fig)


# ===================================================================== scalability
def sec_scal(run):
    G = load_group(run, "scal")
    if not G:
        return log("[skip] scal: no data")
    R = []
    for name, d in G.items():
        impl = name.split("_bs")[0]
        for r in d["records"]:
            r = dict(r, impl=impl)
            R.append(r)
    sizes = sorted({r["lr"] for r in R})
    bss = sorted({r.get("bsize") for r in R if r.get("bsize")})
    Ts = sorted({r["T"] for r in R if "T" in r})
    rows = []
    for T in Ts:
        for c in sizes:
            for bs in bss:
                def sel(impl):
                    return [r for r in R if r["impl"] == impl and r["lr"] == c and r.get("T") == T and r.get("bsize") == bs]
                o, f = sel("orig"), sel("fast")
                if not o and not f:
                    continue
                oo = [r for r in o if not r.get("oom")]
                ff = [r for r in f if not r.get("oom")]
                Q = (o or f)[0]["n_queries"]
                to = np.mean([r["time_s"] for r in oo]) if oo else np.nan
                tf = np.mean([r["time_s"] for r in ff]) if ff else np.nan
                rows.append((T, c, bs, Q, to, tf, to / tf if oo and ff else np.nan,
                             np.mean([r["peak_mb"] for r in oo]) if oo else np.nan, np.mean([r["peak_mb"] for r in ff]) if ff else np.nan,
                             int(any(r.get("oom") for r in o)), int(any(r.get("oom") for r in f)),
                             np.mean([r["share_ratio_chunked"] for r in ff]) if ff else np.nan))
    write_csv("scal_table.csv", ["T", "lr", "bsize", "queries", "time_orig_s", "time_fast_s", "speedup", "peak_orig_mb", "peak_fast_mb",
                                 "orig_oom", "fast_oom", "sharing_chunked"], rows)
    A = np.array(rows, float)
    # --- figures
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
    T0 = Ts[0]
    big = max(bss)
    for impl_col, col, lab in ((4, CO, "original"), (5, CF, "ours")):
        for bs, ls in ((min(bss), ":"), (big, "-")):
            m = (A[:, 0] == T0) & (A[:, 2] == bs) & ~np.isnan(A[:, impl_col])
            ax[0].plot(A[m, 3], A[m, impl_col], ls, marker="o", ms=3, color=col, label=f"{lab}, {bs // 1000}k/call")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("HR queries"); ax[0].set_ylabel(f"time (s), $T$={T0}"); ax[0].legend(frameon=False, fontsize=5.5)
    for c, mk in zip(sizes, "osd^v"):
        for impl_col, col in ((7, CO), (8, CF)):
            m = (A[:, 0] == T0) & (A[:, 1] == c) & ~np.isnan(A[:, impl_col])
            ax[1].plot(A[m, 2], A[m, impl_col] / 1024, mk + ("-" if col == CF else ":"), ms=3, color=col, alpha=0.4 + 0.6 * sizes.index(c) / max(1, len(sizes) - 1))
            mo = (A[:, 0] == T0) & (A[:, 1] == c) & (A[:, 9 if col == CO else 10] == 1)
            ax[1].plot(A[mo, 2], np.full(mo.sum(), ax[1].get_ylim()[1]), "x", color=col, ms=5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("queries per call"); ax[1].set_ylabel("peak GPU memory (GiB)"); ax[1].set_title("x = out of memory", fontsize=6.5)
    piv = np.full((len(sizes), len(bss)), np.nan)
    for r in rows:
        if r[0] == T0:
            piv[sizes.index(r[1]), bss.index(r[2])] = r[6]
    im = ax[2].imshow(piv, cmap="viridis", aspect="auto", origin="lower")
    ax[2].set_xticks(range(len(bss))); ax[2].set_xticklabels([f"{b // 1000}k" for b in bss]); ax[2].set_yticks(range(len(sizes))); ax[2].set_yticklabels([str(s) for s in sizes])
    for i in range(len(sizes)):
        for j in range(len(bss)):
            if not np.isnan(piv[i, j]):
                ax[2].text(j, i, f"{piv[i, j]:.1f}", ha="center", va="center", fontsize=6, color="w")
    ax[2].set_xlabel("queries per call"); ax[2].set_ylabel("LR side (px)"); ax[2].set_title("speed-up", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scalability.pdf")); plt.close(fig)
    # --- cost model over all points where both ran
    pts = []
    for T in Ts:
        for c in sizes:
            for bs in bss:
                o = [r for r in R if r["impl"] == "orig" and r["lr"] == c and r.get("T") == T and r.get("bsize") == bs and not r.get("oom")]
                f = [r for r in R if r["impl"] == "fast" and r["lr"] == c and r.get("T") == T and r.get("bsize") == bs and not r.get("oom")]
                if o and f:
                    pts.append((f[0]["n_queries"], f[0]["n_rollouts_chunked"], T, np.mean([r["time_s"] for r in o]), np.mean([r["time_s"] for r in f])))
    if len(pts) >= 4:
        P = np.array(pts, float)
        co, cf, po, pf = fit_models(P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4])
        log(f"[scal] cost model orig: t = {co[0]:.3e}*Q*T + {co[1]:.3f} (R2 {r2(P[:, 3], po):.4f}); ours: t = {cf[0]:.3e}*R*T + {cf[1]:.3e}*Q*T + {cf[2]:.3f} (R2 {r2(P[:, 4], pf):.4f})")
        share = cf[0] * P[:, 1] * P[:, 2] / np.maximum(pf, 1e-9)
        log(f"[scal] share of our time spent in rollouts: median {np.median(share):.2f}, range {share.min():.2f}-{share.max():.2f}; "
            f"the remainder is per-query decoding and fixed cost")
    # memory model: peak = base + k*bsize (per implementation, largest size)
    for impl, col in (("orig", 7), ("fast", 8)):
        m = (A[:, 0] == T0) & (A[:, 1] == max(sizes)) & ~np.isnan(A[:, col])
        if m.sum() >= 2:
            k, b = np.polyfit(A[m, 2], A[m, col], 1)
            log(f"[scal] peak memory {impl} (LR {max(sizes)}, T={T0}): {b:.0f} MB + {k * 1000:.1f} MB per 1000 queries/call")
    log(f"[scal] OOM cells: original {int(np.nansum(A[:, 9]))}, ours {int(np.nansum(A[:, 10]))} of {len(A)}")


# ===================================================================== identical noise
def expand(feat_q, tok, fourier):
    """per-query generator output [Q,dim,T] from per-token output (the shared implementation returns one row per distinct token)."""
    if tok.shape[0] == feat_q.shape[0] and np.array_equal(tok, feat_q):
        return fourier
    ut, it = np.unique(tok, axis=0, return_inverse=True)
    it = it.reshape(-1)
    first = np.zeros(len(ut), dtype=np.int64)
    first[it[::-1]] = np.arange(len(tok))[::-1]
    uq, iq = np.unique(feat_q, axis=0, return_inverse=True)
    iq = iq.reshape(-1)
    if uq.shape != ut.shape or not np.array_equal(uq, ut):
        raise ValueError("tokens of the two implementations differ")
    return fourier[first][iq]


def phase_from(fq, feat_q, cell_q, Wi, bi, Wp, bp, dim, fused):
    Wf_in, Wc_in = Wi[:, :dim], Wi[:, dim:]
    if fused:
        Wf, Wc, cvec = Wp @ Wf_in, Wp @ Wc_in, Wp @ bi + bp
        ph = np.einsum("qdt,od->qot", fq, Wf) + (feat_q @ Wp.T + cell_q @ Wc.T + cvec)[:, :, None]
    else:
        h = np.einsum("qdt,od->qot", fq, Wf_in) + (cell_q @ Wc_in.T + bi)[:, :, None] + feat_q[:, :, None]
        ph = np.einsum("qct,oc->qot", h, Wp) + bp[None, :, None]
    return ph.transpose(0, 2, 1).reshape(ph.shape[0], -1)            # '... c l -> ... (l c)'


def err(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max()), float(d.mean()), float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))


def resolve(run, f):
    """dump paths are absolute in real runs; tolerate a moved run directory"""
    if os.path.exists(f):
        return f
    alt = os.path.join(run, "noise", os.path.basename(os.path.dirname(f)), os.path.basename(f))
    return alt if os.path.exists(alt) else f


def sec_noise(run):
    G = load_group(run, "noise")
    if not G:
        return log("[skip] noise: no data")
    by = {}
    for name, d in G.items():
        label, sc = name.rsplit("_x", 1)
        by[(label, sc)] = {r["T"]: r["file"] for r in d["records"]}
    rows = []
    scs = sorted({k[1] for k in by}, key=float)
    tests = sorted({k[0] for k in by} - {"orig"})
    for sc in scs:
        ref = by.get(("orig", sc))
        if not ref:
            continue
        for label in tests:
            tst = by.get((label, sc))
            if not tst:
                continue
            for T in sorted(ref):
                if T not in tst:
                    continue
                a, b = np.load(resolve(run, ref[T])), np.load(resolve(run, tst[T]))
                try:
                    fa = expand(a["feat_q"], a["tok"], a["fourier"])
                    fb = expand(b["feat_q"], b["tok"], b["fourier"])
                except ValueError as e:
                    log(f"[noise] {label} x{sc} T={T}: {e}")
                    continue
                g_all = err(fa, fb)
                g_last = err(fa[:, :, -1], fb[:, :, -1])
                per_t = [err(fa[:, :, t], fb[:, :, t])[0] for t in range(fa.shape[-1])]
                W = {k: a[k] for k in ("Wi", "bi", "Wp", "bp")}
                dim = int(a["dim"])
                ph_ref = a["phase"] if "phase" in a.files else None
                ph_fused = phase_from(fb, b["feat_q"], b["cell_q"], W["Wi"], W["bi"], W["Wp"], W["bp"], dim, True)
                ph_unf = phase_from(fb, b["feat_q"], b["cell_q"], W["Wi"], W["bi"], W["Wp"], W["bp"], dim, False)
                if ph_ref is None:
                    ph_ref = phase_from(fa, a["feat_q"], a["cell_q"], W["Wi"], W["bi"], W["Wp"], W["bp"], dim, False)
                p_f, p_u = err(ph_ref, ph_fused), err(ph_ref, ph_unf)
                ra, rb = a["recon"], b["recon"]
                rgb = err(ra * 0.5 * 255, rb * 0.5 * 255)
                rgb_rel = err(ra, rb)[2]
                # C1: queries of one LR pixel receive identical generator outputs in the original
                uq, iq = np.unique(a["feat_q"], axis=0, return_inverse=True)
                iq = iq.reshape(-1)
                first = np.zeros(len(uq), dtype=np.int64)
                first[iq[::-1]] = np.arange(len(iq))[::-1]
                c1 = float(np.abs(fa - fa[first[iq]]).max())
                rows.append((sc, label, T, g_all[0], g_all[1], g_all[2], g_last[0], max(per_t), p_f[0], p_f[1], p_f[2], p_u[0], rgb[0], rgb[1], rgb_rel, c1))
    hdr = ["scale", "test", "T", "gen_max", "gen_mean", "gen_rel", "gen_last_max", "gen_worst_iter_max", "phase_fused_max", "phase_fused_mean",
           "phase_fused_rel", "phase_unfused_max", "rgb_max_8bit", "rgb_mean_8bit", "rgb_rel", "orig_dup_rows_max_diff"]
    write_csv("noise_equivalence.csv", hdr, rows)
    with open(os.path.join(OUT, "tab_noise_equiv.tex"), "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n\\toprule\ntest & $T$ & gen. max & gen. rel. & phase max & RGB max (8-bit) & RGB mean (8-bit) & RGB rel. \\\\\n\\midrule\n")
        for r in rows:
            if r[0] == scs[-1]:
                f.write(f"{r[1].replace('_', ' ')} & {r[2]} & {r[3]:.1e} & {r[5]:.1e} & {r[8]:.1e} & {r[12]:.1e} & {r[13]:.1e} & {r[14]:.1e} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    for r in rows:
        log(f"[noise] x{r[0]} {r[1]:20s} T={r[2]:2d}  generator max {r[3]:.2e} rel {r[5]:.2e} | phase(fused) max {r[8]:.2e} | RGB max {r[12]:.3f} lvl mean {r[13]:.4f} lvl rel {r[14]:.2e} | dup-rows {r[15]:.1e}")


# ===================================================================== arrangements
def sec_exact(run):
    G = load_group(run, "exact")
    if G:
        rows = []
        for name, d in G.items():
            label, sc = name.rsplit("_x", 1)
            for r in d["records"]:
                if r.get("skipped"):
                    continue
                rows.append((sc, label, r["T"], r["variant"], r["n_queries"], r["chunk"], r["max_abs"] * 0.5 * 255, r["mean_abs"] * 0.5 * 255,
                             r["rel_fro"], r["sharing_ratio"], r["time_s"], r["max_abs_other_seed"] * 0.5 * 255))
        write_csv("arrangement_exact.csv", ["scale", "impl", "T", "variant", "queries", "chunk", "max_err_8bit", "mean_err_8bit", "rel_fro",
                                            "sharing_ratio", "time_s", "max_err_other_noise_8bit"], rows)
        with open(os.path.join(OUT, "tab_arrangements.tex"), "w") as f:
            f.write("\\begin{tabular}{llrrrr}\n\\toprule\nimpl. & arrangement & $T$ & sharing $\\rho$ & max err (8-bit) & rel. err \\\\\n\\midrule\n")
            sc0 = sorted({r[0] for r in rows}, key=float)[-1]
            for r in rows:
                if r[0] == sc0 and r[2] == max(x[2] for x in rows):
                    f.write(f"{r[1].replace('_', ' ')} & {r[3].replace('_', ' ')} & {r[2]} & {r[9]:.2f} & {r[6]:.1e} & {r[8]:.1e} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
        worst = {}
        for r in rows:
            worst[r[1]] = max(worst.get(r[1], 0), r[6])
        for k, v in worst.items():
            log(f"[exact] {k:12s}: worst max error over all arrangements/T/scales = {v:.3e} 8-bit levels "
                f"(a different noise draw changes colours by up to {max(x[11] for x in rows if x[1] == k):.2f} levels)")
    else:
        log("[skip] exact: no data")
    S = load_group(run, "stat")
    if S:
        rows = []
        for name, d in S.items():
            impl = name.split("_chunk")[0]
            for r in d["records"]:
                if r.get("oom"):
                    rows.append((impl, r["scale"], r["T"], r["chunk"], r["order"], "", "", "", "", 1))
                    continue
                rows.append((impl, r["scale"], r["T"], r["chunk"], r["order"], r["image"], r["psnr"], r["time_s"], r["sharing_ratio"], 0))
        write_csv("arrangement_stat_raw.csv", ["impl", "scale", "T", "chunk", "order", "image", "psnr", "time_s", "sharing_ratio", "oom"], rows)
        keys = sorted({(r[1], r[3], r[4]) for r in rows})
        out = []
        for sc, ch, od in keys:
            def m(impl, col):
                v = [r[col] for r in rows if r[0] == impl and r[1] == sc and r[3] == ch and r[4] == od and not r[9]]
                return float(np.mean(v)) if v else np.nan
            out.append((sc, ch, od, m("orig", 6), m("fast", 6), m("orig", 7), m("fast", 7), m("orig", 7) / m("fast", 7), m("fast", 8)))
        write_csv("arrangement_stat.csv", ["scale", "chunk", "order", "psnr_orig", "psnr_fast", "time_orig_s", "time_fast_s", "speedup", "sharing_fast"], out)
        for o in out:
            log(f"[stat] x{o[0]} chunk {o[1]:6d} {o[2]:18s} PSNR {o[3]:.3f}/{o[4]:.3f}  time {o[5]:.2f}/{o[6]:.2f}s  speed-up {o[7]:.1f}x  sharing {o[8]:.2f}")
    else:
        log("[skip] stat: no data")


# ===================================================================== adaptive maps
def box_mean(x, k):
    c = np.cumsum(np.cumsum(np.pad(x, ((k, k), (k, k)), mode="edge"), 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * k + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n])[:x.shape[0], :x.shape[1]] / n ** 2


def psnr_of(mse):
    return 10 * np.log10(1.0 / np.maximum(mse, 1e-12))


def sec_advis(run):
    files = sorted(glob.glob(os.path.join(run, "advis", "dump", "*.npz")))
    if not files:
        return log("[skip] advis: no data")
    rows_cls, rows_tile, rows_bd = [], [], []
    for fi, fn in enumerate(files):
        d = np.load(fn)
        gt = d["gt"].astype(np.float32)
        fixed_keys = sorted([k for k in d.files if k.startswith("fixed") and "seed2" not in k], key=lambda k: int(k[5:]))
        Tmax = int(fixed_keys[-1][5:])
        ref = d[f"fixed{Tmax}"].astype(np.float32)
        ref2 = d[f"fixed{Tmax}_seed2"].astype(np.float32)
        nh, nw, tile = (int(v) for v in d["tile_grid"])
        scale = float(d["scale_hr_per_lr"])
        H, Wd = gt.shape[-2:]
        gray = gt.mean(0)
        gy, gx = np.gradient(gray)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        gs = box_mean(grad, 3)
        edge = gs >= np.quantile(gs, 0.8)
        smooth = gs <= np.quantile(gs, 0.4)
        texture = ~edge & ~smooth
        classes = dict(edge=edge, texture=texture, smooth=smooth)
        adaptive_keys = [k for k in d.files if k.startswith("dct_")]
        for ak in adaptive_keys:
            tag = ak[4:]
            rk = f"random_{tag}"
            ad = d[ak].astype(np.float32)
            rd = d[rk].astype(np.float32) if rk in d.files else None
            bm = d[f"budget_{ak}"].astype(np.int32)
            # ---- per-class MSE
            for cname, mask in classes.items():
                def mse(x):
                    return float(((x - gt) ** 2).mean(0)[mask].mean())
                rows_cls.append((fi, tag, cname, psnr_of(mse(ref)), psnr_of(mse(ref2)), psnr_of(mse(ad)), psnr_of(mse(rd)) if rd is not None else np.nan,
                                 float(mask.mean())))
            # ---- per-tile PSNR (tiles of the budget map)
            th, tw = H / nh, Wd / nw
            ys = (np.arange(nh + 1) * th).round().astype(int)
            xs = (np.arange(nw + 1) * tw).round().astype(int)
            e_ref, e_ref2, e_ad = ((ref - gt) ** 2).mean(0), ((ref2 - gt) ** 2).mean(0), ((ad - gt) ** 2).mean(0)
            dt_ad, dt_noise, bud = np.zeros((nh, nw)), np.zeros((nh, nw)), np.zeros((nh, nw))
            for i in range(nh):
                for j in range(nw):
                    sl = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
                    p_ref, p_ref2, p_ad = psnr_of(e_ref[sl].mean()), psnr_of(e_ref2[sl].mean()), psnr_of(e_ad[sl].mean())
                    dt_ad[i, j], dt_noise[i, j] = p_ad - p_ref, p_ref2 - p_ref
                    bud[i, j] = bm[min(ys[i] + 1, H - 1), min(xs[j] + 1, Wd - 1)]
            for thr in (0.5, 1.0, 2.0):
                rows_tile.append((fi, tag, thr, float((dt_ad < -thr).mean()), float((dt_noise < -thr).mean()), float(dt_ad.mean()), float(dt_noise.mean())))
            worst = np.dstack(np.unravel_index(np.argsort(dt_ad, axis=None)[:5], dt_ad.shape))[0]
            log(f"[advis] img{fi} {tag}: tiles worse than 1 dB vs fixed T={Tmax}: adaptive {np.mean(dt_ad < -1) * 100:.1f}% | "
                f"two noise draws of fixed T={Tmax}: {np.mean(dt_noise < -1) * 100:.1f}%; worst tile {dt_ad.min():.2f} dB "
                f"(budget {int(bud[tuple(worst[0])])}, tile {tuple(int(v) for v in worst[0])})")
            for b in sorted({int(x) for x in np.unique(bud)}):
                log(f"        budget {b:3d}: mean tile delta-PSNR {dt_ad[bud == b].mean():+.3f} dB over {int((bud == b).sum())} tiles")
            # ---- tile-boundary artefacts: gradient of the reconstruction across a boundary vs the ground truth
            tix = (np.arange(Wd) * nw) // Wd
            tiy = (np.arange(H) * nh) // H
            def bstat(x):
                out = {}
                # vertical boundaries between columns j and j+1
                for name, mask in (("diff", (tix[1:] != tix[:-1])[None, :] & (bm[:, 1:] != bm[:, :-1])),
                                   ("same", (tix[1:] != tix[:-1])[None, :] & (bm[:, 1:] == bm[:, :-1]))):
                    if mask.any():
                        dr = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean(0)[mask].mean()
                        dg = np.abs(gt[:, :, 1:] - gt[:, :, :-1]).mean(0)[mask].mean()
                        out[name] = float(dr / max(dg, 1e-9))
                for name, mask in (("diff", (tiy[1:] != tiy[:-1])[:, None] & (bm[1:, :] != bm[:-1, :])),
                                   ("same", (tiy[1:] != tiy[:-1])[:, None] & (bm[1:, :] == bm[:-1, :]))):
                    if mask.any():
                        dr = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean(0)[mask].mean()
                        dg = np.abs(gt[:, 1:, :] - gt[:, :-1, :]).mean(0)[mask].mean()
                        out[name + "_h"] = float(dr / max(dg, 1e-9))
                return out
            bs_ad, bs_ref = bstat(ad), bstat(ref)
            rows_bd.append((fi, tag, bs_ad.get("diff", np.nan), bs_ad.get("same", np.nan), bs_ref.get("diff", np.nan), bs_ref.get("same", np.nan),
                            bs_ad.get("diff_h", np.nan), bs_ad.get("same_h", np.nan)))
            # ---- band around different-budget boundaries: error vs interior
            bnd = np.zeros((H, Wd), bool)
            bnd[:, 1:] |= (tix[1:] != tix[:-1])[None, :] & (bm[:, 1:] != bm[:, :-1])
            bnd[1:, :] |= (tiy[1:] != tiy[:-1])[:, None] & (bm[1:, :] != bm[:-1, :])
            band = box_mean(bnd.astype(np.float32), 2) > 0
            if band.any() and (~band).any():
                log(f"        boundary band: MSE adaptive {e_ad[band].mean():.2e} vs fixed {e_ref[band].mean():.2e} "
                    f"(interior {e_ad[~band].mean():.2e} vs {e_ref[~band].mean():.2e})")
            # ---- figures (first adaptive set only per image)
            if ak == adaptive_keys[-1]:
                fig_advis(fi, tag, d, gt, ref, ad, rd, bm, dt_ad, dt_noise, e_ref, e_ad, nh, nw, Tmax)
    write_csv("advis_regions.csv", ["image", "budgets", "region", "psnr_fixedmax", "psnr_fixedmax_seed2", "psnr_adaptive", "psnr_random", "area_fraction"], rows_cls)
    write_csv("advis_tiles.csv", ["image", "budgets", "threshold_db", "frac_tiles_worse_adaptive", "frac_tiles_worse_noise_floor", "mean_delta_adaptive", "mean_delta_noise"], rows_tile)
    write_csv("advis_boundaries.csv", ["image", "budgets", "ratio_diff_budget_v", "ratio_same_budget_v", "ratio_fixed_v_diff", "ratio_fixed_v_same", "ratio_diff_budget_h", "ratio_same_budget_h"], rows_bd)
    for tag in sorted({r[1] for r in rows_cls}):
        for cname in ("edge", "texture", "smooth"):
            v = np.array([r[3:7] for r in rows_cls if r[1] == tag and r[2] == cname], float)
            log(f"[advis] {tag:14s} {cname:8s} PSNR fixed {v[:, 0].mean():.2f} (2nd draw {v[:, 1].mean():.2f}) adaptive {v[:, 2].mean():.2f} random {np.nanmean(v[:, 3]):.2f}")


def fig_advis(fi, tag, d, gt, ref, ad, rd, bm, dt_ad, dt_noise, e_ref, e_ad, nh, nw, Tmax):
    H, Wd = gt.shape[-2:]
    # crops: worst tile for adaptive, and the tile with the highest ground-truth gradient
    i, j = np.unravel_index(np.argmin(dt_ad), dt_ad.shape)
    th, tw = H / nh, Wd / nw
    cs = 96
    def crop(ci, cj):
        cy = int((ci + 0.5) * th); cx = int((cj + 0.5) * tw)
        y0 = int(np.clip(cy - cs // 2, 0, H - cs)); x0 = int(np.clip(cx - cs // 2, 0, Wd - cs))
        return slice(y0, y0 + cs), slice(x0, x0 + cs)
    gy, gx = np.gradient(gt.mean(0))
    g = box_mean(np.sqrt(gx ** 2 + gy ** 2), 4)
    ei, ej = np.unravel_index(np.argmax(g[::int(th), ::int(tw)][:nh, :nw]), (min(nh, g[::int(th), ::int(tw)].shape[0]), min(nw, g[::int(th), ::int(tw)].shape[1])))
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1])
    a = fig.add_subplot(gs[0, 0]); a.imshow(bm, cmap="viridis", interpolation="nearest"); a.set_title("budget map", fontsize=7); a.axis("off")
    a = fig.add_subplot(gs[0, 1]); im = a.imshow(dt_ad, cmap="RdBu", vmin=-3, vmax=3, interpolation="nearest"); a.set_title(f"tile $\\Delta$PSNR (adaptive - fixed {Tmax})", fontsize=7); a.axis("off")
    fig.colorbar(im, ax=a, fraction=0.04)
    a = fig.add_subplot(gs[0, 2]); a.imshow(dt_noise, cmap="RdBu", vmin=-3, vmax=3, interpolation="nearest"); a.set_title("tile $\\Delta$PSNR, 2nd noise draw of fixed", fontsize=7); a.axis("off")
    a = fig.add_subplot(gs[0, 3]); a.hist([dt_ad.ravel(), dt_noise.ravel()], bins=40, range=(-4, 2), label=["adaptive", "noise floor"], color=[CF, CO]); a.set_yscale("log"); a.legend(frameon=False, fontsize=6); a.set_xlabel("tile $\\Delta$PSNR (dB)")
    for r, (ci, cj, title) in enumerate(((i, j, "worst adaptive tile"), (ei, ej, "high-gradient tile"))):
        sl = crop(ci, cj)
        panels = [("GT", gt[:, sl[0], sl[1]].transpose(1, 2, 0)), (f"fixed {Tmax}", ref[:, sl[0], sl[1]].transpose(1, 2, 0)),
                  ("adaptive", ad[:, sl[0], sl[1]].transpose(1, 2, 0)),
                  ("|err| fixed", np.abs(ref - gt)[:, sl[0], sl[1]].mean(0)), ("|err| adaptive", np.abs(ad - gt)[:, sl[0], sl[1]].mean(0)),
                  ("budget", bm[sl[0], sl[1]])]
        sub = gs[1 + r, :].subgridspec(1, len(panels), wspace=0.05)
        for k, (t, im_) in enumerate(panels):
            a = fig.add_subplot(sub[0, k])
            if im_.ndim == 3:
                a.imshow(np.clip(im_, 0, 1), interpolation="nearest")
            elif t == "budget":
                a.imshow(im_, cmap="viridis", interpolation="nearest")
            else:
                a.imshow(im_, cmap="magma", vmin=0, vmax=0.15, interpolation="nearest")
            a.set_title(f"{title}: {t}" if k == 0 else t, fontsize=6); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_advis_img{fi}_{tag}.pdf"), dpi=200); plt.close(fig)


def main():
    global OUT
    run = sys.argv[1]
    OUT = os.path.join(run, "report")
    os.makedirs(OUT, exist_ok=True)
    for fn in (sec_scale, sec_scal, sec_noise, sec_exact, sec_advis):
        try:
            fn(run)
        except Exception as e:                      # one broken section must not hide the others
            import traceback
            log(f"[error] {fn.__name__}: {e!r}\n{traceback.format_exc()[-1500:]}")
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    print("report written to", OUT)


if __name__ == "__main__":
    main()
