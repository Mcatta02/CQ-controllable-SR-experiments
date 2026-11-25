#!/usr/bin/env python3
"""
Extra analysis for the paper (professor items 1.4, 3.1, 3.2, 3.4, 3.6, 3.7). Reads a suite run and writes
LaTeX tables + figures to <out>/ . Works on the existing archive and on new runs (it uses the sharing
statistics and the resolution test only if they are present).

  python extra_analysis.py --results results/<run_dir>  --out paper_extra
  python extra_analysis.py --results full_results_tar.gz --out paper_extra
"""
import argparse
import glob
import json
import os
import tarfile
import tempfile
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def locate(path):
    if os.path.isfile(path) and tarfile.is_tarfile(path):
        tmp = tempfile.mkdtemp(prefix="ana_")
        with tarfile.open(path) as t:
            t.extractall(tmp)
        path = tmp
    c = [r for r, _d, f in os.walk(path) if "manifest.json" in f or "summary.json" in f]
    return sorted(c)[-1]


def J(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def recs(p):
    d = J(p)
    return [r for r in (d or {}).get("records", []) if "time_s" in r and not r.get("oom")]


def ci_mean(x, n=2000):
    x = np.asarray(x, float)
    if x.size < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    idx = rng.integers(0, x.size, (n, x.size))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def ratio_ci(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    idx = rng.integers(0, m, (n, m))
    r = a[idx].mean(1) / b[idx].mean(1)
    return float(a.mean() / b.mean()), float(np.quantile(r, .025)), float(np.quantile(r, .975))


def per_T(rr, key):
    d = defaultdict(list)
    for r in sorted(rr, key=lambda r: (r["T"], r["image"], r["seed"])):
        d[r["T"]].append(r[key])
    return d


def write(path, txt):
    with open(path, "w") as f:
        f.write(txt)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="paper_extra")
    a = ap.parse_args()
    R = locate(a.results)
    os.makedirs(a.out, exist_ok=True)
    report = []

    # ------------------------------------------------ 3.2 quality-runtime, speed-up with CI
    mo, mf = recs(f"{R}/main/orig.json"), recs(f"{R}/main/fast.json")
    Ts = sorted({r["T"] for r in mf})
    ps_o, ps_f = per_T(mo, "psnr"), per_T(mf, "psnr")
    lp_o, lp_f = per_T(mo, "lpips"), per_T(mf, "lpips")
    t_o, t_f = per_T(mo, "time_s"), per_T(mf, "time_s")
    if mo and mf:
        rows = []
        for T in Ts:
            sp, lo, hi = ratio_ci(t_o[T], t_f[T])
            dps = np.mean(ps_f[T]) - np.mean(ps_o[T])
            rows.append((T, np.mean(ps_o[T]), np.mean(ps_f[T]), np.mean(lp_o[T]), np.mean(lp_f[T]),
                         np.mean(t_o[T]), np.mean(t_f[T]), sp, lo, hi, dps))
        tex = ["\\begin{tabular}{r rr rr rr r}", "\\toprule",
               "$T$ & \\multicolumn{2}{c}{PSNR (dB)} & \\multicolumn{2}{c}{LPIPS} & \\multicolumn{2}{c}{Time (s)} & Speed-up \\\\",
               " & Orig. & Ours & Orig. & Ours & Orig. & Ours & [95\\% CI] \\\\", "\\midrule"]
        for T, po, pf, lo_, lf_, to, tf, sp, lo, hi, d in rows:
            tex.append(f"{T} & {po:.3f} & {pf:.3f} & {lo_:.4f} & {lf_:.4f} & {to:.2f} & {tf:.2f} & ${sp:.1f}\\times$ [{lo:.1f}, {hi:.1f}] \\\\")
        tex += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_main_precise.tex", "\n".join(tex))

        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for k, (lab, P, L, Tm, mk) in enumerate([("Original", ps_o, lp_o, t_o, "o-"), ("Ours", ps_f, lp_f, t_f, "s--")]):
            x = [np.mean(Tm[T]) for T in Ts]
            ax[0].plot(x, [np.mean(P[T]) for T in Ts], mk, label=lab)
            ax[1].plot(x, [np.mean(L[T]) for T in Ts], mk, label=lab)
            if k == 0:
                for T, xx in zip(Ts, x):
                    ax[0].annotate(f"T={T}", (xx, np.mean(P[T])), fontsize=6, xytext=(3, -8), textcoords="offset points")
        for axx, yl in zip(ax, ("PSNR (dB)", "LPIPS")):
            axx.set_xscale("log"); axx.set_xlabel("Time per image (s)"); axx.set_ylabel(yl); axx.grid(alpha=.3)
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{a.out}/fig_quality_runtime.pdf"); fig.savefig(f"{a.out}/fig_quality_runtime.png", dpi=200)

        # time needed for a target quality (smallest T whose mean PSNR reaches the target)
        lines = ["\\begin{tabular}{r rr rr}", "\\toprule", "Target PSNR & \\multicolumn{2}{c}{Smallest $T$} & \\multicolumn{2}{c}{Time (s)} \\\\",
                 " & Orig. & Ours & Orig. & Ours \\\\", "\\midrule"]
        pmax = max(np.mean(ps_f[T]) for T in Ts)
        for tgt in sorted({round(min(np.mean(ps_o[T]) for T in Ts) + f * (pmax - min(np.mean(ps_o[T]) for T in Ts)), 2) for f in (.5, .8, .95)}):
            def first(P, Tm):
                for T in Ts:
                    if np.mean(P[T]) >= tgt - 1e-9:
                        return T, np.mean(Tm[T])
                return None, None
            (To, so), (Tf, sf) = first(ps_o, t_o), first(ps_f, t_f)
            if To and Tf:
                lines.append(f"{tgt:.2f} dB & {To} & {Tf} & {so:.2f} & {sf:.2f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_time_to_quality.tex", "\n".join(lines))

    else:
        report.append("no main results in this run: main table, quality-runtime figure and time-to-quality skipped")

    # ------------------------------------------------ 1.4 speed-up vs T and marginal cost (7-point sweep)
    so, sf = recs(f"{R}/speed_T/orig.json"), recs(f"{R}/speed_T/fast.json")
    if so and sf:
        to_, tf_ = per_T(so, "time_s"), per_T(sf, "time_s")
        T7 = sorted(to_)
        lines = ["\\begin{tabular}{r rr r rr}", "\\toprule", "$T$ & \\multicolumn{2}{c}{Time (s)} & Speed-up & \\multicolumn{2}{c}{Marginal cost (ms/component)} \\\\",
                 " & Orig. & Ours & & Orig. & Ours \\\\", "\\midrule"]
        for i, T in enumerate(T7):
            mo_ = mf_ = "--"
            if i:
                dT = T - T7[i - 1]
                mo_ = f"{1000 * (np.mean(to_[T]) - np.mean(to_[T7[i - 1]])) / dT:.0f}"
                mf_ = f"{1000 * (np.mean(tf_[T]) - np.mean(tf_[T7[i - 1]])) / dT:.0f}"
            sp = np.mean(to_[T]) / np.mean(tf_[T])
            lines.append(f"{T} & {np.mean(to_[T]):.2f} & {np.mean(tf_[T]):.2f} & ${sp:.1f}\\times$ & {mo_} & {mf_} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_speedup_vs_T.tex", "\n".join(lines))
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.plot(T7, [np.mean(to_[T]) / np.mean(tf_[T]) for T in T7], "o-")
        ax.set_xscale("log", base=2); ax.set_xlabel("Number of components $T$"); ax.set_ylabel("Speed-up"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{a.out}/fig_speedup_vs_T.pdf"); fig.savefig(f"{a.out}/fig_speedup_vs_T.png", dpi=200)

    # ------------------------------------------------ 1.4 stage breakdown
    sto, stf = J(f"{R}/stages/orig.json"), J(f"{R}/stages/fast.json")
    if sto and stf:
        lines = ["\\begin{tabular}{l r rrrrr r}", "\\toprule",
                 "Impl. & $T$ & Feature & Predictor & (of which recurrent) & Decoder & Other & Total (ms) \\\\", "\\midrule"]
        for lab, d in (("Orig.", sto), ("Ours", stf)):
            for r in d["records"]:
                pred_only = r["predictor_ms"]
                lines.append(f"{lab} & {r['T']} & {r['gen_feat_ms']:.0f} & {pred_only:.0f} & {r['recurrent_ms']:.0f} & "
                             f"{r['decoder_ms']:.0f} & {r['other_ms']:.0f} & {r['total_ms']:.0f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_stages.tex", "\n".join(lines))
        for r in stf["records"]:
            report.append(f"stages fast T={r['T']}: 'other' = {100 * r['other_ms'] / r['total_ms']:.0f}% of {r['total_ms']:.0f} ms")

    # ------------------------------------------------ 3.1 ablation with memory (flag configurations)
    full_flags = set("SFECG")
    names_c = {"S": "sharing", "F": "fusion", "E": "step engine", "C": "compile", "G": "CUDA graph"}
    legacy = {"orig": "Original", "fast_none": "Fast-root code, nothing installed", "fast_graph": "Engine + compile + graph",
              "fast_post": "Sharing + fusion (+compile)", "fast_full": "All (ours)"}

    def label_of(name):
        if name in legacy:
            return legacy[name]
        fl = set(name[4:])
        if fl == full_flags:
            return "All (ours)"
        miss = full_flags - fl
        if len(miss) == 1:
            return "All without " + names_c[next(iter(miss))]
        return " + ".join(names_c[c] for c in "SFECG" if c in fl)

    order = ["orig", "fast_none", "abl_S", "abl_F", "abl_SF", "abl_SFC", "abl_E", "abl_EC", "abl_EG", "abl_ECG",
             "abl_FECG", "abl_SECG", "abl_SFEG", "abl_SFEC", "abl_SFECG", "fast_graph", "fast_post", "fast_full"]
    ab = {k: recs(f"{R}/ablation/{k}.json") for k in order}
    ab = {k: v for k, v in ab.items() if v}
    TA = sorted({r["T"] for r in ab["orig"]}) if "orig" in ab else []
    if TA:
        hdr = " & ".join(f"\\multicolumn{{3}}{{c}}{{$T={T}$}}" for T in TA)
        lines = ["\\begin{tabular}{l " + "rrr " * len(TA) + "}", "\\toprule", f"Configuration & {hdr} \\\\",
                 " & " + " & ".join("Time (s) & Speed-up & Peak (GB)" for _ in TA) + " \\\\", "\\midrule"]
        base = {T: np.mean([r["time_s"] for r in ab["orig"] if r["T"] == T]) for T in TA}
        for k in ab:
            cells = []
            for T in TA:
                sel = [r for r in ab[k] if r["T"] == T]
                if sel:
                    t = np.mean([r["time_s"] for r in sel]); pk = np.mean([r["peak_mb"] for r in sel]) / 1024
                    cells.append(f"{t:.2f} & ${base[T] / t:.1f}\\times$ & {pk:.2f}")
                else:
                    cells.append("-- & -- & --")
            lines.append(f"{label_of(k)} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_ablation_full.tex", "\n".join(lines))
        # compact version for the paper: components alone/combined plus every leave-one-out row
        keep = {"orig", "abl_S", "abl_F", "abl_SF", "abl_ECG", "abl_FECG", "abl_SECG", "abl_SFEG", "abl_SFEC", "abl_SFECG"}
        head = lines[:5]
        body = [l for l, k in zip(lines[5:5 + len(ab)], ab) if k in keep]
        write(f"{a.out}/tab_ablation_memory.tex", "\n".join(head + body + lines[-2:]))
        if "abl_SFECG" in ab:       # contribution of each component = slowdown when it is removed from the full system
            for T in TA:
                tf = np.mean([r["time_s"] for r in ab["abl_SFECG"] if r["T"] == T])
                parts = []
                for c in "SFECG":
                    k = "abl_" + "".join(x for x in "SFECG" if x != c)
                    if k in ab:
                        parts.append(f"{names_c[c]} x{np.mean([r['time_s'] for r in ab[k] if r['T'] == T]) / tf:.2f}")
                report.append(f"ablation T={T}: slowdown when removing -> " + ", ".join(parts))

    # ------------------------------------------------ 3.3 patch-wise budgets versus fixed T = max budget
    ad_files = [f for f in sorted(glob.glob(f"{R}/adaptive/b*.json")) if not f.endswith("spec.json")]
    ad_files.sort(key=lambda f: (-max((recs(f) or [{"budgets": [0]}])[0]["budgets"]), -np.mean((recs(f) or [{"budgets": [0]}])[0]["budgets"])))   # by reference T_max, then mean budget
    if ad_files:
        lab = {"fixed_max": "fixed (max budget)", "fixed": "fixed (same mean $T$)", "sobel": "adaptive, Sobel",
               "variance": "adaptive, variance", "dct": "adaptive, DCT", "random": "adaptive, random"}
        lines = ["\\begin{tabular}{l l r rr rr r}", "\\toprule",
                 "Budgets & Allocation & Mean $T$ & PSNR (dB) & LPIPS & Time (s) & Time saved & $\\Delta$PSNR vs fixed max [95\\% CI] \\\\", "\\midrule"]
        for f in ad_files:
            rr_ = recs(f)
            if not rr_:
                continue
            bud = rr_[0]["budgets"]
            by = defaultdict(dict)
            for r in rr_:
                by[r["method"]][r["image"]] = r
            if "fixed_max" not in by:
                continue
            imgs = sorted(set.intersection(*[set(v) for v in by.values()]))
            ref_p = np.array([by["fixed_max"][i]["psnr"] for i in imgs])
            ref_t = np.array([by["fixed_max"][i]["time_s"] for i in imgs])
            for m in ["fixed_max", "fixed", "sobel", "variance", "dct", "random"]:
                if m not in by:
                    continue
                P = np.array([by[m][i]["psnr"] for i in imgs]); L = np.array([by[m][i]["lpips"] for i in imgs])
                Tm = np.array([by[m][i]["time_s"] for i in imgs]); mT = np.mean([by[m][i]["mean_T"] for i in imgs])
                saved = "--" if m == "fixed_max" else f"{100 * (1 - Tm.mean() / ref_t.mean()):.0f}\\%"
                dcell = "--" if m == "fixed_max" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*ci_mean(P - ref_p))
                lines.append(f"{{{','.join(map(str, bud))}}} & {lab[m]} & {mT:.1f} & {P.mean():.3f} & {L.mean():.4f} & {Tm.mean():.2f} & {saved} & {dcell} \\\\")
                if m in ("sobel", "variance", "dct", "random"):
                    te = np.mean([by[m][i]["t_est_s"] for i in imgs]) * 1000
                    report.append(f"adaptive {bud} {m}: {Tm.mean():.2f}s vs fixed-max {ref_t.mean():.2f}s ({100 * (1 - Tm.mean() / ref_t.mean()):.0f}% saved), "
                                  f"dPSNR {np.mean(P - ref_p):+.3f} dB, difficulty estimation {te:.1f} ms/img")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        write(f"{a.out}/tab_adaptive.tex", "\n".join(lines))
        # two halves for the supplement (the full table is taller than a page): sets ending at 64 | lower references
        mid = [i for i, l in enumerate(lines) if l.strip() == "\\midrule"]
        hd = lines[:mid[0] + 1]
        split = next((i for i in mid[1:] if lines[i + 1].split("&")[0].strip().strip("{}").split(",")[-1] != "64"), None)
        if split:
            write(f"{a.out}/tab_adaptive_a.tex", "\n".join(lines[:split] + ["\\bottomrule", "\\end{tabular}"]))
            write(f"{a.out}/tab_adaptive_b.tex", "\n".join(hd + lines[split + 1:]))

        # compact version for the main text: Sobel only, one row per budget set, against fixed T = 64
        comp = ["\\begin{tabular}{l r rr r r r}", "\\toprule",
                "Budgets & Mean $T$ & PSNR (dB) & LPIPS & Time (s) & Time saved & $\\Delta$PSNR [95\\% CI] \\\\", "\\midrule"]
        last_ref = None
        for f in ad_files:
            rr_ = recs(f)
            by = defaultdict(dict)
            for r in rr_:
                by[r["method"]][r["image"]] = r
            if "fixed_max" not in by or "sobel" not in by:
                continue
            imgs = sorted(set(by["fixed_max"]) & set(by["sobel"]))
            g = lambda m, k: np.array([by[m][i][k] for i in imgs])
            if last_ref != rr_[0]["T_max"]:
                if last_ref is not None:
                    comp.append("\\midrule")
                last_ref = rr_[0]["T_max"]
                comp.append(f"fixed $T={int(rr_[0]['T_max'])}$ & {rr_[0]['T_max']:.0f} & {g('fixed_max', 'psnr').mean():.3f} & "
                            f"{g('fixed_max', 'lpips').mean():.4f} & {g('fixed_max', 'time_s').mean():.2f} & -- & -- \\\\")
            ts, tr = g("sobel", "time_s").mean(), g("fixed_max", "time_s").mean()
            comp.append(f"{{{','.join(map(str, rr_[0]['budgets']))}}} & {g('sobel', 'mean_T').mean():.1f} & {g('sobel', 'psnr').mean():.3f} & "
                        f"{g('sobel', 'lpips').mean():.4f} & {ts:.2f} & {100 * (1 - ts / tr):.0f}\\% & "
                        + "{:+.3f} [{:+.3f}, {:+.3f}]".format(*ci_mean(g("sobel", "psnr") - g("fixed_max", "psnr"))) + " \\\\")
        comp += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_adaptive_compact.tex", "\n".join(comp))

        # quality-versus-time figure: fixed-T curve (main run) and every budget set / difficulty measure.
        # The vertical range is zoomed on the mixed-budget points, because PSNR saturates for large T.
        pts = []
        mk = {"sobel": "o", "variance": "s", "dct": "^"}
        for f in ad_files:
            rr_ = recs(f)
            if not rr_:
                continue
            by = defaultdict(dict)
            for r in rr_:
                by[r["method"]][r["image"]] = r
            imgs = sorted(set.intersection(*[set(v) for v in by.values()]))
            for m in mk:
                if m in by:
                    pts.append((m, np.mean([by[m][i]["time_s"] for i in imgs]), np.mean([by[m][i]["psnr"] for i in imgs]),
                                "{" + ",".join(map(str, rr_[0]["budgets"])) + "}"))
        if pts:
            fig, ax = plt.subplots(figsize=(3.6, 2.9))
            ymin = min(p[2] for p in pts) - 0.015
            ymax = max([p[2] for p in pts] + ([np.mean(ps_f[T]) for T in Ts] if mf else [])) + 0.01
            if mf:
                xs = [np.mean(t_f[T]) for T in Ts]; ys = [np.mean(ps_f[T]) for T in Ts]
                ax.plot(xs, ys, "k.-", lw=1, label="fixed $T$")
                for T, x_, y_ in zip(Ts, xs, ys):
                    if ymin <= y_ <= ymax:
                        ax.annotate(f"$T$={T}", (x_, y_), fontsize=6, xytext=(2, 4), textcoords="offset points")
            seen = set()
            for m, x_, y_, lbl in pts:
                ax.plot(x_, y_, mk[m], ms=4, mfc="none", color="C0", label=("mixed budgets, " + m) if m not in seen else None)
                seen.add(m)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(left=0)
            ax.set_xlabel("Time per image (s)"); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=.3)
            ax.legend(fontsize=6, loc="lower right")
            fig.tight_layout(); fig.savefig(f"{a.out}/fig_adaptive.pdf"); fig.savefig(f"{a.out}/fig_adaptive.png", dpi=200)
    else:
        report.append("no adaptive results in this run (run_suite.py --only adaptive)")

    # ------------------------------------------------ 3.4 scale: measured speed-up vs sharing ratio
    lines = ["\\begin{tabular}{r r r rr r}", "\\toprule", "Scale & $s^2$ & Measured sharing & Orig. (s) & Ours (s) & Speed-up \\\\", "\\midrule"]
    for sc in ("2", "2.5", "3", "3.5", "4"):
        o, f = recs(f"{R}/scale/orig_x{sc}.json"), recs(f"{R}/scale/fast_x{sc}.json")
        if not (o and f):
            continue
        sr = [r["share_ratio_chunked"] for r in f if "share_ratio_chunked" in r]
        shr = f"{np.mean(sr):.1f}" if sr else "n/a"
        to, tf = [r["time_s"] for r in o], [r["time_s"] for r in f]
        sp, lo, hi = ratio_ci(to, tf)
        lines.append(f"$\\times${sc} & {float(sc) ** 2:.2f} & {shr} & {np.mean(to):.2f} & {np.mean(tf):.2f} & ${sp:.1f}\\times$ [{lo:.1f}, {hi:.1f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{a.out}/tab_scale.tex", "\n".join(lines))

    # ------------------------------------------------ 3.6 statistical equivalence over seeds
    eo, ef = recs(f"{R}/equiv/orig.json"), recs(f"{R}/equiv/fast.json")
    if eo and ef:
        lines = ["\\begin{tabular}{r r rr rr}", "\\toprule",
                 "$T$ & seeds & \\multicolumn{2}{c}{PSNR (dB)} & \\multicolumn{2}{c}{LPIPS} \\\\",
                 " & & mean diff. [95\\% CI] & seed std (orig./ours) & mean diff. [95\\% CI] & seed std (orig./ours) \\\\", "\\midrule"]
        for T in sorted({r["T"] for r in eo}):
            res = {}
            for key in ("psnr", "lpips"):
                def mat(rr):
                    d = defaultdict(dict)
                    for r in rr:
                        if r["T"] == T:
                            d[r["image"]][r["seed"]] = r[key]
                    return d
                mo_, mf_ = mat(eo), mat(ef)
                imgs = sorted(set(mo_) & set(mf_))
                sd = sorted({s for i in imgs for s in mo_[i]} & {s for i in imgs for s in mf_[i]})
                diff = [np.mean([mf_[i][s] - mo_[i][s] for s in sd]) for i in imgs]          # per-image, mean over shared seeds
                std_o = np.mean([np.std([mo_[i][s] for s in mo_[i]], ddof=1) for i in imgs if len(mo_[i]) > 1] or [np.nan])
                std_f = np.mean([np.std([mf_[i][s] for s in mf_[i]], ddof=1) for i in imgs if len(mf_[i]) > 1] or [np.nan])
                m, lo, hi = ci_mean(diff)
                res[key] = (m, lo, hi, std_o, std_f, len(sd), len(imgs))
            p, l = res["psnr"], res["lpips"]
            lines.append(f"{T} & {p[5]} & {p[0]:+.4f} [{p[1]:+.4f}, {p[2]:+.4f}] & {p[3]:.4f} / {p[4]:.4f} & "
                         f"{l[0]:+.5f} [{l[1]:+.5f}, {l[2]:+.5f}] & {l[3]:.5f} / {l[4]:.5f} \\\\")
            report.append(f"equiv T={T}: {p[6]} images, {p[5]} shared seeds")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_equivalence.tex", "\n".join(lines))

    # ------------------------------------------------ 3.7 memory table (peak GB and reduction) and resolution
    mem = defaultdict(dict)
    for f in glob.glob(f"{R}/memory/*_bs*.json"):
        if f.endswith("spec.json"):
            continue
        b = os.path.basename(f)[:-5]
        impl, bs = b.split("_bs")
        d = J(f)
        for r in (d or {}).get("records", []):
            if r.get("T") == 64:
                mem[int(bs)][impl] = (None if r.get("oom") else r["peak_mb"] / 1024, None if r.get("oom") else r["ms_per_100k"])
    lines = ["\\begin{tabular}{r rr rr r}", "\\toprule", "Queries/call & \\multicolumn{2}{c}{Peak (GB)} & \\multicolumn{2}{c}{ms / 100k queries} & Peak reduction \\\\",
             " & Orig. & Ours & Orig. & Ours & \\\\", "\\midrule"]
    for bs in sorted(mem):
        o, f = mem[bs].get("orig"), mem[bs].get("fast")
        if f is None:
            continue
        og = f"{o[0]:.2f}" if o and o[0] is not None else "OOM"
        ot = f"{o[1]:.0f}" if o and o[1] is not None else "OOM"
        red = f"${o[0] / f[0]:.1f}\\times$" if o and o[0] else "--"
        lines.append(f"{bs:,} & {og} & {f[0]:.2f} & {ot} & {f[1]:.0f} & {red} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(f"{a.out}/tab_memory_precise.tex", "\n".join(lines))

    ro, rf = recs(f"{R}/resolution/orig.json"), recs(f"{R}/resolution/fast.json")
    if ro and rf:
        lines = ["\\begin{tabular}{r r rr rr}", "\\toprule", "LR size & Queries & \\multicolumn{2}{c}{Time (s)} & \\multicolumn{2}{c}{Peak (GB)} \\\\",
                 " & & Orig. & Ours & Orig. & Ours \\\\", "\\midrule"]
        for o, f in zip(ro, rf):
            lines.append(f"{o['lr_h']}$\\times${o['lr_w']} & {o['n_queries']:,} & {o['time_s']:.2f} & {f['time_s']:.2f} & {o['peak_mb'] / 1024:.2f} & {f['peak_mb'] / 1024:.2f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(f"{a.out}/tab_resolution.tex", "\n".join(lines))
    else:
        report.append("resolution test not in this run (re-run with the patched suite: --only resolution)")

    write(f"{a.out}/notes.txt", "\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()