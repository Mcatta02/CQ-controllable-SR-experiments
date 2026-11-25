"""Pareto frontier (inference time minimised, PSNR maximised) of fixed T in {4,8,16,32,64} against adaptive
allocation (Sobel / variance / DCT / random), from a run of  `run_suite.py --only adaptive`.

  python suite/pareto_report.py --results suite/results/<run_dir> [--main-run <run_dir with main/fast.json>] [--out pareto]

Fixed points come from the adaptive jobs (fixed_max = largest budget, fixed_T<t> = job `fixedT`). If a fixed T in
{4,8,16,32,64} is missing there (older runs), it is taken from <main-run>/main/fast.json on the same images."""
import argparse, glob, json, os, collections
import numpy as np
import pandas as pd

FIXED_T = [4, 8, 16, 32, 64]


def recs(f):
    d = json.load(open(f))
    return [r for r in (d["records"] if isinstance(d, dict) else d) if not r.get("oom") and "psnr" in r]


def frontier(d):
    keep = []
    for i, r in d.iterrows():
        dom = ((d.time <= r.time) & (d.psnr >= r.psnr) & ((d.time < r.time) | (d.psnr > r.psnr))).any()
        if not dom:
            keep.append(i)
    return d.loc[keep].sort_values("time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--main-run", default=None)
    ap.add_argument("--out", default="pareto")
    a = ap.parse_args()
    fixed = collections.defaultdict(list)            # T -> per-image records
    rows, n_img = [], 0
    for f in sorted(glob.glob(f"{a.results}/adaptive/*.json")):
        if f.endswith("spec.json"):
            continue
        r = recs(f)
        if not r:
            continue
        bud = next((x["budgets"] for x in r if x["method"] == "fixed_max"), None)
        for x in r:
            if x["method"] == "fixed_max":
                fixed[int(max(bud))].append(x)
            elif x["method"].startswith("fixed_T"):
                fixed[int(x["method"][7:])].append(x)
        n_img = max(n_img, len({x["image"] for x in r}))
        for m in ("sobel", "variance", "dct", "random"):
            v = [x for x in r if x["method"] == m]
            if v:
                rows.append(("adaptive-" + m, f"budgets={bud}", np.mean([x["psnr"] for x in v]),
                             np.mean([x["lpips"] for x in v]), np.mean([x["time_s"] for x in v]), len(v)))
    missing = [t for t in FIXED_T if t not in fixed]
    if missing and a.main_run:
        mr = recs(f"{a.main_run}/main/fast.json")
        for t in missing:
            fixed[t] = [x for x in mr if x["T"] == t and x["image"] < n_img]
    for t in FIXED_T:
        v = fixed.get(t, [])
        if v:
            by = collections.defaultdict(list)
            for x in v:
                by[x["image"]].append(x)
            ims = sorted(by)
            rows.append(("fixed", f"num_pred={t}", np.mean([np.mean([y["psnr"] for y in by[i]]) for i in ims]),
                         np.mean([np.mean([y["lpips"] for y in by[i]]) for i in ims]),
                         np.mean([np.mean([y["time_s"] for y in by[i]]) for i in ims]), len(ims)))
        else:
            print(f"WARNING: no fixed T={t} (run the `fixedT` job or pass --main-run)")
    df = pd.DataFrame(rows, columns=["type", "label", "psnr", "lpips", "time", "n_images"])
    df.to_csv(a.out + "_points.csv", index=False)
    cols = ["type", "label", "psnr", "lpips", "time"]
    fx, ad = df[df.type == "fixed"], df[df.type.str.startswith("adaptive")]
    out = []
    out.append("=== OVERALL PARETO FRONTIER (fixed T in 4,8,16,32,64 and all adaptive points)")
    out.append(frontier(df)[cols].to_string(index=False))
    out.append("\n=== ADAPTIVE POINTS NOT DOMINATED BY ANY FIXED POINT")
    ok = [i for i, r in ad.iterrows()
          if not ((fx.time <= r.time) & (fx.psnr >= r.psnr) & ((fx.time < r.time) | (fx.psnr > r.psnr))).any()]
    out.append(ad.loc[ok].sort_values("time")[cols].to_string(index=False))
    out.append("\n=== ADAPTIVE POINTS THAT ARE DOMINATED (and by which fixed T)")
    for i, r in ad.sort_values("time").iterrows():
        if i in ok:
            continue
        b = fx[(fx.time <= r.time) & (fx.psnr >= r.psnr) & ((fx.time < r.time) | (fx.psnr > r.psnr))]
        out.append(f"{r.type:18s} {r.label:28s} {r.time:.3f}s {r.psnr:.3f}dB  <- fixed " + ",".join(l.split("=")[1] for l in b.label))
    out.append("\n=== FRONTIER PER METHOD (fixed + that method)")
    for m in sorted(ad.type.unique()):
        out.append("\n" + m)
        out.append(frontier(pd.concat([fx, ad[ad.type == m]]))[cols].to_string(index=False))
    txt = "\n".join(out)
    print(txt)
    open(a.out + "_report.txt", "w").write(txt + "\n")
    # ---- LaTeX tables: fixed T and one row per budget set (PSNR of every allocation at the same time)
    def fmt_set(lbl):
        return "$\\{" + lbl.split("[")[1].split("]")[0].replace(" ", "") + "\\}$"
    piv = df[df.type.str.startswith("adaptive")].pivot_table(index="label", columns="type", values="psnr")
    tim = df[df.type.str.startswith("adaptive")].groupby("label").time.mean()
    nd = {df.loc[i, "label"] for i in ok}
    order = sorted(piv.index, key=lambda l: tim[l])
    def row_set(l):
        v = [f"{piv.loc[l, c]:.3f}" if c in piv.columns and not np.isnan(piv.loc[l, c]) else "--" for c in
             ["adaptive-dct", "adaptive-sobel", "adaptive-variance", "adaptive-random"]]
        best = max((piv.loc[l, c] for c in piv.columns if c != "adaptive-random"))
        star = "$^\\star$" if l in nd else ""
        return f"{fmt_set(l)}{star} & {tim[l]:.2f} & " + " & ".join(v) + " \\\\"
    def table(sets):
        L = ["\\begin{tabular}{l r rrrr}", "\\toprule", "Allocation & Time (s) & DCT & Sobel & Var. & Random \\\\", "\\midrule"]
        for _, r in fx.sort_values("time").iterrows():
            L.append(f"fixed $T={r.label.split('=')[1]}$ & {r.time:.2f} & \\multicolumn{{4}}{{c}}{{{r.psnr:.3f}}} \\\\")
        L.append("\\midrule")
        L += [row_set(l) for l in sets]
        return "\n".join(L + ["\\bottomrule", "\\end{tabular}"])
    # merge fixed rows by time so that frontier reading is easy: keep fixed first, sets by time
    open(a.out + "_table_all.tex", "w").write(table(order))
    open(a.out + "_table_main.tex", "w").write(table([l for l in order if l in nd]))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        mk = {"adaptive-sobel": "o", "adaptive-variance": "s", "adaptive-dct": "^", "adaptive-random": "x"}
        for m, g in ad.groupby("type"):
            ax.scatter(g.time, g.psnr, s=14, marker=mk.get(m, "o"), alpha=.7, label=m.replace("adaptive-", "adaptive, "))
        f_ = fx.sort_values("time")
        ax.plot(f_.time, f_.psnr, "k-o", ms=5, lw=1.2, label="fixed T")
        for _, r in f_.iterrows():
            ax.annotate(r.label.split("=")[1], (r.time, r.psnr), textcoords="offset points", xytext=(4, -9), fontsize=7)
        fr = frontier(df)
        ax.step(fr.time, fr.psnr, where="post", color="r", lw=.8, ls="--", label="Pareto frontier")
        ax.set_xscale("log"); ax.set_xlabel("time per image (s)"); ax.set_ylabel("PSNR (dB)")
        ax.set_ylim(f_.psnr.min() - .1, f_.psnr.max() + .05); ax.legend(fontsize=6); fig.tight_layout()
        fig.savefig(a.out + "_plot.pdf"); fig.savefig(a.out + "_plot.png", dpi=150)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()