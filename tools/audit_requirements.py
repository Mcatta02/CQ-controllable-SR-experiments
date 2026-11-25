#!/usr/bin/env python3
"""
Audit the CQ-SR acceleration paper against the professor's checklist.

Usage
-----
  python audit_requirements.py --results full_results_tar.gz
  python audit_requirements.py --results suite/results/<run_dir> --tex main.tex sec/*.tex
  python audit_requirements.py --results <dir> --tex main.tex --out audit_report.md

--results  results directory (or the .tar.gz of it) produced by your suite
--tex      one or more .tex files of the paper (optional; enables the writing checks)
--out      where to write the markdown report (default: audit_report.md)

Statuses: PASS / PARTIAL / MISSING / SKIPPED (you decided to skip) / INFO.
Re-run it after every new experiment; the report shows what is still open.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
import tarfile
import tempfile

import numpy as np

PASS, PART, MISS, SKIP, INFO = "PASS", "PARTIAL", "MISSING", "SKIPPED", "INFO"
ROWS = []


def add(cid, title, status, detail=""):
    ROWS.append((cid, title, status, detail))


# ----------------------------------------------------------------- loading
def locate_results(path):
    if os.path.isfile(path) and tarfile.is_tarfile(path):
        tmp = tempfile.mkdtemp(prefix="audit_")
        with tarfile.open(path) as t:
            t.extractall(tmp)
        path = tmp
    cands = []
    for root, _dirs, files in os.walk(path):
        if "summary.json" in files or "manifest.json" in files:
            cands.append(root)
    if not cands:
        sys.exit(f"No results directory (summary.json / manifest.json) found under {path}")
    return sorted(cands)[-1]


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def recs(d):
    """measurement records (warm-up 'cold_s' records are excluded)"""
    if not isinstance(d, dict):
        return []
    return [r for r in d.get("records", []) if "time_s" in r]


def spec_of(d):
    return d.get("meta", {}).get("spec", {}) if isinstance(d, dict) else {}


def mean(x):
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


# ----------------------------------------------------------------- checks
def check_reproducibility(R):  # 1.2
    files = [f for f in glob.glob(os.path.join(R, "**", "*.json"), recursive=True)
             if not f.endswith(".spec.json") and "summary" not in f and "manifest" not in f
             and "env.json" not in f and "spec/spec.json" not in f.replace("\\", "/")]
    envs, shas, commits, dirty, bsizes, reps = set(), set(), set(), set(), set(), set()
    for f in files:
        d = jload(f)
        if not isinstance(d, dict) or "meta" not in d:
            continue
        m = d["meta"]
        e = m.get("env", {})
        envs.add((e.get("gpu"), e.get("torch"), e.get("cuda"), e.get("driver")))
        shas.add(m.get("ckpt_sha256"))
        g = m.get("git", {})
        commits.add(g.get("commit"))
        dirty.add(g.get("dirty_files"))
        s = m.get("spec", {})
        if "bsize" in s and not re.search(r"/(memory|kind)/", f.replace("\\", "/")):
            bsizes.add(s["bsize"])
        reps.add(s.get("reps"))
    envfile = jload(os.path.join(R, "env.json")) or {}
    e = envfile.get("env", {})
    add("1.2a", "Hardware and software versions recorded", PASS if e else MISS,
        f"GPU={e.get('gpu')}, torch={e.get('torch')}, CUDA={e.get('cuda')}, driver={e.get('driver')}, "
        f"python={e.get('python')}")
    add("1.2b", "One environment/checkpoint/commit for all runs",
        PASS if len(envs) == 1 and len(shas) == 1 and len(commits) == 1 and dirty <= {0} else PART,
        f"{len(envs)} environment(s), {len(shas)} checkpoint hash(es), {len(commits)} git commit(s), "
        f"dirty_files={sorted(x for x in dirty if x is not None)}. "
        "Commit all changes before the final run so dirty_files=0.")
    add("1.2c", "Eval batch size (queries per call) identical in main/ablation/scale",
        PASS if len(bsizes) == 1 else PART, f"bsize values: {sorted(bsizes)} (queries per call)")
    spec_txt = json.dumps([jload(f) for f in glob.glob(os.path.join(R, "**", "*.spec.json"), recursive=True)])
    has_prec = re.search(r"precision|dtype|tf32|fp16|amp", spec_txt, re.I) is not None
    add("1.2d", "Numerical precision recorded", PASS if has_prec else MISS,
        "No precision/dtype/TF32 field in any spec. State fp32 (and TF32 on/off; the V100 has no TF32) "
        "in the paper and log it in env.json.")
    cold = any("cold_s" in r for r in (jload(os.path.join(R, "main", "fast.json")) or {}).get("records", []))
    add("1.2e", "Timing protocol (warm-up, repetitions)", PART if cold else MISS,
        f"warm-up logged separately as cold_s: {cold}; reps per config: {sorted(x for x in reps if x)}. "
        "Write in the paper: warm-up excluded, number of repetitions, that the GPU was synchronised, "
        "and what is included (encoder + reconstruction + host copy). Verify the synchronisation in the code.")


def check_ablation(R):  # 3.1
    files = sorted(glob.glob(os.path.join(R, "ablation", "*.json")))
    files = [f for f in files if not f.endswith(".spec.json")]
    cfg, Ts, shas = {}, set(), set()
    for f in files:
        d = jload(f)
        rr = recs(d)
        name = os.path.basename(f)[:-5]
        cfg[name] = rr
        Ts |= set(spec_of(d).get("Ts", []))
        shas.add((d or {}).get("meta", {}).get("ckpt_sha256"))
    lines = []
    for name, rr in cfg.items():
        for T in sorted(set(r["T"] for r in rr)):
            sel = [r for r in rr if r["T"] == T]
            lines.append(f"{name} T={T}: {mean(r['time_s'] for r in sel):.2f}s, "
                         f"peak {mean(r['peak_mb'] for r in sel):.0f} MB")
    wanted = {"sharing only", "fusion only", "compile only", "graph only", "sharing+fusion",
              "compile+graph", "sharing+compile+graph", "all"}
    have = {"fast_none": "earlier optimisations only", "fast_graph": "compile+graph (together)",
            "fast_post": "sharing+fusion (together)", "fast_full": "all"}
    present = [have[k] for k in have if k in cfg]
    add("3.1a", "Components evaluated separately (sharing, fusion, compile, CUDA graph) and in combinations",
        PART if len(cfg) >= 3 else MISS,
        f"Configs present: {present}. Sharing and fusion are always switched together, and so are compile "
        "and graph, so no single component can be attributed. Needs separate flags: "
        "sharing only, fusion only, compile only, graph only, and at least sharing+fusion, "
        "sharing+compile+graph, fusion+compile+graph.")
    add("3.1b", "Ablation at T = 4, 16 and 64", PASS if {4, 16, 64} <= Ts else PART,
        f"T covered: {sorted(Ts)}")
    add("3.1c", "Peak GPU memory reported in the ablation", PASS if all("peak_mb" in r for rr in cfg.values() for r in rr) and cfg else MISS,
        "peak_mb exists per record, but the ablation table does not show it yet. " + " | ".join(lines))
    add("3.1d", "Same checkpoint and environment as the main table", PASS if len(shas) == 1 else PART,
        f"{len(shas)} checkpoint hash(es) in ablation/. Ablation uses n_images="
        f"{spec_of(jload(files[0])).get('n_images') if files else '?'} versus 100 in main; say so in the table caption.")


def check_main(R):  # 3.2
    o = jload(os.path.join(R, "main", "orig.json"))
    f = jload(os.path.join(R, "main", "fast.json"))
    Ts = set(r["T"] for r in recs(f))
    nimg = len(set(r["image"] for r in recs(f)))
    add("3.2a", "PSNR, LPIPS and time for T = 4, 8, 16, 32, 64 (original and accelerated)",
        PASS if {4, 8, 16, 32, 64} <= Ts and recs(o) else PART, f"T covered: {sorted(Ts)}; images: {nimg}")
    figs = [os.path.basename(p) for p in glob.glob(os.path.join(R, "figs", "*.png"))]
    q = [x for x in figs if re.search(r"quality|pareto|tradeoff|psnr_vs_time|psnr_time", x)]
    add("3.2b", "Original and accelerated on one PSNR-vs-runtime graph", PASS if q else MISS,
        "No figure with a quality-vs-runtime plot. Existing figures: " + ", ".join(sorted(figs))
        + ". Add one: x = time per image (log), y = PSNR, one curve per implementation, points T=4..64.")
    s = jload(os.path.join(R, "summary.json")) or {}
    fit = s.get("fit", {})
    neg = [k for k, v in fit.items() if v.get("a_s", 0) < 0]
    add("3.2c", "Two-point runtime model replaced by an analysis of all measurements",
        PART if neg else PASS,
        f"The linear fit a+bT gives a negative fixed cost for {neg}, which is physically impossible: the "
        "linear model is wrong (time is not linear in T at small T). Drop the a+bT table; "
        "report the 5-point table, the speed-up vs T (summary.json: speedup_vs_T, 7 points) and the "
        "marginal cost per component between consecutive T values.")
    add("3.2d", "Runtime needed for a given quality", MISS,
        "Compute from the five points: time at which each implementation reaches a given PSNR "
        "(e.g. 28.40 dB is reached at T=16 by both; the accelerated one needs about 5x less time).")
    dec = open(os.path.join(R, "tables", "tab_main.csv")).read() if os.path.exists(os.path.join(R, "tables", "tab_main.csv")) else ""
    add("3.2e", "Decimal precision sufficient to support numerical equivalence", PART,
        "tab_main prints PSNR with 2 decimals, but the PSNR differences are ~1e-4 dB. Print PSNR with 3-4 "
        "decimals and LPIPS with 4, plus the mean difference and its 95% CI (available in summary.json: dpsnr_*).")


def check_scales(R):  # 3.4
    rows = []
    for sc in ["2", "2.5", "3", "3.5", "4"]:
        o = jload(os.path.join(R, "scale", f"orig_x{sc}.json"))
        f = jload(os.path.join(R, "scale", f"fast_x{sc}.json"))
        if not (recs(o) and recs(f)):
            rows.append(f"x{sc}: MISSING")
            continue
        to, tf = mean(r["time_s"] for r in recs(o)), mean(r["time_s"] for r in recs(f))
        rows.append(f"x{sc}: orig {to:.2f}s, fast {tf:.2f}s, speed-up {to / tf:.2f}x, queries/LR px = {float(sc) ** 2:.2f}")
    ok = all("MISSING" not in r for r in rows)
    Ts = spec_of(jload(os.path.join(R, "scale", "fast_x2.json"))).get("Ts")
    add("3.4a", "Scales x2, x2.5, x3, x3.5, x4: time and speed-up", PASS if ok else PART,
        f"All at T={Ts}. " + " | ".join(rows))
    add("3.4b", "Actual rollout-sharing ratio reported per scale", MISS,
        "Not logged. Count queries / unique nearest-LR pixels per image and scale (include the chunk-split "
        "overhead) and log it. Compare it with the expected s^2 and with the measured speed-up. "
        "The measured speed-up is far below s^2, which is the finding to explain.")
    add("3.4c", "Measurements compared with the expected scaling behaviour", MISS,
        "Plot speed-up vs s^2 and add a model: time = rollout part (~|P|*T) + per-query decode part (~Q*T).")


def check_adaptive(R):  # 3.3
    hits = [p for p in glob.glob(os.path.join(R, "**", "*"), recursive=True)
            if re.search(r"adapt|budget|patch", os.path.basename(p), re.I)]
    add("3.3", "Adaptive inference at matched budgets (fixed, random, Sobel, variance, DCT, with overhead)",
        PART if hits else MISS,
        "No adaptive-budget result in the archive. Needed: allocation strategies fixed / random / Sobel / "
        "variance / DCT at the same average T (or equal measured time), PSNR + LPIPS + time, and the time of "
        "difficulty estimation and allocation. Random allocation is the key control: if Sobel does not beat "
        "random at the same budget, present adaptivity as an application, not a contribution.")


def check_baseline(R):  # 3.5
    ab = glob.glob(os.path.join(R, "ablation", "fast_none.json"))
    add("3.5a", "Optimised version of the original as a baseline", PART if ab else MISS,
        "ablation/fast_none ('earlier optimisations only') exists. Describe exactly what it contains and "
        "confirm it is a fair baseline. Consider also a version with torch.compile only, no sharing.")
    add("3.5b", "Comparison with DIIF and LMF", SKIP,
        "Skipped by decision. State it in the limitations and discuss them qualitatively in Related Work.")


def check_equivalence(R):  # 3.6
    sp = spec_of(jload(os.path.join(R, "equiv", "fast.json")))
    seeds = sp.get("seeds", [])
    add("3.6a", "Multiple seeds, PSNR/LPIPS variation quantified", PASS if len(seeds) >= 3 else PART,
        f"seeds={seeds}, images={sp.get('n_images')}, T={sp.get('Ts')}. Use >= 3 seeds.")
    dumps = os.path.join(R, "equiv", "dumps")
    files = glob.glob(os.path.join(dumps, "*.npz"))
    if not files:
        add("3.6b", "Mean/max absolute error between implementations", MISS, "No dumps found.")
        return
    D = {}
    for p in files:
        m = re.match(r"(orig|fast)_T(\d+)_s(\d+)_i(\d+)\.npz", os.path.basename(p))
        if m:
            D[(m[1], int(m[2]), int(m[3]), int(m[4]))] = np.load(p)["crop"].astype("f8")
    out = []
    for T in sorted({k[1] for k in D}):
        cross, floor_o, floor_f = [], [], []
        for (impl, t, s, i), a in D.items():
            if t != T or impl != "orig":
                continue
            b = D.get(("fast", T, s, i))
            if b is not None:
                cross.append(np.abs(a - b))
            a1 = D.get(("orig", T, 1 - s, i)) if s == 0 else None
            if a1 is not None:
                floor_o.append(np.abs(a - a1))
            f0, f1 = D.get(("fast", T, 0, i)), D.get(("fast", T, 1, i))
            if s == 0 and f0 is not None and f1 is not None:
                floor_f.append(np.abs(f0 - f1))
        if cross:
            out.append(f"T={T}: orig-vs-fast same seed: mean {mean(c.mean() for c in cross):.2e}, "
                       f"max {max(c.max() for c in cross):.2e} | orig seed0-vs-seed1: mean "
                       f"{mean(c.mean() for c in floor_o):.2e}, max {max(c.max() for c in floor_o) if floor_o else float('nan'):.2e}"
                       f" | fast seed0-vs-seed1: mean {mean(c.mean() for c in floor_f):.2e}")
    add("3.6b", "Mean and max absolute error orig vs fast (128x128 crops, 4 images)", INFO, " || ".join(out))
    add("3.6c", "Floating-point difference separated from stochastic-position-encoding variation", MISS,
        "With the same seed, orig-vs-fast differs about as much as two seeds of the SAME implementation, "
        "so the SPE noise dominates and exactness is not demonstrated. You need a controlled-noise test on "
        "the released checkpoint: feed the same noise tensors to both implementations and report max/mean "
        "absolute error on the full output (expect ~1e-6 in fp32). The draft reports 7e-7 only for the "
        "retrained variant. Then report the seed-to-seed spread separately.")


def check_memory(R):  # 3.7
    rows, ck = {}, {}
    for f in glob.glob(os.path.join(R, "memory", "*_bs*.json")):
        if f.endswith(".spec.json"):
            continue
        m = re.match(r"(orig|fast|lte)_bs(\d+)\.json", os.path.basename(f))
        d = jload(f)
        if not m or not d:
            continue
        rows[(m[1], int(m[2]))] = d["records"]
        ck.setdefault(m[1], d["meta"].get("ckpt_sha256"))
    impls = sorted({k[0] for k in rows})
    bs = sorted({k[1] for k in rows})
    add("3.7a", "Peak memory vs query batch size", PASS if len(bs) >= 5 else PART, f"impl={impls}, bsize={bs}")
    oom = sorted(k for k, v in rows.items() if any(r.get("oom") for r in v))
    mo = jload(os.path.join(R, "main", "orig.json")) or {}
    main_ok = recs(mo) and not any(r.get("oom") for r in recs(mo))
    bad = [k for k in oom if k[0] == "orig" and main_ok and k[1] <= spec_of(mo).get("bsize", 0)]
    add("3.7b", "Memory results consistent with the main run", PART if bad else PASS,
        f"OOM entries: {oom}. main/orig ran at bsize={spec_of(mo).get('bsize')} without OOM, yet the memory test "
        "reports OOM at 100k-150k. Find out why (different T or code path, or memory left over from a "
        "previous job) before quoting memory ratios." if bad else f"OOM entries: {oom}")
    add("3.7c", "Same checkpoint for all compared implementations", PART,
        "orig and fast share base.pth. The LTE head uses random weights (cost independent of weights); "
        "state this clearly, or call it a speed reference only.")
    add("3.7d", "Dependence on image resolution", MISS,
        "Memory/time runs use one image (n_images=1) and a 200k-point probe. Add 3 image sizes "
        "(e.g. 256x256, 512x512, full DIV2K) at x4.")
    kinds = {spec_of(jload(f)).get("kind") for f in glob.glob(os.path.join(R, "**", "*.json"), recursive=True) if not f.endswith("spec.json")}
    add("3.7e", "Both EDSR and SwinIR encoders (if feasible)", SKIP,
        f"Layer kinds in results: {sorted(k for k in kinds if k)} (these are linear vs conv layers, not encoders). "
        "Optional; skip it and say that only the EDSR model was evaluated.")


def check_stages(R):  # 1.4
    st = jload(os.path.join(R, "stages", "fast.json"))
    so = jload(os.path.join(R, "stages", "orig.json"))
    if not st:
        add("1.4a", "Per-stage time breakdown to explain the speed-up limits", MISS)
        return
    lines = []
    for lab, d in (("orig", so), ("fast", st)):
        for r in (d or {}).get("records", []):
            keys = ["gen_feat_ms", "fid_ms", "predictor_ms", "recurrent_ms", "decoder_ms", "other_ms"]
            lines.append(f"{lab} T={r['T']}: total {r['total_ms']:.0f} ms; "
                         + ", ".join(f"{k[:-3]} {r.get(k, float('nan')):.0f}" for k in keys))
    add("1.4a", "Per-stage time breakdown exists (use it to explain what limits the speed-up)", INFO,
        " || ".join(lines))


# ----------------------------------------------------------------- tex lint
def read_tex(paths):
    txt = ""
    for pat in paths:
        for p in glob.glob(pat):
            with open(p, errors="ignore") as f:
                txt += "\n%%FILE " + p + "\n" + f.read()
    return txt


def section(txt, name):
    m = re.search(r"\\section\*?\{[^}]*" + name + r"[^}]*\}(.*?)(?=\\section|\\bibliography|\Z)", txt, re.S | re.I)
    return m.group(1) if m else ""


def lint_tex(txt, R):
    def lines_with(pattern, flags=re.I, limit=6):
        out = []
        for n, ln in enumerate(txt.splitlines(), 1):
            if ln.lstrip().startswith("%"):
                continue
            if re.search(pattern, ln, flags):
                out.append(f"l.{n}: {ln.strip()[:90]}")
        return out[:limit], len(out)

    hits, n = lines_with(r"TODO|\btbd\b|\\todo|XXX|\?\?")
    add("1.5a", "No TODO / TBD / ?? left", PASS if n == 0 else MISS, f"{n} found. " + " | ".join(hits))

    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", txt, re.S)
    ab = ab.group(1) if ab else ""
    bad = re.findall(r"speculative|negative result|parallel verification", ab, re.I)
    add("1.1a", "Abstract free of the speculative-decoding negative result", PASS if ab and not bad else (PART if not ab else MISS),
        "abstract not found" if not ab else f"matches: {bad}")
    add("1.1b", "Abstract states why reuse preserves quality", PASS if re.search(r"exact|identical|depends only", ab, re.I) else MISS, "")

    intro = section(txt, "Introduction")
    m = re.search(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", intro, re.S)
    items = re.findall(r"\\item", m.group(1)) if m else []
    add("1.1c", "About three contributions in the Introduction", PASS if 0 < len(items) <= 4 else MISS,
        f"{len(items)} items found. Patch-wise budgets mentioned among contributions: "
        f"{bool(m and re.search(r'patch|budget|adaptive', m.group(1), re.I))} (should be a supporting use, not a contribution)")

    rel = section(txt, "Related")
    miss = [k for k in ("DIIF", "LMF", "AnySR") if k.lower() not in rel.lower()]
    add("2.1", "Related Work discusses DIIF, LMF, AnySR", PASS if not miss else MISS, f"missing: {miss}")
    add("2.1b", "No claim that grouping by LR feature is new", PASS if not re.search(r"first to (group|share)|novel (grouping|sharing)", txt, re.I) else MISS, "")

    add("2.2", "Section 4.2 contains conditions, complexity analysis, SPE and batching discussion",
        PASS if all(re.search(p, section(txt, "Faster|exact|sharing"), re.I) for p in ("complexity|O\\(", "batch|chunk", "noise|SPE")) else PART,
        "Check that 4.2 has: separation generator/decoder, conditions for sharing, complexity in Q, |P|, T, L, effect of SPE noise, effect of chunking.")

    exp = section(txt, "Experiments")
    for lab, pat in (("GPU model", r"RTX|A6000|A100|V100|GPU"), ("PyTorch version", r"PyTorch\s*\d"),
                     ("CUDA version", r"CUDA\s*\d"), ("precision", r"single precision|fp32|float32|FP32"),
                     ("batch / queries per call", r"queries per call|batch size"),
                     ("warm-up / timing protocol", r"warm-?up")):
        add("1.2f", f"Setup states: {lab}", PASS if re.search(pat, exp, re.I) else MISS, "")
    env = (jload(os.path.join(R, "env.json")) or {}).get("env", {})
    gpu = env.get("gpu", "")
    for g in ("A6000", "A100", "RTX"):
        if g.lower() in exp.lower() and g.lower() not in gpu.lower():
            add("1.2g", f"Text mentions {g} but results were run on {gpu}", MISS, "Update every timing statement, Fig. 1 and Tables 1-3.")
            break

    hits, n = lines_with(r"SwinIR|variant|6,751|50,304")
    add("1.3", "No mixing of configurations (SwinIR / 'variant' / 6,751 params) with the EDSR model", PASS if n == 0 else MISS,
        f"{n} mentions to check. " + " | ".join(hits))
    add("1.3b", "Checkpoint table (source, encoder, #blocks, T_max, training setup)",
        PASS if re.search(r"EDSR", txt) and re.search(r"400 epochs|T_?\{?\\?max", txt) else MISS,
        "Expected: EDSR (pretrained), 4-layer Linear Transformer, T_max=64, 256 queries/iter, batch 16, Adam 1e-4 halved every 200 epochs, 400 epochs.")

    sec6 = section(txt, "negative|speculative")
    add("2.5", "Speculative decoding reduced to a short paragraph in the main text",
        PASS if len(sec6) < 1500 else PART, f"section length {len(sec6)} characters (aim for < ~1200, move the rest to the supplement)")

    decs = collections.Counter(len(x.split(".")[1]) for x in re.findall(r"\b2\d\.\d+(?=\s*dB)", txt))
    add("1.2h", "Consistent decimals for PSNR in text", PASS if len(decs) <= 1 else PART, f"decimals used: {dict(decs)}")

    hits, n = lines_with(r"\b(always|never|perfectly|guarantee[sd]?|completely|obviously|clearly better|no longer expensive)\b")
    add("1.5b", "Avoid unsupported absolute claims", PASS if n == 0 else PART, f"{n} candidates. " + " | ".join(hits))
    hits, n = lines_with(r"previous work|our earlier|\\cite\{[^}]*\}.*TODO")
    add("1.5c", "Reference [3] complete and cited in third person", PASS if n == 0 else MISS, " | ".join(hits))


# ----------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tex", nargs="*", default=[])
    ap.add_argument("--out", default="audit_report.md")
    a = ap.parse_args()
    R = locate_results(a.results)
    for fn in (check_reproducibility, check_main, check_ablation, check_scales, check_adaptive,
               check_baseline, check_equivalence, check_memory, check_stages):
        try:
            fn(R)
        except Exception as e:  # keep going, but show the problem
            add("ERR", fn.__name__, MISS, f"audit code failed: {e!r}")
    if a.tex:
        txt = read_tex(a.tex)
        if txt:
            lint_tex(txt, R)
        else:
            add("tex", "No .tex files matched", MISS, str(a.tex))
    else:
        add("tex", "Writing checks (1.1, 1.3, 1.5, 2.x)", INFO, "Pass --tex main.tex ... to enable them.")

    order = {MISS: 0, PART: 1, INFO: 2, SKIP: 3, PASS: 4}
    cnt = collections.Counter(r[2] for r in ROWS)
    lines = [f"# Audit of `{R}`", "", "Summary: " + ", ".join(f"{k} {cnt[k]}" for k in (MISS, PART, INFO, SKIP, PASS)), ""]
    for st in (MISS, PART, INFO, SKIP, PASS):
        sel = [r for r in ROWS if r[2] == st]
        if not sel:
            continue
        lines.append(f"## {st}")
        for cid, title, _s, det in sel:
            lines.append(f"- **[{cid}] {title}**" + (f"\n  {det}" if det else ""))
        lines.append("")
    rep = "\n".join(lines)
    with open(a.out, "w") as f:
        f.write(rep)
    print(rep)
    print(f"\n(report written to {a.out})")


if __name__ == "__main__":
    main()
