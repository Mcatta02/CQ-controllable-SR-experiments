"""Re-run the adaptive-vs-fixed experiments and print the Pareto frontier.

  python suite/run_pareto.py                       # full preset (40 images, 12 budget sets + fixed T=4,8,16,32,64)
  python suite/run_pareto.py --preset quick        # smoke test
  python suite/run_pareto.py --report-only <run_dir>   # skip the run, analyse an existing run

Runs  run_suite.py --only adaptive  (all budget sets, Sobel/variance/DCT/random, and the fixed T = 4,8,16,32,64
on the same images), then pareto_report.py on the new results folder. Outputs go to <run_dir>/pareto_*."""
import argparse, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--report-only", default=None, help="existing results folder to analyse")
    ap.add_argument("--main-run", default=None, help="results folder with main/fast.json (fills missing fixed T)")
    a = ap.parse_args()
    run_dir = a.report_only
    if not run_dir:
        before = set(glob.glob(os.path.join(HERE, "results", "*")))
        subprocess.run([sys.executable, os.path.join(HERE, "run_suite.py"), "--preset", a.preset, "--only", "adaptive"], check=True)
        new = sorted(set(glob.glob(os.path.join(HERE, "results", "*"))) - before, key=os.path.getmtime)
        if not new:
            sys.exit("no new results folder found")
        run_dir = new[-1]
    cmd = [sys.executable, os.path.join(HERE, "pareto_report.py"), "--results", run_dir, "--out", os.path.join(run_dir, "pareto")]
    if a.main_run:
        cmd += ["--main-run", a.main_run]
    subprocess.run(cmd, check=True)
    print("\nresults in", run_dir)


if __name__ == "__main__":
    main()
