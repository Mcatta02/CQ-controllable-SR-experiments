"""Re-run the adaptive-vs-fixed experiments and print the Pareto frontier.

  python suite/run_pareto.py                       # full preset (40 images, 12 budget sets + fixed T=4,8,16,32,64)
  python suite/run_pareto.py --preset quick        # smoke test
  python suite/run_pareto.py --report-only <run_dir>   # skip the run, analyse an existing run

Runs  run_suite.py --only adaptive  (all budget sets, Sobel/variance/DCT/random, and the fixed T = 4,8,16,32,64
on the same images), then pareto_report.py on the new results folder. Outputs go to <run_dir>/pareto_*."""
import argparse, os, subprocess, sys, time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--config", default=os.path.join(HERE, "suite_config.yaml"))
    ap.add_argument("--run-id", default=None, help="results folder name (default: <time>_<preset>_pareto_gpu<N>)")
    ap.add_argument("--report-only", default=None, help="existing results folder to analyse")
    ap.add_argument("--main-run", default=None, help="results folder with main/fast.json (fills missing fixed T)")
    a = ap.parse_args()
    run_dir = a.report_only
    if not run_dir:
        # explicit run id, so that runs on several GPUs at the same time cannot pick up each other's folders
        gpu = yaml.safe_load(open(a.config)).get("gpu")
        run_id = a.run_id or f"{time.strftime('%Y%m%d_%H%M')}_{a.preset}_pareto_gpu{gpu}"
        subprocess.run([sys.executable, os.path.join(HERE, "run_suite.py"), "--config", a.config, "--preset", a.preset,
                        "--only", "adaptive", "--run-id", run_id], check=True)
        run_dir = os.path.join(HERE, "results", run_id)
    cmd = [sys.executable, os.path.join(HERE, "pareto_report.py"), "--results", run_dir, "--out", os.path.join(run_dir, "pareto")]
    if a.main_run:
        cmd += ["--main-run", a.main_run]
    subprocess.run(cmd, check=True)
    print("\nresults in", run_dir)


if __name__ == "__main__":
    main()
