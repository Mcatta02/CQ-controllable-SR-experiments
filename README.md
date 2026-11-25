# Fast inference for controllable-quality super-resolution (recurrent LTE)

Code for the paper submitted to NTIRE (anonymous for review).

This repository accelerates inference of the recurrent-LTE model from
[CQ-controllable-SR](https://github.com/kakitamedia/CQ-controllable-SR). That model predicts T Fourier
components one after another, so it costs more as T grows. The accelerated path gives the same
outputs from the same checkpoint and runs faster. It does this by:

- compiling a single recurrent step;
- applying stochastic positional encoding (SPE) from pre-built per-position tables;
- capturing the whole T-step rollout in a CUDA graph;
- fusing the post-processing.

The repository also includes the full benchmark suite that produces every table and figure in
the paper. It compares the unmodified upstream code against this implementation on the same
checkpoint, GPU and images.

## 1. Setup

**Requirements.** You need Linux and an NVIDIA GPU. The paper's results were measured on a
Tesla V100 32 GB with Python 3.12.2, PyTorch 2.5.1 and CUDA 12.1. GPUs with compute capability
below 7.0 also work, but `torch.compile` is then switched off automatically.

```bash
# from the repository root
conda env create -f environment.yml      # also installs the bundled SPE package (others/spe)
conda activate recurrent_lte
```

**Data (DIV2K validation set, 100 images).** Download it once:

```bash
mkdir -p load/div2k && cd load/div2k
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip && unzip DIV2K_valid_HR.zip && cd ../..
# -> load/div2k/DIV2K_valid_HR/0801.png ... 0900.png
```

**Checkpoint.** All experiments use one recurrent-LTE checkpoint, which both implementations
load: 400 epochs, SPE causal transformer, `num_pred=64`. Place it at `pretrained/base.pth`:

```bash
mkdir -p pretrained
# TODO(before release): download link for base.pth
# wget <URL> -O pretrained/base.pth
sha256sum pretrained/base.pth   # 08a8715b1e24c177ca521d380903411463ee726ad313be12dc25a4d67211a094
```

**Upstream baseline.** This is only needed for the comparison suite. Clone the unmodified
upstream repository at the commit we compare against:

```bash
git clone https://github.com/kakitamedia/CQ-controllable-SR third_party/CQ-controllable-SR
git -C third_party/CQ-controllable-SR checkout 695ae3fce0db6d30a5bf8de9e5a5afc60fc5f8e1
```

All paths are set in [suite/suite_config.yaml](suite/suite_config.yaml). Relative paths there are
resolved from the repository root. If your data or checkpoint live elsewhere, edit that file.

**Check the setup.** This takes about 1 minute and prints PASS / WARN / FAIL for each item:

```bash
python suite/env_check.py
```

## 2. Quick start: one test run

This runs ×4 super-resolution on 10 DIV2K images with T = 64, first with the original inference
path and then with the accelerated one:

```bash
python test.py --config configs/test_one.yaml --model pretrained/base.pth --num_pred 64 --save_dir results/orig
python test.py --config configs/test_one.yaml --model pretrained/base.pth --num_pred 64 --save_dir results/fast --fast
```

Both runs print PSNR, LPIPS and the average inference time per image. On a V100 the PSNR is
29.1134 dB for both runs. The average time is about 24.3 s per image for the original path and
5.6 s for the accelerated one. The accelerated average includes the one-time compilation on the
first image. Add `--save_img` to write the output images.

## 3. Reproducing the paper

The suite runs each measurement in its own process, one after another, on an idle GPU. Each job
writes one JSON file to `suite/results/<run_id>/`. The orchestrator also records the
environment, the git commits of both repositories and the checkpoint hash.

```bash
# Smoke test with a few images per experiment (about 40 min on a V100)
python suite/run_suite.py --preset quick --report

# Full run used for the paper (about 6 h on a V100)
python suite/run_suite.py --preset full --report
```

`--report` runs [suite/make_report.py](suite/make_report.py) at the end. It writes
`figs/*.pdf|png`, `tables/*.tex|csv` and `summary.json` into the run directory.

| Experiment (`--only …`) | What it measures |
|---|---|
| `main` | PSNR / LPIPS / time per image, original vs. accelerated, T ∈ {4, 8, 16, 32, 64} (100 images) |
| `equiv` | Output agreement between the two implementations, with the same seeds paired |
| `speed_T` | Time as a function of T |
| `scale` | Speed-up at scales ×2 to ×4 |
| `ablation` | Contribution of each component, alone and in combination |
| `stages` | Time spent in each stage (encoder / predictor / decoder / other) |
| `memory`, `resolution` | Peak memory and time vs. chunk size and image size |
| `kind` | Linear vs. 1×1-conv point-wise layers |
| `adaptive` | Per-tile adaptive T at a matched average T |
| `spec` | Speculative / Jacobi decoding (negative result, [models/speculative.py](models/speculative.py)) |

Useful options:

- `--only main equiv` runs only some experiments.
- `--dry-run` lists the jobs without running them.
- `--resume --run-id <id>` continues an interrupted run.

Further analyses on a finished run:

```bash
python suite/run_pareto.py --report-only suite/results/<run_id>     # quality-vs-time Pareto analysis
python suite/extra_analysis.py --results suite/results/<run_id> --out paper_extra
python suite/make_report.py suite/results/<run_id>                   # re-create figures/tables only
```

## 4. Repository layout

| Path | Contents |
|---|---|
| `models/` | Model code. The accelerated path is in `fast_recurrent.py` (compiled step, SPE tables, CUDA graph) and `fast_post.py` (fused post-processing); `speculative.py` holds the speculative-decoding experiments |
| `suite/` | Benchmark suite: `run_suite.py` (orchestrator), `worker.py` (one measurement), `make_report.py`, `env_check.py`, `difficulty.py` (tile difficulty for adaptive T) |
| `test.py`, `train.py`, `demo.py` | Evaluation (`--fast` turns on the accelerated path), training and demo, as in the upstream repository |
| `configs/` | Training configs (`train-div2k/`) and the test config (`test_one.yaml`) |
| `tests/` | Standalone parity checks between the original and accelerated paths |
| `benchmarks/` | Standalone micro-benchmarks and sweeps used during development |
| `pre_optimized_baseline/` | Model files from before the optimisation, kept for reference |
| `others/spe/` | Stochastic positional encoding package (bundled, installed by `environment.yml`) |

Run all scripts from the repository root, e.g. `python tests/parity_verify.py`.

## 5. Training (optional)

Training is unchanged from upstream and is not needed to reproduce the paper. It needs the
DIV2K training set in `load/div2k/DIV2K_train_HR`.

```bash
WANDB_MODE=disabled python train.py --config configs/train-div2k/recurrent_lte.yaml --name my_run --gpu 0
```

## Acknowledgements

The model, training code and SPE integration come from
[CQ-controllable-SR](https://github.com/kakitamedia/CQ-controllable-SR). SPE (`others/spe`) is
MIT-licensed; see [others/spe/LICENSE](others/spe/LICENSE).
