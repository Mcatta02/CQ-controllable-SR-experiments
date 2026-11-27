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

**Requirements.** You need Linux and an NVIDIA GPU. The paper's results were measured on an
A100 PCIe 40 GB, and the main suite was repeated on a Tesla V100 32 GB and an A100 PCIe 80 GB, all
with Python 3.12.2, PyTorch 2.5.1 and CUDA 12.1. GPUs with compute capability below 7.0 also work,
but `torch.compile` is then switched off automatically.

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
load: 400 epochs, SPE causal transformer, `num_pred=64`. Download `base.pth` from the
[Code Ocean capsule](https://codeocean.com/capsule/7687393/tree/v1) and place it at `pretrained/base.pth`.
Then check that you have the right file:

```bash
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

# Full run used for the paper (about 6 h on a V100, 5 h on an A100)
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

### Extended experiments

A second orchestrator, [suite/run_extra.py](suite/run_extra.py), runs experiments that go beyond the
main suite: larger scales, larger inputs, exactness under different query arrangements and the
local quality of adaptive inference. It uses the same config, the same one-job-per-process rule
and the same result format; its jobs are in [suite/worker_extra.py](suite/worker_extra.py) and its
report in [suite/extra_report.py](suite/extra_report.py).

```bash
python suite/run_extra.py --dry-run                  # list the jobs
python suite/run_extra.py --preset quick             # smoke test (a few minutes per group)
python suite/run_extra.py --preset full              # full run
python suite/run_extra.py --preset full --only scale scal
python suite/run_extra.py --report-only suite/results/extra_<run_id>
```

Results go to `suite/results/extra_<run_id>/`, the report to its `report/` folder. `--resume --run-id <id>`
continues an interrupted run.

| Group (`--only …`) | What it measures | Main outputs in `report/` |
|---|---|---|
| `scale` | Scales ×2 to ×12 at T = 4/16/64: time, PSNR / LPIPS, measured sharing ratio, speed-up against sharing, with a fitted cost model that gives the per-query decoding ceiling on the speed-up | `tab_scale_ext.tex`, `fig_scale_ext.pdf`, `fig_scale_quality.pdf` |
| `scal` | LR sizes 128 to 1024 against 5k to 150k queries per call: time, peak memory, out-of-memory cases, with a fitted time and memory model | `fig_scalability.pdf`, `scal_table.csv` |
| `noise` | Identical-noise comparison of the original and several accelerated configurations: generator outputs at every iteration, phase-head outputs and final RGB | `tab_noise_equiv.tex`, `noise_equivalence.csv` |
| `exact` | Same noise in every call, different query arrangements: shuffled order, queries of one LR pixel split across calls, rectangle / disk / random-subset regions | `tab_arrangements.tex`, `arrangement_exact.csv` |
| `stat` | Full images at non-integer scales, several chunk sizes and query orders, fresh noise per call: sharing ratio, time, PSNR | `arrangement_stat.csv` |
| `advis` | Adaptive inference: budget maps, per-tile PSNR change against a noise floor, edge / texture / smooth statistics, tile-boundary discontinuities | `fig_advis_img*.pdf`, `advis_*.csv` |

**How the identical-noise test works.** The original draws its SPE noise once per `model.query`
call and shares it across all queries of that call. With the same seed:

- `fast_SF` (query sharing and fused head on the original recurrent engine) uses the same random-number stream as the original, so no replay is needed;
- `fast_SFE_replay` and `fast_SFEC_replay` (accelerated engine, eager and compiled) receive the original's noise through `replay_noise` in [models/speculative.py](models/speculative.py);
- `fast_SFEC_ownnoise` uses the engine's own noise draw, so its differences are expected at the noise level;
- the CUDA-graph engine cannot take injected noise. It is covered by the `exact` group instead, which re-seeds before every call.

If `orig` (upstream) and `fast_none` (this repository with the fast path off) already differ, the
two code bases consume random numbers in a different order.

**Limits.**

- In `scal`, LR inputs larger than DIV2K's (about 510×339) are bicubic resizes of DIV2K images, so they measure time and memory only, not quality.
- At ×12 and 1024² LR the original implementation can run out of memory. This is recorded as a result (`oom`), not as a failure.
- The `exact` region tests use one block of full HR rows around the image centre. A region that does not intersect the block is recorded as skipped.

**Other datasets.** [suite/run_datasets.py](suite/run_datasets.py) runs the `main` experiment once
per dataset and scale (T ∈ {4, 16, 64}) and writes `datasets_summary.csv`. It expects HR `.png` images in:

| Name | Folder | Scale |
|---|---|---|
| `div2k_x2`, `div2k_x3`, `div2k_x4` | `load/div2k/DIV2K_valid_HR` | ×2, ×3, ×4 |
| `bsd100` | `load/benchmark/B100/HR` | ×4 |
| `urban100` | `load/benchmark/Urban100/HR` | ×4 |
| `manga109` | `load/benchmark/Manga109/HR` | ×4 |

```bash
python suite/run_datasets.py --datasets div2k_x2 div2k_x3 div2k_x4 bsd100 urban100
```

Manga109 has to be requested from its authors.

**Everything on one GPU.** [suite/run_a100.sh](suite/run_a100.sh) runs the main suite, the Pareto
analysis and the extended experiments in sequence on the given GPU:

```bash
bash suite/run_a100.sh 0
```

## 4. Repository layout

| Path | Contents |
|---|---|
| `models/` | Model code. The accelerated path is in `fast_recurrent.py` (compiled step, SPE tables, CUDA graph) and `fast_post.py` (fused post-processing); `speculative.py` holds the speculative-decoding experiments |
| `suite/` | Benchmark suite: `run_suite.py` (orchestrator), `worker.py` (one measurement), `make_report.py`, `env_check.py`, `difficulty.py` (tile difficulty for adaptive T); extended experiments: `run_extra.py`, `worker_extra.py`, `extra_report.py`, `run_datasets.py`, `run_a100.sh` |
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
[CQ-controllable-SR](https://github.com/kakitamedia/CQ-controllable-SR) and are used with the
author's permission. SPE (`others/spe`) is
MIT-licensed; see [others/spe/LICENSE](others/spe/LICENSE).
