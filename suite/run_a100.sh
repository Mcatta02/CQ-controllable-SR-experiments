#!/usr/bin/env bash
# Full run on one GPU: the standard suite, the Pareto analysis and the extended experiments. Run from the repository root, GPU otherwise idle.
#   bash suite/run_a100.sh [GPU_INDEX]
# Order = priority: (1) main suite on this GPU (same protocol as the V100 run), (2) scalability / scales, (3) correctness, (4) adaptive maps.
# Each step writes its own results folder; use --resume with the printed run id to continue after an interruption.
set -euo pipefail
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
sed -i "s/^gpu: \"[^\"]*\"/gpu: \"$GPU\"/" suite/suite_config.yaml
python suite/env_check.py || true
# 1. main suite (same presets as the V100 run; results folder name carries the GPU name)
python suite/run_suite.py --preset full --report 2>&1 | tee a100_main.log
# 2. Pareto of adaptive inference on this GPU
python suite/run_pareto.py --preset full 2>&1 | tee a100_pareto.log || true
# 3. extra experiments
python suite/run_extra.py --preset full --only scale scal 2>&1 | tee a100_extra_scale.log
python suite/run_extra.py --preset full --only noise exact stat 2>&1 | tee a100_extra_correct.log
python suite/run_extra.py --preset full --only advis 2>&1 | tee a100_extra_advis.log
echo "done. Package: tar czf a100_results.tar.gz suite/results/*A100* suite/results/extra_*"
