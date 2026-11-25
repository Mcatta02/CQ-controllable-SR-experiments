#!/usr/bin/env bash
# Run from your optimized repo root. Usage: GPU=<free gpu number> ./run_ladder.sh
# 24 processes (2 kinds x 4 modes x 3 chunk sizes), a few minutes each.
GPU=${GPU:?set GPU=<free gpu number>}
CKPT=${CKPT:-save/recurrent_lte_paper_repro/epoch-best.pth}
for KIND in linear conv; do
  for MODE in base graph post full; do
    for BS in 5000 10000 100000; do
      CUDA_VISIBLE_DEVICES=$GPU POINTWISE_KIND=$KIND MODE=$MODE BS=$BS TAG=cur CKPT=$CKPT \
        python benchmarks/bench_ladder.py 2>&1 | grep -E "^\[|T="
    done
  done
done
