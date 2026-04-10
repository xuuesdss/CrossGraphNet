#!/bin/bash

set -e

TRAIN=data/subsets/ETH_750_seed42.jsonl

CHAINS=("BSC" "Fantom" "Polygon")
SEEDS=(1 7)

for CHAIN in "${CHAINS[@]}"; do
  TEST=data/subsets/${CHAIN}_750_seed42.jsonl

  for SEED in "${SEEDS[@]}"; do

    OUT=results/crossgraphnet_lite_matrix/eth_to_${CHAIN}_750_seed${SEED}

    echo "========================================="
    echo "Running ETH -> ${CHAIN} | seed=${SEED}"
    echo "Output: ${OUT}"
    echo "========================================="

    PYTHONPATH=. python src/train_crosschain_lite.py \
      --train_path ${TRAIN} \
      --test_path ${TEST} \
      --seed ${SEED} \
      --device cuda \
      --save_dir ${OUT}

  done
done

echo "========================================="
echo "All scale experiments finished."
echo "========================================="