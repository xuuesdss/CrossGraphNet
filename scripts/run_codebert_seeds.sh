#!/usr/bin/env bash
set -e

SEEDS=(7 42)
TARGETS=("BSC" "Polygon" "Fantom")

SRC_TRAIN="data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl"

for seed in "${SEEDS[@]}"; do
  for target in "${TARGETS[@]}"; do

    echo "==============================="
    echo "Seed=${seed} | ETH500 -> ${target}500"
    echo "==============================="

    SRC_TEST="data/train/crossgraphnet_lite_labeled/${target}_500.jsonl"
    OUT_DATA="data/codebert_baseline/eth500_to_${target}500_seed${seed}"
    OUT_RESULT="results/baselines/codebert/eth500_to_${target}500_seed${seed}"

    # 1. 生成 baseline 数据
    python baselines/codebert/prepare_codebert_data_from_labeled.py \
      --src_train ${SRC_TRAIN} \
      --src_test ${SRC_TEST} \
      --out_dir ${OUT_DATA} \
      --seed ${seed}

    # 2. 训练 CodeBERT baseline
    python scripts/train_codebert_baseline.py \
      --train_path ${OUT_DATA}/train.jsonl \
      --val_path ${OUT_DATA}/val.jsonl \
      --test_path ${OUT_DATA}/test.jsonl \
      --output_dir ${OUT_RESULT} \
      --max_length 256 \
      --batch_size 16 \
      --epochs 15 \
      --lr 1e-3 \
      --seed ${seed} \
      --frozen

  done
done

echo "ALL DONE."