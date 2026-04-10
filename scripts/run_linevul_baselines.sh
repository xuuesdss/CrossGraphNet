#!/usr/bin/env bash
set -e

SEEDS=(1 7 42)
TARGETS=(BSC Polygon Fantom)

MODEL_PATH="/home/xu/FedVulGuard/CodeBert"

for TARGET in "${TARGETS[@]}"
do
  TRAIN_LABEL="data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl"
  TEST_LABEL="data/train/crossgraphnet_lite_labeled/${TARGET}_500.jsonl"

  for SEED in "${SEEDS[@]}"
  do
    echo "=========================================="
    echo "LineVul baseline: Eth -> ${TARGET}  seed=${SEED}"
    echo "=========================================="

    DATA_DIR="data/baselines/linevul/eth_to_${TARGET}_500_seed${SEED}"
    OUT_DIR="results/baselines/linevul/eth_to_${TARGET}_500_seed${SEED}"

    python baselines/linevul/prepare_linevul_data_from_labeled.py \
      --train_labeled ${TRAIN_LABEL} \
      --test_labeled ${TEST_LABEL} \
      --out_dir ${DATA_DIR} \
      --seed ${SEED} \
      --val_ratio 0.1

    python baselines/linevul/train_linevul.py \
      --data_dir ${DATA_DIR} \
      --out_dir ${OUT_DIR} \
      --model_name ${MODEL_PATH} \
      --seed ${SEED} \
      --max_length 512 \
      --epochs 5 \
      --train_batch_size 8 \
      --eval_batch_size 8 \
      --lr 2e-5 \
      --weight_decay 0.01 \
      --patience 2

  done
done

echo "======================================"
echo "All LineVul baseline experiments done."
echo "======================================"