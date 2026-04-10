#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=src/train_crosschain_lite.py

EPOCHS=5
BATCH_SIZE=2
DEVICE=cpu

SEEDS=(1 7 42)
TARGETS=(BSC_500 Polygon_500 Fantom_500)

for TARGET in "${TARGETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    TRAIN_PATH="data/train/crossgraphnet_lite_labeled/Ethereum.jsonl"
    TEST_PATH="data/train/crossgraphnet_lite_labeled/${TARGET}.jsonl"

    SAVE_DIR="results/crossgraphnet_lite_matrix/eth_to_${TARGET}_seed${SEED}"

    echo "=================================================="
    echo "[RUN-LITE] Ethereum -> ${TARGET} | seed=${SEED}"
    echo "[SAVE] ${SAVE_DIR}"
    echo "=================================================="

    ${PYTHON} -u ${SCRIPT} \
      --train_path "${TRAIN_PATH}" \
      --test_path "${TEST_PATH}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --save_dir "${SAVE_DIR}"

  done
done

echo "[DONE] all lite runs finished."