#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=src/train_crosschain.py

EPOCHS=5
BATCH_SIZE=2
DEVICE=cpu

SEEDS=(1 7 42)
TARGETS=(BSC_500 Polygon_500 Fantom_500)

for TARGET in "${TARGETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    TRAIN_PATH="data/train/crossgraphnet_lite_labeled/Ethereum.jsonl"
    TEST_PATH="data/train/crossgraphnet_lite_labeled/${TARGET}.jsonl"
    DFG_TRAIN_PATH="data/graphs_dfg/Ethereum.jsonl"
    DFG_TEST_PATH="data/graphs_dfg/${TARGET}.jsonl"

    SAVE_DIR="results/crossgraphnet_full/eth_to_${TARGET}_seed${SEED}"

    echo "=================================================="
    echo "[RUN] Ethereum -> ${TARGET} | seed=${SEED}"
    echo "[SAVE] ${SAVE_DIR}"
    echo "=================================================="

    ${PYTHON} -u ${SCRIPT} \
      --train_path "${TRAIN_PATH}" \
      --test_path "${TEST_PATH}" \
      --dfg_train_path "${DFG_TRAIN_PATH}" \
      --dfg_test_path "${DFG_TEST_PATH}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --device "${DEVICE}" \
      --save_dir "${SAVE_DIR}"

  done
done

echo "[DONE] all runs finished."