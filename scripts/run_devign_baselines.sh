#!/usr/bin/env bash
set -e

# seeds
SEEDS=(1 7 42)

# tasks
TARGETS=(BSC Polygon Fantom)

GRAPH_ETH="data/graphs_ast_norm_uid_v2/Ethereum.jsonl"

for TARGET in "${TARGETS[@]}"
do
  GRAPH_TARGET="data/graphs_ast_norm_uid_v2/${TARGET}.jsonl"

  TRAIN_LABEL="data/train/crossgraphnet_lite_labeled/Ethereum_500.jsonl"
  TEST_LABEL="data/train/crossgraphnet_lite_labeled/${TARGET}_500.jsonl"

  for SEED in "${SEEDS[@]}"
  do
    echo "=========================================="
    echo "Devign baseline: Eth -> ${TARGET}  seed=${SEED}"
    echo "=========================================="

    DATA_DIR="data/baselines/devign/eth_to_${TARGET}_500_seed${SEED}"
    OUT_DIR="results/baselines/devign/eth_to_${TARGET}_500_seed${SEED}"

    # prepare dataset
    python baselines/devign/prepare_devign_data_from_labeled.py \
      --graph_paths \
        ${GRAPH_ETH} \
        ${GRAPH_TARGET} \
      --train_labeled ${TRAIN_LABEL} \
      --test_labeled ${TEST_LABEL} \
      --out_dir ${DATA_DIR} \
      --seed ${SEED} \
      --val_ratio 0.1

    # train model
    python baselines/devign/train_devign.py \
      --data_dir ${DATA_DIR} \
      --out_dir ${OUT_DIR} \
      --seed ${SEED} \
      --epochs 50 \
      --batch_size 32 \
      --emb_dim 64 \
      --hidden_dim 64 \
      --num_layers 3 \
      --dropout 0.2 \
      --lr 1e-3 \
      --patience 10

  done
done

echo "======================================"
echo "All Devign baseline experiments done."
echo "======================================"