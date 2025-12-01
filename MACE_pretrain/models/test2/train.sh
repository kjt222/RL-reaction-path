#!/usr/bin/env bash
set -euo pipefail

# 使用 mace_env 环境，确保依赖齐全
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_env

# 推荐：解除 Torch 的 weights_only 限制，设置 PYTHONPATH
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/MACE_pretrain:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 可按需覆写以下超参（环境变量），例如 EPOCHS=2 bash train.sh
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NEIGHBOR_SAMPLE_SIZE="${NEIGHBOR_SAMPLE_SIZE:-512}"
LMDB_E0_SAMPLES="${LMDB_E0_SAMPLES:-2000}"

# 数据路径（根据本机 oc22_data）
LMDB_TRAIN=${LMDB_TRAIN:-/home/kjt/Data/oc22_data/s2ef-total/train}
LMDB_VAL=${LMDB_VAL:-/home/kjt/Data/oc22_data/s2ef-total/val_id}

cd ../..
python train_mace.py \
  --data_format lmdb \
  --lmdb_train "${LMDB_TRAIN}" \
  --lmdb_val "${LMDB_VAL}" \
  --lmdb_train_max_samples "${TRAIN_MAX_SAMPLES}" \
  --lmdb_val_max_samples "${VAL_MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --neighbor_sample_size "${NEIGHBOR_SAMPLE_SIZE}" \
  --lmdb_e0_samples "${LMDB_E0_SAMPLES}" \
  --output models/test2
