#!/usr/bin/env bash
set -euo pipefail

# 使用 mace_env 环境，确保依赖齐全
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mace_env

# 推荐：解除 Torch 的 weights_only 限制
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
REPO_ROOT="$(cd ../../.. && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 可按需覆写以下超参（环境变量），例如 EPOCHS=2 bash train.sh
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-1000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-200}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-0}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# 数据路径（根据本机 oc22_data）
LMDB_TRAIN=${LMDB_TRAIN:-/home/kjt/Data/oc22_data/s2ef-total/train}
LMDB_VAL=${LMDB_VAL:-/home/kjt/Data/oc22_data/s2ef-total/val_id}
MODEL_JSON=${MODEL_JSON:-"${REPO_ROOT}/models/test2/model.json"}
RUN_DIR=${RUN_DIR:-"${REPO_ROOT}/models/test2/run"}
RUN_CONFIG=${RUN_CONFIG:-"${REPO_ROOT}/models/test2/run.yaml"}

cat > "${RUN_CONFIG}" <<EOF
runs:
  - name: test2_train
    task: train
    backend: mace
    run_dir: ${RUN_DIR}
    data:
      train: ${LMDB_TRAIN}
      val: ${LMDB_VAL}
      train_indices:
        max_samples: ${TRAIN_MAX_SAMPLES}
        shuffle: true
      val_indices:
        max_samples: ${VAL_MAX_SAMPLES}
    train:
      input_json: ${MODEL_JSON}
      epochs: ${EPOCHS}
      batch_size: ${BATCH_SIZE}
      num_workers: ${NUM_WORKERS}
      seed: ${SEED}
      lr: ${LR}
      weight_decay: ${WEIGHT_DECAY}
EOF

cd "${REPO_ROOT}"
python -m frontends.run --config "${RUN_CONFIG}" --run test2_train
