#!/usr/bin/env bash
set -euo pipefail

# 激活环境（可按需修改；已在目标环境下可导出 SKIP_CONDA=1 跳过）
if [[ -z "${SKIP_CONDA:-}" ]]; then
  if [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate mace_env
  else
    echo "WARNING: 未找到 ~/miniconda3/etc/profile.d/conda.sh，跳过自动激活。请手动激活 mace_env。" >&2
  fi
fi
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

# 路径与超参（可通过环境变量覆盖）
CHECKPOINT="${CHECKPOINT:-/home/kjt/mace_project/MACE_pretrain/models/MACE-MP-0-medium/oc22/MACE-MP-0-medium.pt}"
MODEL_JSON="${MODEL_JSON:-$(dirname "$CHECKPOINT")/model.json}"
if [[ ! -f "${MODEL_JSON}" && -f "$(dirname "$CHECKPOINT")/model-oc22.json" ]]; then
  MODEL_JSON="$(dirname "$CHECKPOINT")/model-oc22.json"
fi
LMDB_TRAIN="${LMDB_TRAIN:-/home/kjt/Data/oc22_data/s2ef-total/train}"
LMDB_VAL="${LMDB_VAL:-/home/kjt/Data/oc22_data/s2ef-total/val_id}"
RUN_DIR="${RUN_DIR:-$(dirname "$CHECKPOINT")/finetune}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
LMDB_TRAIN_MAX="${LMDB_TRAIN_MAX:-500000}"
LMDB_VAL_MAX="${LMDB_VAL_MAX:-50000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-0}"

# 优先使用 WSL 仓库根目录，若不可用再按脚本相对路径推算
if [[ -d /home/kjt/mace_project ]]; then
  REPO_ROOT="/home/kjt/mace_project"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
fi

RUN_CONFIG="${RUN_CONFIG:-${REPO_ROOT}/models/MACE-MP-0-medium/oc22/finetune.yaml}"
cat > "${RUN_CONFIG}" <<EOF
runs:
  - name: mace_finetune
    task: finetune
    backend: mace
    run_dir: ${RUN_DIR}
    model_in: ${CHECKPOINT}
    data:
      train: ${LMDB_TRAIN}
      val: ${LMDB_VAL}
      train_indices:
        max_samples: ${LMDB_TRAIN_MAX}
        shuffle: true
      val_indices:
        max_samples: ${LMDB_VAL_MAX}
    train:
      input_json: ${MODEL_JSON}
      epochs: ${EPOCHS}
      batch_size: ${BATCH_SIZE}
      num_workers: ${NUM_WORKERS}
      seed: ${SEED}
      lr: ${LR}
      weight_decay: ${WEIGHT_DECAY}
      plateau_patience: ${PLATEAU_PATIENCE}
EOF

cd "${REPO_ROOT}"
python -m frontends.run --config "${RUN_CONFIG}" --run mace_finetune
