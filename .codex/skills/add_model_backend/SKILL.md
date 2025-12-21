---
name: add_model_backend
description: Add a new PES model backend (e.g., EquiformerV2) while keeping a single training/eval pipeline and stable interfaces.
---

# Add Model Backend

Use this skill when adding a new model backend (e.g., EquiformerV2) to the shared PES pipeline. Keep one training/eval/resume path, switching behavior only via model_type.

## Hard constraints (no add drama)
- Do not add new features, parameters, or defaults unless explicitly requested.
- If requirements are ambiguous, list them as spec holes; do not invent assumptions.
- Maintain global consistency across train/finetune/evaluate/resume/model_loader/metadata/config.
- Avoid duplication; consolidate to a single source of truth when overlap exists.

## Required interface contract
- forward must accept `batch.to_dict()` and flags `training` and `compute_force`.
- outputs must include `energy` and `forces` with the same shapes/units used by the loss.
- if adding embeddings, use `node_embed` and `graph_embed` keys (do not change loss behavior).
- model.json must fully describe the backend (architecture + statistics) and be validated via hash.

## Workflow
1) Extend the model registry with a new builder keyed by model_type.
2) Ensure model.json required fields are enforced consistently (builders + validation).
3) Keep save/load behavior unchanged; update export logic to include backend fields if needed.
4) Keep training entrypoints unified; no new scripts unless explicitly requested.
5) Update docs/Parameters checklist to match actual required fields.
6) Run smoke tests; document any gaps.

## Files to check
- model registry + builders
- train/finetune/evaluate/resume
- model_loader/read_model
- losses + dataloaders
- Parameters/README

## Acceptance commands (required)
Use these smoke tests (adjust paths if repo differs):

```bash
# Train 1 epoch
python model_pretrain/model-utils/train_mace.py \
  --data_format lmdb --lmdb_train $LMDB_TRAIN --lmdb_val $LMDB_VAL \
  --input_json $MODEL_JSON \
  --output_checkpoint /tmp/mace_smoke --output_model /tmp/mace_smoke \
  --epochs 1 --batch_size 2 --num_workers 0 --save_every 1

# Eval
python model_pretrain/model-utils/evaluate.py \
  --input_model /tmp/mace_smoke/mace_smoke_model.pt \
  --data_format lmdb --lmdb_path $LMDB_VAL --batch_size 2 --lmdb_val_max_samples 10

# Resume
python model_pretrain/model-utils/resume.py \
  --input_model /tmp/mace_smoke/mace_smoke_checkpoint.pt \
  --output_checkpoint /tmp/mace_resume --output_model /tmp/mace_resume \
  --epochs 2

# Finetune
python model_pretrain/model-utils/finetune.py \
  --input_model /tmp/mace_smoke/mace_smoke_checkpoint.pt \
  --output_checkpoint /tmp/mace_finetune --output_model /tmp/mace_finetune \
  --data_format lmdb --lmdb_train $LMDB_TRAIN --lmdb_val $LMDB_VAL \
  --epochs 1 --batch_size 2 --num_workers 0
```

If datasets are unavailable, record the validation gap explicitly.
