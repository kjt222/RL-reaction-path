---
name: logic_audit
description: Audit repo changes for global consistency, spec holes, interface drift, and duplication across training/eval/save-load paths.
---

# Logic Audit

Use this skill when the user wants a rigorous audit of change impact, consistency, or design integrity (especially around training, evaluation, save/load, and metadata). Prioritize finding logical gaps, implicit assumptions, and duplication.

## Hard constraints (no add drama)
- Do not add features, parameters, or default behaviors unless explicitly requested.
- If requirements are ambiguous, list them as spec holes; do not invent assumptions.
- Check global consistency across: train, finetune, resume, evaluate, model_loader, metadata, config/docs.
- Avoid duplicate implementations; enforce single source of truth and describe migration steps when needed.

## Audit workflow
1) Identify the change scope and all affected entry points.
2) Build a data-flow map: config -> model registry -> forward outputs -> loss -> save/load -> eval.
3) Enumerate invariants that must remain true (keys, shapes, units, files, hashes).
4) List spec holes (ambiguous requirements or implicit assumptions).
5) Scan for duplication/forks/conflicts; propose a single-source fix without changing behavior.
6) Provide a minimal, reversible change plan.
7) Provide required smoke tests and expected artifacts.

## Output structure
Use the following sections in order:
A. Global data-flow/dependency map
B. Invariants checklist
C. Spec holes
D. Duplication/fork/conflict scan (file + function + reason)
E. Minimal change plan (stepwise, reversible)
F. Acceptance commands (smoke tests + pass criteria)
G. Alignment with RL/AL goals (unified PES interface, multi-backend support)

## Acceptance commands (required)
If the repo uses LMDB for training, include these exact smoke tests (with placeholders):

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

If the repo paths differ, adjust the commands but keep the same coverage (train/eval/resume/finetune). If the datasets are not available, explicitly state the validation gap.
