# parameters.md

Unified run.yaml schema for `frontends/run.py`.

## 1) CommonTaskSpec
Required per run:
- `backend`: `mace` | `equiformer_v2` | `gemnet_oc`
- `task`: `train` | `finetune` | `resume` | `evaluate` | `export`
- `run_dir`: output directory

Optional:
- `model_in`: checkpoint/weights path (required for finetune/resume/export)
- `model.manifest`: manifest.json path (preferred for finetune/resume)
- `model.weights`: state_dict path (optional if manifest has weights.path)
- `device`: `cuda` or `cpu` (default: `cuda`)
- `backend_python`: interpreter for native backend runs (FairChem)
- `backend_env`: extra environment variables (dict)
- `backend_args`: list of extra args for native backend runs

## 2) Data section
Paths may be a single LMDB file or a directory containing multiple `*.lmdb` shards.

Common keys:
- `data.train`
- `data.val`
- `data.test`
- `data.indices`: explicit list of indices for the active split (overrides sampling)
- `data.indices_path`: JSON list of indices (ignored if `data.indices` is set)

Sampling controls:
- `data.train_indices.max_samples`
- `data.train_indices.shuffle`
- `data.train_indices.seed`
- `data.train_indices.indices_path`
- `data.val_indices.max_samples`
- `data.val_indices.shuffle`
- `data.val_indices.seed`
- `data.val_indices.indices_path`

## 3) Train section (core trainer)
Used by MACE (core trainer). Optional unless noted.

Required for MACE:
- `train.input_json`: path to `model.json`

Common knobs:
- `train.epochs`
- `train.batch_size`
- `train.num_workers`
- `train.seed`
- `train.lr`
- `train.weight_decay`
- `train.optimizer`
- `train.amp` (bool)
- `train.accum_steps`
- `train.max_grad_norm`
- `train.save_every`
- `train.log_every` (log progress every N steps; default: ~20 logs/epoch)
- `train.progress_bar` (enable tqdm progress; `auto`/`true`/`false`, default: `auto`)
- `train.progress_mininterval` (tqdm refresh interval in seconds, default: 0.5)
- `train.freeze`: `head_only` to train only head parameters (default: none)

Energy/force loss:
- `train.energy_weight` (default 1.0)
- `train.force_weight` (default 1.0)

Head selection (MACE):
- `train.head_key` (defaults to `omat_pbe`)

Early stop:
- `train.early_stop_factor`

## 4) FairChem backends (EquiformerV2 / GemNet-OC)
Native training is used in Phase 1.

Required:
- `train.config_yml`: FairChem config path

Optional:
- `train.identifier`
- `train.seed`

`model_in` is passed as `--checkpoint` for finetune/resume.

## 5) Export
`task: export` writes standard artifacts from an existing checkpoint:
- Requires `model_in`
- Writes to `run_dir/artifacts/`

## 6) Evaluate / Infer (core runner)
Optional evaluation-only settings:
- `eval.energy_only`: bool (skip forces and evaluate energy only; works for `evaluate` and `infer`)
- `eval.amp`: bool (enable GPU autocast for evaluation/infer; implemented in core)
