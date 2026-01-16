# parameters.md

Unified run.yaml schema for `frontends/run.py`.

## 1) CommonTaskSpec
Required per run:
- `backend`: `mace` | `equiformer_v2` | `gemnet_oc`
- `task`: `train` | `finetune` | `resume` | `evaluate` | `infer` | `export`
- `run_dir`: output directory

Optional:
- `model_in`: checkpoint/weights path (required for finetune/resume/export)
- `model.manifest`: manifest.json path (preferred for finetune/resume)
- `model.weights`: state_dict path (optional if manifest has weights.path)
- `device`: `cuda` or `cpu` (default: `cuda`)
- `backend_python`: interpreter for native backend runs (FairChem)
- `backend_env`: extra environment variables (dict)
- `backend_args`: list of extra args for native backend runs

Repo convention:
- `runs/` only stores YAML configs.
- `run_dir` should point under `models/` (e.g. `models/<model-name>/<run-name>`).

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
- `train.drop_last` (bool, optional; when `train.cycle_steps` is enabled this is forced to `true`)
- `train.cycle_steps` (optimizer updates per cycle; `<=0` disables step-cycles)
- `train.max_grad_norm`
- `train.save_every`
- `train.log_every` (log progress every N steps; default: ~20 logs/epoch)
- `train.progress_bar` (enable tqdm progress; `auto`/`true`/`false`, default: `auto`)
- `train.progress_mininterval` (tqdm refresh interval in seconds, default: 0.5)
- `train.freeze`: `head_only` to train only head parameters (default: none)
- `train.scheduler_step_unit` (`epoch` | `cycle`, default: `epoch`; non-plateau schedulers only)

Cycle behavior:
- If `train.cycle_steps > 0`, checkpoint + eval happen **every cycle**; `train.save_every` applies only when cycle is disabled.

Step definitions (core trainer):
- **micro_step** = one backward() on a micro-batch
- **update_step/global_step** = one optimizer.step() (post-accumulation)
- `train.log_every` and progress bars use **update_step** (not micro-batch)

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

## 7) Experiments: sampling / DFT outbox（不属于 run.yaml）
以下内容属于 `experiments/` 目录，**不通过** `frontends.run` 启动：

采样配置（动作/步长等）：
- `experiments/sampling_rules/config.yaml`

采样输出（JSONL + NPZ）：
- `steps.jsonl` / `basins.jsonl` / `dft_queue.jsonl`
- `structures/*.npz`（StructureStore）

DFT 出队（canonicalize + 去重）：
- 入口：`experiments/mace_pretrain/outbox.py`
- 输出：`dft_submit.jsonl` / `dft_skip.jsonl`

触发规则（高力优先）：
- 入口：`experiments/mace_pretrain/selector.py`
