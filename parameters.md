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

## 7) Experiments: sampling / action quality / DFT outbox（不属于 run.yaml）
以下内容属于 `experiments/` 目录，**不通过** `frontends.run` 启动：

### 7.1 采样配置（`experiments/sampling_rules/config.yaml`）
该文件是采样的主配置入口，分为两层：

1) 动作幅度（顶层 action blocks）：
- `rigid_translate`: `min_step` / `max_step`（Å）
- `rigid_rotate`: `min_deg` / `max_deg`（度）
- `push_pull`: `min_delta` / `max_delta`（Å）
- `dihedral_twist`: `min_deg` / `max_deg`（度），`bond_factor` / `bond_cap`（成键推断参数）
- `jitter`: `sigma`（Å）
- `md`: `enabled` / `integrator` / `min_temp_K` / `max_temp_K` / `min_steps` / `max_steps` / `dt_fs` / `friction`

2) pipeline 链路（`pipeline.*`）：
- `pipeline.action_plugins`: 动作后处理链（当前常用：`noise`）
  - `noise.sigma`: 高斯噪声标准差（Å）
  - `noise.clip`: 逐坐标裁剪上限（Å）
  - `noise.movable_only`: 是否只对 movable 原子加噪声
- `pipeline.validators`: 动作质量验证链（会直接决定动作是否被拒绝并重采样）
  - `min_dist`: 几何硬约束（`min_factor` / `hard_min`）
  - `quality`: 质量门（依赖 `force_pre`，基于力与动作幅度打分）
    - 常用键：`force_source` / `max_force` / `min_force` / `score_threshold`
- `pipeline.trigger`: DFT 候选触发规则（默认基于 `force_pre` 的 max/topK）
- `pipeline.recorders`: 记录器列表（常用：`step_trace` / `basin_registry`）

### 7.2 采样入口（`experiments/mace_pretrain/run_sampling.py`）
采样主入口为 `run_sampling.py`，核心 CLI 参数：
- 结构与输出：
  - `--run_dir`: 输出目录
  - `--structure_json`: 输入结构 JSON（numbers/positions/cell/fixed/tags）
  - `--config`: 采样配置 YAML（通常是 `experiments/sampling_rules/config.yaml`）
  - `--steps`: 最大尝试步数
  - `--target_basins`: 找到 N 个新 basin 后停止
  - `--resume`: 以追加方式续跑
- 模型推理：
  - `--manifest` / `--weights`
  - `--device`（cuda/cpu）
  - `--head`
  - `--amp`（推理 AMP）
- quench：
  - `--quench {none,fire,cg,bfgs,lbfgs}`
  - `--quench_fmax`
  - `--quench_steps`

典型调用（EquiformerV2 / fairchemv2 环境）：
```bash
PYTHONPATH=/home/kjt/projects/RL-reaction-path \
/home/kjt/miniforge3/envs/fairchemv2/bin/python \
experiments/mace_pretrain/run_sampling.py \
  --run_dir runs/sample_loop/sample_0000_eqv2_fire \
  --structure_json tmp/sample_0000.json \
  --config experiments/sampling_rules/config.yaml \
  --manifest models/equiformer_v2_oc22/manifest_bundle/manifest.json \
  --device cuda --amp \
  --quench fire --quench_fmax 0.1 --quench_steps 5000 \
  --target_basins 1 --steps 5000
```

采样输出（JSONL + NPZ）：
- `steps.jsonl` / `basins.jsonl` / `dft_queue.jsonl`
- `structures/*.npz`（StructureStore）
- 采样结束摘要会打印：`attempts_total` / `attempts_rejected`

### 7.3 动作质量诊断（`experiments/action_quality`）
- `action_quality.py`: 从 `steps.jsonl` 汇总动作质量统计
- `report_action_quality.py`: 命令行报告入口

常用用法：
```bash
PYTHONPATH=/home/kjt/projects/RL-reaction-path \
/home/kjt/miniforge3/envs/fairchemv2/bin/python \
experiments/action_quality/report_action_quality.py \
  --run_dir runs/sample_loop/sample_0000_eqv2_fire
```

### 7.4 DFT 出队（canonicalize + 去重）
- 入口：`experiments/mace_pretrain/outbox.py`
- 输出：`dft_submit.jsonl` / `dft_skip.jsonl`

### 7.5 可视化（extxyz 轨迹渲染）
- `experiments/visualization/render_movie.py`（OVITO）
- `experiments/visualization/blender_render.py`（Blender/Cycles，支持 GPU）
- 输入通常来自 `viz/trajectory*.extxyz`
