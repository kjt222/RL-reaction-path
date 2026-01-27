# RL-reaction-path

统一的 PES 训练/评估/推理框架，支持多后端（MACE、EquiformerV2、GemNet-OC），
并提供用于 RL/AL 反应路径探索的实验性采样链路。

## 目录结构
- `backends/` 后端源码（vendor，默认不改）
- `adapters/` 后端适配层（每个模型一个子目录）
- `core/` 后端无关的训练/评估/ckpt/manifest
- `frontends/` CLI 入口（统一调度）
- `experiments/` 实验模块（采样/AL/微调）
  - `experiments/sampling/` 动作+quench+basin 采样链路
  - `experiments/action_quality/` 动作质量验证/噪声插件/诊断报告
  - `experiments/mace_pretrain/` 采样入口 + DFT 出队
  - `experiments/visualization/` 轨迹可视化（extxyz → 单帧/视频）
- `models/` 预训练权重与训练输出（artifacts/checkpoints）
- `runs/` 仅放 YAML 配置（不放输出）
- `Data/` 数据集（不跟踪）
- `tmp/` 临时脚本/一次性实验

## 入口
所有核心任务由统一 CLI 启动：

```bash
python -m frontends.run --config path/to/run.yaml --run <name>
```

前端会根据 backend 分流：
- MACE：使用 core trainer
- EquiformerV2/GemNet-OC：走 FairChem 原生训练，然后导出标准 artifacts

## 标准输出（训练/微调）
每次 run 写到 `run_dir`（通常在 `models/` 下）：
- `run_dir/artifacts/best_model.pt`
- `run_dir/artifacts/manifest.json`
- `run_dir/artifacts/model.json`
- `run_dir/checkpoints/checkpoint.pt`

## 训练（MACE core）
- `task: train/finetune/resume` 使用 core trainer
- `train.input_json` 必填（加载 `model.json`）
- `train.head_key` 选择输出 head（默认 `omat_pbe`）
- `train.freeze: head_only` 只训 head
- `train.progress_bar` / `train.log_every` 控制日志

## 评估 / 推理
- `task: evaluate` 或 `task: infer`
- `eval.energy_only: true` 可跳过 force
- `data.indices` 或 `data.indices_path` 可固定子集

## 实验：采样与微调（experiments）
实验模块不走 `frontends.run`，直接以 Python 入口调用，主要用于：
- 构型采样（动作 → quench → basin）
- 高力触发（DFT 候选队列）
- DFT 出队（canonicalize + 去重）
- 轨迹可视化（extxyz 生成预览帧/视频）

### 采样链路（experiments/sampling）
核心组件：
- 动作：`rigid_translate` / `rigid_rotate` / `push_pull` / `dihedral_twist` / `jitter` / `md`
- 动作插件（action plugins）：动作后处理链（当前默认用于加小幅高斯噪声）
- 动作质量验证（action quality validators）：
  - 几何硬约束：`min_dist` / `fixed` / `bond`
  - 质量门（gate）：`quality`（基于 `force_pre` 与动作幅度打分，低分直接拒绝并重采样）
- quench：ASE `FIRE` / `LBFGS` / `CG` / `BFGS`
- basin：指纹（histogram）为主的稳定 ID
- 记录：`steps.jsonl` / `basins.jsonl` / `dft_queue.jsonl`
- 结构存储：`structures/*.npz`（StructureStore，坐标不内嵌在 JSONL）

最常用入口是 `experiments/mace_pretrain/run_sampling.py`（直接跑完整采样闭环）。

EquiformerV2 / GemNet-OC（fairchemv2 环境）示例：
```bash
PYTHONPATH=/home/kjt/projects/RL-reaction-path \
/home/kjt/miniforge3/envs/fairchemv2/bin/python \
experiments/mace_pretrain/run_sampling.py \
  --structure_json tmp/sample_0000.json \
  --manifest models/equiformer_v2_oc22/manifest_bundle/manifest.json \
  --device cuda --amp \
  --config experiments/sampling_rules/config.yaml \
  --quench fire --quench_fmax 0.1 --quench_steps 5000 \
  --target_basins 1 --steps 5000 \
  --run_dir runs/sample_loop/sample_0000_eqv2_fire
```

采样配置集中在：`experiments/sampling_rules/config.yaml`（通过 `--config` 指定）。
其中 `pipeline.action_plugins` / `pipeline.validators` / `md.*` 是当前最关键的三个调参入口（详见 `parameters.md`）。

### DFT 队列与出队（experiments/mace_pretrain）
- **触发器**：基于 `force_pre`（max/topK），默认阈值见 `selector.py`
- **队列记录**：`dft_queue.jsonl`（含 queue_idx + 结构引用）
- **出队处理**：`outbox.py` 读取队列 → canonicalize → 去重 → 输出
  - `dft_submit.jsonl`（提交 DFT）
  - `dft_skip.jsonl`（去重丢弃）

> 注意：canonicalize 仅在出队阶段使用，StructureStore 与触发记录保持原始坐标。

### 轨迹可视化（experiments/visualization）
- `render_movie.py`：OVITO/CPU 渲染（输入 extxyz，输出 mp4）
- `blender_render.py`：Blender/Cycles 渲染（支持 GPU，输出帧/可选 mp4）

典型输入为采样输出的 `viz/trajectory*.extxyz`。

OVITO 渲染建议使用包装脚本（会自动设置 EGL/offscreen 相关环境变量）：
```bash
/home/kjt/projects/RL-reaction-path/scripts/ovito_render.sh \
  experiments/visualization/render_movie.py \
  --run_dir /path/to/run_dir \
  --renderer opengl --stride 1 --fps 12
```

说明：
- `--renderer opengl` 需要可用的 OpenGL 上下文（通常需要 GUI 会话）；
- 无 GUI/纯 SSH 时，优先使用 `--renderer tachyon` 或 Blender/Cycles。

## 日志规范
- `channel.md`：记录目标/计划/方案
- `implementation.md`：仅记录已实施改动与测试结果
- `daily_log.md`：每日总结（简述即可）

## 环境说明
不同后端可使用不同 Python 环境。对于 FairChem-native 运行，使用
`backend_python` 和 `backend_env`，或在启动前激活对应环境。

## 配置参考
统一 run.yaml 规范见 `parameters.md`。

## DFT 复现参考
OC22 VASP 复现设置与依据见 `docs/vasp_oc22_repro.md`。
