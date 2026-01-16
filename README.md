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
  - `experiments/mace_pretrain/` 高力触发 + DFT 出队
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

### 采样链路（experiments/sampling）
核心组件：
- 动作：平移/旋转/推拉/二面角/抖动
- quench：ASE FIRE / LBFGS（可选）
- basin：指纹/嵌入（轻量版本）
- 记录：`steps.jsonl` / `basins.jsonl` / `dft_queue.jsonl`
- 结构存储：`structures/*.npz`（StructureStore）

建议的调用方式：
```python
from experiments.sampling.selection import build_action_inputs
from experiments.sampling.pipeline import SamplingPipeline

selection_mask, candidates = build_action_inputs(structure)
record = pipeline.run_one(structure, selection_mask=selection_mask, candidates=candidates)
```

### DFT 队列与出队（experiments/mace_pretrain）
- **触发器**：基于 `force_pre`（max/topK），默认阈值见 `selector.py`
- **队列记录**：`dft_queue.jsonl`（含 queue_idx + 结构引用）
- **出队处理**：`outbox.py` 读取队列 → canonicalize → 去重 → 输出
  - `dft_submit.jsonl`（提交 DFT）
  - `dft_skip.jsonl`（去重丢弃）

> 注意：canonicalize 仅在出队阶段使用，StructureStore 与触发记录保持原始坐标。

## 日志规范
- `channel.md`：记录目标/计划/方案
- `implementation.md`：仅记录已实施改动与测试结果
- `daily_log.md`：每日总结（简述即可）

## 环境说明
不同后端可使用不同 Python 环境。对于 FairChem-native 运行，使用
`backend_python` 和 `backend_env`，或在启动前激活对应环境。

## 配置参考
统一 run.yaml 规范见 `parameters.md`。
