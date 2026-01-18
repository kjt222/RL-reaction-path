# Plan Log

## 2026-01-16 Codex
- 目标：查证并说明如何删除 Codex/ChatGPT 对话历史，给出官方步骤并询问用户使用场景（网页/桌面/CLI）。

## 2026-01-16 Codex
- 目标：确认下一步实施顺序与最小闭环（P0→P1→P2→P3），并收敛高力指标与触发规则选择。

## 2026-01-16 Codex
- 目标：确定“可维护的最小数据库”（不复制原始 Data，优先用 indices 映射），用于测试 AL + 采样微调闭环。

## 2026-01-16 Codex
- 目标：在 `experiments/sampling` 实现最小采样闭环骨架（动作/几何/目标生成/验证/管线），不进入 core，动作均匀随机；为后续触发统计提供结构化输出。

## 2026-01-16 Codex
- 目标：补齐采样系统三处关键扩展点的放置位置：Force/PES 接口层、全局盆地图/去重 registry、Recorder 钩子（不改 pipeline 行为，仅预留接口）。

## 2026-01-16 Codex
- 目标：在采样管线中接入 Recorder hook，并实现三种 Recorder（StepTrace / BasinRegistry / ALCandidate），用于分层记录与后续统计。

## 2026-01-16 Codex
- 目标：定义三类 Recorder 的字段规范与最小记录策略（完整性、去重、DFT 触发），在不膨胀磁盘的前提下保证可复现。

## 2026-01-16 Codex
- 目标：评审“主要问题清单”（Force 缺失/触发器/随机性/指纹稳定/收敛处理），给出修复优先级与字段细化方案。

## 2026-01-16 Codex
- 目标：基于问题清单给出明确判断与修复顺序，并说明哪些问题需要先补字段/接口、哪些可后置。

## 2026-01-16 Codex
- 目标：实现 StructureStore（NPZ）与结构引用；更新三类 Recorder（StepTrace/BasinRegistry/DFTCandidate）按引用写入，DFT 仅保存 x_pre。

## 2026-01-16 Codex
- 目标：在 `experiments/mace_pretrain` 新增 DFT 队列近似去重模块（粗桶+RMSD），保持与采样层解耦。

## 2026-01-16 Codex
- 目标：实现 slab 参考系 canonicalize（平移+旋转）并用于 StructureStore/DFT 去重，避免 quench 导致坐标漂移。

## 2026-01-16 Codex
- 目标：将 canonicalize 仅用于 DFT 去重的最终提交阶段，不用于 StructureStore 或触发记录。

## 2026-01-16 Codex
- 目标：设计 DFT 出队流程（不改写原队列），用 queue_idx 续扫；输出 dft_submit.jsonl / dft_skip.jsonl。

## 2026-01-16 Codex
- 目标：实现 slab 参考系的原子分类（S_int/S_surf/S_ads）并在动作采样中禁止 core-core 组合，保证表面/吸附体可动、内部稳定。

## 2026-01-16 Codex
- 目标：接入力统计（pre/min）并实现 mace_pretrain trigger_fn，记录 topK=3/5 与 max/mean，用于 DFT 触发。

## 2026-01-16 Codex
- 目标：更新 README.md 与 parameters.md（补齐 experiments/sampling、DFT outbox、日志规范等；同步删除过期字段）。

## 2026-01-16 Codex
- 目标：核查 P3/P4 是否已完整解决（force 统计与 trigger_fn），给出条件与剩余缺口。

## 2026-01-16 Codex
- 目标：设计 experiments 下的采样/出队入口位置与命名（稳定后再迁移到 frontends）。

## 2026-01-16 Codex
- 目标：实现 experiments/mace_pretrain 入口脚本：run_sampling.py（支持 resume）与 run_outbox.py；run_dir 必填，config 可选。

## 2026-01-18 Codex
- 目标：实现 force_fn loader（manifest+weights）并接入 run_sampling 入口，支持 head/device 参数。

## 2026-01-18 Codex
- 目标：让 SamplingPipeline 支持 force_fn 返回 (E,F)，记录 energy_pre/energy_min，并在 quench 侧补能量输出。

## 2026-01-18 Codex
- 目标：新增 Level-1 几何直方图 basin（元素对分桶），作为硬 ID；保留 embedding 作为软特征。

## 2026-01-18 Codex
- 目标：新增 basin 相似度计算模块（Level-2，embedding+ΔE）；在 basins.jsonl 中可选记录 energy/embedding 特征。

## 2026-01-15 Codex
- 目标：创建“全局常开”日志技能，强制维护 `channel.md` 与 `implementation.md` 的创建与更新。
- 范围：全局（/home/kjt/.codex/skills），用于本仓库所有任务。
- 步骤：新增技能目录与 SKILL.md；在 `implementation.md` 记录本次改动与验收。

## 动作 + Quench + Basin + AL 架构重构（2026-01-15）

### 1. 分析（基于已有信息）
- **问题本质**：表面反应构型空间是高维且有效区域极稀疏；随机扰动/MD 采样大多落在“散架/真空/无效碰撞”，算力利用率极低。
- **模型误差分布**：当前 MACE 的主要误差集中在**高力区域**与**关键原子（tag2/tag1）**，方向一致性（cos）明显偏低；这说明“能力/数据分布缺口”大于“整体缩放问题”。
- **策略必然性**：先做 **动作 → quench → basin_id** 的离散化，把连续空间转成“盆地图”，再做图搜索扩展；并用 **AL** 把 DFT 预算集中投向高力/关键区域。
- **采样限制必要性**：宏动作 + 结构化约束（位点/片段/距离范围）是避免散架的唯一可控手段；单原子自由扰动维度过高且不可控。
- **关键风险**：
  - 只保留 x_min 可能**抹平高力构型**，导致模型难学“反应区”；
  - basin_id 过粗/过细会导致“漏路径”或“图爆炸”；
  - 动作分布偏置会让新盆地率下降、采样陷入低能区域。

### 2. 修改实施方案（分阶段）
> 目标：建立可扩展、可插拔的“动作/弛豫/盆地/AL”闭环，先跑通核心链路，再逐步引入 RL/扩散。

#### 阶段 P0：基线度量与约束定义（1~2 周）
- **目标**：给采样与训练建立统一的评估口径，避免“盲跑”。
- **输出**：
  - 采样日志标准：有效动作率、quench 成功率、新盆地率、重访率。
  - 误差分组指标：按 tags / 高力分位 bins 的 F_RMSE、cos。
  - 散架判定规则（硬约束）：键长上限、吸附体-表面距离范围、最小接触距离等。
- **验收**：能在一次采样轮次中稳定输出上述指标；无效构型被过滤且比例可控。

#### 阶段 P1：动作引擎 + Quench 最小闭环（2~3 周）
- **目标**：动作生成与 quench 成功对接，稳定产出候选构型/盆地。
- **动作接口（可扩展）**：
  - 片段刚体平移/旋转
  - 二面角扭动
  - 成键对推近/拉远
  - 位点 hop
  - 小扰动（同盆地探索）
- **Quench 策略**：
  - 短步数最小化/短弛豫（可配置算法/步数/阈值）
  - **保留 x_pre 与 x_min**（比例可配置），避免高力信息被抹平
- **验收**：固定预算下，新盆地率明显高于随机采样；散架率在可接受阈值内。

#### 阶段 P2：Basin 识别与图结构（2~3 周）
- **目标**：把采样结果离散成“盆地节点 + 动作边”的图。
- **Basin_id 设计**（可插拔）：
  - 结构嵌入（模型 embedding）
  - 能量容差
  - 局部结构指纹/键图
- **图结构**：
  - 节点：x_min + basin_id
  - 边：动作类型 + 参数 + x_pre/x_min 映射
- **验收**：去重稳定、路径可追溯；盆地图规模随采样线性增长且无爆炸。

#### 阶段 P3：AL 选点与回灌（2~4 周）
- **目标**：把 DFT 预算集中到“模型最难学”的区域。
- **选点策略（可组合）**：
  - 高力/高误差（|F| 分位或预测残差）
  - tags 加权（tag2 / 反应区优先）
  - 不确定度（committee / dropout）
  - 新颖性（basin 去重 / embedding 距离）
- **回灌**：将 DFT 点纳入训练集，优先提升反应区方向一致性（cos）。
- **验收**：高力 bins 与 tag2 的误差显著下降；整体 RMSE 随 AL 轮次单调改善。

#### 阶段 P4：RL/扩散（可选加速）
- **目标**：提升“新盆地命中率/跨能垒效率”。
- **原则**：仅在 P1~P3 稳定后引入，避免因引擎不稳定导致策略失效。
- **验收**：相同预算下，新盆地率显著提升且不牺牲稳定性。

### 已确认参数（用于收敛设计）
1. **tags 语义（已确认）**：
   - OCP 默认语义：tag0=slab/bulk，tag1=surface layer，tag2=adsorbate。
   - 依据 val_id_1k 的统计验证：tag2 以 H/C/N/O 为主且 z 高度最高、|F| 最大；tag0 金属比例高且 z 低、|F| 最小；tag1 介于二者之间。
2. **quench 细节**：
   - 最小化器：FIRE 或 L-BFGS（ASE 实现均可）
   - 步数上限：50–200（先用 100）
   - 收敛阈值：fmax=0.1 eV/Å（可在 0.05–0.1 调）
   - 短 MD：可选 5–20 步（低温/小步长），仅作为辅助越过几何卡点
3. **动作参数边界 + top-K**：
   - 刚体平移：0.2–2.0 Å（小扰动 0.2–0.6，大扰动 0.6–2.0）
   - 刚体旋转：5°–45°（大扰动可到 90°，先不启）
   - 二面角扭动：±10°–±60°
   - 推近/拉远：Δr=±0.1–0.8 Å（先用上限 0.5）
   - 位点 hop：位点法向上方 2–4 Å + 0–1 Å 切向扰动
   - top-K：K=8–32；候选混合（官能团/未饱和位点优先 + 少量随机表面点 + 可选 Voronoi/空穴点）；排序=距离合理 + 不重叠 + 多样性
4. **散架判定阈值（硬约束）**：
   - 原子对距离 < 0.8 Å（或 0.7×共价半径和）直接拒绝
   - slab 内部最近邻键长偏离参考 >20–30% 拒绝
   - 吸附体关键键：若不希望断键，限制 <1.5×；若允许解离，则仅限制极端拉断（>2.5×）
   - 解吸判定：adsorbate 质心-表面距离 >6–8 Å 记为无效/降权
5. **basin_id 判定阈值（两级）**：
   - Level-1 快速指纹：元素 + 邻接图 + 局部环境直方图；相似度 >0.98 视为同盆地
   - Level-2 精判：|ΔE_pred|<0.02–0.05 eV（可放宽到 0.1）；embedding 距离取 val 内 5% 分位半径；或局部 RMSD<0.2–0.5 Å
6. **AL 预算与频率**：
   - 每轮 DFT：64 点（范围 32–128）
   - 触发：每 1e4–1e5 个候选或每新增 200–500 新盆地
   - 先跑 3 轮（~200 点）观察 tag2@高力误差变化，再扩到 5–10 轮
7. **评估口径（主指标）**：
   - 主指标：tag2 且 |F| 80–100% 分位的 RMSE/MAE + tag2 的 mean(cos_i)
   - 辅助指标：全局 F_RMSE、tag0 F_RMSE（防整体崩）

### 结构决策（全局组织方案）
- **不引入 core2，不做大规模重排**：现有 `core/` 已按功能分层，调整风险大收益小。
- **新增子包即可**：将 AL/采样/指标新增到 `core/` 下，保持旧 import 兼容。
- **contracts.py 不迁移**：可在新模块内 re-export，但不移动文件，避免破坏依赖。
- **实验策略隔离**：微调 AL 与正式 AL 放在 `experiments/`，仅依赖 `core/` 的稳定接口。

#### 结构草案（尽量完整）
```
core/
  __init__.py
  batch/
  ckpt/
  data/
  eval/
  runner/
  train/
  registry.py
  contracts.py            # 维持原位（核心协议，不迁移）
  losses.py
  manifest.py
  metrics.py              # 原有通用指标
  transforms.py

  # 新增：AL + 指标（采样模块不放 core）
  al/
    __init__.py
    selectors.py          # 选择器接口（score/propose）
    scorers.py            # 通用打分：uncertainty/novelty/high-F
    dataset.py            # 候选池/DFT 队列数据结构
    budget.py             # DFT 预算与轮次调度
    logging.py            # AL 轮次统计日志
  metrics_ext/
    __init__.py
    high_force.py         # 高力分位指标
    topk_force.py         # 每构型 top-K 力指标

experiments/
  mace_pretrain/
    config.yaml
    selector.py           # MACE 微调触发（高力优先）
  al_global/
    config.yaml
    selector.py           # committee/embedding 为主（正式 AL，尚在实验区）
  sampling/
    __init__.py
    pipeline.py           # action -> quench -> basin 主流程
    schema.py             # 结构/动作/结果数据结构
    geometry.py           # 几何工具（旋转/向量/邻接）
    validators.py         # 结构有效性判定（可选钩子）
    recorders.py          # 录制钩子（on_sample）
    forcefield/
      __init__.py
      base.py             # Force/PES 接口层
    graph/
      __init__.py
      registry.py         # 全局去重/盆地图注册
    actions/
      __init__.py
      base.py             # 动作基类/参数协议
      rigid_translate.py
      rigid_rotate.py
      push_pull.py
      dihedral_twist.py
      jitter.py
      targets.py          # 候选对象生成（原子对/二面角）
    quench/
      __init__.py
      base.py             # quench 接口
      ase_fire.py
      ase_lbfgs.py
    basin/
      __init__.py
      base.py             # basin_id 接口
      fingerprint.py      # 快速指纹 Level-1
      embed_energy.py     # embedding + ΔE Level-2
  sampling_rules/
    config.yaml           # 动作参数与权重分布

frontends/
  run.py                  # 统一入口，仅调度 core pipeline

docs/
  design/
    channel.md            # 本方案记录
```

#### 迁移原则
- **不移动旧文件**，只在新增模块中引用 `core.contracts` 等核心协议。
- **最小侵入**：旧训练/评估流程不改或仅加可选 hook。

### 阶段性落地清单（当前仅做“测试AL + 采样微调”）
> 说明：此阶段不做完整 AL 主流程；允许在 `experiments/mace_pretrain` 内部出现与未来 AL 重复的逻辑实现。

#### Step 1：结构搭建（本周）
- 建立 `core/al`、`core/sampling`、`core/metrics_ext` 的空骨架与占位文件。
- 建立 `experiments/mace_pretrain` 与 `experiments/sampling_rules` 的最小配置骨架。
- 仅提供接口与占位实现，不进入训练/评估主流程。

#### Step 2：测试 AL + 采样微调闭环（下一步）
- **采样侧**：在 `experiments/sampling` 中实现 3–4 个基础动作（平移/旋转/推近拉远/二面角），接入 quench 与 basin_id 的最小实现。
- **AL 侧（微调专用）**：实现 `max_F/mean_F` 选点策略；优先选择高力构型（不修改动作）。
- **指标侧**：实现 high-F / top-K 力误差指标，用于微调早停与评估。
- 与现有训练主流程保持松耦合（可作为独立脚本或独立入口执行）。

**Step 2 验收标准**
- 能在纯 Python 环境下实例化 `experiments/sampling` 的动作与管线，并生成 `SampleRecord`。
- dihedral 目标生成可用（结构≥4 原子时可产生候选；无候选时抛出可读错误）。
- 采样模块不进入 core（仅位于 `experiments/sampling`）。

#### Step 3：完整 AL（未来，不在当前阶段）
- 引入 committee / embedding 选点策略（暂放 experiments/al_global，后续可上移至 core）。
- 与图搜索/盆地图扩展深度耦合。

### 建议实施顺序（阶段目标 + 优先级表）
> 当前阶段聚焦：测试AL + 采样微调。完整 AL 与 RL/扩散后置。

**阶段目标概览**
- **P0**：建立评估口径与约束（高力指标、散架判定、日志标准）。
- **P1**：动作 + quench 最小闭环（稳定产生候选构型与 x_min）。
- **P2**：basin_id 去重与盆地图雏形（节点/边结构可追溯）。
- **P3**：微调 AL 触发器（max_F / mean_F）+ 高力指标用于早停。
- **P4**：完整 AL（committee/embedding）与图搜索深度耦合（未来）。

建议的实施顺序
  ┌────────┬──────────────────────────────────────────────┬────────┐
  │ 优先级 │                    任务                      │ 复杂度 │
  ├────────┼──────────────────────────────────────────────┼────────┤
  │ P0     │ 高力指标/散架判定/日志口径落地               │ 低     │
  │ P1     │ 动作库 + quench 最小闭环（x_pre/x_min）      │ 中     │
  │ P2     │ basin_id 去重 + 盆地图骨架                  │ 中     │
  │ P3     │ mace_pretrain 触发器（max_F/mean_F）         │ 低     │
  │ P4     │ 完整 AL（committee/embedding，图搜索耦合）   │ 高     │
  └────────┴──────────────────────────────────────────────┴────────┘

## Epochs -> Cycle (Step-based Training Loop)

### Goal
Replace epoch-based training control with step-based cycles so that validation,
early-stopping, and checkpointing happen at fixed optimizer-step intervals.
This shortens feedback loops when a full epoch is too slow.

### Definition
- **micro_step** = one backward() on a micro-batch.
- **update_step / global_step** = one optimizer.step() (after accumulation).
  - If `accum_steps = 8`, then 8 micro-steps = 1 update_step.
  - **Cycle uses update_step (not micro_step).**

### Proposed Design
1. **New config knobs**
   - `train.cycle_steps`: optimizer steps per cycle (eval/save cadence).
   - `epochs` remains the outer training length (default behavior unchanged).
   - **Disable rule:** if `cycle_steps` is missing or `<= 0`, step-cycle is disabled
     and behavior stays epoch-based (legacy).

2. **Training loop behavior (cycle is scheduling only)**
   - **Keep epoch semantics for data**: dataloader still runs full epochs with normal shuffling.
   - **Cycle does NOT reset or reshuffle data.** It only controls eval/checkpoint cadence.
   - Track `global_step` as **optimizer updates** (post-accumulation).
   - Define `steps_per_epoch = num_batches_per_epoch / accum_steps`
     (guaranteed integer when `drop_last=True` and batch count divisible).
   - At each cycle boundary (every `cycle_steps` update_steps):
     - **Save `checkpoint.pt` first** (so progress is durable even if eval fails).
     - Run validation.
     - Update scheduler (ReduceLROnPlateau) on that val loss.
     - Compare with best; if improved, write `best_model.pt`.
   - If `cycle_steps > steps_per_epoch`, allow it but **warn**:
     cycle may span multiple epochs; eval/save still trigger by update_step.
   - **Early stop** still uses `val_loss` (keeps energy/force weights meaningful) and
     can stop immediately after a cycle eval; by default training still runs full epochs.
   - **Scheduler note:** ReduceLROnPlateau uses `patience` in *number of evals*.
     When eval frequency increases (shorter cycles), `patience` should be scaled up
     accordingly to avoid overly aggressive LR drops.
   - **Checkpoint cadence:** checkpoints are written at cycle boundaries; on crash,
     resume starts from the **last completed cycle** (not exact crash point).

3. **Metrics & logging**
   - Keep current log format, but replace "Epoch" with "Cycle" where applicable.
   - Progress bar covers one cycle (not a full epoch).
   - Log `global_step` as **update_step** (optimizer updates).
   - When showing progress bars, use update_step counts (not micro_step).
   - Log `cycle_index`, `steps_in_cycle` in info lines.

4. **Resume behavior**
   - Checkpoint must store at least:
     - `epoch_idx`
     - `global_step` (update_step)
     - `step_in_epoch` (update_step count within epoch)
     - `micro_step_in_accum` (0..accum_steps-1)
     - `batch_offset` (explicit unit: **batch index**, not sample)
     - `sampler_state` (base_seed + epoch_seed + offset; include rank/world_size if DDP)
   - Resume continues from the **last completed cycle** boundary.
   - **Offset source of truth:** `batch_offset` is authoritative.
     - `step_in_epoch` and `global_step` are redundant checks derived from `batch_offset`.
     - **Formula:** `step_in_epoch = batch_offset // accum_steps`
     - **Invariant:** `batch_offset % accum_steps == 0` (cycle boundary only).
     - Because cycle boundaries occur *after* optimizer.step(), `micro_step_in_accum`
       should always be **0** at resume; otherwise treat as inconsistency and error.
   - **No data repetition within an epoch**: resume must skip already-consumed batches.
     Requires deterministic ordering + offset-based fast-forward.
   - **Config consistency guard:** on resume, verify `batch_size`, `drop_last`,
     `accum_steps`, and dataset length match; otherwise refuse or warn loudly.

### Files Likely to Touch
- `core/train/trainer.py` (loop, logging, save/val schedule)
- `core/ckpt/save_load.py` (checkpoint payload: add global_step/cycle_index/step_in_epoch)
- `core/data/dataloaders.py` (build per-epoch shuffled indices; no DataLoader shuffle)
- `core/data/indices.py` (helper to build deterministic per-epoch permutation)
- `parameters.md` (document new config keys)
- `README.md` (optional: describe cycle-based training)
 - **FairChem reference (no code change)**: FairChem already supports step-based
   eval/save via `optim.eval_every` / `optim.checkpoint_every`, but its `step`
   counts micro-batches. We keep our core loop on optimizer-update steps.

### Acceptance Criteria
- Validation + early stop can run within a few hours (cycle-based), not days.
- `cycle_steps` controls eval/checkpoint cadence; no data reset between cycles.
- Default behavior unchanged when `cycle_steps` is not set.
- Resume continues from the correct cycle without repeating data within an epoch.

### Open Questions
- Default `cycle_steps`: fixed number vs. derived from dataset size?
- If `StepLR` is used: interpret `step_size` in **cycles** (not epochs) or add a new
  `step_size_steps` to preserve old meaning?
### Scheduler Policy (Resolved)
- **ReduceLROnPlateau**: `scheduler.step(val_metric)` only after cycle eval.
### Scheduler Step Unit (Resolved)
- **Default behavior:** step **per epoch** (preserves old semantics).
- **Optional override:** `train.scheduler_step_unit = "cycle"` to step per cycle.
  - This avoids unintentionally speeding up LR changes when enabling cycles.

### Deterministic Data Order (Selected: Option B)
- For each epoch, build a **fixed shuffled index list** using `seed + epoch`.
- DataLoader uses `shuffle=False` and the precomputed indices.
- Resume mid-epoch uses:
  - `epoch` (current epoch index)
  - `step_in_epoch` (optimizer updates completed within epoch)
  - `accum_steps` to compute **micro-batches consumed**
- **Batch offset is authoritative:** `batch_offset` counts batches (not samples).
  - With `drop_last=True`, `sample_offset = batch_offset * batch_size`.
  - Skip the first `sample_offset` indices from the epoch list.
- This avoids the “repeat early samples” bias and guarantees reproducibility.
- **DDP/worker note:** for multi-GPU, generate the full perm **once**, then shard by
  rank; also set `worker_init_fn`/`generator` with fixed seeds so per-worker randomness
  is reproducible.

### Accumulation Boundary Policy (Resolved)
- When `cycle_steps` is enabled, **force `drop_last=True`**.
- This guarantees `len(epoch_batches) % accum_steps == 0` and keeps offsets stable.
- If user sets `drop_last=False` with `cycle_steps`, raise a clear error.

## FairChem 训练损失统一为 per-atom

### 目标
让 FairChem 的**原生训练**（EquiformerV2 / GemNet-OC）在能量 loss 上采用 per‑atom 权重，
保证大小构型对能量 loss 的贡献一致（与 MACE 的 per‑atom 逻辑一致）。

### 现状定位
- 能量 loss 在 FairChem 原生训练里由：
  - `backends/equiformer_v2/fairchem/ocpmodels/trainers/forces_trainer.py::_compute_loss`
  - `backends/equiformer_v2/fairchem/ocpmodels/trainers/energy_trainer.py::_compute_loss`
  计算，当前是**按构型数量平均**（per‑config）。
- loss 实际由 `DDPLoss` 包装，最终会用 `batch_size` 做归一化。

### 方案（最小改动）
1. **新增配置开关**
   - `optim.energy_loss_mode: per_atom | per_config`
   - 默认保持 `per_config`（兼容旧行为）
2. **forces_trainer / energy_trainer 修改**
   - 计算 `natoms = cat(batch.natoms)`，`total_atoms = natoms.sum()`
   - 若 `energy_loss_mode == "per_atom"`：
     - `self.loss_fn["energy"](out["energy"], target, batch_size=total_atoms)`
     - 让 `DDPLoss` 用 **总原子数**做归一化 → per‑atom 权重
   - 若 `per_config`：保持原逻辑不变
3. **DDP 一致性**
   - `DDPLoss` 内部会 all_reduce `batch_size`，因此 per‑atom 权重在多卡下仍一致。
4. **配置落地**
   - 在 EquiformerV2 / GemNet‑OC 的训练 YAML 中显式写：
     - `optim.energy_loss_mode: per_atom`
5. **日志标注（可选）**
   - 启动时打印 `energy_loss_mode`，避免混淆。

### 关注点
- 该方案仍使用现有能量 normalizer（按构型统计）。  
  若要做到“**能量归一化也按 per‑atom**”，需要额外改 normalizer 统计方式，
  属于更大改动，可后续再做。

### 受影响文件
- `backends/equiformer_v2/fairchem/ocpmodels/trainers/forces_trainer.py`
- `backends/equiformer_v2/fairchem/ocpmodels/trainers/energy_trainer.py`
- 相关 FairChem 训练 YAML（`equiformerv2.yml` / `fairchemv2.yml` 等）

## 训练输出必须包含 JSON（manifest / model.json）

### 需求
训练输入来自一个完整模型目录（含 `.pt` + `model.json/manifest.json`），
训练输出到目标目录时必须保留同等完整度：
- **至少要有**：`best_model.pt` + `manifest.json`
- **最好也复制**：`model.json`（确保与训练时使用的 config 一致）

### 现状问题
- core trainer 训练中只写 `checkpoints/*.pt`
- 仅在训练自然结束时才写 `artifacts/manifest.json`
- **中断**时不会自动导出 JSON，导致目标目录只剩 `.pt`，后续管理混乱

### 改动方向
1. **每次保存 checkpoint 时，同步维护 artifacts**
   - 如果 `artifacts/manifest.json` 不存在，立即导出
   - 并将当前使用的 `model.json` 拷贝到 artifacts 目录（例如 `artifacts/model.json`）
2. **保证 resume/中断恢复后仍能导出完整 artifacts**
3. **输出目录语义一致**
   - 训练完成后目标目录应包含：`checkpoints/`, `artifacts/`，且 artifacts 内有 `manifest.json + model.json`

### 受影响文件（预计）
- `core/train/trainer.py`（保存时触发 export）
- `core/ckpt/export.py`（复用现有 export）
- `adapters/mace/read_model.py` 或 `adapter.model_spec`（获取 model.json 路径）

---

## 2026-01-09 Claude1

### 代码审查：核心模块分析

完成了对 adapters/ 和 core/ 关键代码的全面审查。

#### 架构总结

项目采用**插件式架构**，通过 `AdapterBase` 抽象接口统一三个MLP后端：

```
frontends/run.py (统一入口)
       │
       ├─→ core/train/trainer.py (核心训练器 - MACE)
       │         └─→ adapters/mace/adapter.py
       │
       └─→ adapters/fairchem/runner.py (原生FairChem训练)
                 └─→ backends/.../main.py
```

核心契约：
- `CanonicalBatch`: 统一batch格式 (z, pos, cell, pbc, energy, forces, ptr, natoms, head...)
- `ModelOutputs`: 统一输出格式 (energy, forces, node_embed, graph_embed)
- `AdapterBase`: 6个核心方法 (build_model, model_spec, select_head, make_backend_batch, forward, loss)

#### 发现的问题/疑点

**P1. 能量损失模式不一致**
```python
# adapters/mace/adapter.py:365
energy_loss_mode="per_atom",

# adapters/fairchem/adapter_base.py:301
energy_loss_mode="per_config",
```
→ 委员会校准时需要注意损失尺度差异

**P2. 断点续训RNG状态不完整**
`trainer.py` 保存了 `train_state` 但缺少：
- `torch.get_rng_state()`
- `numpy.random.get_state()`
- dataloader worker RNG

→ 对RL的replay buffer精确复现性有影响（关联plan.md C8约束）

**P3. FairChem head_parameters() 覆盖度**
如果模型没有 `energy_block/force_block` 也没有 `out_mlp_E/out_mlp_F`，会抛错。
→ 需确认是否覆盖所有EquiformerV2/GemNet-OC变体

**P4. Embedding捕获方式差异**
- MACE: hook `model.products[-1]`
- FairChem: hook `model.norm` 或 `model.out_mlp_E`

→ RL Policy网络使用embedding时需统一处理

#### 与RL/AL系统对接评估

| plan.md约束 | 当前支持情况 | 备注 |
|-------------|--------------|------|
| C1 Policy可微 | ✅ | adapters返回node_embed/graph_embed |
| C5 Committee校准 | ⚠️ | 损失模式不一致需处理 |
| C8 完整Checkpoint | ⚠️ | 需扩展以支持RL状态 |

#### 改动方向建议

1. **统一energy_loss_mode**: 在FairChem adapter中也支持per_atom模式（或在委员会校准时做转换）

2. **扩展checkpoint格式**: 为RL/AL预留字段
   ```python
   checkpoint["rl_state"] = {
       "value_net_state": ...,
       "policy_net_state": ...,
       "replay_buffer": ...,
       "rng_states": {...},
   }
   ```

3. **统一embedding接口**: 在AdapterBase中定义 `extract_embeddings()` 方法（部分已实现）

**等待用户确认后再进行具体修改。**

## 2026-01-18 Codex
- 目标：撤回/推迟 Level-2 basin 相似度模块（先聚焦硬 ID 与采样链路）。
- 变更：删除 experiments/sampling/graph/similarity.py；暂不提供 basin 相似度计算。

## 2026-01-18 Codex
### CPU quench + GPU 力推理的完整流程（设计）

**目标**  
quench 的优化循环在 CPU 上跑，力/能量由 GPU 模型推理提供。保持接口清晰、可插拔、最少耦合。

**组件职责**
1) **ForceFn（GPU）**  
   - 输入 Structure → 输出 (energy, forces)  
   - 运行在 GPU（torch no_grad），返回 numpy 给 CPU。

2) **ASE Calculator（桥接层）**  
   - 实现 ASE Calculator 接口  
   - 内部调用 ForceFn  
   - 让 ASE 优化器把“能量/力计算”委托给 GPU 模型。

3) **QuenchRunner（CPU）**  
   - ASE FIRE / L-BFGS  
   - 使用上面的 Calculator  
   - 只负责位置更新与收敛判断。

**数据流**
```
structure_pre (CPU)
  └─> ASE Optimizer (CPU)
        └─> calculator.calculate() -> force_fn(structure) (GPU)
              └─> (E, F) numpy -> CPU
        └─> 更新 positions
structure_min + quench stats
```

**落点（代码位置）**
- `experiments/sampling/quench/force_fn_calculator.py`  
  实现 `ForceFnCalculator`（ASE 计算器，调用 force_fn）
- `experiments/sampling/quench/ase_fire.py` / `ase_lbfgs.py`  
  支持：传入 calculator + fixed 约束  
- `experiments/mace_pretrain/run_sampling.py`  
  增加 quench 选项：`--quench {none,fire,lbfgs}` + `--quench_fmax/--quench_steps`  
  组装 `ForceFnCalculator` 并传给 quench，接入 pipeline。

**验收点（只跑链路）**
- steps.jsonl 出现 `force_pre`  
- basins.jsonl 出现 `basin_id`  
- dft_queue.jsonl 出现触发记录  
- quench 打开时 `structure_min` 与 `structure_pre` 可区分

### 采样 quench 默认值调整（执行中）
- 默认：`fmax=0.10 eV/Å`、`max_steps=200`
- 若不收敛：先提到 300；仍不行再放宽 `fmax=0.12–0.15`
- 不收敛样本不进入 basin 统计（仅保留记录）

### JitterAction 可复现性（执行中）
- jitter 随机噪声改为使用 pipeline RNG（采样时生成 seed，apply 时复现）。

---

## 2026-01-16 Claude1

### experiments/sampling 代码审查：盆地网络采样流程

完成对 `experiments/sampling/` 代码的全面审查，分析当前实现与目标需求的匹配度。

#### 当前实现架构

```
x0 ──→ Action ──→ x_pre ──→ Validator ──→ Quench ──→ Basin ──→ SampleRecord
       │                      │              │          │           │
       ├─ JitterAction        ├─ min_dist    ├─ FIRE    ├─ fingerprint
       ├─ RigidTranslate      ├─ fixed_mask  └─ LBFGS   └─ embed_energy
       ├─ RigidRotate         └─ bond_stretch
       ├─ DihedralTwist
       └─ PushPull
```

**核心流程** (`pipeline.py:50-118`):
1. 随机选择 action，最多尝试 max_attempts=5 次
2. 应用 action 得到 x_pre
3. 依次运行 validators，任一失败则 valid=False 且跳过 quench/basin
4. 若 valid，执行 quench 得到 x_min
5. 若有 basin identifier，对 x_min 计算 basin_id
6. 构建 SampleRecord 并通过 recorders 输出

#### 优点

1. **模块化设计好**：Action/Validator/Quench/Basin/Recorder 各自独立，易于扩展
2. **Validator 实现了 C4 约束**：`validate_min_distance_structure` 使用 ASE 共价半径，元素感知 ✓
3. **Recorder 设计灵活**：`ALCandidateRecorder` 支持自定义 trigger_fn，为高F筛选预留接口

#### 问题与批判

**P1. FingerprintBasin 不够鲁棒**
```python
# basin/fingerprint.py:22-28
distances = pairwise_distances(structure.positions)
dist_vals = np.round(distances[iu], self._round_decimals)  # round_decimals=3
digest = hashlib.sha256(payload.tobytes()).hexdigest()[:16]
```
- 问题：0.001Å 精度对热涨落太严格，同一 basin 的不同构型可能产生不同 ID
- 影响：basin 去重失效，图节点爆炸
- 建议：改用 embedding clustering（如 plan.md C2 描述），或放宽精度到 0.01Å

**P2. JitterAction 破坏随机状态可控性**
```python
# actions/jitter.py:27
noise = np.random.normal(scale=sigma, size=(indices.shape[0], 3))  # ❌ 全局RNG
```
- 问题：`apply()` 使用 `np.random` 而非传入的 `rng`
- 影响：结果不可复现，破坏 checkpoint 的 RNG 恢复
- 修复：应改为 `rng.normal(scale=sigma, size=...)`

**P3. Force 信息未在流程中传递**
- `Structure` 和 `SampleRecord` 都没有 `forces` 字段
- 但 AL 筛选需要 force 信息
- 当前需求：基于 max_F 或 mean_top_k_F 筛选高力构型
- 问题：需要额外调用 calculator，或在 quench 过程中保存 force

**P4. trigger_fn 尚未实现**
```python
# mace_pretrain/selector.py
"""Finetune AL selector (max/mean force)."""
# (空)

# al_global/selector.py
"""Global AL selector (future)."""
# (空)
```
- 当前目标无法直接运行，需要实现 force-based trigger

**P5. Quench 不收敛的处理缺失**
- `ASEFIREQuench` 返回 `converged=False` 时，pipeline 仍然计算 basin_id
- 未收敛的 x_min 可能不是真正的局部极小，basin_id 无意义
- 建议：对 converged=False 的 record 做标记或跳过 basin 计算

**P6. BasinRegistry 只做内存去重，无持久化**
- 重启后丢失所有已见 basin
- 长时间运行需要持久化 registry

#### 当前目标适配性评估

| 目标 | 当前支持 | 缺口 |
|------|----------|------|
| 高F构型筛选 | ⚠️ ALCandidateRecorder 框架在 | trigger_fn 未实现，Structure 无 forces |
| Basin 去重 | ⚠️ FingerprintBasin 可用 | 精度问题可能导致假阴性 |
| 后续迁移到 core | ✓ 模块化好 | 需统一 RNG 管理 |

#### 建议实施顺序

1. **紧急**：在 `Structure` 或 `SampleRecord.metrics` 中添加 forces 字段
2. **紧急**：实现 `mace_pretrain/selector.py` 的 force-based trigger
3. **重要**：修复 `JitterAction.apply()` 的 RNG 问题
4. **可选**：对 quench 不收敛做特殊处理
5. **后续**：替换 FingerprintBasin 为 embedding-based clustering

#### 高F筛选的具体实现建议

```python
# mace_pretrain/selector.py
def build_max_force_trigger(threshold: float = 1.0):
    """Trigger when max force exceeds threshold."""
    def _trigger(record: SampleRecord) -> tuple[bool, dict]:
        forces = record.metrics.get("forces")  # 需要在 pipeline 中填充
        if forces is None:
            return False, {}
        max_f = np.linalg.norm(forces, axis=1).max()
        return max_f > threshold, {"max_force": float(max_f)}
    return _trigger

def build_topk_force_trigger(k: int = 10, threshold: float = 0.8):
    """Trigger when mean of top-k forces exceeds threshold."""
    def _trigger(record: SampleRecord) -> tuple[bool, dict]:
        forces = record.metrics.get("forces")
        if forces is None:
            return False, {}
        norms = np.linalg.norm(forces, axis=1)
        topk = np.sort(norms)[-k:]
        mean_topk = float(topk.mean())
        return mean_topk > threshold, {"mean_topk_force": mean_topk}
    return _trigger
```

**等待用户确认后再进行具体修改。**

---

## 2026-01-18 Claude1

### 架构审查：experiments/sampling 采样链路（更新版）

完成对更新后代码的全面审查。之前发现的问题 **大部分已解决**，整体架构 **认可**。

#### 已解决的问题

| 原问题 | 解决方案 | 位置 |
|--------|----------|------|
| P3 Force信息缺失 | `force_fn(structure)` → metrics["force_pre/force_min"] | pipeline.py:120-170 |
| P4 trigger_fn未实现 | `build_max_force_trigger`, `build_topk_force_trigger`, `build_default_trigger` | selector.py |
| P1 FingerprintBasin精度问题 | 新增 `HistogramBasin`（元素对距离直方图），置换不变且更稳定 | basin/histogram.py |
| P5 Quench不收敛 | `if not quench_result.converged: basin_ok = False` | pipeline.py:173-175 |
| P2 JitterAction RNG | 使用 seed 保证可复现 | actions/jitter.py |

#### 架构优点

1. **模块化清晰**：Action/Validator/Quench/Basin/Recorder 五层解耦
2. **数据流明确**：
   ```
   x0 → Action → x_pre → force_fn → Validator → Quench → Basin → SampleRecord
                   ↓                               ↓
              force_pre                     force_min + basin_id
   ```
3. **输出分职责**：
   - `steps.jsonl` - 每步调试/分布分析
   - `basins.jsonl` - 新盆地发现
   - `dft_queue.jsonl` - AL触发点
4. **DFT去重效果好**：canonicalize + RMSD去重，859→19 的压缩率合理
5. **Quench中间步也能触发**：`on_quench_step` 捕获优化轨迹中的高F点

#### 批判性分析（仍存在的问题/隐患）

**H1. BasinRegistry 无持久化（中风险）**
- 现状：`BasinRegistry` 只在内存，重启后丢失所有已见 basin
- 影响：resume 时会重复发现已知 basin，浪费算力
- 缓解：`basins.jsonl` 有日志，但未反序列化回 registry
- 建议：`--resume` 时从 `basins.jsonl` 重建 registry

**H2. HistogramBasin 参数硬编码（低风险）**
```python
bin_width: float = 0.1
max_dist: float = 6.0
```
- 问题：对于金属体系 max_dist=6Å 可能不够；对于有机分子 bin_width=0.1Å 可能太粗
- 建议：参数化，允许配置

**H3. O(n²) 距离计算的扩展性（低风险，当前可接受）**
- HistogramBasin 和 validators 都用 O(n²) 双重循环
- 对于 <200 原子的 slab 没问题
- 对于 >500 原子的大体系会成为瓶颈
- 建议：后续可用 neighbor list 优化

**H4. force_fn 失败无 fallback（低风险）**
```python
# pipeline.py:121-128
try:
    energy_pre, forces_pre = self._split_force_output(self._force_fn(structure_pre))
except Exception as exc:
    flags["force_pre_error"] = str(exc)  # 仅记录，继续运行
```
- 如果 GPU OOM，会一直 fallback 但不中断
- 建议：连续 N 次失败后 raise 或 warn

**H5. NPZ 与 JSONL 一致性（低风险）**
- `StructureStore.put()` 写 NPZ，`JSONLWriter.write()` 写 JSONL
- 如果 NPZ 写入失败但 JSONL 成功，会有悬空引用
- 建议：事务化或先写 NPZ 再写 JSONL

**H6. 动作均匀随机，无自适应（设计待定）**
```python
action = rng.choice(self._actions)  # 均匀分布
```
- 当前：所有动作等概率
- 后续：RL 会需要 `action = policy(state)`
- 建议：预留 `action_sampler` 接口

#### 架构设计评价

| 维度 | 评分 | 说明 |
|------|------|------|
| 模块化 | ✅ 优秀 | 五层解耦，职责清晰 |
| 可测试性 | ✅ 良好 | 各组件可独立测试 |
| 可扩展性 | ⚠️ 中等 | RL接入需要修改 action 采样逻辑 |
| 健壮性 | ⚠️ 中等 | 缺少 registry 持久化、force_fn 连续失败处理 |
| 性能 | ✅ 当前足够 | O(n²) 对小体系可接受 |

#### 结论

**认可当前架构**。对于阶段目标（筛选高F构型 → 微调MACE），设计合理，实现完整。

上述隐患（H1-H6）属于"后续优化"而非"阻塞性问题"，可以在闭环跑通后逐步解决。

**建议优先级**：
1. H1（registry持久化）- resume 场景需要
2. H6（action sampler接口）- RL 接入前需要
3. 其他 - 按需优化
