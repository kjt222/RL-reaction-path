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

## 2026-01-20 Codex
### 目标
- 在 Linux 端提供“半自动”同步，将就绪的 DFT 任务目录经 Tailscale 推送到 Windows 指定目录。

### 方案
- 约定 outbox 目录（默认 `runs/dft_outbox`）。
- 每个任务一个子目录；任务准备完成后写入 `READY` 标记文件。
- 同步脚本扫描 `READY` 且未上传的目录，用 `scp -r` 推送到 Windows。
- 成功写 `_uploaded.flag`，失败写 `_upload_failed.log`（便于重试/排查）。
- Windows 侧负责 EasyConnect + 内网提交；Linux 侧不处理超算连接细节。
- 目标地址与账号通过环境变量覆盖（`WIN_IP`/`WIN_USER`/`WIN_PATH`）。

### 验收
- outbox 中有 `READY` 的任务，脚本可成功上传并生成 `_uploaded.flag`。
- outbox 无 `READY` 时脚本输出“无任务”提示并正常退出。

## 2026-01-20 Codex
### 目标
- 从 OC22 LMDB 训练集抽取 10 个结构，转换为 VASP 结构包用于单点对齐验证。

### 方案
- 使用 LMDB reader 抽取前 10 条样本。
- 每个样本输出一个目录：`POSCAR` + `oc22_forces.npy` + `oc22_energy.txt` + `metadata.json`。
- POSCAR 采用元素分组（按首次出现顺序），forces 同步重排，便于逐原子对比。

### 验收
- 生成 10 个目录（`sample_0000`…`sample_0009`）。
- 每个目录包含 POSCAR 与 OC22 reference (energy/forces)。

## 2026-01-20 Codex
### 目标
- 基于 OC22_dataset 分支与 pymatgen==2020.4.2 的默认配置，生成尽量复现 OC22 的 VASP 输入模板（INCAR/KPOINTS/POTCAR.symbols）。

### 方案
- 解析 OC22_dataset 分支的 MOSurfaceSet（PBE/ENCUT/EDIFFG/KPOINTS 等）与 pymatgen 2020.4.2 的 MPRelaxSet 默认参数。
- 组合成“OC22 slab/adslab”完整 INCAR（显式展开继承项）。
- 对 10 个样本按 OC22 公式写 KPOINTS；生成 POTCAR.symbols（元素顺序与 POSCAR 一致，W 用 W_sv）。

### 验收
- 每个 sample_0000..sample_0009 生成 INCAR/KPOINTS/POTCAR.symbols。
- INCAR 包含 OC22 关键参数（ENCUT=500, EDIFFG=-0.05, ISPIN=2 等）与 MP U 值。

## 2026-01-20 Codex
### 目标
- 设计可视化全流程方案（动作→quench→basin），不改动 core，只在 experiments 增量实现，且可控存储。

### 方案（分阶段）
**P0 需求与边界**
- 明确展示粒度（仅 pre/min vs 全 quench 步）与输出格式（MP4/GIF/OVITO 轨迹）。
- 明确是否需要叠加指标（action 名称、max_F、quench_step 等）。
- 叠加规则：全程显示 `basin_id`（仅 x_min 有真实值；其余帧显示 `basin_id=NA`），若新盆地同时标注 `is_new`。
- 叠加规则：任何触发 DFT 的帧必须显示 `trigger_reason`（含指标/阈值）。
- 触发来源：DFT 触发可出现在 action 后或任意 quench 步。

**P1 数据采集层（Recorder）**
- 在 `experiments/sampling/recorders.py` 新增 `VizRecorder`（或扩展现有 StepTraceRecorder），支持 `on_quench_step`。
- 结构存储使用独立 `viz_structures/` 目录，round_decimals=3（仅用于可视化，主存储保持现状）。
- 输出 `viz_steps.jsonl`：记录帧索引、action、metrics、quench_step、structure_ref、basin_id、is_new、trigger_reason。

**P2 渲染层（离线脚本）**
- 新增 `experiments/visualization/render_movie.py`：读取 `viz_steps.jsonl` + 结构引用，生成帧并合成视频。
- 支持 `--stride`（降帧）、`--fps`、`--overlay`（动作/力信息）。

**P3 验收与可用性**
- 10–20 步小样本验证：能生成 MP4，且能看清动作与 quench 轨迹。
- 空间控制：记录 stride 后尺寸 < 1GB/1k samples。

### 推荐目录结构
- `experiments/visualization/`
  - `render_movie.py`
  - `README.md`
- `runs/<run_id>/viz/`
  - `viz_steps.jsonl`
  - `viz_structures/`
  - `frames/`（可选）
  - `movies/`

### 验收
- 生成 `viz_steps.jsonl` 与 `movies/*.mp4`。
- 单个样本可视化清晰显示动作→quench→min 过程与指标。

## 2026-01-21 Codex
### 目标
- 将可视化渲染切换为 OVITO 高质量渲染（保留 extxyz + 叠加文字），便于科研展示。

### 方案
- render_movie.py 改为使用 OVITO Python API 渲染 PNG 帧，再用 ffmpeg 合成 MP4。
- 默认启用自动成键（可用则显示键），否则退化为仅原子。
- 继续产出 trajectory.extxyz，确保可被 OVITO/VESTA 复用。
- 叠加文字保持英文：stage/action/basin_id/is_new/quench_step/trigger_reason。

### 验收
- 最小化样本能生成 trajectory.extxyz 和 movie.mp4。
- MP4 画质高于散点版（能看到键/更真实原子渲染）。

## 2026-01-21 Codex
### 目标
- 修复可视化可读性与节奏：文字换行不截断、阶段清晰、动作/弛豫区分明显。

### 方案
- OVITO 叠加文字改为多行 overlay，并缩小字号与行距，避免连成一行或被截断。
- 渲染统一正交视角并固定相机方向，降低“溶液感”。
- 按元素自动着色 + 右侧图例，提升结构可辨识度（避免 tag 语义漂移）。
- 仅对 quench_step 降帧；action/min 保留并短暂 hold，突出阶段切换。
 - 叠加新增明显的 phase banner（颜色区分 ACTION/QUENCH/MIN），其余信息下移避免裁切。
 - 新增动作/弛豫分段导出（trajectory_action.extxyz / trajectory_quench.extxyz），支持只导出不渲染。

### 验收
- 同一 run 的 movie.mp4 中能清晰读到 phase/action/basin_id/trigger。
- Action → Quench → Min 时序视觉可区分。

## 2026-01-21 Codex
### 目标
- 渲染过程可观测（打印进度），避免“假卡住”。

### 方案
- render_movie.py 渲染循环加入进度输出（每 N 帧打印一次，可关闭）。

### 验收
- 渲染过程中输出类似 `[render] 200/14000 (1.4%)` 的进度日志。

## 2026-01-22 Codex
- 目标：从 OC22 LMDB(train) 抽取 10 个构型，生成可直接跑 VASP 的输入包（POSCAR/INCAR/KPOINTS/POTCAR.symbols），并保存参考能量/力用于对比；输出到 `Data/oc22_data/oc22_data/temp`。
- 目标：核对公开资料中 OC22 DFT 设定（PBE+U/ENCUT/kpoints/dipole 等），给出“可复现模板 + 不确定项说明”。

## 2026-01-22 Codex
- 目标：完善可视化时间线以明确区分 action/quench（动作插值、阶段标记、慢放节奏），并规划 visualization 模块化结构；不修改采样逻辑。

## 2026-01-22 Codex
- 目标：为可视化严格区分 action/quench，基于“动作参数插值 + 阶段标记 + 慢放节奏”的统一方案，且不修改采样逻辑。
- 方案要点：
  - 动作只保留前后两帧，**在可视化阶段插值**生成 5 帧（不改采样逻辑）。
  - 若记录 `action_type + action_params + target_atoms`，可对目标片段做物理一致插值（平移/旋转/二面角/推近拉远），避免与 quench 混淆。
  - 若不记录参数，只能做全体线性插值，区分度低（不推荐）。
  - **阶段可视化**：phase 文本标签（ACTION/QUENCH/MIN/TRIGGER），颜色区分。
  - **动作高亮**：action 阶段高亮目标原子/画箭头；quench 阶段隐藏箭头，仅显示 fmax/step。
  - **慢放策略**：不降低 fps，而是“重复帧”实现慢放。
    - action：5 帧 × (2~3) 倍重复
    - quench：100 帧 × 2 倍重复
    - trigger：前后各 2 帧 + 触发帧
  - 推荐 fps=20~24；总时长控制在 5~10s。
  - 输出策略：extxyz 分段 (action/quench) + movie 可选；VESTA 用关键帧，OVITO 用短动画。

## 2026-01-22 Codex
- 目标：从 OC22 LMDB 抽取并输出 bulk_id（用于后续查询 MP potcar_spec）。

## 2026-01-23 Codex
- 目标：将 render_movie.py 默认进度输出频率调整为每 20 帧一次，方便渲染过程可视化。

## 2026-01-23 Codex
- 目标：渲染进度输出改为单行进度条样式，避免终端刷屏。

## 2026-01-23 Codex
- 目标: 用 Blender(Cycles GPU 优先) 无头渲染 extxyz 轨迹，生成可播放的 mp4 以替代 OVITO CPU 渲染。
- 范围: 仅渲染指定 run_dir 的 trajectory.extxyz 到 viz_blender 输出目录，不改采样流程。
- 验收: 生成 movie.mp4 并可播放；记录渲染命令与耗时。

## 2026-01-23 Codex
- 目标：改进 Blender 渲染脚本以提高可读性（屏幕叠加文本、固定视角、显示晶胞），并用更直观的 slab 构型做低质量预览渲染验收。
- 计划：
  - 将文本改为 2D overlay（stamp note），避免遮挡结构。
  - 相机改为固定正交/斜角视角，基于首帧结构与晶胞确定视角与缩放。
  - 解析 extxyz Lattice 并渲染晶胞边框（便于判断 slab/真空层）。
  - 选取 temp 中更直观 slab 构型进行低质量渲染，检查单帧效果。

## 2026-01-23 Codex
- 目标：修复 Blender 渲染预览“晶胞不可见/视角偏移/颜色不可读”的问题，并给出可验收的单帧预览。
- 计划：
  - 增强晶胞边框可见性（颜色/半径），确保视野内对齐（按晶胞中心居中或包裹原子）。
  - 增补 Cs/O 等元素颜色，避免杂乱的 hash 色。
  - 调整相机中心与正交尺度，保证晶胞与原子同时入镜。
  - 输出 1 帧低质量预览用于验收（不生成长视频）。

---

## 2026-01-23 Claude1

### 可视化方案设计：论文级 slab+吸附物 渲染

#### 问题分析

当前 OVITO/Blender 渲染的核心问题：

| 问题 | OVITO | Blender | VESTA |
|------|-------|---------|-------|
| 球棍模型 | ❌ 纯球，键很细 | ⚠️ 有键但半径统一 | ✅ |
| 分层着色 | ❌ 按元素着色，基底/吸附物混淆 | ❌ 同上 | ✅ 自动分层 |
| abc轴指示器 | ❌ 无 | ❌ 无 | ✅ |
| 视角控制 | ⚠️ 斜视，结构不清晰 | ⚠️ 同上 | ✅ 沿c轴俯视 |

**根本原因**：当前渲染把 slab 当成"分子云"处理，而不是"基底+界面+吸附物"的分层结构。

#### VESTA 效果好的原因

1. **球棍模型**：球半径 ∝ 共价半径，键粗细适中
2. **分层着色**：金属基底统一颜色，O/吸附物突出
3. **abc轴指示器**：左下角 RGB 箭头
4. **视角**：默认沿c轴略偏，能看到层状结构

#### 设计方案（最强架构师版）

**核心思想**：不做通用渲染器，做**slab+吸附物专用渲染器**

```
experiments/visualization/
├── render_slab.py           # 新：slab专用渲染（静态+动画）
├── render_movie.py          # 现有：OVITO通用渲染
├── blender_render.py        # 现有：Blender通用渲染
├── styles/
│   └── slab_adsorbate.py    # slab着色/分层规则
└── components/
    ├── axis_indicator.py    # abc轴指示器
    ├── layer_detector.py    # 自动检测基底/界面/吸附物
    └── bond_drawer.py       # 球棍模型（元素感知半径）
```

**分层检测算法**（`layer_detector.py`）：
```python
def detect_layers(atoms, z_threshold=2.0):
    """
    基于z坐标和tags自动分层：
    - 基底层(substrate): z < z_surface - z_threshold，或 tags==0
    - 界面层(interface): z_surface ± z_threshold，或 tags==1
    - 吸附物(adsorbate): z > z_surface + z_threshold，或 tags==2
    """
    # 1. 如果有tags，直接用tags
    if hasattr(atoms, 'tags') and atoms.tags.max() > 0:
        return atoms.tags
    # 2. 否则基于z坐标自动检测
    z = atoms.positions[:, 2]
    z_surface = np.percentile(z, 80)  # 表面大约在80%分位
    layers = np.zeros(len(atoms), dtype=int)
    layers[z > z_surface + z_threshold] = 2  # adsorbate
    layers[(z >= z_surface - z_threshold) & (z <= z_surface + z_threshold)] = 1  # interface
    return layers
```

**分层着色规则**：
```python
LAYER_COLORS = {
    0: {"metal": (0.75, 0.65, 0.45),  # 金色基底
        "O": (0.6, 0.2, 0.2)},         # 暗红O
    1: {"metal": (0.8, 0.75, 0.55),   # 亮金界面
        "O": (0.8, 0.3, 0.3)},         # 中红O
    2: {"default": "element",         # 吸附物用元素色
        "highlight": True},            # 高亮显示
}
```

**abc轴指示器**（Blender实现）：
```python
def create_axis_indicator(origin, scale=2.0):
    """在左下角创建 RGB 箭头对应 abc"""
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R=a, G=b, B=c
    labels = ['a', 'b', 'c']
    for i, (color, label) in enumerate(zip(colors, labels)):
        # 创建箭头（cylinder + cone）
        direction = lattice_vectors[i].normalized()
        create_arrow(origin, direction * scale, color, label)
```

**视角控制**：
```python
def compute_slab_view(atoms, cell):
    """计算最佳slab视角"""
    c = cell[2]  # c轴方向
    # 沿c轴略偏（避免完全俯视看不到层状）
    view_dir = c.normalized() + 0.2 * (cell[0] + cell[1]).normalized()
    return view_dir.normalized()
```

#### 推荐实现路径

**Phase 1：快速验证（1天）**
- 用 Blender 现有代码增加：
  1. abc轴指示器
  2. 元素感知球半径（共价半径 × 0.4）
  3. 沿c轴视角
- 输出单帧验证效果

**Phase 2：分层着色（2天）**
- 实现 layer_detector.py
- 基底用统一金属色，吸附物用元素色
- 界面层过渡着色

**Phase 3：论文级输出（1天）**
- 高分辨率渲染（1920×1080, 256 samples）
- 导出 PNG 关键帧（x_in → x_pre → x_min）
- 可选：MP4 动画

#### 工具选择建议

| 用途 | 推荐工具 | 原因 |
|------|----------|------|
| 论文静态图 | Blender（改进后） | 渲染质量高，可精确控制 |
| 快速预览 | VESTA（手动） | 效果好，无需开发 |
| 动画 | Blender（改进后） | GPU加速，批量渲染 |
| 调试 | OVITO | 快速，交互式 |

#### 替代方案：直接用 VESTA

如果开发成本太高，可以：
1. 导出 extxyz → CIF/POSCAR
2. 用 VESTA 打开，手动调整视角
3. 截图/录屏

缺点：难以自动化批量处理

#### 验收标准

1. **球棍模型**：球半径按元素缩放，键清晰可见
2. **分层可辨**：基底/界面/吸附物颜色明显不同
3. **abc轴指示**：左下角有 RGB 箭头
4. **视角合理**：能看到层状结构，不是"溶液感"
5. **信息叠加**：action/quench_step/basin_id 可读

**等待用户确认方向后再实施。**

---

## 2026-01-23 Claude1

### 项目状态分析与下一步优先级

#### 当前状态总览

```
采样链路：    ✅ 完整
  x0 → Action → x_pre → Validator → Quench → Basin → SampleRecord

AL筛选：      ✅ 完整
  force_pre/force_min → trigger_fn → dft_queue.jsonl

DFT去重：     ✅ 完整
  dft_queue.jsonl → canonicalize → RMSD去重 → dft_submit.jsonl

微调闭环：    ❌ 未完成
  dft_submit.jsonl → [DFT计算] → [数据整理] → [微调MACE]
```

#### 发现的关键问题

**P1. 采样只从固定初始结构出发（严重 - 但不阻塞当前目标）**
```python
# run_sampling.py:165-167
for _ in range(int(args.steps)):
    record = pipeline.run_one(structure, ...)  # 每次都是同一个structure
```
- 问题：所有采样围绕初始构型，无法扩展盆地图
- 影响：后续RL需要"从新basin出发"的能力
- **当前可接受**：从一个起点也能采到高F构型，先跑通微调

**P2. validators为空（中等）**
```python
# run_sampling.py:155
validators=[],  # 没有启用碰撞检测
```
- 问题：可能产生原子重叠的无效构型
- 缓解：quench会处理部分，validator应该在quench前过滤
- 建议：启用 `validate_min_distance_structure`

**P3. BasinRegistry无持久化（中等）**
- 问题：重启后丢失已见basin
- 影响：resume时重复发现basin

**P4. 微调数据格式未定义（阻塞）**
- 问题：dft_submit.jsonl 输出后，如何转换为微调输入？
- 缺失：DFT结果回收 → LMDB/NPZ → 微调YAML

#### 优先级排序

| 优先级 | 任务 | 原因 | 阻塞性 |
|--------|------|------|--------|
| **P0** | 微调数据流闭环 | 当前目标是"采样→微调MACE" | 阻塞 |
| P1 | 启用validators | 避免无效构型 | 低 |
| P2 | 从新basin出发采样 | 盆地图扩展需要 | 后续 |
| P3 | BasinRegistry持久化 | resume需要 | 后续 |
| P4 | 可视化改进 | 调试/论文需要 | 非阻塞 |

#### P0：微调数据流设计

当前缺失的环节：

```
dft_submit.jsonl          # ✅ 已有
       ↓
[上传到超算/DFT计算]       # 手动/脚本
       ↓
vasp_results/             # VASP输出（待定义格式）
       ↓
[结果回收+解析]           # ❌ 缺失
       ↓
finetune_data.lmdb        # ❌ 缺失：微调数据格式
       ↓
runs/finetune_xxx.yaml    # ❌ 缺失：微调配置
       ↓
frontends.run finetune    # ✅ 已有
```

**需要实现的模块**：

1. **DFT结果解析器** (`experiments/mace_pretrain/vasp_parser.py`)
   - 输入：VASP输出目录（OUTCAR/vasprun.xml）
   - 输出：能量、力、应力

2. **微调数据整理** (`experiments/mace_pretrain/prepare_finetune.py`)
   - 合并 dft_submit.jsonl（构型）+ DFT结果（标签）
   - 输出：LMDB或NPZ格式

3. **微调配置生成**
   - 基于现有 runs/finetune_oc22_head_full.yaml 模板

#### 建议的下一步行动

**方案A：先手动跑通一遍**
1. 用现有 dft_submit.jsonl 的19个构型
2. 手动准备VASP输入（已有模板）
3. 手动提交DFT计算
4. 手动解析结果
5. 手动整理成微调数据
6. 跑通微调

**方案B：先自动化数据流**
1. 实现 vasp_parser.py
2. 实现 prepare_finetune.py
3. 再跑DFT

**建议**：方案A更快验证闭环，方案B更可持续。取决于DFT计算的周期。

#### 总结

当前采样pipeline**本身没有阻塞性bug**，关键缺口是：

1. **微调数据流**（P0，阻塞）
2. validators未启用（P1，易修复）
3. 采样起点扩展（P2，后续）

**下一步建议**：先定义微调数据格式，决定是手动跑通还是自动化。

## 2026-01-23 Codex

### 可视化近期计划（Blender）
- 解析 extxyz 的 Properties 字段，读出 tags（OC22 的 slab/surface/ads 标签）。
- 渲染策略：元素着色 + tag=0 slab 降权（半透明 + 半径缩小）。
- 添加 VASP 风格 XYZ triad（屏幕角落固定坐标轴）。
- XYZ triad 增加 X/Y/Z 字母标注，便于识别方向。
- 先保证单帧预览可读，再考虑动画与键规则优化。

## 2026-01-24 Codex
- 目标：新增 experiments/diagnostics 动作质量诊断模块（纯计算 + CLI 报告），用于量化动作有效率/收敛率/新盆地率/高力触发率。
- 目标：移除 experiments/sampling/validators.py，改为可复用模块（例如 validate.py），并精简函数命名；更新相关导出接口。

## 2026-01-24 Codex
- Goal: extend sampling quench module to support FIRE/CG/BFGS/LBFGS and expose options in run_sampling for comparison.

## 2026-01-25 Codex
- 目标：用 EquiformerV2 对 sample_0000 进行 5-basin 采样测试（延长步数），并评估并行两路测试的显存可行性。

## 2026-01-25 Codex
- 目标：将 sampling pipeline 做成可插拔架构（recorders/validators/triggers/stoppers），便于调试时快速禁用 DFT 采样、只跑收敛或 basin。
- 设计原则：默认行为不变；通过单一“pipeline 配置”文件控制插件组合，避免新增大量 CLI 选项。
- 计划（P0→P2）：
  - P0（接口/配置）：新增 pipeline 配置模型与解析（YAML/JSON），支持：recorders 列表、各 recorder 阶段开关、trigger 配置、stoppers 配置、validators 列表。
  - P1（实现）：
    - SamplingPipeline 新增 stopper 回调（如 steps/basins/time/trigger 计数）。
    - ALCandidateRecorder 增加阶段过滤（仅 action / 仅 quench / 仅 basin）。
    - run_sampling 只负责加载 pipeline config → 构建插件 → 运行（去掉硬编码）。
  - P2（验收）：
    - 配置1：只启用 StepTrace + BasinRegistry（禁用 DFT）。
    - 配置2：仅在 quench_step 触发 DFT。
    - 通过 sample_0000 做 20–50 steps 烟雾测试，确认：无 dft_queue、basin 正常输出、log 无报错。

## 2026-01-25 Codex
- 计划：修正 OC22 10 个样本的 VASP 输入为**单点计算**（不结构优化）。
- 操作：仅改动 sample_0000~0009 的 INCAR 中优化相关项（IBRION/NSW/ISIF/EDIFFG），其余参数保持 OC22 口径（ENCUT/ISPIN/DFT+U/DIPOL 等）。
- 验收：逐个样本检查 INCAR 已切换为单点，并保留原 POSCAR/KPOINTS/POTCAR.symbols。

## 2026-01-25 Codex
- 计划：将动作质量相关逻辑集中到 `experiments/action_quality`，重构路径并去除 `quench_gate` 名称。
- 变更点：
  - `experiments/diagnostics` 重命名为 `experiments/action_quality`；更新 imports。
  - 将 `experiments/sampling/validate.py` 移入 `experiments/action_quality/validate.py`。
  - 将 force-based gate（原 quench_gate/trigger 逻辑）并入 `validate.py`，改名为 `quality_gate`；移除 quench_gate 配置键。
  - 更新 pipeline/config/run_sampling 以使用 `quality_gate`。
  - 在 action_quality 统计中补充“新角度”指标（invalid 原因分布、动作幅度分布、force_pre 分布等）。
- 验收：
  - 代码引用路径均更新；
  - config.yaml 不再出现 quench_gate；
  - 采样可正常运行（不因 import/配置报错）。

## 2026-01-25 Codex
- 更新：按用户澄清执行重构 —— 将 `experiments/diagnostics` 改名为 `experiments/action_quality`，把 `experiments/sampling/validate.py` 挪入该目录，并将原 `quench_gate` 逻辑合并进 `validate.py`（作为 `quality_gate`），pipeline/config 对应更新。

## 2026-01-25 Codex
- 计划：按要求移除 `quality_gate`（不再在 pipeline 中做门控），同步清理配置键与统计输出。

## 2026-01-25 Codex
- 计划：把 gate 逻辑合并进 `experiments/action_quality/validate.py`，作为“质量验证器”而非独立 gate 插件。
  - 设计：先跑硬性校验（min_dist/fixed/bond），再基于 force_pre + action magnitude 打分；低分直接拒绝（不做概率放行）。
  - 记录：在 flags 中写入 quality_score / quality_reason / quality_reject，便于 action_quality 统计通过率与拒绝原因。
  - Pipeline 调整：在运行 validators 之前写入 action 元信息；在硬校验通过后提前计算 force_pre（供 gate 评分使用）。
- 计划：按新 gate 逻辑完成 4 组采样对比（EquiformerV2；FIRE 0.1/0.15 + 另 2 种 quench），每组目标 1 个 basin、上限 5000 步；记录耗时。

## 2026-01-25 Codex
- 计划：将 gate 逻辑改为“拒绝即重采样”——当动作被 validator/gate 判定为无效时，不进入 quench/basin，而是重新采样下一个动作；同时记录被拒动作用于通过率统计。
- 计划：在 EquiformerV2 上补做两组 fmax=0.1 对比（FIRE + LBFGS），目标 1 个 basin、上限 5000 步，记录耗时。

## 2026-01-26 Codex
- 计划：在 run_sampling 输出摘要中新增 attempts_total / attempts_rejected，用于快速评估 gate 放行率。

## 2026-01-26 Codex
- 计划：对比 OC22 原始能量/力 与新 DFT 单点输出（10 个样本）
- 输出：每样本 dE、dE/atom、力误差（MAE/RMSE），判断是否仅为能量基线偏移

## 2026-01-26 Codex
- 目标：检查仓库/文档是否记录 OC22 计算参数；如无，考虑在 fairchemv2 环境下载相关数据/脚本后再检索（需确认下载范围）。

## 2026-01-27 Codex
- 计划：基于 temp/sample_0000..0009 复制生成 temp2/sample_0000..0009，并在 INCAR 中新增 MAGMOM（仅改变磁性，其他保持一致）。
- 规则：按 POSCAR 元素顺序生成 MAGMOM（U 元素设 5.0，其它 0.0），仅用于对比磁性对误差的影响。

## 2026-01-27 Codex
- 计划：在 `experiments/sampling/actions` 增加 MD 动作（MDAction），用短程 MD 作为“局部探索”动作，MD 结束帧作为 quench 起点。
- 关键设计点：
  - 动作接口需支持“需要 force_fn 的动作”（MDAction 依赖力），建议扩展 action.sample/ apply 的上下文参数。
  - MD 采用 ASE MD（优先 Langevin/Andersen），并严格遵守 fixed 约束。
  - 参数范围先保守：dt≈0.5–1.0 fs，steps≈5–50，T≈50–600 K（可随机采样）。
  - 与 gate/validate 结合：MD 结果同样走 validate，失败直接重采样。
  - 先以调试为目标：增加记录（MD steps/T/dt/终态 max_F），避免大规模无效探索。

## 2026-01-27 Codex
- 计划补充：为动作阶段引入“微小高斯噪声”，用于打破完全确定性、提升局部探索覆盖率。
- 设计要点（先不改代码）：
  - 噪声应只作用于 movable 原子（尊重 fixed/slab 约束）。
  - 噪声在动作之后、validate/gate 之前施加，仍由 gate 负责兜底拒绝。
  - 幅度需保守且可控：建议 `sigma≈0.01–0.03 Å`，并支持按元素/角色缩放。
  - 初始策略：全局常数 sigma + movable mask（最小改动），后续再做分层/元素缩放。

## 2026-01-27 Codex
- 计划：实现 MDAction + 动作后噪声插件（仅 movable），并接入 SamplingPipeline。
- 设计决定：
  - 扩展 ActionContext 以携带 `force_fn`/`ase_calculator`（MD 动作依赖力）。
  - 在 pipeline 的 action.apply 之后引入“action_plugins”链（噪声放这里）。
  - 噪声实现为独立插件函数：仅 movable 掩码，支持 sigma/clip，记录 flags。
  - run_sampling 接入 MDAction，并将噪声插件配置化（来自 sampling_rules/config.yaml）。

## 2026-01-27 Codex
- 计划：集中更新文档与日志，并完成一次可复现的最小验收与提交。
- 范围：
  - 文档：`README.md`、`parameters.md`、`docs/vasp_oc22_repro.md`
  - 日志：`implementation.md`、`daily_log.md`
  - 同步检查：关键采样链路与新配置项（MDAction / action_plugins / action_quality）
- 验收：
  - 关键 Python 文件通过 `py_compile` 语法检查；
  - 文档内容与当前实现一致（不再出现旧路径/旧键名）；
  - 变更按“文档更新”单独 commit 并 push。
