# 项目自定义脚本说明（持续更新中）

本文档用于记录本地新增或修改的自定义脚本，与官方 `README_original.md` 区分。每次调整脚本后请同步更新此文件。

---

## `transit npz to xyz.py`

**功能概述**
- 将 rMD17 数据集中的 `.npz` 原始数据转换为 MACE/ASE 支持的 Extended XYZ。
- 自动检测数据目录：支持 Windows 路径 (`D:\...`) 以及 WSL 映射路径 (`/mnt/d/...`)，允许直接指定单个 `.xyz` 文件或包含多个 `.xyz` 的目录；也可以通过环境变量 `MACE_RMD17_DIR` 手动指定。
- 兼容 rMD17 常见的两套键名：旧版 `R`/`z`/`E`/`F` 与新版 `coords`/`nuclear_charges`/`energies`/`forces`。
- 逐帧写出坐标、力、能量，并在注释行保留 Extended XYZ 所需的 `Properties`、能量、晶格、PBC 等信息（若存在）。

**实现要点**
- 使用 `numpy.load(..., allow_pickle=False)` 安全读取数据键，缺失必需字段会立刻报错，避免破坏标签。
- 通过预制的原子序数 → 元素符号映射转换 `z` 值，遇到未知原子序数会抛出错误。
- 输出目录固定为脚本所在目录下的 `xyz_data/`，逐个 `.npz` 转换成同名 `.xyz`。

**运行方式**
```bash
# Windows PowerShell
python "transit npz to xyz.py"

# WSL/bash（如需自定义数据目录）
export MACE_RMD17_DIR=/mnt/d/D/calculate/MLP/training_set/rmd17/npz_data
python "transit npz to xyz.py"
```

---

## `MACE train rMD17.py`

**功能概述**
- 在 `xyz_data/` 中对 movie-style `.xyz` 文件执行蓄水池采样，随机抽取 500 帧，划分为 450/50 的训练、验证集。
- 使用 ASE + MACE 数据工具构建 `Configuration` 与 `AtomicData`，生成 `torch_geometric` DataLoader。
- 根据官方推荐配置手动实例化 MACE 模型，并编写完整的 PyTorch 训练循环（能量/力损失 + ReduceLROnPlateau 调度）。
- 训练过程中跟踪验证集最优损失，最终仅保存验证集表现最好的模型权重（随附 `best_val_loss`）。

**实现要点**
- 采用 **蓄水池采样** (`reservoir sampling`) 遍历全部帧，避免一次性载入内存，确保全局随机性。
- 用 `KeySpecification` 显式绑定 Extended XYZ 中的 `energy`/`force` 键，保证标签正确导入 `Configuration`。
- 通过 `modules.compute_avg_num_neighbors` 估算平均邻居数，并在构建模型时使用：
  - `hidden_irreps = "128x0e + 128x1o"`
  - `num_polynomial_cutoff = 5`
  - `correlation = 3`
  - 其它超参按推荐默认设置。
- 交互层数默认 3 层，可通过 `--num_interactions` 在命令行灵活调整。
- 训练循环：
  - 每个 epoch 随机打乱训练集并遍历所有 mini-batch。
  - 计算能量/力 MSE（默认能量:力权重为 1:1000，可通过 `--energy_weight` / `--force_weight` 调整），反向传播、更新参数。
  - 使用 `ReduceLROnPlateau(factor=0.8, patience=50)` 按验证损失自动衰减学习率。
  - 若未禁用，训练过程中应用 EMA (`torch_ema.ExponentialMovingAverage`) 对权重做指数滑动平均，可通过 `--ema`/`--no-ema` 和 `--ema_decay` 控制。
  - 记录训练/验证损失及 RMSE 指标；当验证损失刷新最低值时缓存模型权重（默认保存 EMA 权重）。
- **E0s 处理**：优先调用 `data.compute_average_E0s` 对训练集执行最小二乘拟合，获得各元素孤立原子能；若该拟合失效（如所有样本原子组成完全一致），则退回到“在所有带能量标签的样本上计算总能量平均/平均原子数”的全局基线，确保始终可用的能量平移，使网络仅需学习原子化能偏差，从而提升收敛稳定性。

**运行方式示例**
```bash
python "MACE train rMD17.py" \
  --xyz_dir /mnt/d/D/calculate/MLP/MACE项目/xyz_data \
  --output mace_rmd17.pt \
  --sample_size 500 \
  --train_size 450 \
  --epochs 300 \
  --num_interactions 3 \
  --energy_weight 1 \
  --force_weight 1000 \
  --ema_decay 0.99 \
  --batch_size 16
```
> 运行前需激活 `mace_env` 环境，并确保该环境中已安装 MACE 源码及其依赖。

**近期变更**
- 导入 e3nn 之前调用 `torch.serialization.add_safe_globals([slice])`，兼容 PyTorch 2.6+ 的安全加载策略。
- 训练循环返回并保存验证集最优模型权重及 `best_val_loss`，替代“只保留最后一个 epoch”。
- 在 `mace/modules/wrapper_ops.py` 中为 cuEquivariance 增加 `try/except`，在 WSL 环境下 NVML 查询失败时自动回退到基础实现。
- 新增训练集基线能量拟合：自动使用 `compute_average_E0s` 估算 E0s 并传入模型，避免拟合巨大的总能量常数，改善收敛与误差。
- 新增 `--num_interactions`、`--ema`/`--no-ema`、`--ema_decay` 参数，支持自定义交互层数与指数滑动平均（默认开启 EMA，衰减 0.99）。
- 默认损失权重调整为能量:力 = 1:1000，可用 `--energy_weight` / `--force_weight` 自行覆盖。
- `reservoir_sample_atoms` 在读取帧时从 `atoms.calc` 抽取能量并写回 `atoms.info`，确保后续转换/统计 E0s 时能量标签完整；若所有样本确实缺失能量，则明确抛错提示。

---

如脚本功能或参数有更新，请及时同步本文件。
---

## `MACE test rMD17.py`

**功能概述**
- 支持从单个 `.xyz` 文件或目录按蓄水池算法抽样（默认 1000 帧）执行模型评估；不会修改原始数据。
- 载入训练好的 `.pt` 模型并重建与训练阶段一致的 MACE 架构，自动估算/回退 E0s。
- 按输入帧顺序写出带 `pred_energy`、`pred_force` 标签的预测 `.xyz` 文件，默认保存在原 `.xyz` 所在目录下的 `<文件名>_pred.xyz`；也可以通过 `--pred_output` 指定完整输出路径。
- 输出能量与力的 MAE / RMSE 指标，可通过 `--verbose` 查看批次级预测。

**运行方式示例**
```bash
python "MACE test rMD17.py" \
  --xyz_path /mnt/d/D/calculate/MLP/MACE项目/xyz_data/rmd17_aspirin.xyz \
  --model /mnt/d/D/calculate/MLP/MACE项目/mace_models/mace_03.pt \
  --sample_size 1000 \
  --batch_size 64 \
  --num_interactions 3 \
  --cutoff 5.0
```
> 请确保评估参数（如 `--num_interactions`、`--cutoff`）与训练阶段一致；需要在 CPU 上评估可指定 `--device cpu`。
- MACE test rMD17.py 在导入 e3nn 之前同样执行 	orch.serialization.add_safe_globals([slice])，以兼容 PyTorch 2.6+ 的安全策略。
- MACE test rMD17.py 在导入 e3nn 之前同样执行 	orch.serialization.add_safe_globals([slice])，以兼容 PyTorch 2.6+ 的安全策略。

## `sample train and test set.py`

**功能概述**
- 从单个 `.xyz` 文件中随机抽取 train/validation 和 test 两个子集（默认各 1000 帧），两者互不重叠；若数据量不足直接报错。
- 自动补全缺失的能量标签：如果 `atoms.info` 中没有 energy，但 `atoms.calc` 可提供，就写回 info 以保持标签完整。
- 使用 `--output_prefix` 控制输出路径，分别写出 `<prefix>_train.xyz` 和 `<prefix>_test.xyz`，遵循 Extended XYZ 格式（含 `Properties` 注释、坐标和力）。

**运行方式示例**
```bash
python "sample train and test set.py" \
  --input_xyz /mnt/d/D/calculate/MLP/MACE项目/xyz_data/rmd17_aspirin.xyz \
  --output_prefix /mnt/d/D/calculate/MLP/MACE项目/splits/aspirin \
  --train_size 1000 \
  --test_size 1000 \
  --seed 42
```

## `MACE inference rMD17.py`

**功能概述**
- `MACE inference rMD17.py` 在导入 e3nn 前同样执行 `torch.serialization.add_safe_globals([slice])`，确保与 PyTorch 2.6+ 的安全策略兼容。
- 与评估脚本类似，从 `.xyz` 文件（或目录）随机抽取一定数量的帧，载入训练好的 `.pt` 模型，并输出模型对每帧的预测能量与力。
- 即使输入数据缺失能量/力标签，脚本也会补全预测结果并写入新的 `.xyz` 文件（包含 `pred_force` 属性和 `pred_energy=...` 注释）。
- 当请求的 `--sample_size` 大于实际帧数时，会自动退回到使用全部帧并给出日志提示，避免报错中断。

**运行方式示例**
```bash
python "MACE inference rMD17.py" \
  --xyz_path /mnt/d/D/calculate/MLP/MACE项目/xyz_data/rmd17_aspirin.xyz \
  --model /mnt/d/D/calculate/MLP/MACE项目/mace_models/mace_03.pt \
  --output /mnt/d/D/calculate/MLP/MACE项目/predictions/aspirin_pred.xyz \
  --sample_size 1000 \
  --batch_size 64 \
  --num_interactions 3 \
  --cutoff 5.0
```
> 输出文件会包含原子坐标、预测力以及结构层面的 `pred_energy` 注释；请根据训练设置保持 `--num_interactions`、`--cutoff` 等参数一致。
## `MACE inference rMD17.py`（更新）

**功能概述**
- 在导入 e3nn 前执行 `torch.serialization.add_safe_globals([slice])`，兼容 PyTorch 2.6+。
- 默认仅抽取 1 帧（可通过参数调整）进行推理，输出模型预测的能量与力。
- 无论输入是否包含标签，都会生成带有 `pred_force` 和 `pred_energy=...` 注释的预测 `.xyz`。
- 若 `--sample_size` 超过实际帧数，将自动使用全部可用帧并记录告警，不会抛出异常。

**运行方式示例**
```bash
python "MACE inference rMD17.py" \
  --xyz_path /mnt/d/D/calculate/MLP/MACE项目/xyz_data/rmd17_aspirin.xyz \
  --model /mnt/d/D/calculate/MLP/MACE项目/mace_models/mace_03.pt \
  --output /mnt/d/D/calculate/MLP/MACE项目/predictions/aspirin_pred.xyz \
  --sample_size 1 \
  --batch_size 1 \
  --num_interactions 3 \
  --cutoff 5.0
```
> 输出 `.xyz` 将包含原子坐标、预测力与结构级 `pred_energy` 注释；请确保核心模型参数与训练设置保持一致。
- `sample train and test set.py` 修复了写回 Extended XYZ 时的标签问题：当检测到 `force` 数组时会正确追加 `:force:R:3`，确保输出训练/测试集保留力标签。
- `MACE test rMD17.py` 现直接依赖模型检查点中保存的 E0s，不再重新拟合，以保持与推理脚本一致的基线处理。
