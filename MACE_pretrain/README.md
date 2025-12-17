# MACE_pretrain

基于 MACE 的预训练/微调/续训/评估脚本，模型结构由 `model.json` 主导。核心脚本在 `model-utils/`，通用工具在项目根。

```
MACE_pretrain/
├── model-utils/
│   ├── train_mace.py     # 预训练入口
│   ├── finetune.py       # 微调入口
│   ├── resume.py         # 断点续训入口
│   ├── evaluate.py       # 评估入口（支持 --use_ema）
│   ├── losses.py         # per-atom / per-comp 损失与指标
│   ├── models.py         # JSON -> 模型构建
│   └── model_loader.py   # 统一的加载/保存/校验
├── optimizer.py          # 参数分组（bias/norm/scale/shift no_decay）+ 调度器闭包
├── dataloader/           # xyz / lmdb 数据管道
├── metadata.py           # E0/邻居统计工具
└── read_model.py         # 从权重导出/校验 model.json
```

## 核心行为
- **损失口径统一**：loss = wE * (能量 per-atom MSE) + wF * (力 per-component MSE)；验证同口径。RMSE/MAE 用全局 SSE/Count 计算，额外输出未除原子数的能量 MAE（energy_mae_cfg）便于对比总量误差。
- **日志指标**：同时打印 per-atom / per-comp RMSE/MAE 与“未除原子数”的能量 MAE，便于直接对比总量误差。
- **数据加载**
  - XYZ：要求 forces 存在，否则报错。
  - LMDB：pbc 优先读存储值；采样种子来自 args；缺失 key 在建索引时过滤，运行期不重采样；缺 key 直接报错。
  - dataloader 不再计算 e0/z_table/avg_num_neighbors，这些放在 JSON。
- **优化/调度**
  - `optimizer.py` 自动分组：bias/norm/scale/shift/标量 no_decay，其余 decay。
  - 调度器闭包 `scheduler_step(val_loss)` 适配 Plateau/StepLR，训练循环无需区分 `step(val_loss)` 与 `step()`。
- **EMA 与保存格式**
  - 训练/微调/续训可选 EMA；best 由 EMA（开启时）或 raw（关闭时）决定。
  - `checkpoint.pt`：当前 raw `model_state_dict` + `train_state`（含 optimizer/scheduler/ema_state_dict、lmdb_indices、config）+ `best_model_state_dict`（最佳，EMA 优先）+ CPU 版 `model`。
  - `best_model.pt`：最优权重（EMA 优先）+ CPU 版 `model`。`best_model_ema.pt` 已移除。
  - 评估可对 checkpoint 使用 `--use_ema` 选择加载 ema_state_dict；best_model.pt 不受开关影响（其自身即最优权重）。
- **JSON 规范化**：train 构建模型后会导出规范化后的 model.json 覆盖原文件，resume/finetune 直接使用该规范化版本做严格校验。
- **显存优化**：验证/评估仅在 forward 时启用梯度，随后 `detach()`，减少无谓图占用。

## 基本用法
准备好同目录的 `model.json`：
- 训练
  ```bash
  PYTHONPATH=$(pwd) python model-utils/train_mace.py \
    --data_format lmdb \
    --lmdb_train /path/to/train --lmdb_val /path/to/val \
    --output /path/to/run_dir \
    --batch_size 32 --epochs 100 --lr 1e-3
  ```
- 续训
  ```bash
  PYTHONPATH=$(pwd) python model-utils/resume.py \
    --checkpoint_dir /path/to/run_dir --epochs 150
  ```
- 微调
  ```bash
  PYTHONPATH=$(pwd) python model-utils/finetune.py \
    --checkpoint /path/to/run_dir/checkpoint.pt \
    --data_format lmdb --lmdb_train ... --lmdb_val ... \
    --lr 5e-5 --epochs 20
  ```
- 评估（可选 `--use_ema`）
  ```bash
  PYTHONPATH=$(pwd) python model-utils/evaluate.py \
    --checkpoint /path/to/run_dir/checkpoint.pt \
    --data_format lmdb --lmdb_path /path/to/val \
    --use_ema            # 可选，仅 checkpoint 才支持开关
  # evaluate 不需要 --model_json；best_model.pt 自身即模型+权重
  ```

## 设计选择与提示
- **JSON 校验**：`build_model_with_json` 在有 nn.Module 时强制 JSON 校验（除 E0 差异外报错），失败才回退 nn.Module；evaluate 仅要求 checkpoint 含 nn.Module。
- **随机性/复现**：LMDB 抽样用外部 seed；抽样索引写入 `train_state.lmdb_indices` 以便 resume/复现。
- **权重衰减**：no_decay 关键词 bias/norm/scale/shift + 标量/1D 权重名匹配，避免把物理尺度正则到 0。
- **数据校验**：LMDB 缺元素/缺 key 直接报错；XYZ 缺 forces 报错；pbc 按存储值。
- **日志**：当样本数 ≤10 时 evaluate 会额外输出逐样本能量误差与力 RMSE。

更多字段/CLI 细节见 `Parameters.md`，模型导出与元数据处理见 `read_model.py` / `metadata.py`。***
