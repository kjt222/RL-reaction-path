# model_pretrain

基于 MACE 的预训练/微调/续训/评估脚本，模型结构由 `<name>_model.json` 主导。核心脚本在 `model-utils/`，通用工具在项目根。

```
model_pretrain/
├── model-utils/
│   ├── train_mace.py     # 预训练入口
│   ├── finetune.py       # 微调入口
│   ├── resume.py         # 断点续训入口
│   ├── evaluate.py       # 评估入口
│   ├── losses.py         # per-atom / per-comp 损失与指标
│   ├── models/           # JSON -> 模型构建
│   └── model_loader.py   # 统一的加载/保存/校验
├── optimizer.py          # 参数分组（bias/norm/scale/shift no_decay）+ 调度器闭包
├── dataloader/           # xyz / lmdb 数据管道
├── metadata.py           # E0/邻居统计工具
└── read_model.py         # 从权重导出/校验 model.json
```

## 核心行为
- **损失口径统一**：loss = wE * (能量 per-atom MSE) + wF * (力 per-component MSE)；验证同口径。RMSE/MAE 用全局 SSE/Count 计算，额外输出未除原子数的能量 MAE（energy_mae_cfg）便于对比总量误差。
- **日志指标**：训练/微调/评估日志同时打印 per-atom / per-comp RMSE/MAE，并额外打印未除原子数的能量绝对误差 `|E| cfg` 便于总量对比。
- **数据加载**
  - XYZ：要求 forces 存在，否则报错。
  - LMDB：pbc 优先读存储值；采样种子来自 args；缺失 key 在建索引时过滤，运行期不重采样；缺 key 直接报错；默认覆盖 OC22 元素，可用 `--elements` 自定义。
  - dataloader 不再计算 e0/z_table/avg_num_neighbors，这些放在 JSON。
- **优化/调度**
  - `optimizer.py` 自动分组：bias/norm/scale/shift/标量 no_decay，其余 decay。
  - 调度器闭包 `scheduler_step(val_loss)` 适配 Plateau/StepLR，训练循环无需区分 `step(val_loss)` 与 `step()`。
- **EMA 与保存格式**
  - 训练/微调/续训可选 EMA；best 由 EMA（开启时）或 raw（关闭时）决定。
  - `*_checkpoint.pt`：当前 raw `model_state_dict` + `train_state`（含 optimizer/scheduler/ema_state_dict、lmdb_indices、config）+ `model_json_text`/`model_json_hash` + `code_version` + `run_name`。
  - `*_model.pt`：最优权重（EMA 优先）+ CPU 版 `model`。
  - 评估按 `--input_model` 指定权重前向；如需评估最佳权重，请直接指向 `*_model.pt`。
  - 输出命名：`--output_checkpoint/--output_model` 传目录时自动生成 `<dir_name>_checkpoint.pt` / `<dir_name>_model.pt`；传文件则按文件名保存，并在同目录生成 `<name>_model.json`。
- **JSON 规范化**：train 构建模型后会导出规范化后的 `<name>_model.json` 写入输出位置，resume/finetune 使用该版本做严格校验。
- **微调输出**：finetune 会将来源的 `model.json` 复制到输出位置，保证后续校验与评估可用。
- **显存优化**：验证/评估仅在 forward 时启用梯度，随后 `detach()`；自动过滤 None 键，避免未计算量导致的 `None.detach()` 报错；整体减少无谓图占用。

## 基本用法
准备好 `model.json` 并在参数中显式指定：
- 训练
  ```bash
  PYTHONPATH=$(pwd) python model-utils/train_mace.py \
    --data_format lmdb \
    --lmdb_train /path/to/train --lmdb_val /path/to/val \
    --input_json /path/to/run/run_model.json \
    --output_checkpoint /path/to/run \
    --output_model /path/to/run \
    --batch_size 32 --epochs 100 --lr 1e-3
  ```
- 续训
  ```bash
  PYTHONPATH=$(pwd) python model-utils/resume.py \
    --input_model /path/to/run/run_checkpoint.pt \
    --output_checkpoint /path/to/run \
    --output_model /path/to/run \
    --epochs 150
  ```
- 微调
  ```bash
  PYTHONPATH=$(pwd) python model-utils/finetune.py \
    --input_model /path/to/run/run_checkpoint.pt \
    --output_checkpoint /path/to/finetune \
    --output_model /path/to/finetune \
    --data_format lmdb --lmdb_train ... --lmdb_val ... \
    --lr 5e-5 --epochs 20
  ```
- 评估
  ```bash
  PYTHONPATH=$(pwd) python model-utils/evaluate.py \
    --input_model /path/to/run/run_checkpoint.pt \
    --data_format lmdb --lmdb_path /path/to/val
  # evaluate 可选 --input_json，用于显式指定 model.json
  ```

## 设计选择与提示
- **JSON 校验**：加载时先校验 `model_json_hash`，若可加载 nn.Module 再做二次校验；无法解析 model.json 直接报错。
- **随机性/复现**：LMDB 抽样用外部 seed；抽样索引写入 `train_state.lmdb_indices` 以便 resume/复现。
- **权重衰减**：no_decay 关键词 bias/norm/scale/shift + 标量/1D 权重名匹配，避免把物理尺度正则到 0。
- **数据校验**：LMDB 缺元素/缺 key 直接报错；XYZ 缺 forces 报错；pbc 按存储值。
- **日志**：当样本数 ≤10 时 evaluate 会额外输出逐样本能量误差与力 RMSE。

更多字段/CLI 细节见 `Parameters.md`，模型导出与元数据处理见 `read_model.py` / `metadata.py`。***
