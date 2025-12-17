# Parameters Checklist

核对 `model.json` / checkpoint / CLI 时常用的字段与约定。

## 1) model.json 必备字段
- 结构：`model_type`、`hidden_irreps`、`MLP_irreps`、`num_channels`、`num_interactions`、`correlation`、`max_ell`、`num_radial_basis`、`num_polynomial_cutoff`、`radial_type`、`gate`、`scaling`
- 统计：`z_table`、`cutoff`、`avg_num_neighbors`、`e0_values`
- 建议用 `read_model.py --write-json` 从权重直接导出；需要重算 E0 可用 `metadata.py`。

## 2) checkpoint / best 格式
- `checkpoint.pt`（train/finetune/resume统一）：
  - `model_state_dict`：当前 raw 权重
  - `train_state`: `optimizer_state_dict`、`scheduler_state_dict`、`ema_state_dict`（如启用）、`epoch`、`best_val_loss`、`config`、`lmdb_indices`
  - `best_model_state_dict`: 当前记录的最优权重（开启 EMA 时为 EMA，否则 raw），另带 `best_epoch` 方便 resume
  - `model`: CPU 版模型副本
  - 不再存 metadata，结构/统计依赖同目录的 `model.json`
- `best_model.pt`：最优权重（开启 EMA 时为 EMA，否则 raw）+ CPU 模型
- `best_model_ema.pt` 已移除

## 3) 常用 CLI
- 数据：`--data_format (xyz|lmdb)`、`--xyz_dir`、`--lmdb_*`、`--batch_size`、`--num_workers`、`--seed`
- 采样：`--sample_size`（xyz）、`--lmdb_train_max_samples`、`--lmdb_val_max_samples`
- 损失权重：`--energy_weight`、`--force_weight`
- 训练控制：`--epochs`、`--lr`、`--weight_decay`、`--save_every`、`--output`
- 微调/续训：`--checkpoint`（必填）+ `model.json`，`--reuse_indices`（finetune），`--progress` 控制进度条
- 评估：`--checkpoint`、`--lmdb_path`/`--xyz_dir`、`--use_ema`（从 checkpoint 的 ema_state_dict 评估），不需要 `--model_json`

## 4) 检查要点
- JSON 与权重结构一致（hidden_irreps/max_ell/path_count 等）；缺失 forces 的 XYZ 直接报错。
- LMDB：pbc 用存储值；缺 key/缺元素报错；采样索引用外部 seed，可在 checkpoint 的 `lmdb_indices` 复现；coverage 默认为 OC22 元素表，若需自定义传 `--elements`。
- 权重衰减：bias/norm/scale/shift/标量自动 no_decay，其余 decay；调度器通过闭包统一 `scheduler_step(val_loss)`。
- EMA：开启时 best 取 EMA；checkpoint 始终保存 raw+ema_state_dict 便于 resume；评估可用 `--use_ema` 切换；best_model.pt 自身即模型+最优权重（不再有 best_model_ema.pt）。
- LMDB 复现：resume/finetune 使用 checkpoint 保存的 lmdb_indices，coverage_zs 默认为 OC22（可用 --elements 覆盖），若传入 resume_indices 必须包含 train/val，否则报错。
- 权重衰减：标量/1D（常见 bias/scale）默认 no_decay，额外可用 no_decay_keywords 覆盖；避免 no_decay=0 的极端情况。
- JSON 规范化：train 构建模型后导出规范化 JSON 覆盖原文件，后续 resume/finetune 直接使用；strict 校验缺键/不一致会报错。
