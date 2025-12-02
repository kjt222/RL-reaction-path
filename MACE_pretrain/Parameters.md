# Parameters Checklist

便于核对 model.json / checkpoint / 训练脚本的参数，按用途分组列出常用字段。

## 1) model.json 关键字段
- 模型结构：`model_type`、`hidden_irreps`、`MLP_irreps`、`num_channels`、`num_interactions`、`correlation`、`max_ell`/`max_L`、`num_radial_basis`/`num_bessel_basis`、`num_polynomial_cutoff`/`num_cutoff_basis`、`radial_type`、`gate`、`scaling`
- 数据统计：`z_table`、`cutoff`、`avg_num_neighbors`、`e0_values`
- 生成方式：推荐用 `read_model.py --write-json <ckpt>` 直接从权重导出；重算 E0 可用 `metadata.py` 的 `recompute_e0s_from_lmdb`，未覆盖的元素保留旧值。

## 2) checkpoint.pt 常见内容
- 权重：`model_state_dict`（当前）、`best_model_state_dict`（最优，如有）
- 训练状态（用于 resume/finetune）：`optimizer_state_dict`、`scheduler_state_dict`、`ema_state_dict`、`epoch`、`best_val_loss`、`lmdb_indices`（train/val 采样索引）、`config`（完整运行参数）
- 备注：不再写入 metadata；结构/统计依赖同目录 `model.json`

## 3) 训练 / finetune / 评估脚本常用 CLI 参数
- 数据与并行：`--data_format (xyz|lmdb)`、`--xyz_dir`、`--lmdb_train`、`--lmdb_val`、`--lmdb_path`、`--batch_size`、`--num_workers`、`--device`、`--seed`
- 子集/统计：`--lmdb_train_max_samples`、`--lmdb_val_max_samples`、`--lmdb_val_max_samples`（evaluate）`--lmdb_e0_samples`、`--neighbor_sample_size`、`--sample_size`（xyz）
- 模型/损失：`--energy_weight`、`--force_weight`、（其余超参通常来自 checkpoint/metadata）
- 训练控制：`--epochs`、`--lr`、`--weight_decay`、`--save_every`、`--output`
- 续训/微调：`--checkpoint`（必填，支持任意 .pt）、`--output`（默认 checkpoint 父目录/finetune）、`--reuse_indices`（finetune），其余超参同训练
- 评估：`--checkpoint`、`--lmdb_val_max_samples`、`--elements`（如果需要显式元素列表）

## 4) 检查建议
- 推理/评估：确保 `model.json` 与权重真实架构一致（例如 `hidden_irreps`、`max_ell` 等），`z_table`/`e0_values`/`avg_num_neighbors` 完整。`read_model.validate_json_against_checkpoint` 可直接对比。
- 使用 `read_model.py --write-json` 可从权重直接导出 `model.json`，会自动填充推断出的关键字段（`hidden_irreps`/`max_ell`/`num_channels`/`num_radial_basis`/`num_interactions`/产品 path_count 等），避免手工补全；需要新 E0 可先导出再用 `metadata.py` 重算。
- 训练/finetune：命令行超参（batch_size、num_workers、lr、epochs、max_samples、neighbor_sample_size 等）与记录在 `config` 的值保持一致；resume/finetune 时核对 `lmdb_indices` 是否按需复用。
- 如果切换到 nightly Torch 或新 CUDA 版本，确认 PyG 扩展与当前 torch ABI 匹配（必要时源码编译）。
