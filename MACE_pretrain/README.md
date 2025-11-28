# MACE_pretrain

该目录封装了一套用于 OC22/XYZ 数据的 MACE 预训练与评估脚本，基于 `README_original.md` 的官方流程与 `README_new.md` 的自定义经验。文件结构：

```
MACE_pretrain/
├── train_mace.py
├── evaluate.py
├── models.py
└── dataloader/
    ├── xyz_loader.py
    └── lmdb_loader.py
```

## 数据后端
- **Extended XYZ**：沿用 `MACE train rMD17.py` 的蓄水池采样、`KeySpecification`、`compute_e0s` 回退策略。
- **OC22 LMDB**：解析 `data.000X.lmdb`，PyG Data → ASE Atoms → MACE Config；支持 `--lmdb_e0_samples`、`--neighbor_sample_size`。若 checkpoint 缺少 `z_table` / `avg_num_neighbors` 会回退估算并给出警告。当前版本的 `LmdbAtomicDataset` 在构建 `--lmdb_*_max_samples` 子集时，会**优先为每个元素保留至少一个样本**，若 `max_samples` 低于元素种类数（OC22 为 57）会直接报错；采样索引会被保存到 checkpoint 的 `lmdb_indices` 中，`--resume` 会强制复用相同子集，避免随机性导致继续训练失败。

## 2025-11-29 更新
1. `evaluate.py` 强制配置日志且启动时打印提示；当数据条目 ≤10 时会输出逐样本能量误差与力 RMSE，使用 `batch` 索引聚合，避免 reshape 崩溃。
2. `read_model_structure.py` 采用占位类 + `weights_only=True` 加载，规避 NVML/扩展副作用，≤200 长度张量直接打印完整数值便于检查 E0。
3. 模型资源：新增 `models/MACE-MP-0-medium/processed_oc22_v2/`（含 `checkpoint.pt`/`best_model.pt`/`metadata.json`），淘汰旧的 `processed_oc22` 目录并移除冗余 `raw/e0_oc22.json`。
4. 环境：补充 `mace_env.yml`（完整 Conda 导出）与 `pip_filtered.txt`（过滤后的 `pip freeze`），与原 `environment.yml` 最小依赖列表并存。

## 2025-11-26 更新
1. 新增 `resume.py`：专职断点续训入口，加载 checkpoint 内的 config、优化器/调度器/EMA 状态、`lmdb_indices`、元数据与模型权重，从 `epoch+1` 继续；`train_mace.py` 不再提供 `--resume`。
2. `train_mace.py` 现在总是按当前的 `--lmdb_*_max_samples` 重新采样 LMDB 子集；若要复用旧子集，请用 `finetune.py --reuse_indices` 或 `resume.py`，其中 `max_samples` 在复用时不再裁剪/扩充，只会提示。
3. `reuse_indices` 打开时，`max_samples` 只提示不生效；想换子集请去掉该开关，让数据集按新的 `max_samples` 重新抽样并写入新的 `lmdb_indices`。
4. `metadata.py` 增强：有 metadata 的 checkpoint 会在同目录自动写出 `metadata.json`；若目录已有 JSON 与 checkpoint metadata 不一致会报错；若 checkpoint 缺 metadata，可在信任前提下用 JSON 补齐。建议为外部下载的模型（如 MACE-MP-0）在模型目录手工写入 `metadata.json`，至少包含 `z_table`、`avg_num_neighbors`、`e0_values`、`cutoff`、`num_interactions`，可选附加架构字段（如 `model_type`、`hidden_irreps`、`num_radial_basis`、`num_polynomial_cutoff`/`num_cutoff_basis`、`max_ell`、`correlation`、`num_channels`、`radial_type`、`gate`）以及训练/数据标签，便于复现与校验。

## 2025-11-18 更新
1. **LMDB 容错**：当某个 shard 缺失 key 时不再中止训练，而是告警并重采样其它样本，最多尝试 8 次；构建采样索引时也会跳过坏样本并记录数量。
2. **子集追踪**：checkpoint 内新增 `lmdb_indices` 字段，包含 train/val 的 `(shard_idx, local_idx)` 列表；resume 时自动注入，确保断点续训不会改变样本集合。
3. **单精度默认**：`torch.set_default_dtype(torch.float32)`（此前是 float64），显著降低显存占用与算力压力。如需双精度，可自行修改 `train_mace.py` / `evaluate.py`。
4. **周期保存**：`--save_every` 默认改为 `1`，即每个 epoch 都写 `checkpoint.pt`，最大化断点恢复精度。可按需调大或设为 0 关闭。
5. **推荐命令**：若在 WSL+NVIDIA 环境中训练，建议附加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片，示例命令见下文。
6. **元数据解耦**：新增 `metadata.py`，checkpoint 分三类信息：
   - 模型参数：`model_state_dict`（可用于继续训练或微调）。
   - 元数据（跟模型绑定，微调也不可变）：`metadata` 包含 `z_table`、`avg_num_neighbors`、`e0_values`、`cutoff`、`num_interactions`，兼容旧字段。
   - 训练状态（可选恢复，可替换）：`train_state` 包括优化器/调度器/EMA 状态、`epoch`、`best_val_loss`、`lmdb_indices`、`config`。
  `train_mace.py`/`evaluate.py` 均通过 `load_checkpoint` 读取；保存用 `save_checkpoint`，避免重复堆砌字段并确保旧格式可读。
7. **微调脚本**：新增 `finetune.py`，可指向已训练目录（含 `checkpoint.pt` 与 `best_model.pt`/`best.pt`），自动读取元数据、从最佳权重起步，允许重新指定数据路径、学习率、是否复用优化器/调度器状态及 LMDB 采样索引。
8. **微调默认收敛策略**：`finetune.py` 采用 AdamW、ReduceLROnPlateau 默认 patience=4，梯度裁剪默认 1.0，减少震荡与刷屏的 tqdm（动态列宽、0.5s 刷新间隔）。旧 checkpoint 若缺 metadata 会根据当前数据重估并写回新的 checkpoint，后续可直接复用。

## 训练脚本 `train_mace.py`
- 支持 `xyz`/`lmdb`；保存 checkpoint 时写入 `model_state_dict`、`best_val_loss`、`avg_num_neighbors`、`z_table`、`e0_values`、`cutoff`、`num_interactions`，确保评估/推理沿用训练统计。
- 提供 `--lmdb_train_max_samples` / `--lmdb_val_max_samples` 随机抽取指定数量的样本做 smoke test，无需复制/删除原始 LMDB。
- **自动 checkpoint**：
  - `--output` 指向目录，内部维护 `checkpoint.pt`（周期保存当前状态，默认每 1 个 epoch 触发，可用 `--save_every` 调整或设 0 关闭）和 `best_model.pt`（每次刷新 val 最优时覆盖）。覆盖式写入，不会堆积历史文件。
- `checkpoint.pt` 内包含模型、优化器、调度器、EMA、当前/最佳指标，以及完整运行配置 `config`，并存储 `lmdb_indices`（train/val 的采样索引）。若要续训或复用子集，请使用 `resume.py` 或在 `finetune.py` 中启用 `--reuse_indices`；`train_mace.py` 本身只负责全新训练。
- `LmdbAtomicDataset` 会在每个 DataLoader worker 内独立打开 LMDB 句柄，可安全使用 `num_workers>0`；如在 `/mnt/d` 上仍遇到 I/O 瓶颈，可将数据复制到 WSL 本地磁盘（如 `/home/<user>/oc22_data`），或暂时 `--num_workers 0`。
- 训练循环默认开启 tqdm 进度条，可通过 `--no-progress` 关闭。
- 默认 dtype 为 float32，可在 `train_mace.py` 顶部修改。
- 运行示例（WSL 环境，含显存碎片优化）：
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_mace.py \
    --data_format lmdb \
    --lmdb_train /path/to/train \
    --lmdb_val /path/to/val_id \
    --batch_size 32 --epochs 5 --num_workers 4 \
    --neighbor_sample_size 512 --lmdb_e0_samples 2000 \
    --lmdb_train_max_samples 10000 --lmdb_val_max_samples 2000 \
    --output mace_oc22.pt
  ```
- 若在本机（例如 RTX 5060 Ti）上复现，可参考当前默认跑法：
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/kjt/mace_project/MACE_pretrain/train_mace.py \
    --data_format lmdb \
    --lmdb_train /home/kjt/Data/oc22_data/s2ef-total/train \
    --lmdb_val /home/kjt/Data/oc22_data/s2ef-total/val_id \
    --lmdb_train_max_samples 50000 --lmdb_val_max_samples 2000 \
    --batch_size 8 --num_workers 2 --epochs 400000 \
    --neighbor_sample_size 2048 --lmdb_e0_samples 50000 \
    --output "/mnt/d/D/calculate/MLP/MACE项目/MACE_pretrain/models/test1"
  ```

## 续训脚本 `resume.py`
- 用途：原封不动从 checkpoint 继续训练，加载模型、元数据、`config`、优化器/调度器/EMA 状态、`lmdb_indices`，从保存的 `epoch+1` 接着跑。`train_mace.py` 的旧 `--resume` 已移除，请改用此脚本。
- 使用示例：
  ```bash
  python MACE_pretrain/resume.py \
    --checkpoint_dir /path/to/run_dir \
    --epochs 120 \  # 可选，提升总轮数；若不填则用 checkpoint 中的设定
    --output /path/to/run_dir  # 可选，默认覆盖原目录
  ```
  如果需要复用相同的 LMDB 子集，直接使用 checkpoint 内的 `lmdb_indices`；如果想重新采样，请重新跑 `train_mace.py`（无 `--reuse_indices` 概念）或 `finetune.py` 时去掉 `--reuse_indices`。

## 评估脚本 `evaluate.py`
- 载入 .pt 时复用保存的 `z_table`、`avg_num_neighbors` 等；若同目录存在 `metadata.json` 会强校验一致性（`best_model.pt` 会优先读取旁边 `checkpoint.pt` 的 metadata）。
- 支持 `--lmdb_val_max_samples` 随机抽取验证子集，避免全量扫描。
- 数据条目 ≤10 时额外输出逐样本能量误差与力 RMSE；更大数据集仅汇总 Loss / Energy/Force RMSE / R²。
- 示例：
  ```bash
  python evaluate.py \
    --checkpoint mace_oc22.pt \
    --data_format lmdb \
    --lmdb_path /path/to/val_ood \
    --batch_size 32 --num_workers 4
  ```

## 微调脚本 `finetune.py`
- 适用于“从最优权重继续训练、调低学习率/更换数据/优化器状态”的场景。
- 读取 `--checkpoint_dir` 下的 `checkpoint.pt` 与 `best_model.pt`/`best.pt`，从最佳模型起步，元数据保持不变；数据路径/学习率/优化器状态可重新指定。
- `--reuse_indices` 会复用 checkpoint 中保存的 LMDB 子集；此时 `max_samples` 不会裁剪/扩充，只会提示；想换子集请去掉该开关让数据重新采样。
- 示例（使用 best_model，重设 lr，复用 LMDB 子集）：
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune.py \
    --checkpoint_dir /path/to/run_dir \
    --data_format lmdb \
    --lmdb_train /new/train/path \
    --lmdb_val /new/val/path \
    --batch_size 8 --epochs 20 --lr 5e-5 \
    --reuse_indices \
    --output /path/to/run_dir/finetune
  ```

## 常见问题与排查
1. **PyG 版本不兼容**：OC22 LMDB 基于旧版 PyG 保存，`lmdb_loader.py` 的 `_upgrade_pyg_data` 会从底层 `__dict__`/`_store` 重建 `Data`。若仍报错，可确认 `torch-geometric` / `torch-scatter` 等与 PyTorch 版本匹配（PyTorch 2.9.0 + CUDA 12.8 对应 `https://data.pyg.org/whl/torch-2.9.0+cu128.html`）。
2. **LMDB 键缺失**：OC22 LMDB 使用普通字符串键（"0"、"1"…），`LmdbAtomicDataset` 已改用 `key = f"{local_idx}".encode("ascii")` 读取。
3. **训练极慢**：
   - 数据位于 `/mnt/d/...` 时，WSL ↔ Windows I/O 缓慢；建议将 LMDB 拷贝到 WSL 本地 (`/home/kjt/...`) 或暂时设置 `--num_workers 0`。
   - 可以用 `--lmdb_train_max_samples` / `--lmdb_val_max_samples` 先跑小规模 smoke test，确认流程无误后再切到全量数据。
   - 6GB GPU 需使用极小 batch（如 3），导致单卡训练一轮需数小时。若可能，建议换用 16GB GPU 并调大 batch。
4. **lmdb_e0_samples / neighbor_sample_size**：
   - `--lmdb_e0_samples` 仅控制 E0 拟合时抽样数量；无需大于实际 LMDB 条目数。
   - `--neighbor_sample_size` 控制 avg_num_neighbors 的估算样本量。checkpoint 中存的是最终统计值，评估/推理不会重新计算。
5. **torch-geometric 依赖**：`torch-scatter`/`torch-sparse` 等需与 PyTorch+CUDA 版本匹配，安装方式请参考 PyG 官方安装指引。

## 环境/依赖
- `environment.yml`：精简依赖（PyTorch 2.4 + CUDA 12.1），适合新环境快速创建。
- `mace_env.yml`：完整 Conda 导出，包含当前开发机的 CUDA/torch-geometric/cuequivariance 版本，适合一键复刻。
- `pip_filtered.txt`：过滤后的 `pip freeze`，便于纯 pip 环境对齐版本。

## 与原 README 的关系
- `README_original.md`：提供了 MACE 模型结构、OC22 数据说明；本目录继承其超参设置与数据格式。
- `README_new.md`：总结了本地脚本的蓄水池采样、E0 拟合、EMA、最佳 checkpoint 保存等经验，已在 `xyz_loader.py`/`train_mace.py` 中实现。

## 目录/路径提示
- OC22 数据无需放入 `ocp` 根目录，可直接通过 `--lmdb_train`/`--lmdb_val` 指向绝对路径。
- 若在 WSL 中想要可视化浏览文件，可在 Windows 资源管理器输入 `\\wsl$\Ubuntu\home\kjt`（按实际发行版/用户名替换）。

以上内容总结了当前代码状态及使用心得，后续若有新问题（如新增元素、推理脚本等）请继续在此文档中补充。
