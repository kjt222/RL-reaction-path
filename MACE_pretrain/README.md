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

## 2025-11-18 更新
1. **LMDB 容错**：当某个 shard 缺失 key 时不再中止训练，而是告警并重采样其它样本，最多尝试 8 次；构建采样索引时也会跳过坏样本并记录数量。
2. **子集追踪**：checkpoint 内新增 `lmdb_indices` 字段，包含 train/val 的 `(shard_idx, local_idx)` 列表；resume 时自动注入，确保断点续训不会改变样本集合。
3. **单精度默认**：`torch.set_default_dtype(torch.float32)`（此前是 float64），显著降低显存占用与算力压力。如需双精度，可自行修改 `train_mace.py` / `evaluate.py`。
4. **周期保存**：`--save_every` 默认改为 `1`，即每个 epoch 都写 `checkpoint.pt`，最大化断点恢复精度。可按需调大或设为 0 关闭。
5. **推荐命令**：若在 WSL+NVIDIA 环境中训练，建议附加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片，示例命令见下文。

## 训练脚本 `train_mace.py`
- 支持 `xyz`/`lmdb`；保存 checkpoint 时写入 `model_state_dict`、`best_val_loss`、`avg_num_neighbors`、`z_table`、`e0_values`、`cutoff`、`num_interactions`，确保评估/推理沿用训练统计。
- 提供 `--lmdb_train_max_samples` / `--lmdb_val_max_samples` 随机抽取指定数量的样本做 smoke test，无需复制/删除原始 LMDB。
- **自动 checkpoint/续训**：
  - `--output` 指向目录，内部维护 `checkpoint.pt`（周期保存当前状态，默认每 1 个 epoch 触发，可用 `--save_every` 调整或设 0 关闭）和 `best_model.pt`（每次刷新 val 最优时覆盖）。覆盖式写入，不会堆积历史文件。
  - `checkpoint.pt` 内包含模型、优化器、调度器、EMA、当前/最佳指标，以及完整运行配置 `config`。最新版还会存储 `lmdb_indices`（train/val 的采样索引），`--resume <目录>` 会自动复用这些索引，避免因随机采样导致的元素缺失或 KeyError。若想保留旧存档，可换新的 `--output` 目录。
  - 中断后恢复会从最近一次 checkpoint 的 epoch+1 开始，若想减少丢失进度，将 `--save_every` 调小。
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

## 评估脚本 `evaluate.py`
- 载入 .pt 时复用保存的 `z_table`、`avg_num_neighbors` 等；输出 Loss、Energy/Force RMSE、R²。
- 示例：
  ```bash
  python evaluate.py \
    --checkpoint mace_oc22.pt \
    --data_format lmdb \
    --lmdb_path /path/to/val_ood \
    --batch_size 32 --num_workers 4
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

## 与原 README 的关系
- `README_original.md`：提供了 MACE 模型结构、OC22 数据说明；本目录继承其超参设置与数据格式。
- `README_new.md`：总结了本地脚本的蓄水池采样、E0 拟合、EMA、最佳 checkpoint 保存等经验，已在 `xyz_loader.py`/`train_mace.py` 中实现。

## 目录/路径提示
- OC22 数据无需放入 `ocp` 根目录，可直接通过 `--lmdb_train`/`--lmdb_val` 指向绝对路径。
- 若在 WSL 中想要可视化浏览文件，可在 Windows 资源管理器输入 `\\wsl$\Ubuntu\home\kjt`（按实际发行版/用户名替换）。

以上内容总结了当前代码状态及使用心得，后续若有新问题（如新增元素、推理脚本等）请继续在此文档中补充。
