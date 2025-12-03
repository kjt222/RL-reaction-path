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
- **OC22 LMDB**：解析 `data.000X.lmdb`，PyG Data → ASE Atoms → MACE Config。使用固定覆盖清单（55 个 OC22 元素）逐条扫描，确保每种元素至少一条样本；`max_samples` 低于 55 会报错，扫描全量仍缺元素也会报错。映射 `z_table` 优先使用模型的 `z_table`（可大于 55），覆盖集合不做截断；采样索引写入 checkpoint 的 `lmdb_indices`，`--resume`/`--reuse_indices` 可复用。

## 2025-12-03 更新
- LMDB 覆盖策略固定为 OC22 55 元素清单，构建子集时先扫全量保证每种元素至少一条，缺失元素直接报错；映射 z_table 可使用模型的 z_table（可大于 55），不再依赖采样检测元素。
- `finetune.py` 移除无效的 `--lmdb_e0_samples`/`--neighbor_sample_size` 参数，调用 dataloader 时自动传入 `model.json` 的 z_table 对齐索引，覆盖集合与模型 z_table 不匹配会报错。
- `metadata.override_e0_from_json` 支持同时覆盖 state_dict 与 nn.Module，并可直接保存至指定 pt 文件。

## 2025-12-02 更新
- `metadata.py` 精简为两件事：1) `override_e0_from_json` 用 JSON 的 E0 写回 state_dict；2) `recompute_e0s_from_lmdb` 从 LMDB 采样（默认 50 万条）最小二乘重算 E0，未覆盖的元素保留旧值，CLI 用法：
  ```bash
  PYTHONPATH=$(pwd)/..:$(pwd) python metadata.py \
    --lmdb_dir /path/to/train \
    --model_json models/MACE-MP-0-medium/oc22/model-oc22.json \
    --output_json models/MACE-MP-0-medium/oc22/model-oc22.json \
    --max_samples 500000
  ```
- 校验逻辑移到 `read_model.validate_json_against_checkpoint`：导出 checkpoint 内 nn.Module 的完整 JSON 后逐字段对比。`finetune.py`/`resume.py`/`evaluate.py` 统一采用此校验，除 `e0_values` 差异会警告外，其余不一致一律报错。
- `finetune.py` 接口简化：必需 `--checkpoint` 指向任意 .pt，同目录 `model.json` 默认使用；`--checkpoint_dir` 与 `--override_e0_from_json` 已移除，输出默认写在 `<checkpoint父目录>/finetune`。
- `finetune.py` 使用与 `train_mace.py` 相同的 LMDB dataloader（`prepare_lmdb_dataloaders`），data loader 本身负责覆盖样本中的元素；`model.json` 只参与模型构建校验和提供统计/架构元数据，不再决定子集中必须含有哪些元素。

## 2025-11-30 更新
- 新增 `models/MACE-MP-0-medium/raw/model.json`：由 `read_model.py` 直接解析官方原始权重 (`raw/MACE-MP-0-medium.pt`) 生成，字段采用 Parameters.md 中的命名（含 hidden_irreps=128x0e+128x1o+128x2e、avg_num_neighbors、z_table、e0_values 等），后续推理/评估请优先依赖该 JSON 而非旧 metadata。
- `read_model.py` 现支持递归打印子模块属性/缓冲，能在包装器内部读取 avg_num_neighbors、irreps 等信息，侦探模式同时反推 hidden_irreps/num_channels/num_radial_basis，并在写出 `model.json` 时自动填充推断出的关键字段（如 hidden_irreps/max_ell/num_channels/num_radial_basis/num_interactions/path_count），无需手动补全模型结构。使用前请确保 `PYTHONPATH` 覆盖项目根且 `unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`。
- 新增 `models/MACE-MP-0-large/raw/model.json` 与 `raw/MACE-MP-0-large.pt`，同样由 `read_model.py` 导出，可与 medium 版相同流程使用。
- 模型资产改为 `model.json` + 纯权重：`process_model.py` 现在仅用 `model.json`/`--e0_file` 覆盖 state_dict 的 E0 并输出 checkpoint/best_model（不再写 metadata）。
- 清理资源：删除了 `models/MACE-MP-0-medium/processed*` 等 metadata 目录，统一只保留 raw 权重 + `model.json`。`finetune.py`/`resume.py` 现已适配 `model.json`：用 `read_model.validate_json_against_checkpoint` 严格校验，先按 JSON 重建模型，失败再回退到 checkpoint 内置的 nn.Module。

## 2025-11-29 更新
1. `evaluate.py` 强制配置日志且启动时打印提示；当数据条目 ≤10 时会输出逐样本能量误差与力 RMSE，使用 `batch` 索引聚合，避免 reshape 崩溃。
2. （已废弃）`read_model_structure.py` 功能已并入 `read_model.py` 的递归打印/侦探模式。
3. 模型资源改动：淘汰旧的 `processed_oc22` 系列目录并移除冗余 `raw/e0_oc22.json`，请改用 raw 权重 + `model.json`。
4. 环境：补充 `mace_env.yml`（完整 Conda 导出）与 `pip_filtered.txt`（过滤后的 `pip freeze`），与原 `environment.yml` 最小依赖列表并存。
5. WSL NVML 兼容：各脚本内部已 stub NVML/cuequivariance，运行时建议 `CUDA_VISIBLE_DEVICES="" CUEQUIVARIANCE_DISABLE_NVML=1`，并确保未设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`。
6. PyG 本地编译：若使用 nightly Torch（如 2.10.dev + cu128），需自行源码编译 `torch-scatter`/`torch-sparse`/`torch-spline-conv`/`torch-cluster`/`torch-geometric` 以匹配 ABI，可用 `CUDA_HOME=/usr/local/cuda-12.8 pip install --no-binary=:all: ...`。

## 2025-11-26 更新
1. 新增 `resume.py`：专职断点续训入口，加载 checkpoint 内的 config、优化器/调度器/EMA 状态、`lmdb_indices`、元数据与模型权重，从 `epoch+1` 继续；`train_mace.py` 不再提供 `--resume`。
2. `train_mace.py` 现在总是按当前的 `--lmdb_*_max_samples` 重新采样 LMDB 子集；若要复用旧子集，请用 `finetune.py --reuse_indices` 或 `resume.py`，其中 `max_samples` 在复用时不再裁剪/扩充，只会提示。
3. `reuse_indices` 打开时，`max_samples` 只提示不生效；想换子集请去掉该开关，让数据集按新的 `max_samples` 重新抽样并写入新的 `lmdb_indices`。
4. （旧流程，已由 `model.json` 取代）`metadata.py` 会在有 metadata 的 checkpoint 同目录写出 `metadata.json`；缺 metadata 时可在信任前提下用 JSON 补齐。当前推荐直接使用 `model.json`（见 2025-11-30 更新）。

## 2025-11-18 更新
1. **LMDB 容错**：当某个 shard 缺失 key 时不再中止训练，而是告警并重采样其它样本，最多尝试 8 次；构建采样索引时也会跳过坏样本并记录数量。
2. **子集追踪**：checkpoint 内新增 `lmdb_indices` 字段，包含 train/val 的 `(shard_idx, local_idx)` 列表；resume 时自动注入，确保断点续训不会改变样本集合。
3. **单精度默认**：`torch.set_default_dtype(torch.float32)`（此前是 float64），显著降低显存占用与算力压力。如需双精度，可自行修改 `train_mace.py` / `evaluate.py`。
4. **周期保存**：`--save_every` 默认改为 `1`，即每个 epoch 都写 `checkpoint.pt`，最大化断点恢复精度。可按需调大或设为 0 关闭。
5. **推荐命令**：若在 WSL+NVIDIA 环境中训练，建议附加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 减少显存碎片，示例命令见下文。
6. **元数据解耦**（已迁移为 `model.json` 流程，见 2025-11-30 更新）：新增 `metadata.py`，checkpoint 分三类信息：
   - 模型参数：`model_state_dict`（可用于继续训练或微调）。
   - 元数据（跟模型绑定，微调也不可变）：`metadata` 包含 `z_table`、`avg_num_neighbors`、`e0_values`、`cutoff`、`num_interactions`，兼容旧字段。
   - 训练状态（可选恢复，可替换）：`train_state` 包括优化器/调度器/EMA 状态、`epoch`、`best_val_loss`、`lmdb_indices`、`config`。
  `train_mace.py`/`evaluate.py` 均通过 `load_checkpoint` 读取；保存用 `save_checkpoint`，避免重复堆砌字段并确保旧格式可读。
7. **微调脚本**：`finetune.py` 现直接接受任意 `--checkpoint` 路径（同目录 `model.json`），默认输出到 `<checkpoint父目录>/finetune`，可重设数据路径/超参，校验 JSON 严格（非 E0 不一致即报错，E0 差异仅警告）。
8. **微调默认收敛策略**：`finetune.py` 采用 AdamW、ReduceLROnPlateau 默认 patience=4，梯度裁剪默认 1.0，减少震荡与刷屏的 tqdm（动态列宽、0.5s 刷新间隔）。旧 checkpoint 若缺 metadata 会根据当前数据重估并写回新的 checkpoint，后续可直接复用。

## 模型准备（model.json 主导）
1. **导出 JSON**：官方权重可用 `read_model.py` 生成 `model.json`，如
   ```bash
   python read_model.py models/MACE-MP-0-medium/raw/MACE-MP-0-medium.pt \
     --write-json models/MACE-MP-0-medium/raw/model.json
   ```
   大模型同理替换路径。
2. **写入 E0 并生成纯权重**：用 `process_model.py` 将 `model.json`（或 `--e0_file`）中的 E0 覆盖到 state_dict，并输出不含 metadata 的 `checkpoint.pt`/`best_model.pt`：
   ```bash
   python process_model.py \
     --input models/MACE-MP-0-medium/raw/MACE-MP-0-medium.pt \
     --output_dir models/MACE-MP-0-medium/raw \
     --model_json models/MACE-MP-0-medium/raw/model.json
   ```
3. **保持 JSON 同步**：训练/评估/部署时请将 `model.json` 与 checkpoint 放在同一目录，或提前复制到输出目录（如 `models/test1/model.json`）。当前脚本默认信任 `model.json` 作为唯一元数据。

## 训练脚本 `train_mace.py`
- 运行前必须在 `--output` 目录放置 `model.json`（架构 + 统计量），脚本会按 JSON 构建模型；checkpoint 不再写入 metadata，JSON 是唯一的元数据来源。推荐直接拷贝 `models/MACE-MP-0-medium/raw/model.json` 或用 `read_model.py` 生成。
- 支持 `xyz`/`lmdb`，提供 `--lmdb_train_max_samples` / `--lmdb_val_max_samples` 随机抽取子集做 smoke test。
- **自动 checkpoint**：`--output` 现在视为目录；周期写 `checkpoint.pt`（state_dict + train_state + 一个 CPU 模型副本），最佳时覆盖 `best_model.pt`（同样仅包含权重+模型，不含 metadata），默认每个 epoch 保存，可用 `--save_every` 调整或 0 关闭。
- `LmdbAtomicDataset` 会在每个 DataLoader worker 内独立打开 LMDB 句柄，可安全使用 `num_workers>0`；如在 `/mnt/d` 上仍遇到 I/O 瓶颈，可将数据复制到 WSL 本地磁盘（如 `/home/<user>/oc22_data`），或暂时 `--num_workers 0`。
- 训练循环默认开启 tqdm 进度条，可通过 `--no-progress` 关闭；默认 dtype 为 float32。
- 运行示例（先准备 output 目录的 `model.json`，含显存碎片优化）：
  ```bash
  mkdir -p models/test1
  cp models/MACE-MP-0-medium/raw/model.json models/test1/
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_mace.py \
    --data_format lmdb \
    --lmdb_train /path/to/train \
    --lmdb_val /path/to/val_id \
    --batch_size 32 --epochs 5 --num_workers 4 \
    --neighbor_sample_size 512 --lmdb_e0_samples 2000 \
    --lmdb_train_max_samples 10000 --lmdb_val_max_samples 2000 \
    --output models/mace_oc22
  ```
- 若在本机（例如 RTX 5060 Ti）上复现，可参考当前默认跑法（假设 `models/test1` 已含 `model.json`）：
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
- 需要 `checkpoint.pt` + 同目录的 `model.json`；启动用 `read_model.validate_json_against_checkpoint` 严格校验（除 E0 外不一致即报错），按 JSON 重建模型并加载权重，失败回退到 checkpoint 内置 nn.Module。
- 保存逻辑与 `train_mace.py` 对齐：`checkpoint.pt` / `best_model.pt` 都包含权重 + 一个 CPU 版模型副本 + 训练状态（epoch/config/lmdb_indices/优化器等）。
- 示例：
  ```bash
  python MACE_pretrain/resume.py \
    --checkpoint_dir /path/to/run_dir \
    --epochs 120 \  # 可选，提升总轮数；若不填则用 checkpoint 中的设定
    --output /path/to/run_dir  # 可选，默认覆盖原目录
  ```
-  如果需要复用相同的 LMDB 子集，直接使用 checkpoint 内的 `lmdb_indices`；如果想重新采样，请重新跑 `train_mace.py`（无 `--reuse_indices` 概念）或 `finetune.py` 时去掉 `--reuse_indices`。

## 评估脚本 `evaluate.py`
- 需要与 checkpoint 同目录的 `model.json`（或用 `--model_json` 指定）；启动时用 `read_model.validate_json_against_checkpoint` 校验 JSON 与 checkpoint，差异（除 E0 外）会报错；按 JSON 重建模型并严格加载权重，若 strict 失败才回退到 checkpoint 内置的 nn.Module。
- `z_table`/`cutoff` 等直接从模型推断并用于 DataLoader 元素校验。
- 支持 `--lmdb_val_max_samples` 随机抽取验证子集，避免全量扫描。
- 数据条目 ≤10 时额外输出逐样本能量误差与力 RMSE；更大数据集仅汇总 Loss / Energy/Force RMSE / R²。
- 示例：
  ```bash
  python evaluate.py \
    --checkpoint models/MACE-MP-0-medium/raw/MACE-MP-0-medium.pt \
    --model_json models/MACE-MP-0-medium/raw/model.json \
    --data_format lmdb \
    --lmdb_path /path/to/val_ood \
    --batch_size 32 --num_workers 4
  ```

## 微调脚本 `finetune.py`
- 需要指定 `--checkpoint`（任意 .pt）+ 同目录的 `model.json`；启动时用 `read_model.validate_json_against_checkpoint` 严格校验（非 E0 不一致报错，E0 差异仅警告），按 JSON 重建模型并加载权重，失败回退 checkpoint 内置 nn.Module。
- 支持 `--model_json` 指定其他 JSON；`--reuse_indices` 复用 checkpoint 的 LMDB 子集；其他超参可重设。
- 输出默认写到 `<checkpoint父目录>/finetune` 下的 `checkpoint.pt` / `best_model.pt`（包含权重、CPU 模型副本与训练状态）。
- 新增 `models/MACE-MP-0-medium/oc22/finetune.sh`，封装常用路径/超参（默认 `model-oc22.json` + `MACE-MP-0-medium.pt`、`num_workers=0`、`lmdb_train_max_samples=500000`、`plateau_patience=4`），方便在本机快速重现实验。
- 示例（重设 lr，复用 LMDB 子集）：
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune.py \
    --checkpoint /path/to/run_dir/best_model.pt \
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
6. **CPU 运行崩溃**：在 WSL 中给 `resume.py` 设置 `CUDA_VISIBLE_DEVICES=""` 曾出现 glibc “double free detected in tcache 2” 崩溃；若需要仅用 CPU，建议关闭该变量后重启 Shell 或显式在脚本里设置 `device=cpu`，避免通过屏蔽 CUDA 的方式触发崩溃。

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
