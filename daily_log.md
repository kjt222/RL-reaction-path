# Daily Log


## 日志模板（简版）
- 今日目标/假设：
- 执行与结果（简述）：
- 结论/决策：
- 下一步：
## 2026-01-15
- 初始化日志机制；确认后续需持续记录可复现实验思路。
- 现状回顾：阶段性工作主要是下载模型/后端、统一训练接口与误差计算；核心方法论与闭环尚未成型。
- 对 OC22 相关 MACE 进行了误差诊断：区分缩放问题与方向问题；确认误差主要集中在高力区域与 adsorbate（tag2）原子。
- 基于 val_id_1k 的统计验证 OCP tags 语义：tag0=slab/bulk，tag1=surface layer，tag2=adsorbate。
- 设计并记录“动作+quench+basin+AL”方案与阶段性落地清单，形成清晰的架构决策（见 channel.md）。
- 落地结构骨架：新增 core/al、core/sampling、core/metrics_ext 与 experiments 目录（占位文件，未接入主流程）。
## 2026-01-16
- 采样模块落地到 experiments/sampling：schema/geometry/validators/pipeline + 动作实现（平移/旋转/推拉/二面角/jitter）。
- 新增采样扩展点：forcefield 接口、盆地图 registry、Recorder 钩子；实现三类 recorder（steps/basins/dft_queue）。
- 引入 StructureStore（NPZ）与结构引用；Recorder 改为写结构引用而非内嵌坐标。
- DFT 队列：增加 queue_idx 续扫；新增出队处理（canonicalize+去重+submit/skip 输出）。
- 新增 DFT 去重模块（粗桶+RMSD），与采样解耦；默认 RMSD 阈值 0.08 Å。
- 新增 slab 参考系 canonicalize（cell/PCA 轴 + anchor 平移），仅用于 DFT 出队阶段。
- 实现 slab 几何分层与 core/movable mask 推断；push_pull/dihedral 支持 core-core 过滤。
- 接入力统计：记录 force_pre/force_min（topK=3/5），quench 输出最后一步 forces。
- 新增触发器构建：max_F / topK_mean 触发；默认阈值 max_F>0.7 或 top5>0.35。
- 增加便捷构建函数 build_action_inputs，自动生成 selection_mask 与 candidates。
- 更新 README.md / parameters.md，同步 experiments 采样与 DFT 出队说明。
## 2026-01-18
- 修复 SamplingPipeline 缩进错误；支持 basin 对象 identify；未收敛 quench 不进入 basin。
- 引入 ForceFnCalculator：CPU quench + GPU 力推理；FIRE/LBFGS 支持 fixed/tags 约束与回调。
- 采样入口新增 quench/amp/target_basins 统计与汇总；EquiformerV2 跑通采样至 5 个 basin。
- DFT 触发扩展为包含 quench 中间构型；候选数量显著上升并进入出队去重。
- DFT 去重切换为全局 RMSD（默认 0.18 Å），将候选压缩到约 20。
- JitterAction 改为 seed 伪随机，确保可复现。

## 2026-01-27
- 文档与日志统一：`README.md` / `parameters.md` / `docs/vasp_oc22_repro.md` 与当前采样链路对齐（`--config`、MDAction、action_plugins、action_quality）。
- 关键决策被明确记录：对标 OC22 标签时必须单点（不得结构优化）；`LDAU/MAGMOM/KPOINTS` 必须按物种顺序逐样本展开。
- 采样质量方向收敛：将 gate 语义收敛为“动作质量验证 + 重采样”，强调通过率与拒绝原因统计。
