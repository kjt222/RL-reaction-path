# Implementation Log

## 2026-01-15
- Changes: 添加全局常开日志技能，强制记录 channel.md 与 implementation.md；更新 channel.md 记录计划条目。
- Files: /home/kjt/.codex/skills/project-logging/SKILL.md, channel.md, Agent-sum.md, implementation.md
- Tests: 未运行（结构与文档变更）
- Results: 技能已创建；channel.md 已记录计划；implementation.md 具备记录规范。
- Notes: 日志技能用于全局常开，后续按模板记录。
## 2026-01-16
- Changes: 记录本次对话的计划条目（删除历史对话指引）。
- Files: channel.md
- Tests: 未运行（仅文档变更）
- Results: 计划已记录。
- Notes: 待回复用户并引用官方说明。
## 2026-01-16
- Changes: 记录“下一步实施顺序/指标收敛”计划条目。
- Files: channel.md
- Tests: 未运行（仅文档变更）
- Results: 计划已记录。
- Notes: 等待用户确认高力指标与触发规则。
## 2026-01-16
- Changes: 记录“最小数据库”计划条目。
- Files: channel.md
- Tests: 未运行（仅文档变更）
- Results: 计划已记录。
- Notes: 等待用户确认最小数据库范围与形式。
## 2026-01-16
- Changes: 将采样模块从 core 迁移到 experiments/sampling，更新结构草案与步骤描述。
- Files: channel.md, experiments/sampling, core/sampling
- Tests: 未运行（结构与文档变更）
- Results: 采样模块已迁移；core 不再包含 sampling。
- Notes: 按用户要求采样逻辑不进入 core。
## 2026-01-16
- Changes: 在 experiments/sampling 实现最小采样闭环骨架（schema/geometry/validators/pipeline 与 4 个动作），补充 quench/basin 最小实现与结构设计更新。
- Files: experiments/sampling/__init__.py, experiments/sampling/schema.py, experiments/sampling/geometry.py, experiments/sampling/validators.py, experiments/sampling/pipeline.py, experiments/sampling/actions/{base,rigid_translate,rigid_rotate,push_pull,dihedral_twist,jitter,targets,__init__}.py, experiments/sampling/quench/{base,ase_fire,ase_lbfgs,__init__}.py, experiments/sampling/basin/{base,fingerprint,embed_energy,__init__}.py, channel.md
- Tests: python3 - <<'PY' (sampling pipeline smoke test)
- Results: 采样管线可运行，输出 SampleRecord 且 basin_id 生成正常。
- Notes: 动作均匀随机；默认参数待后续小样本统计校准。
## 2026-01-16
- Changes: 更新采样动作默认参数（更保守步进）；二面角成键判定改为共价半径因子+可选上限；jitter 更小；新增/扩展 validators（共价半径下限、固定原子、关键键拉断）；补充 sampling_rules 默认配置。
- Files: experiments/sampling/actions/{rigid_translate,rigid_rotate,push_pull,dihedral_twist,jitter}.py, experiments/sampling/actions/targets.py, experiments/sampling/validators.py, experiments/sampling_rules/config.yaml
- Tests: python3 - <<'PY' (sampling pipeline smoke test)
- Results: 采样管线仍可运行，输出 SampleRecord 正常。
- Notes: bond_cap 默认 null；Top-K 统计后续在触发器中使用。
## 2026-01-16
- Changes: 在 experiments/sampling 增加 Force/PES 接口层、全局盆地图注册、Recorder 钩子；更新结构草案。
- Files: experiments/sampling/forcefield/{__init__,base}.py, experiments/sampling/graph/{__init__,registry}.py, experiments/sampling/recorders.py, channel.md
- Tests: python3 - <<'PY' (import smoke test)
- Results: 新模块可导入；占位接口已就位。
- Notes: 仅预留扩展点，pipeline 尚未使用。
## 2026-01-16
- Changes: 采样管线接入 recorder hook；实现 StepTrace/BasinRegistry/ALCandidate 记录器与 JSONL 输出；更新 recorders 以支持 numpy 序列化。
- Files: experiments/sampling/pipeline.py, experiments/sampling/recorders.py, channel.md
- Tests: python3 - <<'PY' (recorder + pipeline smoke test)
- Results: 采样管线可写出 trace/basin/al 三类日志文件。
- Notes: Recorder 默认不写结构；BasinRegistryRecorder 保存 x_min 代表结构。
## 2026-01-16
- Changes: 记录“问题清单评审与修复顺序”计划条目。
- Files: channel.md
- Tests: 未运行（仅文档变更）
- Results: 计划已记录。
- Notes: 待输出具体判断与优先级建议。
## 2026-01-16
- Changes: 新增 StructureStore（NPZ 持久化）与 StructureRef；更新三类 Recorder 使用结构引用写入，DFT 只记录 x_pre；新增结构索引 JSONL。
- Files: experiments/sampling/structure_store.py, experiments/sampling/recorders.py, channel.md
- Tests: 未运行（结构变更）
- Results: 结构存储与 Recorder 引用写入已实现。
- Notes: BasinRegistry 仍可回退为内嵌结构（未提供 store 时）。
## 2026-01-16
- Changes: 在 mace_pretrain 增加 DFT 队列近似去重模块（粗桶+RMSD）并提供 trigger 包装器。
- Files: experiments/mace_pretrain/dedup.py, channel.md
- Tests: 未运行（结构变更）
- Results: DFT 去重逻辑可在 trigger 层解耦使用。
- Notes: 默认不做对齐，假设固定基底同系；阈值可配置。
## 2026-01-16
- Changes: 调整 DFT 队列去重 RMSD 默认阈值为 0.08 Å（movable 原子 RMSD）。
- Files: experiments/mace_pretrain/dedup.py
- Tests: 未运行（参数变更）
- Results: 默认阈值已按要求更新。
- Notes: 仍采用粗桶+RMSD 结构级去重。
## 2026-01-16
- Changes: 新增 slab 参考系 canonicalize（cell/PCA 轴 + anchor 平移），并接入 StructureStore 与 DFT 去重流程。
- Files: experiments/sampling/canonicalize.py, experiments/sampling/structure_store.py, experiments/mace_pretrain/dedup.py, channel.md
- Tests: 未运行（结构变更）
- Results: 结构可投影到 slab 参考系用于存储与去重。
- Notes: anchor 优先 fixed，缺失时用底部层几何估计。
## 2026-01-16
- Changes: 取消 StructureStore 与 DFT 去重的 canonicalize 接入，仅保留 canonicalize 模块以备 DFT 提交阶段使用。
- Files: experiments/sampling/structure_store.py, experiments/mace_pretrain/dedup.py, channel.md
- Tests: 未运行（结构变更）
- Results: 结构存储与去重均基于原始坐标；canonicalize 仅保留实现。
- Notes: DFT 提交阶段将单独调用 canonicalize。
## 2026-01-16
- Changes: 为 DFT 队列新增 queue_idx 续扫能力；新增 DFT 出队处理（canonicalize + 去重 + submit/skip 输出）；StructureStore 支持 NPZ 读取。
- Files: experiments/sampling/recorders.py, experiments/sampling/structure_store.py, experiments/mace_pretrain/outbox.py
- Tests: 未运行（功能变更）
- Results: DFT 队列记录与出队处理逻辑已实现。
- Notes: 出队默认读取 dft_queue.jsonl，输出 dft_submit.jsonl / dft_skip.jsonl。
## 2026-01-16
- Changes: 新增 slab 几何分类与 core/movable mask 推断；push_pull/dihedral 支持 core_mask 过滤以避免 core-core 组合。
- Files: experiments/sampling/selection.py, experiments/sampling/actions/push_pull.py, experiments/sampling/actions/dihedral_twist.py
- Tests: 未运行（功能变更）
- Results: 动作采样可通过 core_mask 约束 slab 内部不参与组合。
- Notes: selection_mask 仍需由调用方提供（建议用 infer_slab_masks）。
## 2026-01-16
- Changes: 增加力统计计算（pre/min，topK=3/5）并接入采样管线；Quench 输出最后一步 forces；新增触发器构建函数；改进 slab 法向推断与 push_pull 过滤健壮性。
- Files: experiments/sampling/force_metrics.py, experiments/sampling/schema.py, experiments/sampling/quench/ase_fire.py, experiments/sampling/quench/ase_lbfgs.py, experiments/sampling/pipeline.py, experiments/mace_pretrain/selector.py, experiments/sampling/selection.py, experiments/sampling/canonicalize.py, experiments/sampling/actions/push_pull.py
- Tests: 未运行（功能变更）
- Results: 采样记录包含 force_pre/force_min 统计；触发器可读 metrics；push_pull 不会死循环。
- Notes: force_min 优先复用 quench forces，缺失时可回退 force_fn。
## 2026-01-16
- Changes: 添加默认触发器（max_F>0.7 或 top5>0.35）与 OR 组合器；新增 build_action_inputs 便捷生成 selection_mask/candidates。
- Files: experiments/mace_pretrain/selector.py, experiments/sampling/selection.py
- Tests: 未运行（功能变更）
- Results: 可直接使用默认阈值触发与简化的动作输入构建。
- Notes: 触发阈值后续可按触发率微调。
## 2026-01-16
- Changes: 更新 README.md / parameters.md（补齐 experiments 采样与 DFT 出队说明）；删除误添加的 experiments/sampling/outbox.py。
- Files: README.md, parameters.md, experiments/sampling/outbox.py
- Tests: 未运行（文档与结构变更）
- Results: 文档与目录说明已同步。
- Notes: 采样/出队仍位于 experiments/mace_pretrain。
