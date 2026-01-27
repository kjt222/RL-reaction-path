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
## 2026-01-16
- Changes: 新增 experiments 入口脚本（采样与出队）；采样入口要求 force_path，出队支持 RMSD 阈值与关闭 canonicalize。
- Files: experiments/mace_pretrain/run_sampling.py, experiments/mace_pretrain/run_outbox.py, channel.md
- Tests: 未运行（新增入口脚本）
- Results: experiments 入口已就位，便于后续迁移到 frontends。
- Notes: run_sampling.py 中 force_fn 仍需按具体模型实现。
## 2026-01-18
- Changes: 新增 force_fn loader（manifest+weights）；run_sampling 改为使用 manifest/weights/device/head，并增加 resume 输出保护。
- Files: experiments/mace_pretrain/force_fn.py, experiments/mace_pretrain/run_sampling.py, channel.md
- Tests: 未运行（功能变更）
- Results: 采样入口可直接用 manifest 进行推理。
- Notes: 需要确保 manifest.backend 已注册且 weights 可读。
## 2026-01-18
- Changes: SamplingPipeline 支持 force_fn 返回 (E,F)；记录 energy_pre/energy_min；Quench 输出能量；force_fn 返回标量能量。
- Files: experiments/sampling/schema.py, experiments/sampling/quench/ase_fire.py, experiments/sampling/quench/ase_lbfgs.py, experiments/sampling/pipeline.py, experiments/mace_pretrain/force_fn.py, channel.md
- Tests: 未运行（功能变更）
- Results: 力与能量统计可在采样链路中同时记录。
- Notes: embed_energy basin 仍需上层填充 embedding。
## 2026-01-18
- Changes: 新增 Level-1 元素对距离直方图 basin（HistogramBasin），并导出到 basin 包。
- Files: experiments/sampling/basin/histogram.py, experiments/sampling/basin/__init__.py, channel.md
- Tests: 未运行（功能变更）
- Results: 几何硬 ID 可用，置换不变性更稳。
- Notes: 需要在采样入口显式选择该 basin 实现。
## 2026-01-18
- Changes: 采样入口默认使用 HistogramBasin 作为 Level-1 硬 ID。
- Files: experiments/mace_pretrain/run_sampling.py
- Tests: 未运行（参数变更）
- Results: 默认 basin_id 已切换为直方图实现。
## 2026-01-18
- Changes: 增加 CPU quench + GPU 力推理（ForceFnCalculator）；quench 支持中间步触发；未收敛跳过 basin；全局 RMSD 去重默认 0.18；JitterAction seed 可复现；Equiformer 适配修复 ptr 设备。
- Files: experiments/sampling/quench/force_fn_calculator.py, experiments/sampling/quench/ase_fire.py, experiments/sampling/quench/ase_lbfgs.py, experiments/sampling/quench/base.py, experiments/sampling/pipeline.py, experiments/sampling/recorders.py, experiments/sampling/actions/jitter.py, experiments/mace_pretrain/run_sampling.py, experiments/mace_pretrain/force_fn.py, experiments/mace_pretrain/dedup.py, experiments/mace_pretrain/outbox.py, experiments/mace_pretrain/run_outbox.py, adapters/fairchem/adapter_base.py
- Tests: 
  - `PYTHONPATH=... conda run -n equiformerv2 python3 experiments/mace_pretrain/run_sampling.py --run_dir runs/sample loop/sample_loop_eqv2_quenchsteps_20260118_155319 --structure_json /tmp/oc22_sample.json --steps 200 --target_basins 5 --manifest models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200 --amp`
  - `PYTHONPATH=... conda run -n equiformerv2 python3 experiments/mace_pretrain/run_outbox.py --run_dir runs/sample loop/sample_loop_eqv2_quenchsteps_20260118_155319`
- Results: 采样到 5 个 basin；dft_queue 触发 859；去重后 dft_submit 19。
## 2026-01-20
- Changes: 新增 DFT outbox 同步脚本（扫描 READY，scp 推送到 Windows，写上传标记/失败日志），支持环境变量覆盖目标。
- Files: scripts/sync_dft_jobs.py
- Tests: 未运行（尚未配置 Windows SSH / Tailscale 连通性验证）
- Results: 待验证
- Notes: 默认 outbox=`runs/dft_outbox`，标记文件 `READY` / `_uploaded.flag`。
## 2026-01-20
- Changes: 按要求移除 Linux 端 DFT outbox 同步脚本。
- Files: scripts/sync_dft_jobs.py
- Tests: 未运行（删除文件）
- Results: 脚本已删除，scripts/ 目录为空则一并移除。
## 2026-01-20
- Changes: 从 OC22 LMDB 抽取 10 个样本并导出 VASP 用结构包（POSCAR/oc22_forces.npy/oc22_energy.txt/metadata.json）。
- Files: Data/oc22_data/oc22_data/temp/sample_0000..sample_0009 (数据输出)
- Tests: `PYTHONPATH=... python3 /tmp/extract_oc22_vasp.py`
- Results: 已生成 10 个样本目录。
- Notes: POSCAR 按元素分组并重排 forces；需自行补 INCAR/KPOINTS/POTCAR。
## 2026-01-20
- Changes: 基于 OC22_dataset + pymatgen 2020.4.2 生成 OC22 口径 VASP 输入（INCAR/KPOINTS/POTCAR.symbols），为 10 个样本补齐输入。
- Files: Data/oc22_data/oc22_data/temp/sample_0000..sample_0009/INCAR, KPOINTS, POTCAR.symbols; /tmp/oc22_make_vasp_inputs.py
- Tests: `python3 /tmp/oc22_make_vasp_inputs.py`
- Results: 已为 10 个样本写入 INCAR/KPOINTS/POTCAR.symbols。
- Notes: POTCAR 实体需按 symbols 顺序手动拼接；INCAR 已包含 DIPOL（按结构质心自动计算）。
## 2026-01-20
- Changes: 新增可视化记录与离线渲染；采样端输出 viz_steps + viz_structures；渲染脚本支持 extxyz 与 MP4（叠加 basin_id/is_new/DFT 触发原因）。
- Files: experiments/sampling/recorders.py, experiments/mace_pretrain/run_sampling.py, experiments/visualization/render_movie.py, channel.md
- Tests: `python3 experiments/visualization/render_movie.py --run_dir runs/viz_test_tmp --fps 2 --max_frames 3`
- Results: 生成 extxyz 与 MP4 成功（随后清理测试目录）。
- Notes: MP4 依赖 ffmpeg；已在脚本内做偶数分辨率缩放。
## 2026-01-21
- Changes: 修复 MACE 适配中 ptr device 不一致导致的 CUDA 报错。
- Files: adapters/mace/adapter.py
- Tests: 采样运行（见下）
- Results: GPU 采样可正常进入 quench + basin。
- Notes: 将 `build_ptr(batch)` 结果显式移动到 `device`。
## 2026-01-21
- Changes: 使用 OC22 LMDB 样本进行采样并生成可视化输出（MACE 与 GemNet 各一组，fmax=0.2）。
- Files: /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111/*; /media/kjt/kjt-ssd/RL-runs/oc22_sample_gemnet_20260121_105136/*
- Tests:
  - `conda run -n mace ... run_sampling.py --target_basins 50 --quench_fmax 0.2`
  - `conda run -n fairchemv2 ... run_sampling.py --target_basins 50 --quench_fmax 0.2`
  - `python3 experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111 --fps 12`
  - `python3 experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_gemnet_20260121_105136 --fps 12`
- Results:
  - MACE: steps=86, basins=50, quench_converged=50, quench_unconverged=36, dft_candidates=11817, elapsed≈545s.
  - GemNet: steps=68, basins=50, quench_converged=50, quench_unconverged=18, dft_candidates=9701, elapsed≈487s.
  - 两个 run 均生成 `viz/trajectory.extxyz` 与 `viz/movie.mp4`。
## 2026-01-21
- Changes: render_movie.py 改为使用 OVITO 渲染高质量帧，保留 extxyz 输出与文字叠加。
- Files: experiments/visualization/render_movie.py
- Tests: `python3 experiments/visualization/render_movie.py --run_dir /tmp/ovito_viz_test --fps 12`
- Results: 失败（当前环境缺少 OVITO Python API：ModuleNotFoundError: ovito）；/tmp/ovito_viz_test 已清理。
- Notes: 需在可用的 conda 环境中安装/启用 ovito 后再重试。
## 2026-01-21
- Changes: render_movie.py 新增 --renderer 参数（tachyon/opengl）并在 OVITO 叠加文字渲染时按指定渲染器输出。
- Files: experiments/visualization/render_movie.py
- Tests:
  - `/home/kjt/miniforge3/envs/visualization/bin/python experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111 --fps 12 --max_frames 300 --renderer opengl`
  - `QT_QPA_PLATFORM=offscreen LIBGL_ALWAYS_SOFTWARE=1 /home/kjt/miniforge3/envs/visualization/bin/python experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111 --fps 12 --max_frames 50 --renderer opengl`
- Results: OpenGL 渲染失败（无可用 OpenGL 上下文 / 缺少图形驱动），需在具备 OpenGL 的图形会话中运行或改用 tachyon + 降帧。
- Notes: OpenGL 仅在可用图形/GL 环境下生效；当前环境建议使用 --stride/--max_frames 控制耗时。
## 2026-01-21
- Tests: `DISPLAY=:0 /home/kjt/miniforge3/envs/visualization/bin/python experiments/visualization/render_movie.py --run_dir /tmp/ovito_opengl_test --fps 12 --renderer opengl`
- Results: OpenGL 渲染在小样本下仍触发 Signal(6) 崩溃；已清理 /tmp/ovito_opengl_test。
## 2026-01-21
- Changes: OVITO 渲染优化（多行叠加、小字+紧行距、正交视角、tag 颜色区分、action/min 帧短暂停留、仅 quench 降帧）。
- Files: experiments/visualization/render_movie.py, channel.md
- Tests: `/home/kjt/miniforge3/envs/visualization/bin/python experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111 --fps 12 --stride 10 --width 1280 --height 720 --renderer tachyon`
- Results: 失败（/media 下无写权限，无法写入 trajectory.extxyz/trajectory_render.extxyz）。
## 2026-01-21
- Changes: 渲染改为元素着色（ColorByType），减少原子半径并加入元素颜色图例；移除 tag 着色逻辑。
- Files: experiments/visualization/render_movie.py
- Tests: not run (pending user re-render in GUI session with write access)
- Results: not run
## 2026-01-21
- Changes: render_movie.py 渲染循环加入进度输出（--log_every）。
- Files: experiments/visualization/render_movie.py, channel.md
- Tests: not run (waiting for user run)
- Results: not run
## 2026-01-21
- Tests: `QT_QPA_PLATFORM=xcb /home/kjt/miniforge3/envs/visualization/bin/python experiments/visualization/render_movie.py --run_dir /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111 --fps 12 --stride 1 --width 1280 --height 720 --renderer opengl --log_every 200`
- Results: 成功生成 `viz/movie.mp4`（14649 帧，约 20:20 时长）。
## 2026-01-21
- Changes: 叠加文字下移并缩小字号；新增 phase banner（颜色区分）；渲染帧结构传递 stage 信息。
- Files: experiments/visualization/render_movie.py, channel.md
- Tests: not run (waiting for user re-render)
- Results: not run
## 2026-01-21
- Changes: 视角改为斜向正交；输出分段轨迹（trajectory_action.extxyz / trajectory_quench.extxyz）；支持仅导出 extxyz（--skip_movie）。
- Files: experiments/visualization/render_movie.py, channel.md
- Tests: not run (waiting for user run)
- Results: not run
## 2026-01-21
- Changes: 从 OC22 LMDB 随机抽样筛选小体系，导出 5 个候选构型 extxyz + summary.csv。
- Files: tmp/oc22_candidates/*
- Tests: `/home/kjt/miniforge3/envs/mace/bin/python` 脚本读取 LMDB 并导出候选。
- Results: 生成 5 个小体系（23–25 原子）候选结构。
## 2026-01-21
- Changes: 从 OC22 LMDB 筛选 slab+adsorbate 构型（nads=1），导出到 temp 目录供采样。
- Files: Data/oc22_data/oc22_data/temp/oc22_slab_idx5640310_n120.extxyz
- Tests: `/home/kjt/miniforge3/envs/mace/bin/python` 随机抽样筛选（nads<=3、tags含0/1、vacuum较大）。
- Results: 选中 idx=5640310，natoms=120，vacuum≈28.565 Å。
## 2026-01-21
- Changes: 进一步筛选更直观的 slab+adsorbate 候选（nads=1、元素更少），导出 3 个备选。
- Files: Data/oc22_data/oc22_data/temp/oc22_slab_idx5782456_n98_e2.extxyz; Data/oc22_data/oc22_data/temp/oc22_slab_idx2985847_n98_e2.extxyz; Data/oc22_data/oc22_data/temp/oc22_slab_idx336801_n97_e2.extxyz; Data/oc22_data/oc22_data/temp/summary_slab_candidates.csv
- Tests: `/home/kjt/miniforge3/envs/mace/bin/python` 随机筛选（nads=1、vacuum>=10、natoms 60–140、元素数<=2）。
- Results: 生成 3 个二元体系候选（98/98/97 原子，vacuum≈18–19 Å）。
## 2026-01-21
- Changes: 选定更直观构型并按“adsorbate+slab”命名规范重命名。
- Files: Data/oc22_data/oc22_data/temp/oc22_slab_CsO_ads_CsO_slab.extxyz
- Tests: not run
- Results: 原文件 oc22_slab_idx5782456_n98_e2.extxyz 已重命名。

## 2026-01-22
- Changes: Exported 10 OC22 LMDB samples to VASP-ready folders (POSCAR/INCAR/KPOINTS/POTCAR.symbols + ref energy/forces).
- Files: Data/oc22_data/oc22_data/temp/vasp_10 (sample_*_idx*/{POSCAR,INCAR,KPOINTS,POTCAR.symbols,metadata.json,oc22_ref_*}, manifest.json)
- Tests: not run (data export only)
- Results: Export completed; note torch CUDA warning from host env when reading LMDB.

## 2026-01-22
- Changes: Added OC22 VASP reproduction notes with parameter rationale and clarified sample sets; linked from README.
- Files: docs/vasp_oc22_repro.md, README.md
- Tests: not run (docs only)
- Results: N/A

## 2026-01-22
- Changes: Fixed INCAR formatting for temp/sample_0000~0009 (LDAU arrays per POTCAR order; DIPOL commas removed).
- Files: Data/oc22_data/oc22_data/temp/sample_0000~sample_0009/INCAR
- Tests: not run (file rewrite only)
- Results: Updated INCAR now uses VASP-compatible LDAU lists and DIPOL spacing.

## 2026-01-22
- Changes: Rewrote LDAU lists in temp/sample_0000~0009 using whitespace-split POTCAR.symbols order (handles single-line POTCAR.symbols); forced LDAU=.TRUE. and per-element arrays.
- Files: Data/oc22_data/oc22_data/temp/sample_0000~sample_0009/INCAR
- Tests: not run (file rewrite only)
- Results: LDAUL/LDAUU/LDAUJ now match POTCAR.symbols length; DIPOL spacing preserved.

## 2026-01-23
- Changes: Set default render progress logging interval to 20 frames.
- Files: experiments/visualization/render_movie.py
- Tests: not run (arg default change only)
- Results: N/A

## 2026-01-23
- Changes: Switched render progress logging to a single-line progress bar.
- Files: experiments/visualization/render_movie.py
- Tests: not run (log formatting change only)
- Results: N/A

## 2026-01-23
- Changes: disable Cycles denoising in Blender render script to avoid OIDN runtime error.
- Files: experiments/visualization/blender_render.py
- Tests: blender --background --python experiments/visualization/blender_render.py -- --extxyz /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111/viz/trajectory_quench.extxyz --out_dir /home/kjt/projects/RL-reaction-path/tmp/viz_blender_quench --width 1280 --height 720 --samples 64 --stride 1 --fps 12 --write_mp4
- Results: partial render started; timed out after 120s (frames were being generated ~5s/frame). Full quench trajectory has 14047 frames, so full render at current settings would take ~20h.

## 2026-01-23
- Changes: added simple bond rendering in Blender (covalent radii + scale factor) and bond material; bonds rebuilt per frame.
- Files: experiments/visualization/blender_render.py
- Tests: /usr/bin/blender --background --python experiments/visualization/blender_render.py -- --extxyz /media/kjt/kjt-ssd/RL-runs/oc22_sample_mace_20260121_104111/viz/trajectory_quench.extxyz --out_dir /home/kjt/projects/RL-reaction-path/tmp/viz_blender_quench --width 1280 --height 720 --samples 32 --stride 5 --fps 12 --write_mp4
- Results: command ran ~120s before tool timeout; generated frames up to frame_00025 in tmp/viz_blender_quench/frames. Full render estimated to take hours; needs to be run interactively.
## 2026-01-23
- Changes: updated Blender renderer to add element color map (Cs/O/etc), stronger cell material (emission/radius), wrap atoms in cell for rendering, tweak stamp style and ortho scale.
- Files: experiments/visualization/blender_render.py
- Tests: 
  - /usr/bin/blender --background --python experiments/visualization/blender_render.py -- --extxyz /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/oc22_slab_CsO_ads_CsO_slab.extxyz --out_dir /home/kjt/projects/RL-reaction-path/tmp/viz_blender_preview --width 960 --height 540 --samples 4 --stride 1 --max_frames 1 --radius 0.6
- Results: preview frame generated at /home/kjt/projects/RL-reaction-path/tmp/viz_blender_preview/frames/frame_00000.png; cell edges still not visible in output.
- Notes: may need explicit cell line scaling/visibility or repositioning for slab cells.
## 2026-01-23
- Changes: further increased cell edge visibility (radius/emission), reduced stamp font size, wrap first-frame atoms into cell before rendering.
- Files: experiments/visualization/blender_render.py
- Tests:
  - /usr/bin/blender --background --python experiments/visualization/blender_render.py -- --extxyz /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/oc22_slab_CsO_ads_CsO_slab.extxyz --out_dir /home/kjt/projects/RL-reaction-path/tmp/viz_blender_preview --width 960 --height 540 --samples 4 --stride 1 --max_frames 1 --radius 0.6
- Results: preview regenerated at /home/kjt/projects/RL-reaction-path/tmp/viz_blender_preview/frames/frame_00000.png (cell edges still not visible).

## 2026-01-23
- Changes: parse extxyz Properties (tags) and apply slab down-weighting (alpha+scale); add VASP-style XYZ triad; propagate tags through bond/wrap/render paths.
- Files: experiments/visualization/blender_render.py
- Tests: not run (awaiting user-selected preview run)
- Results: N/A

## 2026-01-23
- Changes: added X/Y/Z letter labels to triad so directions are explicit.
- Files: experiments/visualization/blender_render.py
- Tests: not run (awaiting user preview)
- Results: N/A

## 2026-01-24
- Changes: removed sampling validators module, added reusable sampling validate helpers with shorter names; added action-quality diagnostics (summary + CLI report) for steps.jsonl; updated exports.
- Files: experiments/sampling/validators.py (deleted); experiments/sampling/validate.py; experiments/sampling/__init__.py; experiments/diagnostics/action_quality.py; experiments/diagnostics/report_action_quality.py; experiments/diagnostics/__init__.py.
- Tests: not run (waiting for user request).
- Results: n/a.
- Notes: logging updated per request.

## 2026-01-24
- Changes: added ASE CG and BFGS quench implementations; exposed quench choices (fire/cg/bfgs/lbfgs) in MACE sampling CLI; updated quench exports.
- Files: experiments/sampling/quench/ase_cg.py; experiments/sampling/quench/ase_bfgs.py; experiments/sampling/quench/__init__.py; experiments/mace_pretrain/run_sampling.py.
- Tests: not run (not requested).
- Results: n/a.

## 2026-01-24
- Changes: created temp input /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json for sampling test (no repo files changed).
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/equiformerv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200`
- Results:
  - equiformerv2 env failed: torch ImportError `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`.
  - fairchemv2 env failed: CUDA init error (cudaGetDeviceCount error 304); `nvidia-smi -L` failed (`Failed to initialize NVML: Unknown Error`).
- Notes: GPU not accessible in this session; sampling test not completed.

## 2026-01-24
- Changes: ran quench comparison on sample_0000 with EquiformerV2 (fairchemv2 env) at fmax=0.1 (steps=1) for FIRE/BFGS/LBFGS; CG failed due to missing ASE optimizer.
- Files: /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json (temp); /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire; /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_bfgs; /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs.
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 1 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_bfgs --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 1 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench bfgs --quench_fmax 0.1 --quench_steps 200`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 1 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 200`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_cg --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 1 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench cg --quench_fmax 0.1 --quench_steps 200`
- Results:
  - fire: steps=1 basins=0 quench_converged=0 quench_unconverged=1 dft_candidates=202 elapsed=37.6s.
  - bfgs: steps=1 basins=0 quench_converged=0 quench_unconverged=1 dft_candidates=179 elapsed=41.4s.
  - lbfgs: steps=1 basins=0 quench_converged=0 quench_unconverged=1 dft_candidates=202 elapsed=37.7s.
  - cg: failed (ImportError: cannot import name 'CG' from ase.optimize; ASECGQuench reports ASE required).
- Notes: temp outputs removed after review.

## 2026-01-25
- Changes: attempted parallel EquiformerV2 sampling runs to reach 5 basins (fire + lbfgs) with extended steps.
- Files: /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire5/run.log; /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs5/run.log.
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire5 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 200 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs5 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 200 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 200`
- Results: both parallel runs failed with CUDA OOM (each process ~10–12GiB; not enough for parallel on 24GiB). No basins reached.
- Notes: parallel EquiformerV2 sampling not feasible on single 3090 without reducing memory or running serially.

## 2026-01-25
- Changes: launched parallel EquiformerV2 sampling runs with AMP enabled (fire + lbfgs) targeting 5 basins.
- Files: /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire5_amp2/run.log; /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs5_amp2/run.log.
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_fire5_amp2 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 200 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200 --amp`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/tmp/sample_0000_eqv2_lbfgs5_amp2 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 200 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 200 --amp`
- Results: running (AMP enabled; both processes active; each ~7.9GiB VRAM). Will update when finished.

## 2026-01-25
- Changes: temporarily removed VizRecorder from sampling pipeline (no per-step viz capture); removed viz output cleanup checks.
- Files: /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py.
- Tests: not run (requested to stop running jobs and restart).
- Results: n/a.
- Notes: attempted to delete tmp run dirs but blocked by tool policy; user should remove manually.

## 2026-01-25
- Changes: relaunched EquiformerV2 sampling (AMP) with steps=500 and target_basins=5 for FIRE and L-BFGS, no per-step viz.
- Files: /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire5_amp_steps500/run.log; /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs5_amp_steps500/run.log; /home/kjt/projects/RL-reaction-path/tmp/amp_parallel_pids.txt.
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire5_amp_steps500 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 500 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200 --amp`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs5_amp_steps500 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 500 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 200 --amp`
- Results: running (not monitored per request).

## 2026-01-25
- Results (sample_0000 eqv2 AMP runs): both runs exited early without "Sampling finished".
  - fire: steps.jsonl lines=50, basins=0, dft_queue=9541; last write 02:13, run.log only startup warnings.
  - lbfgs: steps.jsonl lines=50, basins=0, dft_queue=10220; last write 02:13, run.log only startup warnings.
- Notes: likely abnormal termination (no summary line); wall time ~34 min from run.log ctime (01:39) to steps.jsonl mtime (02:13).

## 2026-01-25
- Changes: removed previous incomplete eqv2 runs (power loss), relaunched parallel AMP runs with same settings.
- Files: /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire5_amp_steps500/run.log; /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs5_amp_steps500/run.log; /home/kjt/projects/RL-reaction-path/tmp/amp_parallel_pids.txt.
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire5_amp_steps500 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 500 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 200 --amp`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs5_amp_steps500 --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --steps 500 --target_basins 5 --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 200 --amp`
- Results: running (not monitored per request).

## 2026-01-25
- Results: previous restart attempt failed immediately with CUDA error 304 (cudaGetDeviceCount). Cleared outputs and relaunched with CUDA_VISIBLE_DEVICES=0; both processes now running.

## 2026-01-25
- Tests: action quality report on running AMP runs.
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/diagnostics/report_action_quality.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire5_amp_steps500`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/diagnostics/report_action_quality.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs5_amp_steps500`
- Results: steps~31/32, valid_rate=1.0, quench_converged_rate=0.0, new_basin_rate=0.0, trigger_rate_max/topk=1.0.

## 2026-01-25
- Changes: made sampling pipeline components configurable (recorders/validators/triggers/stoppers); added stage-filtered DFT recorder; added stopper framework and pipeline config helpers.
- Files: experiments/sampling/recorders.py, experiments/sampling/plugins.py, experiments/sampling/stoppers.py, experiments/mace_pretrain/run_sampling.py
- Tests: not run (config-only refactor)
- Results: n/a
- Changes: added quench_gate (configurable) and stopper support; updated default sampling_rules to disable DFT recorder and enable quench_gate thresholds.
- Files: experiments/sampling/pipeline.py, experiments/sampling/plugins.py, experiments/sampling/stoppers.py, experiments/mace_pretrain/run_sampling.py, experiments/sampling_rules/config.yaml
- Tests: not run (config/flow change)
- Results: n/a

## 2026-01-25
- Changes: Updated OC22 temp samples (sample_0000~0009) to **single-point** VASP settings by setting IBRION=-1 and NSW=0.
- Files: Data/oc22_data/oc22_data/temp/sample_0000/INCAR, sample_0001/INCAR, sample_0002/INCAR, sample_0003/INCAR, sample_0004/INCAR, sample_0005/INCAR, sample_0006/INCAR, sample_0007/INCAR, sample_0008/INCAR, sample_0009/INCAR
- Tests: not run (input generation change only)
- Results: INCAR now no longer requests ionic relaxation; other OC22 parameters preserved.
- Notes: VASP still outputs forces in OUTCAR for single-point runs; no extra flag required.

## 2026-01-25
- Changes: renamed action quality module path and integrated quality gate into validate; removed quench_gate naming and config key; added extra action quality metrics.
- Files: experiments/action_quality/action_quality.py, experiments/action_quality/validate.py, experiments/action_quality/report_action_quality.py, experiments/action_quality/__init__.py, experiments/sampling/__init__.py, experiments/sampling/pipeline.py, experiments/sampling/plugins.py, experiments/mace_pretrain/run_sampling.py, experiments/sampling_rules/config.yaml, channel.md
- Tests: not run (refactor/config change only)
- Results: n/a

## 2026-01-25
- Changes: removed quality_gate from sampling pipeline/config and dropped related reporting.
- Files: experiments/sampling/pipeline.py, experiments/sampling/plugins.py, experiments/mace_pretrain/run_sampling.py, experiments/sampling_rules/config.yaml, experiments/action_quality/action_quality.py, experiments/action_quality/report_action_quality.py, experiments/action_quality/validate.py, channel.md
- Tests: not run (refactor/config change only)
- Results: n/a

## 2026-01-25
- Changes: sampling gate now resamples on rejection (invalid/gate-fail attempts emit records and retry up to max_attempts); run_one no longer double-emits.
- Files: /home/kjt/projects/RL-reaction-path/experiments/sampling/pipeline.py
- Tests:
  - `nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --input /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/sample_0000/POSCAR --model /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle --backend equiformer_v2 --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire_fmax0p1_resample_20260125c --sample_config /home/kjt/projects/RL-reaction-path/experiments/sampling_rules/config.yaml --quench fire --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --max_steps 5000 --amp`
  - `nohup /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --input /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/sample_0000/POSCAR --model /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle --backend equiformer_v2 --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs_fmax0p1_resample_20260125c --sample_config /home/kjt/projects/RL-reaction-path/experiments/sampling_rules/config.yaml --quench lbfgs --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --max_steps 5000 --amp`
- Results: running (logs in /home/kjt/projects/RL-reaction-path/tmp/sample_logs_20260125c).

## 2026-01-25
- Results: initial resample runs failed due to missing PYTHONPATH (ModuleNotFoundError: experiments). Relaunched with PYTHONPATH set.
- Tests:
  - `nohup env PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --input /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/sample_0000/POSCAR --model /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle --backend equiformer_v2 --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire_fmax0p1_resample_20260125c --sample_config /home/kjt/projects/RL-reaction-path/experiments/sampling_rules/config.yaml --quench fire --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --max_steps 5000 --amp`
  - `nohup env PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --input /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp/sample_0000/POSCAR --model /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle --backend equiformer_v2 --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs_fmax0p1_resample_20260125c --sample_config /home/kjt/projects/RL-reaction-path/experiments/sampling_rules/config.yaml --quench lbfgs --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --max_steps 5000 --amp`
- Results: running (logs in /home/kjt/projects/RL-reaction-path/tmp/sample_logs_20260125c).

## 2026-01-25
- Changes: generated structure_json from sample_0000 POSCAR for run_sampling.
- Files: /home/kjt/projects/RL-reaction-path/tmp/poscar_to_json.py, /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json
- Tests:
  - `nohup env PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire_fmax0p1_resample_20260125d --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench fire --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --steps 5000 --amp`
  - `nohup env PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py --run_dir /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs_fmax0p1_resample_20260125d --structure_json /home/kjt/projects/RL-reaction-path/tmp/sample_0000.json --manifest /home/kjt/projects/RL-reaction-path/models/equiformer_v2_oc22/manifest_bundle/manifest.json --device cuda --quench lbfgs --quench_fmax 0.1 --quench_steps 5000 --target_basins 1 --steps 5000 --amp`
- Results: running (logs in /home/kjt/projects/RL-reaction-path/tmp/sample_logs_20260125d).

## 2026-01-25
- Results (resample gate, fmax=0.1, EquiformerV2):
  - FIRE: `steps=240`, `basins=1`, `elapsed=474.9s` (stop_reason=target_basins).
  - LBFGS: `steps=240`, `basins=1`, `elapsed=425.4s` (stop_reason=target_basins).
  - Attempt-level stats from steps.jsonl (both runs): total attempts=1196, valid=1, rejected=1195; quality_gate rejects=1134 (~94.8% of attempts), other rejects=61.
- Files:
  - /home/kjt/projects/RL-reaction-path/tmp/sample_logs_20260125d/fire_0p1_resample.log
  - /home/kjt/projects/RL-reaction-path/tmp/sample_logs_20260125d/lbfgs_0p1_resample.log
  - /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_fire_fmax0p1_resample_20260125d/steps.jsonl
  - /home/kjt/projects/RL-reaction-path/runs/sample_loop/sample_0000_eqv2_lbfgs_fmax0p1_resample_20260125d/steps.jsonl

## 2026-01-26
- Changes: added attempts_total / attempts_rejected summary counts in run_sampling output (from steps.jsonl).
- Files: /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py
- Tests: not run (summary-only change)
- Results: N/A

## 2026-01-26
- Changes: 无代码改动；完成 DFT vs OC22 结果对比
- Files: (analysis) /tmp/dft_vs_oc22_compare.py
- Tests: /home/kjt/miniforge3/envs/mace/bin/python /tmp/dft_vs_oc22_compare.py
- Results: 10 个样本中 5 个能量/力几乎一致；5 个样本能量偏差 3–16 eV 且力误差显著（详见对话回复）
- Notes: 元素数>样本数，E0 拟合欠定，仅供参考

## 2026-01-27
- Changes: 复制 temp/sample_0000..0009 到 temp2，并在 INCAR 中新增 MAGMOM（按元素顺序；U 元素=5.0，其它=0.0）。
- Files:
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0000/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0001/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0002/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0003/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0004/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0005/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0006/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0007/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0008/INCAR
  - /home/kjt/projects/RL-reaction-path/Data/oc22_data/oc22_data/temp2/sample_0009/INCAR
- Tests: not run (数据准备)
- Results: temp2 生成完成，原 temp 未改动

## 2026-01-27
- Changes: 引入 MD 动作与“动作后噪声”插件链（在 action 之后、validators/force/quench 之前）；采样入口接入 action_plugins 与 MDAction（仅在 md.enabled=true 且 ASE 可用时启用）。
- Files:
  - /home/kjt/projects/RL-reaction-path/experiments/action_quality/noise.py
  - /home/kjt/projects/RL-reaction-path/experiments/sampling/actions/md.py
  - /home/kjt/projects/RL-reaction-path/experiments/sampling/actions/__init__.py
  - /home/kjt/projects/RL-reaction-path/experiments/sampling/pipeline.py
  - /home/kjt/projects/RL-reaction-path/experiments/sampling/plugins.py
  - /home/kjt/projects/RL-reaction-path/experiments/mace_pretrain/run_sampling.py
  - /home/kjt/projects/RL-reaction-path/experiments/action_quality/validate.py
  - /home/kjt/projects/RL-reaction-path/experiments/action_quality/action_quality.py
  - /home/kjt/projects/RL-reaction-path/experiments/sampling_rules/config.yaml
- Tests:
  - `/home/kjt/miniforge3/envs/fairchemv2/bin/python -m py_compile experiments/sampling/pipeline.py experiments/action_quality/noise.py experiments/sampling/actions/md.py experiments/sampling/plugins.py experiments/mace_pretrain/run_sampling.py experiments/action_quality/validate.py experiments/action_quality/action_quality.py`
  - `/home/kjt/miniforge3/envs/fairchemv2/bin/python - <<'PY' ... MDAction + DummyCalc smoke ... PY`
  - `/home/kjt/miniforge3/envs/fairchemv2/bin/python - <<'PY' ... SamplingPipeline + noise plugin smoke ... PY`
- Results: py_compile 通过；MDAction 与噪声插件在最小结构上可运行，record.flags 中可见 `noise_sigma`。

## 2026-01-27
- Changes: 统一文档与日志，修正文档中的旧参数键与旧路径，并明确 OC22 对标语义（单点而非结构优化）。
- Files:
  - /home/kjt/projects/RL-reaction-path/channel.md
  - /home/kjt/projects/RL-reaction-path/README.md
  - /home/kjt/projects/RL-reaction-path/parameters.md
  - /home/kjt/projects/RL-reaction-path/docs/vasp_oc22_repro.md
  - /home/kjt/projects/RL-reaction-path/daily_log.md
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python -m py_compile experiments/sampling/pipeline.py experiments/sampling/actions/md.py experiments/sampling/plugins.py experiments/action_quality/validate.py experiments/action_quality/noise.py experiments/mace_pretrain/run_sampling.py`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --help | sed -n '1,40p'`
- Results: 语法检查通过；CLI 帮助输出与文档示例一致（使用 `--config`）。

## 2026-01-27
- Changes: 提交前复验（显式暂存，避开 `runs/` 与数据目录）。
- Tests:
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python -m py_compile adapters/mace/adapter.py experiments/mace_pretrain/run_sampling.py experiments/sampling/pipeline.py experiments/sampling/plugins.py experiments/sampling/actions/md.py experiments/action_quality/validate.py experiments/action_quality/noise.py experiments/action_quality/action_quality.py experiments/action_quality/report_action_quality.py experiments/sampling/quench/ase_cg.py experiments/sampling/quench/ase_bfgs.py experiments/sampling/stoppers.py experiments/sampling/recorders.py experiments/visualization/render_movie.py experiments/visualization/blender_render.py`
  - `PYTHONPATH=/home/kjt/projects/RL-reaction-path /home/kjt/miniforge3/envs/fairchemv2/bin/python experiments/mace_pretrain/run_sampling.py --help | sed -n '1,80p'`
- Results: 通过。
