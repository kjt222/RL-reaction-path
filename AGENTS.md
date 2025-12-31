# AGENTS

## Project goal and scope
- Provide a unified PES training/eval/inference pipeline across multiple backends (MACE, EquiformerV2, GemNet-OC) for RL/AL reaction-path workflows.
- Keep backend-specific logic isolated in adapters while core pipelines remain backend-agnostic.
- Standardize artifacts (manifest + weights + checkpoints) for portability and reproducibility.

## Repository layout (core vs vendor)
- Core code (safe to edit): `frontends/`, `core/`, `adapters/`, `docs/`, `parameters.md`, `README.md`.
- Vendor/backends (do not edit unless explicitly requested): `backends/` (mace/equiformer_v2/escaip sources).
- Data and outputs (do not modify/commit): `Data/`, `runs/`, `models/`, `__pycache__/`.

## Running (common commands)
- Train/finetune/resume/export/eval: `python -m frontends.run --config path/to/run.yaml --run <name>`.
- Example run config: see `runs/finetune_acc8_test.yaml` and schema in `parameters.md`.
- Evaluate vs infer: `task: evaluate` requires forces; use `task: infer` for energy-only.
- Export standard artifacts: `task: export` with `model_in` and `run_dir`.

## GPU and environment
- GPU (CUDA) recommended for training; CPU is acceptable for small eval/infer.
- MACE environment: `mace_env.yml`.
- FairChem (EquiformerV2/GemNet-OC) environment: `equiformerv2.yml` / `fairchemv2.yml`.
- For native FairChem runs, set `backend_python`/`backend_env` in run.yaml or activate the env before running.

## Code conventions
- Naming: `snake_case` for functions/vars, `CamelCase` for classes, `UPPER_SNAKE` for constants.
- Logging: use `logging` in core/adapters (`LOGGER = logging.getLogger(__name__)`); `print` is only for CLI/entrypoints.
- Errors: fail fast with clear `ValueError`/`KeyError` messages; avoid silent fallbacks.
- Commits: short imperative subject, optional scope prefix (e.g., `core: fix manifest io`).
- Tests: not mandatory by default, but required when touching interface/format logic; otherwise include smoke commands.

## Key constraints (interfaces/protocols)
- Manifest schema is defined in `core/manifest.py`; keep fields backward compatible.
- Checkpoint format is defined in `core/ckpt/save_load.py`; keep keys compatible.
- Canonical batch contract in `core/contracts.py` and `core/data/io_lmdb.py` must remain stable.
- Artifact layout is defined in `core/runner/layout.py` and `core/ckpt/export.py`.
- Unified run.yaml schema is defined in `parameters.md` and `core/runner/spec.py`.

## Workflow
- Use `rg` to locate the entry point before opening files; do not scan the full repo by default.
- Make small, isolated changes; re-run a focused smoke test for the touched path.
- Provide a clear diff summary and note any validation gaps.

## Safety and privacy
- Do not print or log secrets/keys; scrub env vars in examples.
- Do not commit large files, datasets, or model binaries; keep them in `Data/` or `models/`.
- Avoid modifying vendor code under `backends/` unless explicitly asked.

## Hard rules
- Training/eval artifacts must be symmetric and standard: `run_dir/artifacts/best_model.pt`, `run_dir/artifacts/manifest.json`, `run_dir/checkpoints/checkpoint.pt`.
- Only adapters handle backend differences; CLI parameters and core contracts stay unified.
- Default to `rg` + targeted file reads; no full-repo scans.
- 以后回答一律使用中文。
