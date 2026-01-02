# RL-reaction-path

Unified PES training/eval/inference pipeline with multiple backends (MACE, EquiformerV2, GemNet-OC)
for RL/AL workflows.

## Layout
- `backends/` vendor code (kept unchanged)
- `adapters/` backend adapters (one folder per model)
- `core/` backend-agnostic data, training, eval, ckpt, manifest
- `frontends/` CLI entrypoints (single entry)
- `models/` pretrained checkpoints and model configs
- `runs/` run outputs
- `Data/` datasets (not tracked)

## Entry point
All tasks are launched via the frontend:

```bash
python -m frontends.run --config path/to/run.yaml --run <name>
```

The frontend dispatches to:
- Core trainer for MACE.
- Native FairChem trainer for EquiformerV2/GemNet-OC (Phase 1 bridge), then exports standard artifacts.

## Standard outputs
Each run writes:
- `run_dir/artifacts/best_model.pt`
- `run_dir/artifacts/manifest.json`
- `run_dir/checkpoints/checkpoint.pt`
- `run_dir/logs/`

## Training (MACE core)
- `task: train` or `task: finetune` uses the core trainer.
- `train.input_json` is required to load `model.json`.
- `train.head_key` selects the output head (defaults to `omat_pbe`).
- `train.freeze: head_only` freezes the trunk and trains only head parameters.
- `train.progress_bar` and `train.log_every` control per-epoch visibility.

## Evaluation
- `task: evaluate` supports `eval.energy_only: true` to skip forces.
- `data.indices` or `data.indices_path` can pin a deterministic subset.

## Environment notes
Different backends can use different Python environments. For FairChem-native runs, use
`backend_python` and `backend_env` in the run config, or activate the env before launching.

## Config reference
See `parameters.md` for the unified run.yaml schema and backend-specific fields.
