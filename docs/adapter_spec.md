# Adapter and Manifest Specification

## Scope
This spec defines a minimal, shared contract so multiple backends (MACE and EquiformerV2) can produce
consistent metrics and embeddings while keeping training logic backend-specific.

## Goals
- Keep a single core training/finetune/resume loop; adapters only supply model/batch/loss details.
- Unify error metrics (energy/forces) across backends.
- Standardize model export as state_dict + manifest.
- Standardize run_dir outputs (artifacts/checkpoints/logs).
- Allow importing legacy .model/.pt (nn.Module) as a source.

## Non-goals
- No new training features or CLI flags.
- No rewriting vendor training internals unless explicitly requested.
- No changes to dataset formats or preprocessing pipelines.

## Directory Boundaries
- Vendor code stays under `backends/*/` and is not modified.
- Project code lives outside vendor trees (e.g., `core/`, `adapters/`, `frontends/`).
- Frontend modules do not shadow backend package names (avoid `mace.py`, `equiformer.py` at top level).

## Canonical Data Contract
`CanonicalBatch` is the shared data contract for adapters and metrics.

Required keys:
- `z`: int64, shape [N]. Atomic numbers for all atoms in the batch.
- `pos`: float32, shape [N, 3]. Positions in Angstrom.
- `energy`: float32, shape [B]. Total energy per configuration in eV.
- `forces`: float32, shape [N, 3]. Forces in eV/Angstrom.
- `ptr`: int64, shape [B + 1]. Cumulative atom counts per configuration.

Optional keys:
- `cell`: float32, shape [B, 3, 3] or [3, 3]. Unit cell in Angstrom.
- `pbc`: bool, shape [B, 3] or [3]. Periodic boundary flags.
- `natoms`: int64, shape [B]. Alternative to `ptr` if `ptr` is absent.

## Output Contract
`ModelOutputs` must include:
- `energy`: float32, shape [B]. Total energy per configuration (eV).
- `forces`: float32, shape [N, 3]. Forces (eV/Angstrom).
- `node_embed`: float32, shape [N, D]. Per-atom latent.
- `graph_embed`: float32, shape [B, D]. Per-config latent.

Embedding rules:
- MACE: `node_embed` is the output of the last interaction block (post-product, pre-readout).
- EquiformerV2: `node_embed` is the output of the last transformer block (post-norm, pre-energy head).
- GemNet-OC: `node_embed` uses the output of `out_mlp_E` (per-atom hidden before energy head).
- `graph_embed` uses mean pooling per configuration:
  g_b = (1 / N_b) * sum_{i in b} h_i

If a backend does not expose a stable node embedding, the adapter should return a
fallback (e.g., zeros) and set `embedding.node_embed_layer` to null in the manifest.

Backend-specific extraction notes:
- MACE forward returns a dict with keys `energy`, `forces`, `node_energy`, `node_feats`, etc.
  `node_feats` is the concatenation of per-interaction outputs after the product blocks.
  The adapter should extract the last interaction output (preferred: forward hook on
  `model.products[-1]`, or slice `node_feats` using irreps dims).
- EquiformerV2 forward returns `(energy, forces)` when `regress_forces=True`.
  Per-node representation is `x.embedding` after the last `TransBlockV2` and `self.norm`,
  where `x` is an `SO3_Embedding` with shape `[N, num_coefficients, sphere_channels]`.
  The adapter should flatten to `[N, D]` (D = num_coefficients * sphere_channels) to
  satisfy `node_embed` shape.

## Metrics Contract
All backends must report the same metrics with the same formulas:

Energy (per-atom):
- Error per atom: (E_pred - E_true) / N (signed; abs only for MAE)
- RMSE: sqrt(mean(error^2) weighted by N)
- MAE: mean(abs(error)) weighted by N

Energy (per-config):
- MAE_cfg: mean over configs of abs(E_pred - E_true)

Forces (per-component):
- RMSE: sqrt(mean((F_pred - F_true)^2)) over all components
- MAE: mean(abs(F_pred - F_true)) over all components

Units note:
- Metrics assume energies in eV and forces in eV/Angstrom. If a backend outputs
  normalized values, the adapter must apply the backend normalizer/linref before
  computing metrics.

## Run Directory Layout
All backend tasks write under a single run_dir:
- `artifacts/`: standard outputs (`best_model.pt`, `manifest.json`).
- `checkpoints/`: backend-native checkpoints and interim artifacts.
- `logs/`: runtime logs (when supported by the backend).

## Manifest Contract
The manifest is a JSON file stored alongside weights.

Required fields:
- `schema_version`: string
- `backend`: string (`mace`, `equiformer_v2`, `gemnet_oc`)
- `backend_version`: string (pip version or git commit)
- `source`: { `path`, `format`, `sha256` }
- `weights`: { `path`, `format`, `sha256`, `dtype` }
- `rebuildable`: bool
- `io`: { `energy_unit`, `force_unit`, `energy_is_total`: true }
- `embedding`: { `node_embed_layer`, `graph_pool`: `mean` }

Optional fields:
- `config`: backend-specific config (e.g., model.json or fairchem config)
- `head`: selected head for multi-head models
- `normalizer`: avg_num_neighbors / linref / scale-shift info
- `notes`: free text

### Handling legacy .model/.pt (nn.Module)
If the model does not provide a config:
- Write a trace-only manifest with `rebuildable: false`.
- Capture whatever is available from the module (heads, z_table, cutoff, etc).

If a backend provides an export API:
- Load once, export to a rebuildable format (config + state_dict),
  and set `rebuildable: true` in the manifest.

## Adapter Interface
Adapters provide a minimal, shared surface for core training and inference.

Core training hooks:
- `build_model(cfg) -> nn.Module`
- `model_spec(cfg, model=None) -> dict`
- `select_head(cfg, model) -> str | None`
- `make_backend_batch(cbatch, device) -> backend_batch`
- `forward(model, backend_batch) -> ModelOutputs`
- `loss(outputs, cbatch) -> (loss, logs)`

Optional native bridge (FairChem):
- `native_train(spec) -> run_dir`
- `export_artifacts(native_run_dir, run_dir) -> None`

Inference hooks (frontends):
- `load(source, device) -> ModelBundle`
- `predict(model_bundle, batch) -> ModelOutputs`
- `extract_embeddings(model_bundle, batch) -> EmbeddingOutputs`
- `export_manifest(model_bundle) -> dict`

## Invariants (Must Not Change)
- MACE training requires a valid model.json and must keep strict JSON validation.
- Core training/output layout remains stable (artifacts/checkpoints/logs).
- No new defaults or parameters unless explicitly requested.

## Implementation Notes
- Shared logic lives under `core/` and `adapters/`; CLI entrypoints remain under `frontends/`.
- `core/runner` owns CommonTaskSpec parsing and dispatch; backends implement wrappers only.
- Backends remain the only place where training is implemented.
