"""Shared FairChem adapter base."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from core.batch.batch_utils import batch_index_from_ptr, build_ptr, mean_pool
from core.contracts import CanonicalBatch, EmbeddingOutputs, ModelBundle, ModelOutputs
from core.manifest import ArtifactRef, Manifest

from adapters.base import AdapterBase


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_fairchem_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    backend_root = repo_root / "backends" / "equiformer_v2" / "fairchem"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _get_backend_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("ocp-models")
    except Exception:
        setup_path = Path(__file__).resolve().parents[2] / "backends" / "equiformer_v2" / "fairchem" / "setup.py"
        if setup_path.exists():
            text = setup_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if "version=" in line:
                    return line.split("version=")[-1].strip().strip("\"'")
        return "unknown"


def _build_manifest(
    source_path: Path,
    backend: str,
    backend_version: str,
    rebuildable: bool,
    config: Optional[dict],
    embedding_layer: Optional[str],
    backend_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    source = ArtifactRef(path=str(source_path), format=source_path.suffix.lstrip("."), sha256=_sha256_file(source_path))
    weights = ArtifactRef(path=str(source_path), format="torch", sha256=source.sha256)
    manifest = Manifest(
        schema_version="v1",
        backend=backend,
        backend_version=backend_version,
        source=source,
        weights=weights,
        rebuildable=rebuildable,
        io={"energy_unit": "eV", "force_unit": "eV/Angstrom", "energy_is_total": True},
        embedding={"node_embed_layer": embedding_layer, "graph_pool": "mean"},
        config=config,
        backend_state=backend_state,
    )
    return manifest.to_dict()


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.dim() == 0:
            return float(value.item())
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _serialize_normalizers(normalizers: Dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, normalizer in normalizers.items():
        try:
            state = normalizer.state_dict()
        except Exception:
            continue
        payload[str(key)] = {k: _to_jsonable(v) for k, v in state.items()}
    return payload


def _resolve_embedding_module(model: torch.nn.Module) -> tuple[Optional[torch.nn.Module], Optional[str]]:
    if hasattr(model, "norm") and isinstance(model.norm, torch.nn.Module):
        return model.norm, "norm_output"
    if hasattr(model, "out_mlp_E") and isinstance(model.out_mlp_E, torch.nn.Module):
        return model.out_mlp_E, "out_mlp_E"
    return None, None


def _canonical_to_data(batch: CanonicalBatch, device: torch.device, model) -> tuple[Any, torch.Tensor]:
    from torch_geometric.data import Data

    z = torch.as_tensor(batch["z"], device=device)
    pos = torch.as_tensor(batch["pos"], device=device)
    tags = batch.get("tags")
    fixed = batch.get("fixed")
    ptr = build_ptr(batch)
    batch_index, counts = batch_index_from_ptr(ptr, device=device)

    natoms = counts.to(device)

    cell = batch.get("cell")
    pbc = batch.get("pbc")
    cell_t = None
    pbc_t = None
    if cell is not None:
        cell_t = torch.as_tensor(cell, device=device)
        if cell_t.dim() == 2:
            cell_t = cell_t.unsqueeze(0).repeat(natoms.numel(), 1, 1)
    if pbc is not None:
        pbc_t = torch.as_tensor(pbc, device=device)
        if pbc_t.dim() == 1:
            pbc_t = pbc_t.unsqueeze(0).repeat(natoms.numel(), 1)
    if getattr(model, "use_pbc", False):
        if cell_t is None or pbc_t is None:
            raise ValueError("Model expects PBC but batch missing cell/pbc")

    tags_t = torch.as_tensor(tags, device=device).to(torch.long) if tags is not None else None
    fixed_t = torch.as_tensor(fixed, device=device).to(torch.long) if fixed is not None else None

    data = Data(
        pos=pos,
        atomic_numbers=z,
        batch=batch_index,
        natoms=natoms,
        cell=cell_t,
        pbc=pbc_t,
        tags=tags_t,
        fixed=fixed_t,
    )
    return data, batch_index


class FairchemAdapterBase(AdapterBase):
    backend_name = ""
    default_node_embed_layer = "norm_output"

    def __init__(self, train_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(train_cfg=train_cfg)
        self._config: Optional[dict] = None
        self._model: Optional[torch.nn.Module] = None
        self._normalizers: Dict[str, Any] = {}
        self._embedding_layer: Optional[str] = None
        self._scale_dict: Optional[Dict[str, Any]] = None
        self._aux_files: Optional[Dict[str, str]] = None

    def build_model(self, cfg: Mapping[str, Any]) -> torch.nn.Module:
        _ensure_fairchem_imports()
        import ocpmodels.models  # noqa: F401
        from ocpmodels.common.registry import registry

        config = cfg.get("config")
        config_yml = cfg.get("config_yml") or cfg.get("config_path")
        if config is None and config_yml:
            import yaml

            with Path(str(config_yml)).expanduser().open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("FairChem build_model requires config dict or config_yml path")

        model_name = config["model"]
        model_attributes = config.get("model_attributes", {})
        num_targets = config.get("task", {}).get("num_targets", model_attributes.get("num_targets", 1))
        bond_feat_dim = model_attributes.get("num_gaussians", 50)

        model_cls = registry.get_model_class(model_name)
        model = model_cls(None, bond_feat_dim, num_targets, **model_attributes)

        self._config = config
        self._model = model
        _module, layer_name = _resolve_embedding_module(model)
        self._embedding_layer = layer_name
        return model

    def model_spec(self, cfg: Mapping[str, Any], model: torch.nn.Module | None = None) -> dict[str, Any]:
        config = self._config or cfg.get("config")
        embedding_layer = self._embedding_layer or cfg.get("embedding_layer") or self.default_node_embed_layer
        return {
            "backend_version": _get_backend_version(),
            "config": config,
            "head": None,
            "io": {"energy_unit": "eV", "force_unit": "eV/Angstrom", "energy_is_total": True},
            "embedding": {"node_embed_layer": embedding_layer, "graph_pool": "mean"},
        }

    def select_head(self, cfg: Mapping[str, Any], model: torch.nn.Module) -> str | None:
        _ = cfg
        _ = model
        return None

    def make_backend_batch(self, cbatch: CanonicalBatch, device: torch.device) -> Any:
        model = self._model
        if model is None:
            raise ValueError("build_model must be called before make_backend_batch")
        data, batch_index = _canonical_to_data(cbatch, device, model)
        compute_forces = cbatch.get("forces") is not None and not bool(cbatch.get("energy_only"))
        return {"data": data, "batch_index": batch_index, "compute_forces": compute_forces}

    def forward(self, model: torch.nn.Module, backend_batch: Any) -> ModelOutputs:
        _ensure_fairchem_imports()
        data = backend_batch["data"]
        batch_index = backend_batch["batch_index"]
        compute_forces = bool(backend_batch.get("compute_forces", False))

        node_embed: Optional[torch.Tensor] = None

        def _capture(_module, _inputs, output):
            nonlocal node_embed
            node_embed = output

        embed_module, embed_layer = _resolve_embedding_module(model)
        self._embedding_layer = embed_layer
        handle = embed_module.register_forward_hook(_capture) if embed_module is not None else None
        orig_regress = getattr(model, "regress_forces", None)
        try:
            if not compute_forces and hasattr(model, "regress_forces"):
                model.regress_forces = False
            with torch.set_grad_enabled(compute_forces):
                out = model(data)
        finally:
            if orig_regress is not None:
                model.regress_forces = orig_regress
            if handle is not None:
                handle.remove()

        if isinstance(out, tuple):
            energy, forces = out
        else:
            energy, forces = out, None

        if node_embed is None:
            num_nodes = int(data.atomic_numbers.shape[0])
            node_embed = torch.zeros((num_nodes, 1), device=energy.device if torch.is_tensor(energy) else data.pos.device)
        node_embed = node_embed.reshape(node_embed.shape[0], -1)
        graph_embed = mean_pool(node_embed, batch_index.to(node_embed.device))
        return {
            "energy": energy,
            "forces": forces,
            "node_embed": node_embed,
            "graph_embed": graph_embed,
        }

    def loss(self, outputs: ModelOutputs, cbatch: CanonicalBatch) -> tuple[torch.Tensor, dict[str, float]]:
        from core.losses import compute_energy_force_loss

        config = self._config or {}
        optim_cfg = config.get("optim", {}) if isinstance(config, dict) else {}
        task_cfg = config.get("task", {}) if isinstance(config, dict) else {}

        energy_loss = self.train_cfg.get("loss_energy") or optim_cfg.get("loss_energy", "mae")
        force_loss = self.train_cfg.get("loss_force") or optim_cfg.get("loss_force", "mae")
        energy_weight = float(self.train_cfg.get("energy_weight", optim_cfg.get("energy_coefficient", 1.0)))
        force_weight = float(self.train_cfg.get("force_weight", optim_cfg.get("force_coefficient", 1.0)))

        train_on_free_atoms = bool(task_cfg.get("train_on_free_atoms", False))
        tag_specific_weights = task_cfg.get("tag_specific_weights") or None

        return compute_energy_force_loss(
            outputs,
            cbatch,
            energy_weight=energy_weight,
            force_weight=force_weight,
            energy_loss=str(energy_loss),
            force_loss=str(force_loss),
            energy_loss_mode="per_config",
            train_on_free_atoms=train_on_free_atoms,
            tag_specific_weights=tag_specific_weights,
        )

    def native_train(self, spec: Any) -> Any:
        from adapters.fairchem.runner import run_task as run_equiformer

        result = run_equiformer(spec)
        return result.run_dir

    def export_artifacts(self, native_run_dir: Any, run_dir: Any) -> None:
        from adapters.fairchem.runner import _find_latest_checkpoint
        from core.ckpt.export import export_bundle
        from core.runner.layout import artifacts_dir

        best_ckpt = _find_latest_checkpoint(Path(native_run_dir))
        if best_ckpt is None:
            raise FileNotFoundError("No FairChem checkpoint found to export.")
        bundle = self.load(str(best_ckpt), device="cpu")
        export_bundle(
            bundle,
            output_dir=artifacts_dir(Path(run_dir)),
            weights_name="best_model.pt",
            manifest_name="manifest.json",
        )

    @classmethod
    def supports_source(cls, source_path: str) -> bool:
        return source_path.endswith(".pt") or source_path.endswith(".pth")

    def load(self, source_path: str, device: str) -> ModelBundle:
        _ensure_fairchem_imports()
        import ocpmodels.models  # noqa: F401
        from ocpmodels.common.registry import registry
        from ocpmodels.modules.normalizer import Normalizer
        from ocpmodels.modules.scaling.compat import load_scales_compat

        model_path = Path(source_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if config is None:
            raise ValueError("Checkpoint missing config; cannot rebuild FairChem model")

        model_name = config["model"]
        model_attributes = config.get("model_attributes", {})
        num_targets = config.get("task", {}).get(
            "num_targets", model_attributes.get("num_targets", 1)
        )
        bond_feat_dim = model_attributes.get("num_gaussians", 50)

        model_cls = registry.get_model_class(model_name)
        model = model_cls(None, bond_feat_dim, num_targets, **model_attributes)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        ckpt_key_count = next(iter(state_dict)).count("module")
        mod_key_count = next(iter(model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            state_dict = {key_count_diff * "module." + k: v for k, v in state_dict.items()}
        elif key_count_diff < 0:
            prefix = "module." * abs(key_count_diff)
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        scale_dict = checkpoint.get("scale_dict") if isinstance(checkpoint, dict) else None
        if scale_dict:
            load_scales_compat(model, scale_dict)

        normalizers: Dict[str, Normalizer] = {}
        if isinstance(checkpoint, dict) and "normalizers" in checkpoint:
            for key, value in checkpoint["normalizers"].items():
                normalizer = Normalizer(mean=0.0, std=1.0, device="cpu")
                normalizer.load_state_dict(value)
                normalizers[key] = normalizer

        model.to(device)
        model.eval()

        _module, layer_name = _resolve_embedding_module(model)
        self._embedding_layer = layer_name

        backend_state: dict[str, Any] = {}
        normalizer_state = _serialize_normalizers(normalizers)
        if normalizer_state:
            backend_state["normalizers"] = normalizer_state
        if scale_dict:
            backend_state["scale_dict"] = _to_jsonable(scale_dict)

        aux_files: dict[str, str] = {}
        linref_candidate = model_path.parent / "oc22_linfit_coeffs.npz"
        if linref_candidate.exists():
            aux_files["linref"] = str(linref_candidate)
        metadata_candidate = model_path.parent / "oc22_metadata.pkl"
        if metadata_candidate.exists():
            aux_files["metadata"] = str(metadata_candidate)
        if aux_files:
            backend_state["aux_files"] = aux_files

        manifest = _build_manifest(
            source_path=model_path,
            backend=self.backend_name,
            backend_version=_get_backend_version(),
            rebuildable=True,
            config=config,
            embedding_layer=self._embedding_layer or self.default_node_embed_layer,
            backend_state=backend_state or None,
        )
        self._normalizers = normalizers
        self._scale_dict = scale_dict
        self._aux_files = aux_files or None
        self._config = config
        self._model = model
        extras = {
            "normalizers": normalizers,
            "config": config,
            "scale_dict": scale_dict,
            "aux_files": aux_files or None,
        }
        return ModelBundle(model=model, manifest=manifest, backend=self.backend_name, device=device, extras=extras)

    def load_from_manifest(self, manifest_path: str, device: str, weights_path: Optional[str] = None) -> ModelBundle:
        _ensure_fairchem_imports()
        import ocpmodels.models  # noqa: F401
        from ocpmodels.common.registry import registry
        from ocpmodels.modules.normalizer import Normalizer
        from ocpmodels.modules.scaling.compat import load_scales_compat

        manifest_path_obj = Path(manifest_path).expanduser().resolve()
        if not manifest_path_obj.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

        with manifest_path_obj.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        config = manifest.get("config")
        if config is None:
            raise ValueError("Manifest missing config; cannot rebuild FairChem model")

        model_name = config["model"]
        model_attributes = config.get("model_attributes", {})
        num_targets = config.get("task", {}).get(
            "num_targets", model_attributes.get("num_targets", 1)
        )
        bond_feat_dim = model_attributes.get("num_gaussians", 50)

        model_cls = registry.get_model_class(model_name)
        model = model_cls(None, bond_feat_dim, num_targets, **model_attributes)

        weights_path_resolved = weights_path or manifest.get("weights", {}).get("path")
        if not weights_path_resolved:
            raise ValueError("Weights path missing; set model.weights or weights.path in manifest")
        weights_path_obj = Path(weights_path_resolved).expanduser()
        if not weights_path_obj.is_absolute():
            weights_path_obj = manifest_path_obj.parent / weights_path_obj
        weights_path_obj = weights_path_obj.resolve()
        if not weights_path_obj.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path_obj}")

        obj = torch.load(weights_path_obj, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            state_dict = obj["state_dict"]
        else:
            state_dict = obj

        ckpt_key_count = next(iter(state_dict)).count("module")
        mod_key_count = next(iter(model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            state_dict = {key_count_diff * "module." + k: v for k, v in state_dict.items()}
        elif key_count_diff < 0:
            prefix = "module." * abs(key_count_diff)
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)

        backend_state = manifest.get("backend_state", {}) if isinstance(manifest, dict) else {}
        scale_dict = None
        normalizer_state: dict[str, Any] = {}
        aux_files = None
        if isinstance(backend_state, dict) and backend_state.get("path") and backend_state.get("format") == "torch":
            state_path = Path(str(backend_state["path"]))
            if not state_path.is_absolute():
                state_path = manifest_path_obj.parent / state_path
            state_path = state_path.resolve()
            state_obj = torch.load(state_path, map_location="cpu", weights_only=False)
            if isinstance(state_obj, dict):
                scale_dict = state_obj.get("scale_dict")
                normalizer_state = state_obj.get("normalizers", {})
                aux_files = state_obj.get("aux_files")
        elif isinstance(backend_state, dict):
            scale_dict = backend_state.get("scale_dict")
            normalizer_state = backend_state.get("normalizers", {})
            aux_files = backend_state.get("aux_files")

        if scale_dict:
            load_scales_compat(model, scale_dict)

        normalizers: Dict[str, Normalizer] = {}
        if isinstance(normalizer_state, dict):
            for key, value in normalizer_state.items():
                normalizer = Normalizer(mean=0.0, std=1.0, device="cpu")
                state = {k: torch.as_tensor(v) for k, v in value.items()}
                normalizer.load_state_dict(state)
                normalizers[key] = normalizer

        model.to(device)
        model.eval()

        _module, layer_name = _resolve_embedding_module(model)
        self._embedding_layer = layer_name
        self._normalizers = normalizers
        self._scale_dict = scale_dict
        self._aux_files = aux_files or None
        self._config = config
        self._model = model

        extras = {
            "normalizers": normalizers,
            "config": config,
            "scale_dict": scale_dict,
            "aux_files": aux_files,
        }
        return ModelBundle(model=model, manifest=manifest, backend=self.backend_name, device=device, extras=extras)

    def predict(self, bundle: ModelBundle, batch: CanonicalBatch) -> ModelOutputs:
        _ensure_fairchem_imports()

        model = bundle.model
        data, batch_index = _canonical_to_data(batch, torch.device(bundle.device), model)

        extras = bundle.extras or {}
        energy_only = bool(batch.get("energy_only") or extras.get("energy_only"))
        orig_regress = getattr(model, "regress_forces", None)

        node_embed: Optional[torch.Tensor] = None

        def _capture(_module, _inputs, output):
            nonlocal node_embed
            node_embed = output

        embed_module, embed_layer = _resolve_embedding_module(model)
        self._embedding_layer = embed_layer
        handle = embed_module.register_forward_hook(_capture) if embed_module is not None else None
        try:
            if energy_only and hasattr(model, "regress_forces"):
                model.regress_forces = False
            with torch.no_grad():
                out = model(data)
        finally:
            if energy_only and orig_regress is not None:
                model.regress_forces = orig_regress
            if handle is not None:
                handle.remove()

        if isinstance(out, tuple):
            energy, forces = out
        else:
            energy, forces = out, None

        if node_embed is None:
            num_nodes = int(data.atomic_numbers.shape[0])
            node_embed = torch.zeros((num_nodes, 1), device=energy.device if torch.is_tensor(energy) else data.pos.device)
        node_embed = node_embed.reshape(node_embed.shape[0], -1)
        graph_embed = mean_pool(node_embed, batch_index.to(node_embed.device))

        result: ModelOutputs = {
            "energy": energy,
            "forces": forces,
            "node_embed": node_embed,
            "graph_embed": graph_embed,
        }
        return result

    def extract_embeddings(self, bundle: ModelBundle, batch: CanonicalBatch) -> EmbeddingOutputs:
        outputs = self.predict(bundle, batch)
        return {
            "node_embed": outputs["node_embed"],
            "graph_embed": outputs["graph_embed"],
        }

    def export_manifest(self, bundle: ModelBundle) -> dict[str, Any]:
        return dict(bundle.manifest)
