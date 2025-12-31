"""MACE adapter (no training logic)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from core.batch.batch_utils import batch_index_from_ptr, build_ptr, mean_pool
from core.contracts import CanonicalBatch, EmbeddingOutputs, ModelBundle, ModelOutputs
from core.manifest import ArtifactRef, Manifest
from core.registry import register_adapter

from ..base import AdapterBase


DEFAULT_MACE_HEAD = "omat_pbe"


def _resolve_head(model: torch.nn.Module, requested: Optional[str] = None) -> tuple[list[str], str]:
    heads = getattr(model, "heads", None)
    if heads is None:
        heads = [DEFAULT_MACE_HEAD]
    heads_list = list(heads)
    if requested is not None:
        if requested not in heads_list:
            raise ValueError(f"Requested head {requested} not in model heads: {heads_list}")
        return heads_list, requested
    if DEFAULT_MACE_HEAD in heads_list:
        return heads_list, DEFAULT_MACE_HEAD
    raise ValueError(f"Model heads {heads_list} do not include required head {DEFAULT_MACE_HEAD}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_mace_imports() -> None:
    adapter_root = Path(__file__).resolve().parent
    repo_root = adapter_root.parents[1]
    vendor_root = repo_root / "backends" / "mace" / "mace-torch"
    if str(adapter_root) not in sys.path:
        sys.path.insert(0, str(adapter_root))
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))


def _load_model_json(model_path: Path) -> tuple[Optional[dict], Optional[Path]]:
    from .modeling.model_loader import resolve_input_json_path, load_model_json

    try:
        json_path = resolve_input_json_path(model_path)
    except FileNotFoundError:
        return None, None
    meta, _text, _hash = load_model_json(json_path)
    return meta, json_path


def _normalize_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_config_value(val) for key, val in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        if value is torch.nn.functional.silu:
            return "silu"
        if value is torch.nn.functional.relu:
            return "relu"
        name = getattr(value, "__name__", None)
        return name if name else str(value)
    return str(value)


def _extract_official_config(model: torch.nn.Module) -> Optional[dict]:
    try:
        from mace.tools.scripts_utils import extract_config_mace_model
    except Exception:
        return None
    try:
        config = extract_config_mace_model(model)
    except Exception:
        return None
    if not isinstance(config, dict) or config.get("error"):
        return None
    normalized = {key: _normalize_config_value(val) for key, val in config.items()}
    if hasattr(model, "apply_cutoff"):
        normalized.setdefault("apply_cutoff", bool(model.apply_cutoff))
    if hasattr(model, "interactions") and getattr(model, "interactions", None):
        try:
            edge_irreps_first = getattr(model.interactions[0], "edge_irreps", None)
        except Exception:
            edge_irreps_first = None
        if edge_irreps_first is not None:
            normalized["edge_irreps_first"] = _normalize_config_value(edge_irreps_first)
    normalized.setdefault("model_type", model.__class__.__name__)
    if "atomic_numbers" in normalized and "z_table" not in normalized:
        normalized["z_table"] = normalized["atomic_numbers"]
    if "atomic_energies" in normalized and "e0_values" not in normalized:
        normalized["e0_values"] = normalized["atomic_energies"]
    if "num_bessel" in normalized and "num_radial_basis" not in normalized:
        normalized["num_radial_basis"] = normalized["num_bessel"]
    return normalized


def _build_manifest(
    source_path: Path,
    backend_version: str,
    rebuildable: bool,
    config: Optional[dict],
    head: Optional[str],
    notes: Optional[str] = None,
) -> dict[str, Any]:
    source = ArtifactRef(path=str(source_path), format=source_path.suffix.lstrip("."), sha256=_sha256_file(source_path))
    weights = ArtifactRef(path=str(source_path), format="torch", sha256=source.sha256)
    manifest = Manifest(
        schema_version="v1",
        backend="mace",
        backend_version=backend_version,
        source=source,
        weights=weights,
        rebuildable=rebuildable,
        io={"energy_unit": "eV", "force_unit": "eV/Angstrom", "energy_is_total": True},
        embedding={"node_embed_layer": "products[-1]", "graph_pool": "mean"},
        config=config,
        head=head,
        notes=notes,
    )
    return manifest.to_dict()


def _canonical_to_atomic_data(
    batch: CanonicalBatch,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[Any, torch.Tensor, str]:
    from mace import data, tools
    from mace.tools.torch_geometric.batch import Batch

    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.get_default_dtype()
    default_dtype = torch.get_default_dtype()
    if model_dtype != default_dtype:
        torch.set_default_dtype(model_dtype)

    z = torch.as_tensor(batch["z"], device=device)
    pos = torch.as_tensor(batch["pos"], device=device)
    ptr = build_ptr(batch)
    batch_index, counts = batch_index_from_ptr(ptr, device=z.device)

    if hasattr(model, "atomic_numbers"):
        z_list = [int(zv) for zv in model.atomic_numbers.view(-1).tolist()]
    else:
        z_list = sorted({int(zv) for zv in z.view(-1).tolist()})
    z_table = tools.AtomicNumberTable(z_list)

    if hasattr(model, "r_max"):
        cutoff = float(model.r_max)
    else:
        raise ValueError("Model missing r_max; cannot build neighborhoods")

    requested_head = batch.get("head") if isinstance(batch, dict) else None
    heads, head_name = _resolve_head(model, requested_head)

    energy = batch.get("energy")
    forces = batch.get("forces")
    cell = batch.get("cell")
    pbc = batch.get("pbc")

    data_list = []
    try:
        for i in range(counts.numel()):
            start = int(ptr[i].item())
            end = int(ptr[i + 1].item())
            z_i = z[start:end].detach().cpu().numpy().astype(np.int64)
            pos_i = pos[start:end].detach().cpu().numpy()

            props: Dict[str, Any] = {}
            if energy is not None:
                energy_i = float(torch.as_tensor(energy)[i].item())
                props["energy"] = energy_i
            if forces is not None:
                forces_i = torch.as_tensor(forces)[start:end].detach().cpu().numpy()
                props["forces"] = forces_i

            cell_i = None
            if cell is not None:
                cell_t = torch.as_tensor(cell)
                cell_i = cell_t[i].detach().cpu().numpy() if cell_t.dim() == 3 else cell_t.detach().cpu().numpy()

            pbc_i = None
            if pbc is not None:
                pbc_t = torch.as_tensor(pbc)
                pbc_i = pbc_t[i].detach().cpu().numpy() if pbc_t.dim() == 2 else pbc_t.detach().cpu().numpy()
                pbc_i = tuple(bool(x) for x in pbc_i)

            cfg = data.Configuration(
                atomic_numbers=z_i,
                positions=pos_i,
                properties=props,
                property_weights={},
                cell=cell_i,
                pbc=pbc_i,
                weight=1.0,
                config_type="Default",
                head=head_name,
            )
            data_list.append(data.AtomicData.from_config(cfg, z_table=z_table, cutoff=cutoff, heads=heads))
    finally:
        if model_dtype != default_dtype:
            torch.set_default_dtype(default_dtype)

    batch_obj = Batch.from_data_list(data_list)
    return batch_obj, batch_index, head_name


class MaceAdapter(AdapterBase):
    backend_name = "mace"

    def __init__(self, train_cfg: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(train_cfg=train_cfg)
        self._model_config: Optional[dict] = None
        self._selected_head: Optional[str] = None
        self._model: Optional[torch.nn.Module] = None

    def build_model(self, cfg: Mapping[str, Any]) -> torch.nn.Module:
        _ensure_mace_imports()
        from .modeling.model_loader import load_model_json
        from .modeling.models import attach_model_metadata, build_model_from_json

        input_json = cfg.get("input_json") or cfg.get("model_json") or cfg.get("config_json")
        if not input_json:
            raise ValueError("MACE build_model requires train.input_json (model.json)")
        json_path = Path(str(input_json)).expanduser().resolve()
        meta, _text, _hash = load_model_json(json_path)
        model = build_model_from_json(meta)
        attach_model_metadata(model, meta)
        self._model_config = meta
        self._model = model
        return model

    def model_spec(self, cfg: Mapping[str, Any], model: torch.nn.Module | None = None) -> dict[str, Any]:
        _ensure_mace_imports()
        config = self._model_config
        if config is None and model is not None:
            config = _extract_official_config(model)
        head = self._selected_head or cfg.get("head_key") or cfg.get("target_keys") or cfg.get("head")
        import mace

        return {
            "backend_version": getattr(mace, "__version__", "unknown"),
            "config": config,
            "head": head,
            "io": {"energy_unit": "eV", "force_unit": "eV/Angstrom", "energy_is_total": True},
            "embedding": {"node_embed_layer": "products[-1]", "graph_pool": "mean"},
        }

    def select_head(self, cfg: Mapping[str, Any], model: torch.nn.Module) -> str | None:
        requested = cfg.get("head_key") or cfg.get("target_keys") or cfg.get("head")
        _heads, head = _resolve_head(model, requested)
        self._selected_head = head
        return head

    def make_backend_batch(self, cbatch: CanonicalBatch, device: torch.device) -> Any:
        model = self._model
        if model is None:
            raise ValueError("build_model must be called before make_backend_batch")
        batch_local = dict(cbatch)
        if self._selected_head and "head" not in batch_local:
            batch_local["head"] = self._selected_head
        compute_forces = batch_local.get("forces") is not None and not bool(batch_local.get("energy_only"))
        batch_obj, batch_index, _head = _canonical_to_atomic_data(batch_local, model, device)
        batch_obj = batch_obj.to(device)
        return {
            "data": batch_obj,
            "batch_index": batch_index.to(device),
            "compute_forces": compute_forces,
        }

    def forward(self, model: torch.nn.Module, backend_batch: Any) -> ModelOutputs:
        _ensure_mace_imports()
        batch_obj = backend_batch["data"]
        batch_index = backend_batch["batch_index"]
        compute_forces = bool(backend_batch.get("compute_forces", False))

        node_embed: Optional[torch.Tensor] = None

        def _capture(_module, _inputs, output):
            nonlocal node_embed
            node_embed = output

        handle = model.products[-1].register_forward_hook(_capture)
        try:
            with torch.set_grad_enabled(compute_forces):
                outputs = model(batch_obj.to_dict(), training=model.training, compute_force=compute_forces)
        finally:
            handle.remove()

        if node_embed is None:
            raise RuntimeError("Failed to capture node embeddings from MACE model")

        energy = outputs["energy"]
        forces = outputs.get("forces")
        graph_embed = mean_pool(node_embed, batch_index.to(node_embed.device))
        return {
            "energy": energy,
            "forces": forces,
            "node_embed": node_embed,
            "graph_embed": graph_embed,
        }

    def loss(self, outputs: ModelOutputs, cbatch: CanonicalBatch) -> tuple[torch.Tensor, dict[str, float]]:
        from core.losses import compute_energy_force_loss

        energy_weight = float(self.train_cfg.get("energy_weight", 1.0))
        force_weight = float(self.train_cfg.get("force_weight", 1.0))
        energy_loss = str(self.train_cfg.get("loss_energy", "mse"))
        force_loss = str(self.train_cfg.get("loss_force", "mse"))

        return compute_energy_force_loss(
            outputs,
            cbatch,
            energy_weight=energy_weight,
            force_weight=force_weight,
            energy_loss=energy_loss,
            force_loss=force_loss,
            energy_loss_mode="per_atom",
        )

    @classmethod
    def supports_source(cls, source_path: str) -> bool:
        return source_path.endswith(".pt") or source_path.endswith(".model")

    def load(self, source_path: str, device: str) -> ModelBundle:
        torch.serialization.add_safe_globals([slice])
        _ensure_mace_imports()
        from .modeling.model_loader import load_checkpoint_artifacts, load_for_eval
        model_path = Path(source_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        json_meta, json_path = _load_model_json(model_path)
        rebuildable = json_meta is not None

        if rebuildable:
            model, _ckpt_obj = load_for_eval(model_path, json_path)
        else:
            state_dict, module, _train_state, _raw = load_checkpoint_artifacts(model_path)
            if module is None:
                raise ValueError("Checkpoint missing nn.Module and model.json; cannot load")
            model = module
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()

        import mace

        _heads, head = _resolve_head(model, None)
        self._model = model
        self._selected_head = head

        notes = None
        config = json_meta
        if config is not None:
            self._model_config = config
        if config is None:
            config = _extract_official_config(model)
            if config is not None:
                notes = "config extracted via mace.tools.scripts_utils.extract_config_mace_model; rebuild not guaranteed"

        manifest = _build_manifest(
            source_path=model_path,
            backend_version=getattr(mace, "__version__", "unknown"),
            rebuildable=rebuildable,
            config=config,
            head=head,
            notes=notes,
        )
        return ModelBundle(model=model, manifest=manifest, backend=self.backend_name, device=device)

    def load_from_manifest(self, manifest_path: str, device: str, weights_path: Optional[str] = None) -> ModelBundle:
        torch.serialization.add_safe_globals([slice])
        _ensure_mace_imports()
        from .modeling.models import attach_model_metadata, build_model_from_json

        manifest_path_obj = Path(manifest_path).expanduser().resolve()
        if not manifest_path_obj.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

        with manifest_path_obj.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        config = manifest.get("config")
        if config is None:
            raise ValueError("Manifest missing config; cannot rebuild MACE model")

        model = build_model_from_json(config)

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
        state_dict: Optional[dict] = None
        if isinstance(obj, dict):
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                state_dict = obj
            else:
                state_dict = obj.get("model_state_dict") or obj.get("state_dict")
                maybe_module = obj.get("model")
                if state_dict is None and isinstance(maybe_module, torch.nn.Module):
                    state_dict = maybe_module.state_dict()
        elif isinstance(obj, torch.nn.Module):
            state_dict = obj.state_dict()

        if state_dict is None:
            raise ValueError(f"Unsupported weights object in {weights_path_obj}")

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
        attach_model_metadata(model, config)

        model.to(device)
        model.eval()
        _heads, head = _resolve_head(model, None)
        self._model = model
        self._selected_head = head
        return ModelBundle(model=model, manifest=manifest, backend=self.backend_name, device=device)

    def predict(self, bundle: ModelBundle, batch: CanonicalBatch) -> ModelOutputs:
        _ensure_mace_imports()

        model = bundle.model
        batch_local = dict(batch)
        extras = bundle.extras or {}
        if "head" not in batch_local and "head" in extras:
            batch_local["head"] = extras["head"]
        if "energy_only" not in batch_local and "energy_only" in extras:
            batch_local["energy_only"] = extras["energy_only"]

        batch_obj, batch_index, _head = _canonical_to_atomic_data(batch_local, model, torch.device(bundle.device))
        batch_obj = batch_obj.to(bundle.device)

        node_embed: Optional[torch.Tensor] = None

        def _capture(_module, _inputs, output):
            nonlocal node_embed
            node_embed = output

        handle = model.products[-1].register_forward_hook(_capture)
        compute_forces = not bool(batch_local.get("energy_only"))
        try:
            with torch.set_grad_enabled(compute_forces):
                outputs = model(batch_obj.to_dict(), training=False, compute_force=compute_forces)
        finally:
            handle.remove()

        if node_embed is None:
            raise RuntimeError("Failed to capture node embeddings from MACE model")

        energy = outputs["energy"].detach()
        forces = outputs.get("forces")
        forces = forces.detach() if forces is not None else None
        node_embed = node_embed.detach()
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


def register() -> None:
    register_adapter(MaceAdapter.backend_name, MaceAdapter)
