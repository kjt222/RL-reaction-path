"""Target transforms for normalization/linref."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from core.batch.batch_utils import batch_index_from_ptr, build_ptr


class TargetTransform:
    def fit(self, _loader) -> None:
        return None

    def apply_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return dict(batch)

    def inverse_outputs(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
        return dict(outputs)

    def state_dict(self) -> dict[str, Any]:
        return {}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.item())
        return float(torch.as_tensor(value).view(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _as_float(value[0], default=default)
    return float(value)


def _normalizer_state(obj: Any) -> Optional[Mapping[str, Any]]:
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        try:
            return obj.state_dict()
        except Exception:
            return None
    if isinstance(obj, Mapping):
        return obj
    return None


def _extract_normalizer_stats(normalizers: Mapping[str, Any]) -> tuple[float, float, float, float, bool]:
    energy_state = _normalizer_state(normalizers.get("target"))
    force_state = _normalizer_state(normalizers.get("grad_target"))
    if not energy_state and not force_state:
        return 0.0, 1.0, 0.0, 1.0, False

    energy_mean = _as_float(energy_state.get("mean") if energy_state else None, default=0.0)
    energy_std = _as_float(energy_state.get("std") if energy_state else None, default=1.0)
    force_mean = _as_float(force_state.get("mean") if force_state else None, default=0.0)
    force_std = _as_float(force_state.get("std") if force_state else None, default=1.0)
    use_norm = bool(energy_state or force_state)
    return energy_mean, energy_std, force_mean, force_std, use_norm


def _load_linref_coeffs(path: str | Path) -> torch.Tensor:
    import numpy as np

    path_obj = Path(path).expanduser().resolve()
    coeff = np.load(path_obj, allow_pickle=True)["coeff"]
    return torch.as_tensor(coeff, dtype=torch.float32)


def _config_flag(config: Mapping[str, Any], *path: str, default: Optional[bool] = None) -> Optional[bool]:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, Mapping):
            return default
        cursor = cursor.get(key)
    if cursor is None:
        return default
    return bool(cursor)


def _config_value(config: Mapping[str, Any], *path: str) -> Optional[Any]:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    return cursor


@dataclass
class StandardScalerTransform(TargetTransform):
    energy_mean: float = 0.0
    energy_std: float = 1.0
    force_mean: float = 0.0
    force_std: float = 1.0
    fitted: bool = False

    def fit(self, loader) -> None:
        energy_sum = 0.0
        energy_sq = 0.0
        energy_count = 0.0
        force_sum = 0.0
        force_sq = 0.0
        force_count = 0.0

        for batch in loader:
            energy = batch.get("energy")
            ptr = batch.get("ptr")
            if energy is not None and ptr is not None:
                energy_t = torch.as_tensor(energy).view(-1)
                ptr_t = torch.as_tensor(ptr)
                counts = (ptr_t[1:] - ptr_t[:-1]).to(energy_t)
                per_atom = energy_t / counts
                energy_sum += float(per_atom.sum().item())
                energy_sq += float((per_atom.pow(2)).sum().item())
                energy_count += int(per_atom.numel())

            forces = batch.get("forces")
            if forces is not None:
                forces_t = torch.as_tensor(forces).view(-1)
                force_sum += float(forces_t.sum().item())
                force_sq += float((forces_t.pow(2)).sum().item())
                force_count += int(forces_t.numel())

        if energy_count:
            mean = energy_sum / energy_count
            var = max(energy_sq / energy_count - mean**2, 0.0)
            self.energy_mean = mean
            self.energy_std = var**0.5 if var > 0 else 1.0
        if force_count:
            mean = force_sum / force_count
            var = max(force_sq / force_count - mean**2, 0.0)
            self.force_mean = mean
            self.force_std = var**0.5 if var > 0 else 1.0

        self.fitted = True

    def apply_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        energy = batch.get("energy")
        ptr = batch.get("ptr")
        if energy is not None and ptr is not None:
            energy_t = torch.as_tensor(energy)
            ptr_t = torch.as_tensor(ptr)
            counts = (ptr_t[1:] - ptr_t[:-1]).to(energy_t)
            per_atom = energy_t / counts
            normed = (per_atom - self.energy_mean) / self.energy_std
            out["energy"] = normed * counts

        forces = batch.get("forces")
        if forces is not None:
            forces_t = torch.as_tensor(forces)
            out["forces"] = (forces_t - self.force_mean) / self.force_std
        return out

    def inverse_outputs(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(outputs)
        energy = outputs.get("energy")
        ptr = batch.get("ptr")
        if energy is not None and ptr is not None:
            energy_t = torch.as_tensor(energy)
            ptr_t = torch.as_tensor(ptr)
            counts = (ptr_t[1:] - ptr_t[:-1]).to(energy_t)
            per_atom = energy_t / counts
            denorm = per_atom * self.energy_std + self.energy_mean
            out["energy"] = denorm * counts

        forces = outputs.get("forces")
        if forces is not None:
            forces_t = torch.as_tensor(forces)
            out["forces"] = forces_t * self.force_std + self.force_mean
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": "standard",
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "force_mean": self.force_mean,
            "force_std": self.force_std,
            "fitted": self.fitted,
        }


@dataclass
class FairchemTransform(TargetTransform):
    energy_mean: float = 0.0
    energy_std: float = 1.0
    force_mean: float = 0.0
    force_std: float = 1.0
    use_normalizer: bool = True
    linref_coeff: Optional[torch.Tensor] = None
    apply_linref: bool = False

    def _linref_energy(self, batch: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.linref_coeff is None:
            raise ValueError("linref_coeff is required when apply_linref is True")
        z = torch.as_tensor(batch.get("z"), device=device).to(torch.long)
        if z.numel() == 0:
            return torch.zeros((0,), device=device, dtype=dtype)
        coeff = self.linref_coeff.to(device=device, dtype=dtype)
        if int(z.max().item()) >= coeff.numel():
            raise ValueError("lin_ref missing coefficients for element Z")
        ptr = build_ptr(batch).to(device)
        batch_index, counts = batch_index_from_ptr(ptr, device=device)
        linref_atom = coeff[z]
        energy = torch.zeros(counts.numel(), device=device, dtype=dtype)
        if batch_index.numel():
            energy.index_add_(0, batch_index, linref_atom)
        return energy

    def apply_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(batch)
        energy = batch.get("energy")
        if energy is not None and (self.apply_linref or self.use_normalizer):
            energy_t = torch.as_tensor(energy)
            if self.apply_linref:
                linref_energy = self._linref_energy(batch, energy_t.device, energy_t.dtype)
                energy_t = energy_t - linref_energy
            if self.use_normalizer:
                energy_t = (energy_t - self.energy_mean) / self.energy_std
            out["energy"] = energy_t

        forces = batch.get("forces")
        if forces is not None and self.use_normalizer:
            forces_t = torch.as_tensor(forces)
            out["forces"] = (forces_t - self.force_mean) / self.force_std
        return out

    def inverse_outputs(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(outputs)
        energy = outputs.get("energy")
        if energy is not None and (self.apply_linref or self.use_normalizer):
            energy_t = torch.as_tensor(energy)
            if self.use_normalizer:
                energy_t = energy_t * self.energy_std + self.energy_mean
            if self.apply_linref:
                linref_energy = self._linref_energy(batch, energy_t.device, energy_t.dtype)
                energy_t = energy_t + linref_energy
            out["energy"] = energy_t

        forces = outputs.get("forces")
        if forces is not None and self.use_normalizer:
            forces_t = torch.as_tensor(forces)
            out["forces"] = forces_t * self.force_std + self.force_mean
        return out

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": "fairchem",
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "force_mean": self.force_mean,
            "force_std": self.force_std,
            "use_normalizer": self.use_normalizer,
            "apply_linref": self.apply_linref,
        }
        if self.linref_coeff is not None:
            payload["linref_coeff"] = self.linref_coeff.detach().cpu().tolist()
        return payload


def _build_from_transform_cfg(transform_cfg: Mapping[str, Any]) -> TargetTransform:
    mode = str(transform_cfg.get("mode", "identity")).lower()
    if mode in {"identity", "none"}:
        return TargetTransform()
    if mode == "standard":
        transform = StandardScalerTransform()
        for key in ("energy_mean", "energy_std", "force_mean", "force_std"):
            if key in transform_cfg:
                setattr(transform, key, float(transform_cfg[key]))
                transform.fitted = True
        return transform
    if mode == "fairchem":
        coeff = transform_cfg.get("linref_coeff")
        coeff_tensor = torch.as_tensor(coeff, dtype=torch.float32) if coeff is not None else None
        return FairchemTransform(
            energy_mean=float(transform_cfg.get("energy_mean", 0.0)),
            energy_std=float(transform_cfg.get("energy_std", 1.0)),
            force_mean=float(transform_cfg.get("force_mean", 0.0)),
            force_std=float(transform_cfg.get("force_std", 1.0)),
            use_normalizer=bool(transform_cfg.get("use_normalizer", True)),
            linref_coeff=coeff_tensor,
            apply_linref=bool(transform_cfg.get("apply_linref", False)),
        )
    raise ValueError(f"Unsupported transform mode: {mode}")


def _build_fairchem_transform(
    manifest: Optional[Mapping[str, Any]],
    extras: Optional[Mapping[str, Any]],
) -> Optional[TargetTransform]:
    normalizers = {}
    aux_files = {}
    config = {}
    if extras:
        normalizers = extras.get("normalizers") or {}
        aux_files = extras.get("aux_files") or {}
        config = extras.get("config") or {}
    if manifest and not config:
        config = manifest.get("config") or {}

    energy_mean, energy_std, force_mean, force_std, use_norm = _extract_normalizer_stats(normalizers)

    if not use_norm and isinstance(config, Mapping):
        dataset_cfg = config.get("dataset") if isinstance(config.get("dataset"), Mapping) else {}
        train_cfg = dataset_cfg.get("train") if isinstance(dataset_cfg, Mapping) else {}
        if isinstance(train_cfg, Mapping) and train_cfg.get("normalize_labels"):
            energy_mean = float(train_cfg.get("target_mean", 0.0))
            energy_std = float(train_cfg.get("target_std", 1.0))
            force_mean = float(train_cfg.get("grad_target_mean", 0.0))
            force_std = float(train_cfg.get("grad_target_std", 1.0))
            use_norm = True

    linref_path = None
    if isinstance(aux_files, Mapping):
        linref_path = aux_files.get("linref") or aux_files.get("lin_ref")
    if linref_path is None and isinstance(config, Mapping):
        linref_path = _config_value(config, "dataset", "train", "lin_ref") or _config_value(
            config, "dataset", "val", "lin_ref"
        )

    use_energy_lin_ref = False
    if isinstance(config, Mapping):
        use_energy_lin_ref = bool(
            _config_flag(config, "model", "use_energy_lin_ref", default=False)
            or _config_flag(config, "model_attributes", "use_energy_lin_ref", default=False)
        )

    apply_linref = bool(linref_path) and not use_energy_lin_ref
    coeff_tensor = _load_linref_coeffs(linref_path) if apply_linref else None

    if not use_norm and coeff_tensor is None:
        return None

    return FairchemTransform(
        energy_mean=energy_mean,
        energy_std=energy_std,
        force_mean=force_mean,
        force_std=force_std,
        use_normalizer=use_norm,
        linref_coeff=coeff_tensor,
        apply_linref=apply_linref,
    )


def build_transform(
    cfg: Mapping[str, Any],
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> TargetTransform:
    transform_cfg = cfg.get("transform")
    if transform_cfg:
        return _build_from_transform_cfg(transform_cfg)

    manifest_normalizer = manifest.get("normalizer") if isinstance(manifest, Mapping) else None
    if isinstance(manifest_normalizer, Mapping):
        return _build_from_transform_cfg(manifest_normalizer)

    fairchem_transform = _build_fairchem_transform(manifest, extras)
    if fairchem_transform is not None:
        return fairchem_transform

    return TargetTransform()
