"""Inspect a MACE checkpoint/.pt file and print structural metadata."""

from __future__ import annotations

# ==============================================================================
# 强力屏蔽 NVML：在任何第三方 import 之前占坑 pynvml，防止 cuequivariance 导入真实 NVML
# ==============================================================================
import sys
from types import SimpleNamespace


class _NVMLStub:
    class _FakeMem:
        total = 24 * 1024**3
        free = 20 * 1024**3
        used = 4 * 1024**3

    def nvmlInit(self, *a, **k):
        return None

    def nvmlShutdown(self, *a, **k):
        return None

    def nvmlDeviceGetCount(self, *a, **k):
        return 1

    def nvmlDeviceGetHandleByIndex(self, idx, *a, **k):
        return idx

    def nvmlDeviceGetCudaComputeCapability(self, handle):
        return (8, 0)

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return 300000

    def nvmlDeviceGetName(self, handle):
        return b"Mock GPU for NVML stub"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return self._FakeMem()

    def __getattr__(self, name):
        return lambda *args, **kwargs: 0


sys.modules["pynvml"] = _NVMLStub()
_STUB = SimpleNamespace()
for _name in [
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "cuequivariance_ops",
    "cuequivariance_ops.triton",
    "cuequivariance_ops.triton.cache_manager",
    "cuequivariance_ops.triton.tuning_decorator",
    "cuequivariance_ops.triton.autotune_aot",
]:
    sys.modules[_name] = _STUB
# ==============================================================================

import argparse
import logging
from typing import Any, Mapping
import json
from pathlib import Path
import math
import tempfile

import torch
import torch.nn as nn

# 允许加载包含 slice 的安全全局，用于兼容 Torch >=2.6 的 weights_only 默认行为
torch.serialization.add_safe_globals([slice])

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _infer_and_suggest_config(state_dict: Mapping[str, Any]) -> None:
    """根据权重形状反推部分模型配置，结果输出到日志。"""
    LOGGER.info("=" * 60)
    LOGGER.info("侦探模式：根据权重形状反推架构")

    target_suffix = "symmetric_contractions.contractions.0.weights_max"
    found_key = next((k for k in state_dict.keys() if k.endswith(target_suffix)), None)

    if found_key:
        shape = state_dict[found_key].shape
        LOGGER.info("捕获权重: %s 形状=%s", found_key, shape)
        if len(shape) >= 2:
            num_paths = shape[1]
            if num_paths == 23:
                LOGGER.info("诊断：路径数=23，包含 l=2 (D 波) 通道，建议 hidden_irreps=\"128x0e + 128x1o + 128x2e\"，max_ell≈3")
            elif num_paths == 8:
                LOGGER.info("诊断：路径数=8，仅 l=0,1 通道，建议 hidden_irreps=\"128x0e + 128x1o\"，max_ell≈2")
            else:
                LOGGER.warning("未知路径数=%d，无法匹配标准配置", num_paths)
    else:
        LOGGER.warning("未找到收缩权重键，无法推断 hidden_irreps。")

    if "node_embedding.linear.output_mask" in state_dict:
        num_channels = state_dict["node_embedding.linear.output_mask"].numel()
        LOGGER.info("推测 num_channels=%d (node_embedding.linear.output_mask 长度)", num_channels)
    if "radial_embedding.bessel_fn.bessel_weights" in state_dict:
        num_radial_basis = state_dict["radial_embedding.bessel_fn.bessel_weights"].shape[0]
        LOGGER.info("推测 num_radial_basis=%d (bessel_weights 长度)", num_radial_basis)
    if "radial_embedding.cutoff_fn.p" in state_dict:
        shape = state_dict["radial_embedding.cutoff_fn.p"].shape
        if len(shape) > 0:
            num_polynomial_cutoff = shape[0]
            LOGGER.info("推测 num_polynomial_cutoff=%d (cutoff_fn.p 长度)", num_polynomial_cutoff)
        else:
            LOGGER.info("cutoff_fn.p 是标量，无法推断 num_polynomial_cutoff（可能为常数项）")

    LOGGER.info("=" * 60)


def _to_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() <= 200:
            return x.tolist()
        return f"tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    if isinstance(x, (list, tuple)):
        return list(x)
    return x


def _log_fields(container: Mapping[str, Any], name: str, keys: list[str]) -> None:
    LOGGER.info("--- %s ---", name)
    for k in keys:
        if k in container:
            LOGGER.info("%-24s %s", k + ":", _to_list(container[k]))


def _irrep_str(x: Any) -> Any:
    try:
        return str(x)
    except Exception:
        return x


def _parse_max_ell_from_irreps(irreps_str: str) -> int | None:
    try:
        from e3nn import o3  # local import to avoid hard dep at import time
        return max(m.l for m in o3.Irreps(irreps_str))
    except Exception:
        return None


def _infer_from_state_dict(sd: Mapping[str, Any]) -> dict:
    """Best-effort 推断核心架构/统计字段（命名遵循 Parameters.md）。"""
    inferred: dict[str, Any] = {}
    # hidden_irreps + max_ell: 根据 contraction 路径数粗判
    path_counts = [
        int(t.shape[1])
        for k, t in sd.items()
        if isinstance(t, torch.Tensor) and k.endswith("contractions.0.weights_max") and t.dim() >= 2
    ]
    if path_counts:
        max_path = max(path_counts)
        if max_path >= 14:
            inferred["hidden_irreps"] = "128x0e+128x1o+128x2e"
            inferred["max_ell"] = 2
        elif max_path >= 8:
            inferred["hidden_irreps"] = "128x0e+128x1o"
            inferred["max_ell"] = 1
    # num_channels
    if "node_embedding.linear.output_mask" in sd:
        inferred["num_channels"] = int(sd["node_embedding.linear.output_mask"].numel())
    # num_radial_basis
    if "radial_embedding.bessel_fn.bessel_weights" in sd:
        inferred["num_radial_basis"] = int(sd["radial_embedding.bessel_fn.bessel_weights"].shape[0])
    # num_polynomial_cutoff
    if "radial_embedding.cutoff_fn.p" in sd:
        t = torch.as_tensor(sd["radial_embedding.cutoff_fn.p"])
        shape = t.shape
        if len(shape) >= 1 and shape[0] > 0:
            inferred["num_polynomial_cutoff"] = int(shape[0])
        else:
            inferred["num_polynomial_cutoff"] = 1
    # num_interactions
    layers = {
        int(k.split(".")[1])
        for k in sd.keys()
        if k.startswith("interactions.") and k.split(".")[1].isdigit()
    }
    if layers:
        inferred["num_interactions"] = max(layers) + 1
    # avg_num_neighbors
    for k in sd:
        if "avg_num_neighbors" in k:
            try:
                inferred["avg_num_neighbors"] = float(torch.as_tensor(sd[k]).item())
                break
            except Exception:
                continue
    # cutoff
    for k in sd:
        if k.endswith("r_max"):
            try:
                inferred["cutoff"] = float(torch.as_tensor(sd[k]).item())
                break
            except Exception:
                continue
    return inferred


def _export_model_json(model: nn.Module, json_path: Path) -> None:
    """提取/推断完整的模型 JSON，字段命名遵循 Parameters.md。"""
    meta: dict[str, Any] = {}
    meta["model_type"] = type(model).__name__
    state_dict = model.state_dict()
    # 顶层
    if hasattr(model, "atomic_numbers"):
        z = getattr(model, "atomic_numbers")
        try:
            meta["z_table"] = [int(zv) for zv in z.tolist()]
        except Exception:
            meta["z_table"] = [int(zv) for zv in z]
    if hasattr(model, "atomic_energies_fn") and hasattr(model.atomic_energies_fn, "atomic_energies"):
        meta["e0_values"] = _to_list(model.atomic_energies_fn.atomic_energies)
    if hasattr(model, "r_max"):
        meta["cutoff"] = float(torch.as_tensor(getattr(model, "r_max")).item())
    if hasattr(model, "num_interactions"):
        try:
            meta["num_interactions"] = int(torch.as_tensor(getattr(model, "num_interactions")).item())
        except Exception:
            meta["num_interactions"] = int(getattr(model, "num_interactions"))
    # radial
    try:
        meta["num_radial_basis"] = len(model.radial_embedding.bessel_fn.bessel_weights)
        meta["radial_type"] = "bessel"
    except Exception:
        pass
    try:
        meta["num_polynomial_cutoff"] = int(torch.as_tensor(model.radial_embedding.cutoff_fn.p).shape[0])
    except Exception:
        try:
            meta["num_polynomial_cutoff"] = int(torch.as_tensor(model.radial_embedding.cutoff_fn.p).item())
        except Exception:
            pass
    # interactions
    interactions = []
    if hasattr(model, "interactions"):
        for blk in model.interactions:
            info = {}
            for attr in ["node_feats_irreps", "edge_attrs_irreps", "edge_feats_irreps", "target_irreps", "hidden_irreps"]:
                if hasattr(blk, attr):
                    info[attr] = _irrep_str(getattr(blk, attr))
            if hasattr(blk, "radial_MLP"):
                info["radial_MLP"] = getattr(blk, "radial_MLP")
            if hasattr(blk, "avg_num_neighbors"):
                info["avg_num_neighbors"] = float(torch.as_tensor(getattr(blk, "avg_num_neighbors")).item())
            interactions.append(info)
        meta["interactions"] = interactions
    # 顶层 hidden_irreps/MLP_irreps/correlation/gate/radial_type/avg_num_neighbors：若存在直接写，否则从子结构/缓冲补齐
    if "hidden_irreps" not in meta:
        if hasattr(model, "hidden_irreps_str"):
            meta["hidden_irreps"] = str(getattr(model, "hidden_irreps_str"))
        elif hasattr(model, "hidden_irreps"):
            meta["hidden_irreps"] = _irrep_str(getattr(model, "hidden_irreps"))
        elif interactions:
            meta["hidden_irreps"] = interactions[0].get("hidden_irreps")
    if "MLP_irreps" not in meta:
        if hasattr(model, "MLP_irreps_str"):
            meta["MLP_irreps"] = str(getattr(model, "MLP_irreps_str"))
        elif hasattr(model, "MLP_irreps"):
            meta["MLP_irreps"] = _irrep_str(getattr(model, "MLP_irreps"))
        else:
            for rd in meta.get("readouts", []):
                if rd.get("hidden_irreps") is not None:
                    meta["MLP_irreps"] = rd["hidden_irreps"]
                    break
    if "correlation" not in meta:
        for attr in ["correlation_meta", "correlation"]:
            if hasattr(model, attr):
                try:
                    meta["correlation"] = int(torch.as_tensor(getattr(model, attr)).item())
                except Exception:
                    meta["correlation"] = getattr(model, attr)
                break
    if "gate" not in meta:
        if hasattr(model, "gate_str"):
            meta["gate"] = str(getattr(model, "gate_str"))
        else:
            gate_attr = getattr(model, "gate", None)
            if callable(gate_attr):
                # 粗判常见 gate
                if gate_attr == torch.nn.functional.silu:
                    meta["gate"] = "silu"
                elif gate_attr == torch.nn.functional.relu:
                    meta["gate"] = "relu"
            elif gate_attr is not None:
                meta["gate"] = str(gate_attr)
    if "radial_type" not in meta and hasattr(model, "radial_type_str"):
        meta["radial_type"] = str(getattr(model, "radial_type_str"))
    if "avg_num_neighbors" not in meta:
        for blk in meta.get("interactions", []):
            if "avg_num_neighbors" in blk:
                meta["avg_num_neighbors"] = blk["avg_num_neighbors"]
                break
    if "avg_num_neighbors" not in meta and hasattr(model, "avg_num_neighbors_meta"):
        try:
            meta["avg_num_neighbors"] = float(torch.as_tensor(getattr(model, "avg_num_neighbors_meta")).item())
        except Exception:
            meta["avg_num_neighbors"] = getattr(model, "avg_num_neighbors_meta")
    # products
    products = []
    if hasattr(model, "products"):
        for prod in model.products:
            pinfo = {}
            try:
                pinfo["irreps_out"] = _irrep_str(prod.symmetric_contractions.irreps_out)
            except Exception:
                pass
            try:
                pinfo["linear_out"] = int(torch.as_tensor(prod.linear.output_mask).numel())
            except Exception:
                pass
            # contractions path counts
            contrs = []
            try:
                for contr in prod.symmetric_contractions.contractions:
                    if hasattr(contr, "weights_max"):
                        # 优先用 weights_max 的第二维（路径数）
                        contrs.append({"path_count": int(contr.weights_max.shape[1])})
                    elif hasattr(contr, "U_matrix_3"):
                        # 退而求其次，用 U_matrix_3 的最后一维
                        shape = contr.U_matrix_3.shape
                        contrs.append({"path_count": int(shape[-1])})
            except Exception:
                pass
            if contrs:
                pinfo["contractions"] = contrs
            products.append(pinfo)
        meta["products"] = products
    # readouts
    readouts = []
    if hasattr(model, "readouts"):
        for rd in model.readouts:
            rinfo = {}
            for attr in ["irreps_in", "irreps_out", "hidden_irreps"]:
                if hasattr(rd, attr):
                    rinfo[attr] = _irrep_str(getattr(rd, attr))
            readouts.append(rinfo)
        meta["readouts"] = readouts
    # scale_shift
    if hasattr(model, "scale_shift"):
        ss = model.scale_shift
        try:
            meta["scale_shift"] = {
                "scale": float(torch.as_tensor(ss.scale).item()),
                "shift": float(torch.as_tensor(ss.shift).item()),
            }
        except Exception:
            pass
    # 结合 state_dict 推断缺失字段
    inferred = _infer_from_state_dict(state_dict)
    for k, v in inferred.items():
        meta.setdefault(k, v)
    # 缓冲区补齐关键超参
    if "max_ell" not in meta and hasattr(model, "max_ell_meta"):
        try:
            meta["max_ell"] = int(torch.as_tensor(getattr(model, "max_ell_meta")).item())
        except Exception:
            meta["max_ell"] = getattr(model, "max_ell_meta")
    if "num_radial_basis" not in meta and hasattr(model, "num_radial_basis_meta"):
        try:
            meta["num_radial_basis"] = int(torch.as_tensor(getattr(model, "num_radial_basis_meta")).item())
        except Exception:
            meta["num_radial_basis"] = getattr(model, "num_radial_basis_meta")
    if "num_polynomial_cutoff" not in meta and hasattr(model, "num_polynomial_cutoff_meta"):
        try:
            meta["num_polynomial_cutoff"] = int(torch.as_tensor(getattr(model, "num_polynomial_cutoff_meta")).item())
        except Exception:
            meta["num_polynomial_cutoff"] = getattr(model, "num_polynomial_cutoff_meta")
    # 如果 interactions 带有 edge_attrs_irreps，可细化 max_ell
    for blk in meta.get("interactions", []):
        ir = blk.get("edge_attrs_irreps") or blk.get("target_irreps")
        max_ell = _parse_max_ell_from_irreps(ir) if ir else None
        if max_ell is not None:
            meta["max_ell"] = max(meta.get("max_ell", 0), max_ell)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    LOGGER.info("Wrote model.json to %s", json_path)


def _inspect_module(model: nn.Module, prefix: str = "") -> None:
    LOGGER.info("%sDetected nn.Module: %s", prefix, type(model).__name__)
    attrs = [
        "r_max",
        "cutoff",
        "num_interactions",
        "avg_num_neighbors",
        "max_ell",
        "num_channels",
        "num_radial_basis",
        "num_bessel_basis",
        "num_polynomial_cutoff_basis",
        "correlation",
        "interaction_cls",
    ]
    LOGGER.info("--- Module attributes ---")
    for a in attrs:
        if hasattr(model, a):
            LOGGER.info("%s%-24s %s", prefix, a + ":", getattr(model, a))
    if hasattr(model, "atomic_numbers"):
        z = getattr(model, "atomic_numbers")
        try:
            z_list = z.tolist()
        except Exception:
            z_list = z
        LOGGER.info("%satomic_numbers (len=%d): %s", prefix, len(z_list), z_list)
    if hasattr(model, "atomic_energies_fn") and hasattr(model.atomic_energies_fn, "atomic_energies"):
        e0s = model.atomic_energies_fn.atomic_energies
        LOGGER.info("%satomic_energies_fn.atomic_energies: %s", prefix, _to_list(e0s))
    for name in ("model_kwargs", "kwargs", "config"):
        if hasattr(model, name):
            val = getattr(model, name)
            if isinstance(val, Mapping):
                LOGGER.info("%s--- %s ---", prefix, name)
                for k, v in val.items():
                    LOGGER.info("%s%-24s %s", prefix, k + ":", _to_list(v))
    visible = {k: v for k, v in vars(model).items() if not k.startswith("_")}
    if visible:
        LOGGER.info("%s--- __dict__ (visible keys) ---", prefix)
        for k, v in visible.items():
            LOGGER.info("%s%-24s %s", prefix, k + ":", _to_list(v))
    # 打印可见的 buffers（如 avg_num_neighbors 等）
    try:
        buffers = list(model.named_buffers())
        if buffers:
            LOGGER.info("%s--- Buffers ---", prefix)
            for name, buf in buffers:
                LOGGER.info("%s%-24s %s", prefix, name + ":", _to_list(buf))
    except Exception:
        pass
    # 若存在内部嵌套模型，递归检查
    for child_name in ("model", "interatomic_potential"):
        if hasattr(model, child_name):
            child = getattr(model, child_name)
            if isinstance(child, nn.Module):
                LOGGER.info("%s发现内部嵌套模型 (%s)，深入检查...", prefix, child_name)
                _inspect_module(child, prefix=prefix + "  ")
    # 遍历直接子模块，防止遗漏存放在 _modules 的子层
    try:
        for name, child in model.named_children():
            if child is model:
                continue
            LOGGER.info("%s子模块: %s (%s)", prefix, name, type(child).__name__)
            _inspect_module(child, prefix=prefix + "  ")
    except Exception:
        pass


def _inspect_dict(obj: Mapping[str, Any]) -> None:
    LOGGER.info("Detected mapping/dict with keys: %s", list(obj.keys()))

    common_keys = [
        "r_max",
        "cutoff",
        "num_interactions",
        "avg_num_neighbors",
        "z_table",
        "atomic_numbers",
        "e0s",
        "hidden_irreps",
        "correlation",
        "interaction_cls",
        "num_channels",
    ]
    _log_fields(obj, "Top-level fields", common_keys)

    # 若 checkpoint 内含 metadata 字段，优先打印其内容
    if "metadata" in obj and isinstance(obj["metadata"], Mapping):
        _log_fields(
            obj["metadata"],
            "metadata",
            [
                "r_max",
                "cutoff",
                "num_interactions",
                "avg_num_neighbors",
                "hidden_irreps",
                "max_ell",
                "num_channels",
                "num_radial_basis",
                "num_polynomial_cutoff",
                "z_table",
                "e0_values",
                "MLP_irreps",
                "gate",
                "radial_type",
                "model_type",
            ],
        )

    mk = obj.get("model_kwargs") or obj.get("config") or {}
    if isinstance(mk, Mapping):
        _log_fields(
            mk,
            "model_kwargs/config",
            [
                "r_max",
                "cutoff",
                "num_interactions",
                "avg_num_neighbors",
                "hidden_irreps",
                "max_ell",
                "num_channels",
                "num_radial_basis",
                "num_bessel_basis",
                "num_polynomial_cutoff_basis",
                "correlation",
                "interaction_cls",
                "atomic_numbers",
                "z_table",
                "e0s",
            ],
        )

    sd = obj.get("state_dict")
    if sd is None and any(k.startswith("interactions.") for k in obj):
        sd = obj
    if isinstance(sd, Mapping):
        interaction_layers = {
            int(k.split(".")[1])
            for k in sd.keys()
            if k.startswith("interactions.") and k.split(".")[1].isdigit()
        }
        if interaction_layers:
            LOGGER.info("Detected interaction layers: %s (count=%d)", sorted(interaction_layers), len(interaction_layers))
        sample_keys = [k for k in sd.keys() if "interactions" in k][:5]
        if sample_keys:
            LOGGER.info("Sample state_dict keys: %s", sample_keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MACE .pt/.model file and print structural metadata.")
    parser.add_argument("path", type=str, help="Path to the .pt/.model checkpoint")
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Optional path to write extracted model.json (checkpoint dict with 'model' 也支持)",
    )
    args = parser.parse_args()

    LOGGER.info("Loading %s", args.path)
    try:
        obj = torch.load(args.path, map_location="cpu", weights_only=False)
    except Exception as e:
        LOGGER.error("Failed to load checkpoint with weights_only=False: %s", e)
        raise

    if isinstance(obj, nn.Module):
        _inspect_module(obj)
        if args.write_json:
            _export_model_json(obj, Path(args.write_json))
        try:
            _infer_and_suggest_config(obj.state_dict())
        except Exception as e:  # pragma: no cover - best effort diagnostics
            LOGGER.warning("推断架构时出错（Module）：%s", e)
    elif isinstance(obj, Mapping):
        _inspect_dict(obj)
        sd = obj.get("model_state_dict") or obj.get("state_dict") or obj
        # 如果字典里自带 nn.Module，支持写出 JSON
        module = obj.get("model") if isinstance(obj, dict) else None
        if args.write_json and isinstance(module, nn.Module):
            _export_model_json(module, Path(args.write_json))
        if isinstance(sd, Mapping):
            try:
                _infer_and_suggest_config(sd)
            except Exception as e:  # pragma: no cover - best effort diagnostics
                LOGGER.warning("推断架构时出错（Dict）：%s", e)
            if args.write_json and module is None:
                # 纯 state_dict 情况下无法完整写出 model.json（缺 z_table/e0_values）
                raise ValueError("write-json 需要 checkpoint 内含 nn.Module（键 'model'）；纯 state_dict 无法写出完整 JSON。")
    else:
        LOGGER.info("Unknown object type: %s", type(obj))
    LOGGER.info("Object repr: %s", repr(obj)[:500])


def _norm_val(v: Any):
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().tolist()
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_val(v[k]) for k in sorted(v)}
    return v


def _diff_json(a: Any, b: Any, path: str = "") -> list[str]:
    """递归 diff，支持 list/tuple 按元素比较（浮点带容差）。"""
    diffs: list[str] = []
    na, nb = _norm_val(a), _norm_val(b)

    if isinstance(na, dict) and isinstance(nb, dict):
        keys = set(na) | set(nb)
        for k in sorted(keys):
            va = na.get(k, "<missing>")
            vb = nb.get(k, "<missing>")
            diffs.extend(_diff_json(va, vb, f"{path}.{k}" if path else k))
        return diffs

    if isinstance(na, (list, tuple)) and isinstance(nb, (list, tuple)):
        if len(na) != len(nb):
            diffs.append(f"{path}: len {len(na)} != {len(nb)}")
            return diffs
        for i, (va, vb) in enumerate(zip(na, nb)):
            idx_path = f"{path}[{i}]" if path else f"[{i}]"
            diffs.extend(_diff_json(va, vb, idx_path))
        return diffs

    if isinstance(na, float) and isinstance(nb, float):
        if not math.isclose(na, nb, rel_tol=0, abs_tol=1e-6):
            diffs.append(f"{path}: {na} != {nb}")
    else:
        if na != nb:
            diffs.append(f"{path}: {na} != {nb}")
    return diffs


def validate_json_against_checkpoint(json_path: Path | str, checkpoint_path: Path | str) -> tuple[bool, list[str]]:
    """用 read_model 的导出逻辑比对 json 与 checkpoint，返回 (ok, diffs)。"""
    json_path = Path(json_path)
    checkpoint_path = Path(checkpoint_path)
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)

    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    module = obj.get("model") if isinstance(obj, Mapping) else obj if isinstance(obj, nn.Module) else None
    if module is None:
        raise ValueError("Checkpoint 不包含 nn.Module，无法用 read_model 导出进行对比。")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        _export_model_json(module, tmp_path)
        with tmp_path.open("r", encoding="utf-8") as f:
            exported = json.load(f)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

    # 严格对比：不再用目标 JSON 补齐缺失键，缺键也视为差异。
    diffs = _diff_json(json_meta, exported)
    ok = len(diffs) == 0
    return ok, diffs


if __name__ == "__main__":
    main()
