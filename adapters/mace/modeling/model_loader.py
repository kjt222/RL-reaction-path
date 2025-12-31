"""Shared helpers for loading/saving MACE models and checkpoints."""

from __future__ import annotations

import copy
import json
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch

from .models import build_model_from_json, attach_model_metadata
from ..read_model import _export_model_json, _diff_json

LOGGER = logging.getLogger(__name__)

# 允许加载包含 slice 的安全全局，用于兼容 Torch >=2.6 的 weights_only 默认行为
torch.serialization.add_safe_globals([slice])

_CHECKPOINT_SUFFIX = "_checkpoint.pt"
_MODEL_SUFFIX = "_model.pt"
_LEGACY_BESTMODEL_SUFFIX = "_bestmodel.pt"
_MODEL_JSON_SUFFIX = "_model.json"


def _canonical_json_text(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_text(data: Mapping[str, Any]) -> str:
    return _canonical_json_text(data)


def hash_text(text: str) -> str:
    return _hash_text(text)


def load_model_json(path: Path) -> tuple[dict, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"未找到 model.json: {path}")
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    text = _canonical_json_text(meta)
    return meta, text, _hash_text(text)


def get_code_version(repo_dir: Path | None = None) -> dict[str, Any]:
    repo_dir = repo_dir or Path(__file__).resolve().parents[3]

    def _run(cmd: list[str]) -> str | None:
        try:
            result = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:  # pragma: no cover - best effort
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    describe = _run(["git", "describe", "--tags", "--always"])
    dirty_raw = _run(["git", "status", "--porcelain"])
    is_dirty = None if dirty_raw is None else bool(dirty_raw)
    return {
        "git_commit": commit,
        "git_describe": describe,
        "is_dirty": is_dirty,
    }


def run_name_from_output_dir(output_dir: Path) -> str:
    return output_dir.name


def derive_run_name_from_checkpoint(checkpoint_path: Path) -> str | None:
    name = checkpoint_path.name
    for suffix in (_CHECKPOINT_SUFFIX, _MODEL_SUFFIX, _LEGACY_BESTMODEL_SUFFIX):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if checkpoint_path.suffix.lower() == ".pt":
        return checkpoint_path.stem
    return None


def _is_pt_file(path: Path) -> bool:
    return path.suffix.lower() == ".pt"


def resolve_output_checkpoint_path(path: Path) -> Path:
    if _is_pt_file(path):
        return path
    return path / f"{path.name}{_CHECKPOINT_SUFFIX}"


def resolve_output_model_path(path: Path) -> Path:
    if _is_pt_file(path):
        return path
    return path / f"{path.name}{_MODEL_SUFFIX}"


def resolve_input_json_path(input_model: Path, input_json: Path | None = None) -> Path:
    if input_json is not None:
        if not input_json.exists():
            raise FileNotFoundError(f"未找到 model.json: {input_json}")
        return input_json
    run_name = derive_run_name_from_checkpoint(input_model)
    base_dir = input_model.parent
    if run_name:
        candidate = base_dir / f"{run_name}{_MODEL_JSON_SUFFIX}"
        if candidate.exists():
            return candidate
    legacy = base_dir / "model.json"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"未找到 model.json: {base_dir}")


def resolve_output_json_paths(
    output_checkpoint: Path | None = None,
    output_model: Path | None = None,
) -> list[Path]:
    paths: list[Path] = []
    if output_checkpoint is not None:
        resolved_ckpt = resolve_output_checkpoint_path(output_checkpoint)
        run_name = derive_run_name_from_checkpoint(resolved_ckpt) or resolved_ckpt.stem
        paths.append(resolved_ckpt.parent / f"{run_name}{_MODEL_JSON_SUFFIX}")
    if output_model is not None:
        resolved_model = resolve_output_model_path(output_model)
        run_name = derive_run_name_from_checkpoint(resolved_model) or resolved_model.stem
        candidate = resolved_model.parent / f"{run_name}{_MODEL_JSON_SUFFIX}"
        if candidate not in paths:
            paths.append(candidate)
    if not paths:
        raise ValueError("必须提供 output_checkpoint 或 output_model 以生成输出 JSON 路径。")
    return paths


def resolve_output_paths(
    output_checkpoint: Path | None,
    output_model: Path | None,
) -> tuple[Path | None, Path | None]:
    resolved_ckpt = resolve_output_checkpoint_path(output_checkpoint) if output_checkpoint is not None else None
    resolved_model = resolve_output_model_path(output_model) if output_model is not None else None
    return resolved_ckpt, resolved_model


def checkpoint_path_for_run(output_dir: Path, run_name: str | None = None) -> Path:
    run_name = run_name or run_name_from_output_dir(output_dir)
    return output_dir / f"{run_name}{_CHECKPOINT_SUFFIX}"


def best_model_path_for_run(output_dir: Path, run_name: str | None = None) -> Path:
    run_name = run_name or run_name_from_output_dir(output_dir)
    return output_dir / f"{run_name}{_MODEL_SUFFIX}"


def model_json_path_for_run(output_dir: Path, run_name: str | None = None) -> Path:
    run_name = run_name or run_name_from_output_dir(output_dir)
    return output_dir / f"{run_name}{_MODEL_JSON_SUFFIX}"


def resolve_model_json_path(base_dir: Path, run_name: str | None = None) -> Path:
    if run_name:
        candidate = base_dir / f"{run_name}{_MODEL_JSON_SUFFIX}"
        if candidate.exists():
            return candidate
    legacy = base_dir / "model.json"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"未找到 model.json: {base_dir}")


def resolve_checkpoint_path(checkpoint_dir: Path) -> Path:
    run_name = run_name_from_output_dir(checkpoint_dir)
    candidate = checkpoint_dir / f"{run_name}{_CHECKPOINT_SUFFIX}"
    if candidate.exists():
        return candidate
    legacy = checkpoint_dir / "checkpoint.pt"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_dir}")


def validate_model_json_hash(
    json_text: str,
    json_hash: str,
    checkpoint_obj: Mapping[str, Any] | None,
) -> None:
    if checkpoint_obj is None:
        raise ValueError("Checkpoint 缺少校验信息，无法验证 model.json。")
    expected_hash = checkpoint_obj.get("model_json_hash")
    expected_text = checkpoint_obj.get("model_json_text")
    if expected_hash is None or expected_text is None:
        raise ValueError("Checkpoint 缺少 model_json_hash/model_json_text，无法验证 model.json。")
    if expected_hash != json_hash:
        raise ValueError(f"model.json hash 不一致: checkpoint={expected_hash} current={json_hash}")
    if expected_text != json_text:
        raise ValueError("model.json 内容与 checkpoint 记录不一致。")


def validate_json_against_module(json_meta: Mapping[str, Any], module: torch.nn.Module) -> None:
    tmp_path = None
    try:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp_path = Path(tmp.name)
        tmp.close()
        _export_model_json(module, tmp_path)
        with tmp_path.open("r", encoding="utf-8") as f:
            exported = json.load(f)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass

    diffs = _diff_json(json_meta, exported)
    if diffs:
        raise ValueError(f"model.json 与模型不一致: {diffs}")


def load_checkpoint_artifacts(path: Path) -> tuple[dict, torch.nn.Module | None, dict, dict]:
    """通用的 checkpoint 解析，返回 (state_dict, module, train_state, raw_dict)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state_dict: dict | None = None
    module: torch.nn.Module | None = None
    train_state: dict = {}

    if isinstance(obj, dict):
        state_dict = obj.get("model_state_dict") or obj.get("state_dict")
        maybe_module = obj.get("model")
        if isinstance(maybe_module, torch.nn.Module):
            module = maybe_module
        train_state = obj.get("train_state") or {}
        if state_dict is None and isinstance(module, torch.nn.Module):
            state_dict = module.state_dict()
    elif isinstance(obj, torch.nn.Module):
        module = obj
        state_dict = obj.state_dict()
    else:
        raise ValueError(f"Unsupported checkpoint object type: {type(obj)} from {path}")

    if state_dict is None:
        raise ValueError(f"No state_dict found in checkpoint: {path}")
    return state_dict, module, train_state, obj if isinstance(obj, dict) else {}


def build_model_with_json(
    json_path: Path,
    checkpoint_path: Path,
    state_dict: dict,
    module_fallback: torch.nn.Module | None = None,
    checkpoint_obj: Mapping[str, Any] | None = None,
) -> Tuple[torch.nn.Module, dict]:
    """按 JSON 构建模型；先做 JSON hash 校验，若有 module 再做二次校验。"""
    json_meta, json_text, json_hash = load_model_json(json_path)

    if checkpoint_obj is None:
        try:
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"无法读取 checkpoint 进行 JSON 校验: {checkpoint_path}") from exc
    validate_model_json_hash(json_text, json_hash, checkpoint_obj if isinstance(checkpoint_obj, Mapping) else None)

    if module_fallback is not None:
        validate_json_against_module(json_meta, module_fallback)
        try:
            module_fallback.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            LOGGER.warning("nn.Module 严格加载 state_dict 失败，将尝试 non-strict：%s", exc)
            module_fallback.load_state_dict(state_dict, strict=False)
        attach_model_metadata(module_fallback, json_meta)
        return module_fallback.float(), json_meta

    # 无 nn.Module，只能按 JSON 重建
    try:
        model = build_model_from_json(json_meta)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError("仅有 state_dict 且无法按 model.json 重建/加载模型。") from exc
    attach_model_metadata(model, json_meta)
    return model, json_meta


def load_for_eval(model_path: Path, input_json: Path | None = None) -> tuple[torch.nn.Module, dict]:
    """评估专用加载：优先用 checkpoint hash 校验；需要可解析的 model.json。"""
    state_dict, module, _train_state, obj = load_checkpoint_artifacts(model_path)

    json_path = resolve_input_json_path(model_path, input_json)
    json_meta, json_text, json_hash = load_model_json(json_path)
    validate_model_json_hash(json_text, json_hash, obj if isinstance(obj, Mapping) else None)

    if module is None:
        model = build_model_from_json(json_meta)
    else:
        validate_json_against_module(json_meta, module)
        model = module

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        LOGGER.warning("nn.Module 严格加载 state_dict 失败，将尝试 non-strict：%s", exc)
        model.load_state_dict(state_dict, strict=False)
    attach_model_metadata(model, json_meta)
    return model, obj if isinstance(obj, dict) else {}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    train_state: dict,
    model_state_dict: dict | None = None,
    ema_state_dict: dict | None = None,
    model_json_text: str | None = None,
    model_json_hash: str | None = None,
    code_version: dict | None = None,
    run_name: str | None = None,
) -> None:
    state = model_state_dict or {k: v.cpu() for k, v in model.state_dict().items()}
    if ema_state_dict is not None:
        # 确保 train_state 带上 ema_state_dict 方便 resume
        train_state = dict(train_state)
        train_state["ema_state_dict"] = ema_state_dict
    payload = {
        "model_state_dict": state,
        "train_state": train_state,
        "model_json_text": model_json_text,
        "model_json_hash": model_json_hash,
        "code_version": code_version,
        "run_name": run_name,
    }
    torch.save(payload, path)


def save_best_model(
    path: Path,
    model: torch.nn.Module,
    best_state_dict: dict | None,
    fallback_state: dict | None = None,
    model_state_dict: dict | None = None,
    model_json_text: str | None = None,
    model_json_hash: str | None = None,
    code_version: dict | None = None,
    run_name: str | None = None,
) -> None:
    best_state = best_state_dict if best_state_dict is not None else model_state_dict if model_state_dict is not None else fallback_state
    if best_state is None:
        raise ValueError("best_state_dict 为空或未提供 fallback_state，无法保存 best_model。")
    best_model_copy = copy.deepcopy(model).cpu()
    best_model_copy.load_state_dict(best_state, strict=False)
    torch.save(
        {
            "model_state_dict": best_state,
            "best_model_state_dict": best_state_dict,
            "model": best_model_copy,
            "model_json_text": model_json_text,
            "model_json_hash": model_json_hash,
            "code_version": code_version,
            "run_name": run_name,
        },
        path,
    )


__all__ = [
    "canonical_json_text",
    "hash_text",
    "load_model_json",
    "get_code_version",
    "run_name_from_output_dir",
    "derive_run_name_from_checkpoint",
    "resolve_output_checkpoint_path",
    "resolve_output_model_path",
    "resolve_input_json_path",
    "resolve_output_json_paths",
    "resolve_output_paths",
    "checkpoint_path_for_run",
    "best_model_path_for_run",
    "model_json_path_for_run",
    "resolve_model_json_path",
    "resolve_checkpoint_path",
    "validate_model_json_hash",
    "validate_json_against_module",
    "load_checkpoint_artifacts",
    "build_model_with_json",
    "load_for_eval",
    "save_checkpoint",
    "save_best_model",
]
