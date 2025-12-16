"""Shared helpers for loading/saving MACE models and checkpoints."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Tuple

import torch

from models import build_model_from_json
from read_model import validate_json_against_checkpoint

LOGGER = logging.getLogger(__name__)

# 允许加载包含 slice 的安全全局，用于兼容 Torch >=2.6 的 weights_only 默认行为
torch.serialization.add_safe_globals([slice])


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
) -> Tuple[torch.nn.Module, dict]:
    """按 JSON 构建模型；如果提供 module_fallback 则要求 JSON 校验通过后加载 state_dict."""
    with json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)

    # 有 nn.Module 时，要求 JSON 校验通过，否则报错
    if module_fallback is not None:
        ok, diffs = validate_json_against_checkpoint(json_path, checkpoint_path)
        if not ok:
            raise ValueError(f"model.json 与 checkpoint 不一致: {diffs}")
        try:
            module_fallback.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            LOGGER.warning("nn.Module 严格加载 state_dict 失败，将尝试 non-strict：%s", exc)
            module_fallback.load_state_dict(state_dict, strict=False)
        return module_fallback.float(), json_meta

    # 无 nn.Module，只能按 JSON 重建
    try:
        model = build_model_from_json(json_meta)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ValueError("仅有 state_dict 且无法按 model.json 重建/加载模型。") from exc
    return model, json_meta


def load_for_eval(checkpoint_path: Path, use_ema: bool = False) -> tuple[torch.nn.Module, dict]:
    """评估专用加载：必须包含 nn.Module，否则报错。use_ema=True 时优先加载 ema_state_dict（如存在）。"""
    state_dict, module, train_state, obj = load_checkpoint_artifacts(checkpoint_path)
    if use_ema and isinstance(train_state, dict) and train_state.get("ema_state_dict") is not None:
        LOGGER.info("Evaluate: use_ema=True，加载 ema_state_dict。")
        state_dict = train_state["ema_state_dict"]
    if module is None:
        raise ValueError("Checkpoint 缺少可用的 nn.Module（键 'model'），evaluate 仅支持包含模型实例的 checkpoint。")
    try:
        module.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        LOGGER.warning("nn.Module 严格加载 state_dict 失败，将尝试 non-strict：%s", exc)
        module.load_state_dict(state_dict, strict=False)
    return module, obj if isinstance(obj, dict) else {}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    train_state: dict,
    model_state_dict: dict | None = None,
    ema_state_dict: dict | None = None,
    best_state_dict: dict | None = None,
) -> None:
    module_copy = copy.deepcopy(model).cpu()
    state = model_state_dict or {k: v.cpu() for k, v in model.state_dict().items()}
    if ema_state_dict is not None:
        # 确保 train_state 带上 ema_state_dict 方便 resume
        train_state = dict(train_state)
        train_state["ema_state_dict"] = ema_state_dict
    torch.save(
        {
            "model_state_dict": state,
            "train_state": train_state,
            "best_model_state_dict": best_state_dict,
            "model": module_copy,
        },
        path,
    )


def save_best_model(
    path: Path,
    model: torch.nn.Module,
    best_state_dict: dict | None,
    fallback_state: dict | None = None,
    model_state_dict: dict | None = None,
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
        },
        path,
    )


__all__ = [
    "load_checkpoint_artifacts",
    "build_model_with_json",
    "load_for_eval",
    "save_checkpoint",
    "save_best_model",
]
