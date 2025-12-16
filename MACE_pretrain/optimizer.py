"""Optimizer/scheduler helpers with sensible param-group defaults."""

from __future__ import annotations

import logging
from typing import Iterable

import torch

LOGGER = logging.getLogger(__name__)


_DEFAULT_NO_DECAY_KEYWORDS = {
    "bias",
    "norm",
    "layernorm",
    "batchnorm",
    "scale",
    "shift",
    "running_mean",
    "running_var",
}


def _is_no_decay(name: str, param: torch.nn.Parameter, keywords: set[str]) -> bool:
    # 0D 标量：常用于物理缩放系数，默认不做衰减
    if param.ndim == 0:
        return True
    lower = name.lower()
    # 仅依赖命名关键字来决定 no_decay，避免一刀切地豁免所有 1D 权重
    return any(key in lower for key in keywords)


def build_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    extra_no_decay: Iterable[str] | None = None,
) -> list[dict]:
    keywords = set(_DEFAULT_NO_DECAY_KEYWORDS)
    if extra_no_decay:
        keywords.update([k.lower() for k in extra_no_decay])

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        target = no_decay_params if _is_no_decay(name, param, keywords) else decay_params
        target.append(param)

    param_groups: list[dict] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    total = len(decay_params) + len(no_decay_params)
    LOGGER.info(
        "参数分组：decay=%d (%.1f%%), no_decay=%d (%.1f%%)",
        len(decay_params),
        100.0 * len(decay_params) / total if total else 0.0,
        len(no_decay_params),
        100.0 * len(no_decay_params) / total if total else 0.0,
    )
    return param_groups


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    opt_name = getattr(cfg, "optimizer", "adamw").lower()
    lr = getattr(cfg, "lr", 1e-3)
    weight_decay = getattr(cfg, "weight_decay", 0.0)
    betas = getattr(cfg, "betas", (0.9, 0.999))
    eps = getattr(cfg, "eps", 1e-8)
    momentum = getattr(cfg, "momentum", 0.9)

    param_groups = build_param_groups(model, weight_decay, extra_no_decay=getattr(cfg, "no_decay_keywords", None))

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    LOGGER.info("构建优化器 %s | lr=%.3e weight_decay=%.2e", optimizer.__class__.__name__, lr, weight_decay)
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    sched_name = getattr(cfg, "scheduler", "plateau")
    if sched_name is None:
        LOGGER.info("未启用调度器。")
        return None, lambda *_args, **_kwargs: None
    sched_name = sched_name.lower()

    if sched_name == "plateau":
        factor = getattr(cfg, "lr_factor", 0.8)
        patience = getattr(cfg, "plateau_patience", getattr(cfg, "patience", 50))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )
        def scheduler_step(metric=None):
            scheduler.step(metric)
    elif sched_name == "step":
        step_size = getattr(cfg, "step_size", 30)
        gamma = getattr(cfg, "gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        def scheduler_step(metric=None):  # metric ignored
            scheduler.step()
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")

    LOGGER.info("构建调度器 %s", scheduler.__class__.__name__)
    return scheduler, scheduler_step


def load_optimizer_state(optimizer: torch.optim.Optimizer, state_dict, logger=LOGGER) -> bool:
    if not state_dict:
        return False
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("加载优化器状态失败，将使用新优化器：%s", exc)
        return False


def load_scheduler_state(scheduler, state_dict, logger=LOGGER) -> bool:
    if scheduler is None or not state_dict:
        return False
    try:
        scheduler.load_state_dict(state_dict)
        return True
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("加载调度器状态失败，将使用新调度器：%s", exc)
        return False


__all__ = [
    "build_param_groups",
    "build_optimizer",
    "build_scheduler",
    "load_optimizer_state",
    "load_scheduler_state",
]
