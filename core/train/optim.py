"""Optimizer/scheduler helpers for core trainer."""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

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


def _get(cfg: Mapping[str, object], name: str, default: object) -> object:
    if cfg is None:
        return default
    return cfg.get(name, default)


def _is_no_decay(name: str, param: torch.nn.Parameter, keywords: set[str]) -> bool:
    if param.ndim <= 1:
        return True
    lower = name.lower()
    return any(key in lower for key in keywords)


def build_param_groups(
    model: torch.nn.Module,
    weight_decay: float,
    extra_no_decay: Iterable[str] | None = None,
) -> list[dict]:
    keywords = set(_DEFAULT_NO_DECAY_KEYWORDS)
    if extra_no_decay:
        keywords.update([str(k).lower() for k in extra_no_decay])

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
        "Param groups: decay=%d (%.1f%%), no_decay=%d (%.1f%%)",
        len(decay_params),
        100.0 * len(decay_params) / total if total else 0.0,
        len(no_decay_params),
        100.0 * len(no_decay_params) / total if total else 0.0,
    )
    return param_groups


def build_optimizer(model: torch.nn.Module, cfg: Mapping[str, object]) -> torch.optim.Optimizer:
    opt_name = str(_get(cfg, "optimizer", "adamw")).lower()
    lr = float(_get(cfg, "lr", 1e-3))
    weight_decay = float(_get(cfg, "weight_decay", 0.0))
    betas = _get(cfg, "betas", (0.9, 0.999))
    eps = float(_get(cfg, "eps", 1e-8))
    momentum = float(_get(cfg, "momentum", 0.9))

    param_groups = build_param_groups(model, weight_decay, extra_no_decay=_get(cfg, "no_decay_keywords", None))

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=0.0)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=0.0)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    LOGGER.info("Optimizer %s | lr=%.3e weight_decay=%.2e", optimizer.__class__.__name__, lr, weight_decay)
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Mapping[str, object]):
    sched_name = _get(cfg, "scheduler", "plateau")
    if sched_name is None:
        LOGGER.info("Scheduler disabled.")
        return None, lambda *_args, **_kwargs: None
    sched_name = str(sched_name).lower()

    if sched_name == "plateau":
        factor = float(_get(cfg, "lr_factor", 0.8))
        patience = int(_get(cfg, "plateau_patience", _get(cfg, "patience", 50)))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
        )

        def scheduler_step(metric=None):
            scheduler.step(metric)

    elif sched_name == "step":
        step_size = int(_get(cfg, "step_size", 30))
        gamma = float(_get(cfg, "gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        def scheduler_step(metric=None):
            _ = metric
            scheduler.step()

    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")

    scheduler_step.state_dict = scheduler.state_dict
    scheduler_step.load_state_dict = scheduler.load_state_dict

    LOGGER.info("Scheduler %s", scheduler.__class__.__name__)
    return scheduler, scheduler_step


def load_optimizer_state(optimizer: torch.optim.Optimizer, state_dict, logger=LOGGER) -> bool:
    if not state_dict:
        return False
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to load optimizer state: %s", exc)
        return False


def load_scheduler_state(scheduler, state_dict, logger=LOGGER) -> bool:
    if scheduler is None or not state_dict:
        return False
    try:
        scheduler.load_state_dict(state_dict)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to load scheduler state: %s", exc)
        return False


__all__ = [
    "build_param_groups",
    "build_optimizer",
    "build_scheduler",
    "load_optimizer_state",
    "load_scheduler_state",
]
