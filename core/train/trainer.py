"""Core training loop using adapter interface."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from core.ckpt.export import export_standard_artifacts
from core.ckpt.save_load import load_checkpoint, load_weights, save_best_model, save_checkpoint
from core.metrics import accumulate_metrics, finalize_metrics, init_metrics_state
from core.runner.layout import artifacts_dir, standard_checkpoint_paths
from core.runner.spec import BackendRunResult, CommonTaskSpec
from core.train.optim import build_optimizer, build_scheduler, load_optimizer_state, load_scheduler_state
from core.transforms import build_transform, TargetTransform
from core.data.dataloaders import build_dataloaders


LOGGER = logging.getLogger(__name__)


def _as_tensor(value, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if value is None:
            moved[key] = None
        elif isinstance(value, (int, float, bool, str)):
            moved[key] = value
        else:
            moved[key] = _as_tensor(value, device)
    return moved


def _batch_size(batch: Mapping[str, Any]) -> int:
    ptr = batch.get("ptr")
    if ptr is not None:
        ptr_t = torch.as_tensor(ptr)
        counts = ptr_t[1:] - ptr_t[:-1]
        return int(counts.numel())
    natoms = batch.get("natoms")
    if natoms is not None:
        natoms_t = torch.as_tensor(natoms)
        return int(natoms_t.numel())
    energy = batch.get("energy")
    if energy is not None:
        return int(torch.as_tensor(energy).numel())
    return 1


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 0:
        seconds = 0
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_progress(progress: float | None, width: int = 24) -> str:
    if progress is None:
        return "[?]"
    progress = max(0.0, min(1.0, progress))
    filled = int(round(progress * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _resolve_log_every(value: Any, total_steps: int | None) -> int | None:
    try:
        requested = int(value) if value is not None else 0
    except (TypeError, ValueError):
        requested = 0
    if requested > 0:
        return requested
    if total_steps is None or total_steps <= 0:
        return None
    return max(1, total_steps // 20)


def _resolve_progress_bar(value: Any) -> bool:
    if value is None:
        mode = "auto"
    elif isinstance(value, bool):
        mode = "tqdm" if value else "none"
    else:
        mode = str(value).strip().lower()
    if mode in {"none", "false", "0", "off"}:
        return False
    if tqdm is None:
        return False
    if mode == "auto":
        return sys.stderr.isatty()
    return True


def _init_loss_state() -> dict[str, float | None]:
    return {
        "energy_sum": 0.0,
        "energy_count": 0.0,
        "force_sum": 0.0,
        "force_count": 0.0,
        "energy_weight": None,
        "force_weight": None,
        "fallback_sum": 0.0,
        "fallback_count": 0.0,
    }


def _update_loss_state(
    state: dict[str, float | None],
    logs: Mapping[str, float] | None,
    loss: torch.Tensor,
    batch_size: int,
) -> None:
    if logs and "energy_loss_sum" in logs:
        state["energy_sum"] = float(state["energy_sum"] or 0.0) + float(logs.get("energy_loss_sum", 0.0))
        state["energy_count"] = float(state["energy_count"] or 0.0) + float(logs.get("energy_count", 0.0))
        state["force_sum"] = float(state["force_sum"] or 0.0) + float(logs.get("force_loss_sum", 0.0))
        state["force_count"] = float(state["force_count"] or 0.0) + float(logs.get("force_count", 0.0))
        if state["energy_weight"] is None and "energy_weight" in logs:
            state["energy_weight"] = float(logs.get("energy_weight", 1.0))
        if state["force_weight"] is None and "force_weight" in logs:
            state["force_weight"] = float(logs.get("force_weight", 1.0))
        return

    state["fallback_sum"] = float(state["fallback_sum"] or 0.0) + float(loss.item()) * batch_size
    state["fallback_count"] = float(state["fallback_count"] or 0.0) + float(batch_size)


def _finalize_loss(state: Mapping[str, float | None]) -> float:
    energy_count = float(state.get("energy_count") or 0.0)
    force_count = float(state.get("force_count") or 0.0)
    if energy_count > 0.0 or force_count > 0.0:
        energy_weight = float(state.get("energy_weight") or 1.0)
        force_weight = float(state.get("force_weight") or 1.0)
        energy_term = float(state.get("energy_sum") or 0.0) / energy_count if energy_count else 0.0
        force_term = float(state.get("force_sum") or 0.0) / force_count if force_count else 0.0
        return energy_weight * energy_term + force_weight * force_term

    fallback_count = float(state.get("fallback_count") or 0.0)
    if fallback_count > 0.0:
        return float(state.get("fallback_sum") or 0.0) / fallback_count
    return 0.0


def _apply_freeze_policy(adapter: Any, model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> None:
    policy = train_cfg.get("freeze")
    if policy is None:
        return
    policy_str = str(policy).strip().lower()
    if not policy_str or policy_str in {"none", "false", "0"}:
        return
    if policy_str != "head_only":
        raise ValueError(f"Unsupported train.freeze policy: {policy}")

    for param in model.parameters():
        param.requires_grad = False

    head_params_fn = getattr(adapter, "head_parameters", None)
    if head_params_fn is None:
        raise ValueError("train.freeze=head_only requires adapter.head_parameters")
    head_params = list(head_params_fn(model))
    if not head_params:
        raise ValueError("train.freeze=head_only found no head parameters to train")
    for param in head_params:
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOGGER.info("Freeze policy %s enabled: trainable=%d/%d", policy_str, trainable, total)


def _train_epoch(
    model: torch.nn.Module,
    adapter: Any,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    accum_steps: int,
    max_grad_norm: float | None,
    use_amp: bool,
    transform: TargetTransform,
    log_every: int | None = None,
    epoch: int | None = None,
    use_tqdm: bool = False,
    tqdm_mininterval: float = 0.5,
) -> tuple[float, dict[str, float]]:
    model.train()
    metrics_state = init_metrics_state()
    loss_state = _init_loss_state()

    total_steps = None
    try:
        total_steps = len(loader)
    except TypeError:
        total_steps = None
    log_every = _resolve_log_every(log_every, total_steps)
    progress_every = log_every or 1
    epoch_start = time.time()

    progress_bar = None
    iterable = loader
    if use_tqdm and tqdm is not None:
        desc = f"Train {epoch}" if epoch is not None else "Train"
        progress_bar = tqdm(
            loader,
            total=total_steps,
            desc=desc,
            ascii=True,
            dynamic_ncols=True,
            mininterval=tqdm_mininterval,
            leave=False,
        )
        iterable = progress_bar
        log_every = None

    optimizer.zero_grad(set_to_none=True)
    step = 0
    for step, cbatch_raw in enumerate(iterable, start=1):
        cbatch = transform.apply_batch(cbatch_raw)
        cbatch = _move_batch_to_device(cbatch, device)
        backend_batch = adapter.make_backend_batch(cbatch, device)

        with torch.set_grad_enabled(True):
            context = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else torch.enable_grad()
            with context:
                outputs = adapter.forward(model, backend_batch)
                loss, logs = adapter.loss(outputs, cbatch)
                loss_for_backward = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        if step % accum_steps == 0:
            if max_grad_norm:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = _batch_size(cbatch)
        _update_loss_state(loss_state, logs, loss, batch_size)

        outputs_detached = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()
        }
        outputs_physical = transform.inverse_outputs(outputs_detached, cbatch_raw)
        accumulate_metrics(metrics_state, outputs_physical, cbatch_raw)

        if progress_bar is not None and progress_every and (step % progress_every == 0 or step == 1):
            progress_bar.set_postfix_str(f"loss={float(loss.item()):.6f}")

        if log_every and (step % log_every == 0 or (total_steps and step == total_steps)):
            elapsed = time.time() - epoch_start
            avg_step = elapsed / step if step else 0.0
            eta = (total_steps - step) * avg_step if total_steps else None
            progress = step / total_steps if total_steps else None
            pct = progress * 100.0 if progress is not None else 0.0
            bar = _format_progress(progress)
            epoch_label = f"{epoch}" if epoch is not None else "?"
            LOGGER.info(
                "Train epoch %s %s step %d/%s (%.1f%%) loss=%.6f eta=%s",
                epoch_label,
                bar,
                step,
                total_steps if total_steps is not None else "?",
                pct,
                float(loss.item()),
                _format_duration(eta),
            )

    if progress_bar is not None:
        progress_bar.close()

    if step and step % accum_steps != 0:
        if max_grad_norm:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = _finalize_loss(loss_state)
    metrics = finalize_metrics(metrics_state)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def _eval_epoch(
    model: torch.nn.Module,
    adapter: Any,
    loader,
    device: torch.device,
    transform: TargetTransform,
    log_every: int | None = None,
    epoch: int | None = None,
    use_tqdm: bool = False,
    tqdm_mininterval: float = 0.5,
) -> tuple[float, dict[str, float]]:
    model.eval()
    metrics_state = init_metrics_state()
    loss_state = _init_loss_state()

    total_steps = None
    try:
        total_steps = len(loader)
    except TypeError:
        total_steps = None
    log_every = _resolve_log_every(log_every, total_steps)
    progress_every = log_every or 1
    epoch_start = time.time()

    progress_bar = None
    iterable = loader
    if use_tqdm and tqdm is not None:
        desc = f"Eval {epoch}" if epoch is not None else "Eval"
        progress_bar = tqdm(
            loader,
            total=total_steps,
            desc=desc,
            ascii=True,
            dynamic_ncols=True,
            mininterval=tqdm_mininterval,
            leave=False,
        )
        iterable = progress_bar
        log_every = None

    for step, cbatch_raw in enumerate(iterable, start=1):
        cbatch = transform.apply_batch(cbatch_raw)
        cbatch = _move_batch_to_device(cbatch, device)
        backend_batch = adapter.make_backend_batch(cbatch, device)
        needs_grad = cbatch.get("forces") is not None
        with torch.set_grad_enabled(needs_grad):
            outputs = adapter.forward(model, backend_batch)
            loss, logs = adapter.loss(outputs, cbatch)

        batch_size = _batch_size(cbatch)
        _update_loss_state(loss_state, logs, loss, batch_size)

        outputs_detached = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in outputs.items()
        }
        outputs_physical = transform.inverse_outputs(outputs_detached, cbatch_raw)
        accumulate_metrics(metrics_state, outputs_physical, cbatch_raw)

        if progress_bar is not None and progress_every and (step % progress_every == 0 or step == 1):
            progress_bar.set_postfix_str(f"loss={float(loss.item()):.6f}")

        if log_every and (step % log_every == 0 or (total_steps and step == total_steps)):
            elapsed = time.time() - epoch_start
            avg_step = elapsed / step if step else 0.0
            eta = (total_steps - step) * avg_step if total_steps else None
            progress = step / total_steps if total_steps else None
            pct = progress * 100.0 if progress is not None else 0.0
            bar = _format_progress(progress)
            epoch_label = f"{epoch}" if epoch is not None else "?"
            LOGGER.info(
                "Eval  epoch %s %s step %d/%s (%.1f%%) loss=%.6f eta=%s",
                epoch_label,
                bar,
                step,
                total_steps if total_steps is not None else "?",
                pct,
                float(loss.item()),
                _format_duration(eta),
            )

    if progress_bar is not None:
        progress_bar.close()

    avg_loss = _finalize_loss(loss_state)
    metrics = finalize_metrics(metrics_state)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def run_task(spec: CommonTaskSpec, adapter: Any, export_artifacts: bool = True) -> BackendRunResult:
    device = torch.device(spec.device)
    train_cfg = spec.train
    epochs = int(train_cfg.get("epochs", 1))
    accum_steps = int(train_cfg.get("accum_steps", 1))
    max_grad_norm = train_cfg.get("max_grad_norm")
    max_grad_norm = float(max_grad_norm) if max_grad_norm else None
    use_amp = bool(train_cfg.get("amp", False))
    use_amp = use_amp and device.type == "cuda"

    train_loader, val_loader, sampled_indices = build_dataloaders(spec.data, train_cfg)

    bundle = None
    loaded_from_manifest = False
    if spec.model_manifest and spec.mode in {"finetune", "resume"}:
        load_from_manifest = getattr(adapter, "load_from_manifest", None)
        if load_from_manifest is None:
            raise ValueError(f"Adapter {adapter.__class__.__name__} does not support manifest loading")
        bundle = load_from_manifest(spec.model_manifest, device="cpu", weights_path=spec.model_weights)
        model = bundle.model
        loaded_from_manifest = True
    else:
        model = adapter.build_model(train_cfg)
    adapter.select_head(train_cfg, model)
    model.to(device)

    transform = build_transform(
        train_cfg,
        manifest=bundle.manifest if bundle is not None else None,
        extras=bundle.extras if bundle is not None else None,
    )
    if getattr(transform, "fit", None):
        transform.fit(train_loader)

    _apply_freeze_policy(adapter, model, train_cfg)

    optimizer = build_optimizer(model, train_cfg)
    scheduler_obj, scheduler_step = build_scheduler(optimizer, train_cfg)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    start_epoch = 1
    best_metric = float("inf")
    best_epoch = 0

    if spec.mode in {"finetune", "resume"} and spec.model_in:
        if spec.mode == "resume":
            ckpt = load_checkpoint(spec.model_in, map_location="cpu")
            model.load_state_dict(load_weights(spec.model_in, map_location="cpu"), strict=False)
            load_optimizer_state(optimizer, ckpt.get("optimizer_state_dict"))
            if scheduler_step is not None:
                load_scheduler_state(scheduler_step, ckpt.get("scheduler_state_dict"))
            if scaler is not None and ckpt.get("scaler_state_dict"):
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", best_metric))
            best_epoch = int(ckpt.get("best_epoch", 0))
        else:
            if not loaded_from_manifest:
                model.load_state_dict(load_weights(spec.model_in, map_location="cpu"), strict=False)

    ckpt_paths = standard_checkpoint_paths(spec.run_dir)
    best_ckpt_path = ckpt_paths["best_model"]
    last_ckpt_path = ckpt_paths["checkpoint"]

    scheduler_patience = getattr(scheduler_obj, "patience", None)
    early_stop_factor = int(train_cfg.get("early_stop_factor", 0))
    early_stop_window = scheduler_patience * early_stop_factor if scheduler_patience and early_stop_factor > 0 else None

    save_every = int(train_cfg.get("save_every", 1))
    log_every = train_cfg.get("log_every")
    use_tqdm = _resolve_progress_bar(train_cfg.get("progress_bar"))
    tqdm_mininterval = float(train_cfg.get("progress_mininterval", 0.5))
    epoch_time_sum = 0.0
    epoch_count = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        train_loss, train_metrics = _train_epoch(
            model,
            adapter,
            train_loader,
            optimizer,
            scaler,
            device,
            accum_steps,
            max_grad_norm,
            use_amp,
            transform,
            log_every=log_every,
            epoch=epoch,
            use_tqdm=use_tqdm,
            tqdm_mininterval=tqdm_mininterval,
        )
        train_time = time.time() - epoch_start
        val_start = time.time()
        val_loss, val_metrics = _eval_epoch(
            model,
            adapter,
            val_loader,
            device,
            transform,
            log_every=log_every,
            epoch=epoch,
            use_tqdm=use_tqdm,
            tqdm_mininterval=tqdm_mininterval,
        )
        val_time = time.time() - val_start
        epoch_time = train_time + val_time
        epoch_time_sum += epoch_time
        epoch_count += 1
        eta_total = (epochs - epoch) * (epoch_time_sum / epoch_count)

        if scheduler_step is not None:
            scheduler_step(val_loss)

        LOGGER.info(
            "Epoch %4d | Train Loss %.6f | Val Loss %.6f | Val MAE(E %.6f) | Val RMSE (E %.6f, F %.6f)",
            epoch,
            train_loss,
            val_loss,
            val_metrics.get("energy_mae", 0.0),
            val_metrics.get("energy_rmse", 0.0),
            val_metrics.get("force_rmse", 0.0),
        )
        LOGGER.info(
            "Epoch %4d timing | train=%s | val=%s | total=%s | eta_total=%s",
            epoch,
            _format_duration(train_time),
            _format_duration(val_time),
            _format_duration(epoch_time),
            _format_duration(eta_total),
        )

        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            save_best_model(best_ckpt_path, model)

        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                last_ckpt_path,
                model,
                optimizer,
                scheduler_step.state_dict() if scheduler_step is not None else None,
                scaler.state_dict() if scaler is not None else None,
                epoch=epoch,
                best_metric=best_metric,
                best_epoch=best_epoch,
                config=train_cfg,
                normalizer=transform.state_dict(),
                extra={"sampled_indices": sampled_indices},
            )

        if early_stop_window is not None and (epoch - best_epoch) >= early_stop_window:
            LOGGER.info("Early stopping triggered (best epoch %d).", best_epoch)
            break

    # Final checkpoint snapshot
    save_checkpoint(
        last_ckpt_path,
        model,
        optimizer,
        scheduler_step.state_dict() if scheduler_step is not None else None,
        scaler.state_dict() if scaler is not None else None,
        epoch=epoch,
        best_metric=best_metric,
        best_epoch=best_epoch,
        config=train_cfg,
        normalizer=transform.state_dict(),
        extra={"sampled_indices": sampled_indices},
    )

    # Export standard artifacts
    if export_artifacts:
        if best_ckpt_path.exists():
            state_dict = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=False)
        export_standard_artifacts(
            adapter=adapter,
            model=model,
            cfg=train_cfg,
            output_dir=artifacts_dir(spec.run_dir),
            weights_name="best_model.pt",
            manifest_name="manifest.json",
            normalizer=transform.state_dict(),
            head=None,
        )

    return BackendRunResult(run_dir=spec.run_dir, checkpoint_path=last_ckpt_path, best_model_path=best_ckpt_path)
