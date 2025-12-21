"""Finetune a MACE model from a provided model weight file."""

from __future__ import annotations
import sitecustomize  # noqa: F401



import argparse
import contextlib
import json
import logging
import types
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.serialization
try:  # PyG>=2.3
    from torch_geometric.loader import DataLoader as PYGDataLoader
except ImportError:  # PyG<=2.2
    from torch_geometric.dataloader import DataLoader as PYGDataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

# Ensure project root (MACE_pretrain) is on sys.path to import optimizer/dataloader from root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mace import tools
from mace.tools import torch_geometric as mace_tg

from dataloader import prepare_xyz_dataloaders, prepare_lmdb_dataloaders
from losses import compute_train_loss, init_metrics_state, accumulate_metrics, finalize_metrics
from optimizer import build_optimizer, build_scheduler, load_scheduler_state
from model_loader import (
    load_checkpoint_artifacts,
    build_model_with_json,
    save_checkpoint,
    save_best_model,
    canonical_json_text,
    hash_text,
    get_code_version,
    derive_run_name_from_checkpoint,
    resolve_input_json_path,
    resolve_output_json_paths,
    resolve_output_paths,
)

torch.serialization.add_safe_globals([slice])
torch.set_default_dtype(torch.float32)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a trained MACE model")
    parser.add_argument("--input_model", type=Path, required=True, help="起始模型权重路径（.pt 文件）")
    parser.add_argument("--input_json", type=Path, help="model.json 路径（默认按 input_model 推断）")
    parser.add_argument("--output_checkpoint", type=Path, required=True, help="checkpoint 输出路径（.pt 或目录）")
    parser.add_argument("--output_model", type=Path, required=True, help="best model 输出路径（.pt 或目录）")
    parser.add_argument("--reuse_scheduler_state", action="store_true", help="加载旧调度器状态（默认不加载，重新创建）")
    parser.add_argument("--reuse_indices", action="store_true", help="复用模型来源保存的 lmdb_indices（仅 LMDB）")
    parser.add_argument("--data_format", choices=["xyz", "lmdb"], help="数据格式，不指定则使用模型来源保存的 config")
    parser.add_argument("--xyz_dir", type=Path, help="XYZ 数据路径")
    parser.add_argument("--lmdb_train", type=Path, help="训练 LMDB 目录")
    parser.add_argument("--lmdb_val", type=Path, help="验证 LMDB 目录")
    parser.add_argument("--lmdb_train_max_samples", type=int, default=None)
    parser.add_argument("--lmdb_val_max_samples", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--train_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪阈值（<=0 禁用），默认 1.0。")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cutoff", type=float, help="默认沿用模型来源元数据")
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--force_weight", type=float, default=1000.0)
    parser.add_argument("--lr", type=float, default=1.0e-4, help="微调学习率")
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--num_interactions", type=int, help="默认沿用模型来源元数据")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--ema", dest="ema", action="store_true", help="启用 EMA（默认）")
    parser.add_argument("--no-ema", dest="ema", action="store_false", help="禁用 EMA")
    parser.add_argument("--elements", type=int, nargs="+", help="可选元素列表（LMDB）")
    parser.add_argument("--progress", dest="progress", action="store_true", help="显示进度条（默认）")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="关闭进度条")
    parser.add_argument("--plateau_patience", type=int, default=4, help="ReduceLROnPlateau 的 patience（默认 4）。")
    parser.add_argument("--early_stop_factor", type=int, default=5, help="early-stop 窗口 = 调度器 patience * early_stop_factor，设 0 关闭")
    parser.add_argument("--save_every", type=int, default=1, help="每 N 个 epoch 保存 checkpoint，0 关闭周期保存")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(ema=True, progress=True)
    return parser.parse_args()




def evaluate(
    model,
    loader,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics_state = init_metrics_state()

    for batch in loader:
        batch = batch.to(device)
        # Force computation relies on autograd, so keep gradients enabled.
        with torch.set_grad_enabled(True):
            outputs = model(batch.to_dict(), training=False, compute_force=True)
        # 评估阶段不反传，立刻切断图以节省显存，并过滤 None
        outputs = {k: v.detach() for k, v in outputs.items() if v is not None}
        loss = compute_train_loss(
            outputs,
            batch,
            types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight),
        )

        batch_size = batch.energy.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        accumulate_metrics(metrics_state, outputs, batch, cfg=types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight))

    if total_samples == 0:
        return {}

    finalized = finalize_metrics(metrics_state, energy_weight=energy_weight, force_weight=force_weight)
    finalized["loss"] = total_loss / total_samples
    return finalized


def train(
    model,
    optimizer: torch.optim.Optimizer,
    scheduler_step,
    scheduler_obj,
    train_loader,
    valid_loader,
    device: torch.device,
    epochs: int,
    energy_weight: float,
    force_weight: float,
    ema: ExponentialMovingAverage | None,
    show_progress: bool,
    early_stop_factor: int,
    save_every: int,
    output_checkpoint_path: Path | None,
    output_model_path: Path | None,
    config: dict | None,
    lmdb_indices: dict | None = None,
    clip_grad_norm: float | None = None,
    model_json_text: str | None = None,
    model_json_hash: str | None = None,
    code_version: dict | None = None,
    start_epoch: int = 1,
    best_state_dict: dict | None = None,
    best_val_loss: float = float("inf"),
    best_epoch: int = 0,
):
    last_state_dict: dict | None = None
    latest_train_state: dict | None = None
    last_epoch = start_epoch - 1

    scheduler_patience = getattr(scheduler_obj, "patience", None)
    if scheduler_patience is not None and early_stop_factor > 0:
        early_stop_window = scheduler_patience * early_stop_factor
    else:
        early_stop_window = None

    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        model.train()
        train_metrics_state = init_metrics_state()
        train_loss_sum = 0.0
        train_sample_count = 0

        iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            if show_progress
            else train_loader
        )
        for batch in iterator:
            batch = batch.to(device)
            batch_size = batch.energy.shape[0]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.to_dict(), training=True, compute_force=True)
            loss = compute_train_loss(
                outputs,
                batch,
                types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight),
            )
            loss.backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update()

            accumulate_metrics(train_metrics_state, outputs, batch, cfg=types.SimpleNamespace(energy_weight=energy_weight, force_weight=force_weight))
            train_loss_sum += loss.item() * batch_size
            train_sample_count += batch_size

        finalized_train = finalize_metrics(train_metrics_state, energy_weight=energy_weight, force_weight=force_weight)
        avg_train_loss = train_loss_sum / train_sample_count if train_sample_count > 0 else 0.0

        context = ema.average_parameters() if ema is not None else contextlib.nullcontext()
        with context:
            with torch.enable_grad():
                val_metrics = evaluate(
                    model,
                    valid_loader,
                    device,
                    energy_weight,
                    force_weight,
                )

        val_loss = float(val_metrics.get("loss", 0.0))
        val_energy_rmse = float(val_metrics.get("energy_rmse", 0.0))
        val_force_rmse = float(val_metrics.get("force_rmse", 0.0))
        val_energy_mae = float(val_metrics.get("energy_mae", 0.0))
        val_force_mae = float(val_metrics.get("force_mae", 0.0))
        val_energy_mae_cfg = float(val_metrics.get("energy_mae_cfg", 0.0))

        scheduler_step(val_loss)
        LOGGER.info(
            "Epoch %4d | Train Loss %.6f | Val Loss %.6f | Val RMSE (E/atom %.6f, F/comp %.6f) "
            "| Val MAE (E/atom %.6f, F/comp %.6f) | Val |E| cfg %.6f | LR %.6e",
            epoch,
            avg_train_loss,
            val_loss,
            val_energy_rmse,
            val_force_rmse,
            val_energy_mae,
            val_force_mae,
            val_energy_mae_cfg,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with ema.average_parameters() if ema is not None else torch.no_grad():
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
        last_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        train_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_step.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "lmdb_indices": lmdb_indices,
        }
        latest_train_state = train_state

        if output_checkpoint_path is not None:
            output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if output_model_path is not None:
            output_model_path.parent.mkdir(parents=True, exist_ok=True)
        if output_checkpoint_path is not None or output_model_path is not None:
            if val_loss <= best_val_loss and best_state_dict is not None:
                if output_model_path is not None:
                    model_run_name = derive_run_name_from_checkpoint(output_model_path)
                    save_best_model(
                        output_model_path,
                        model,
                        best_state_dict,
                        model_state_dict=best_state_dict,
                        model_json_text=model_json_text,
                        model_json_hash=model_json_hash,
                        code_version=code_version,
                        run_name=model_run_name,
                    )
                    LOGGER.info("Saved new best model to %s", output_model_path)
            if save_every > 0 and epoch % save_every == 0 and output_checkpoint_path is not None:
                ckpt_run_name = derive_run_name_from_checkpoint(output_checkpoint_path)
                save_checkpoint(
                    output_checkpoint_path,
                    model,
                    train_state,
                    model_state_dict=last_state_dict,
                    ema_state_dict=train_state.get("ema_state_dict"),
                    model_json_text=model_json_text,
                    model_json_hash=model_json_hash,
                    code_version=code_version,
                    run_name=ckpt_run_name,
                )
                LOGGER.info("Saved checkpoint at epoch %d to %s", epoch, output_checkpoint_path)

        if early_stop_window is not None:
            epochs_since_best = epoch - best_epoch
            if epochs_since_best >= early_stop_window:
                LOGGER.info(
                    "Early stopping triggered after %d epochs without val improvement "
                    "(best epoch %d, window %d).",
                    epochs_since_best,
                    best_epoch,
                    early_stop_window,
                )
                break

    if best_state_dict is None and last_state_dict is not None:
        best_state_dict = last_state_dict
    if last_state_dict is None:
        last_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    if latest_train_state is None:
        latest_train_state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_step.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "epoch": last_epoch,
            "best_val_loss": best_val_loss,
            "config": config,
            "lmdb_indices": lmdb_indices,
        }

    return best_state_dict, best_val_loss, {
        "model_state_dict": last_state_dict,
        "best_model_state_dict": best_state_dict,
        "best_val_loss": best_val_loss,
        "train_state": latest_train_state,
    }


def _resolve_paths_from_config(args: argparse.Namespace, config: dict | None) -> None:
    if config is None:
        return
    if args.data_format is None and "data_format" in config:
        args.data_format = config["data_format"]
    if args.xyz_dir is None and config.get("xyz_dir"):
        args.xyz_dir = Path(config["xyz_dir"])
    if args.lmdb_train is None and config.get("lmdb_train"):
        args.lmdb_train = Path(config["lmdb_train"])
    if args.lmdb_val is None and config.get("lmdb_val"):
        args.lmdb_val = Path(config["lmdb_val"])
    if args.lmdb_train_max_samples is None and config.get("lmdb_train_max_samples") is not None:
        args.lmdb_train_max_samples = config["lmdb_train_max_samples"]
    if args.lmdb_val_max_samples is None and config.get("lmdb_val_max_samples") is not None:
        args.lmdb_val_max_samples = config["lmdb_val_max_samples"]
    if args.batch_size == parser_defaults()["batch_size"] and "batch_size" in config:
        args.batch_size = config["batch_size"]
    if args.num_workers == parser_defaults()["num_workers"] and "num_workers" in config:
        args.num_workers = config["num_workers"]


def parser_defaults() -> dict:
    return {
        "batch_size": 16,
        "num_workers": 0,
    }


def main() -> None:
    args = parse_args()
    tools.set_seeds(args.seed)
    code_version = get_code_version()

    model_path = args.input_model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    output_checkpoint = args.output_checkpoint.expanduser()
    output_model = args.output_model.expanduser()
    resolved_ckpt, resolved_model = resolve_output_paths(
        output_checkpoint,
        output_model,
    )
    if resolved_ckpt is None or resolved_model is None:
        raise ValueError("必须提供 output_checkpoint 与 output_model。")

    model_json_path = resolve_input_json_path(model_path, args.input_json)
    output_json_paths = resolve_output_json_paths(output_checkpoint, output_model)

    ckpt_state_dict, ckpt_module, ckpt_train_state, ckpt_raw = load_checkpoint_artifacts(model_path)
    raw_config = ckpt_train_state.get("config") or (ckpt_raw.get("raw", {}).get("config") if isinstance(ckpt_raw, dict) else None)
    _resolve_paths_from_config(args, raw_config)

    # 如未显式提供，尽量用 model.json 填充关键超参
    with model_json_path.open("r", encoding="utf-8") as f:
        json_meta = json.load(f)
    for output_json in output_json_paths:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(json_meta, f, ensure_ascii=True, indent=2)
    model_json_text = canonical_json_text(json_meta)
    model_json_hash = hash_text(model_json_text)
    if args.cutoff is None and "cutoff" in json_meta:
        args.cutoff = float(json_meta["cutoff"])
    if args.num_interactions is None and "num_interactions" in json_meta:
        args.num_interactions = int(json_meta["num_interactions"])
    model_zs = [int(z) for z in json_meta.get("z_table", [])]
    z_table = tools.AtomicNumberTable(model_zs) if model_zs else None

    # 数据加载：完全信任 model.json，不再重新计算 E0/avg_num_neighbors
    resume_indices = ckpt_train_state.get("lmdb_indices") if args.reuse_indices else None
    lmdb_indices = None
    if args.data_format is None:
        raise ValueError("data_format 未指定，且模型来源 config 中缺失。")

    if args.data_format == "xyz":
        raise ValueError("finetune 目前仅在 LMDB 流程下跳过统计计算；如需 XYZ 微调请先实现无统计量加载。")
    elif args.data_format == "lmdb":
        if args.lmdb_train is None or args.lmdb_val is None:
            raise ValueError("--lmdb_train/--lmdb_val 必须提供或在模型来源 config 中存在")
        if z_table is None:
            raise ValueError("model.json 缺少 z_table，无法构建 LMDB dataloader。")
        train_loader, valid_loader, train_indices, val_indices = prepare_lmdb_dataloaders(
            args,
            z_table=z_table,
            resume_indices=resume_indices,
            coverage_zs=getattr(args, "elements", None),
            seed=args.seed,
        )
        lmdb_indices = {"train": train_indices, "val": val_indices}
    else:
        raise ValueError(f"Unsupported data format: {args.data_format}")

    # 权重来源：仅使用指定的模型权重
    module_fallback = ckpt_module
    state_dict_for_load = ckpt_state_dict
    load_path = model_path
    LOGGER.info("加载指定模型权重: %s", model_path)
    best_state_dict = None

    model, _ = build_model_with_json(
        model_json_path,
        load_path,
        state_dict_for_load,
        module_fallback,
        checkpoint_obj=ckpt_raw if isinstance(ckpt_raw, dict) else None,
    )

    # 权重：仅使用指定模型权重作为起始
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = build_optimizer(model, args)

    scheduler_obj, scheduler_step = build_scheduler(optimizer, args)
    if args.reuse_scheduler_state and ckpt_train_state.get("scheduler_state_dict") is not None:
        scheduler_state = ckpt_train_state.get("scheduler_state_dict")
        if scheduler_state is not None:
            load_scheduler_state(scheduler_obj, scheduler_state, LOGGER)
            LOGGER.info("已加载调度器状态。")

    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay) if args.ema else None
    if ema is not None:
        LOGGER.info("EMA 启用；微调不加载旧 EMA 状态。")
    else:
        LOGGER.info("EMA 关闭。")

    best_val_loss = float(ckpt_train_state.get("best_val_loss", float("inf")))

    best_state_dict, best_val_loss, last_ckpt = train(
        model=model,
        optimizer=optimizer,
        scheduler_step=scheduler_step,
        scheduler_obj=scheduler_obj,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=args.epochs,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        ema=ema,
        show_progress=args.progress,
        early_stop_factor=args.early_stop_factor,
        save_every=args.save_every,
        output_checkpoint_path=resolved_ckpt,
        output_model_path=resolved_model,
        config=vars(args),
        lmdb_indices=lmdb_indices,
        clip_grad_norm=args.clip_grad_norm,
        model_json_text=model_json_text,
        model_json_hash=model_json_hash,
        code_version=code_version,
        start_epoch=1,
        best_state_dict=best_state_dict,
        best_val_loss=best_val_loss,
        best_epoch=0,
    )

    if resolved_ckpt is not None or resolved_model is not None:
        final_train_state = last_ckpt.get("train_state") or {}
        final_train_state.setdefault("config", vars(args))
        final_train_state.setdefault("lmdb_indices", lmdb_indices)
        final_train_state.setdefault("epoch", args.epochs)
        final_model_state = last_ckpt.get("model_state_dict") or {k: v.cpu() for k, v in model.state_dict().items()}
        final_best = best_state_dict or final_model_state

        if resolved_ckpt is not None:
            ckpt_run_name = derive_run_name_from_checkpoint(resolved_ckpt)
            save_checkpoint(
                resolved_ckpt,
                model,
                final_train_state,
                model_state_dict=final_model_state,
                ema_state_dict=final_train_state.get("ema_state_dict"),
                model_json_text=model_json_text,
                model_json_hash=model_json_hash,
                code_version=code_version,
                run_name=ckpt_run_name,
            )
        if resolved_model is not None:
            model_run_name = derive_run_name_from_checkpoint(resolved_model)
            save_best_model(
                resolved_model,
                model,
                final_best,
                model_state_dict=final_best,
                model_json_text=model_json_text,
                model_json_hash=model_json_hash,
                code_version=code_version,
                run_name=model_run_name,
            )
        LOGGER.info(
            "Finetune 完成，checkpoint 保存在 %s，best 模型保存在 %s，best_val_loss=%.6f",
            resolved_ckpt,
            resolved_model,
            best_val_loss,
        )


if __name__ == "__main__":
    main()
