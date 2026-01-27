"""Sampling entrypoint for MACE AL experiments (skeleton)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from experiments.mace_pretrain.force_fn import build_force_fn
from experiments.sampling.actions.dihedral_twist import DihedralTwistAction
from experiments.sampling.actions.jitter import JitterAction
from experiments.sampling.actions.md import MDAction
from experiments.sampling.actions.push_pull import PushPullAction
from experiments.sampling.actions.rigid_rotate import RigidRotateAction
from experiments.sampling.actions.rigid_translate import RigidTranslateAction
from experiments.sampling.basin import HistogramBasin
from experiments.sampling.graph.registry import BasinRegistry
from experiments.sampling.pipeline import SamplingPipeline
from experiments.sampling.plugins import (
    build_action_plugins,
    build_recorders,
    build_stoppers,
    build_trigger,
    build_validators,
    list_output_files,
    recorder_enabled,
    split_action_pipeline_config,
)
from experiments.sampling.quench.ase_bfgs import ASEBFGSQuench
from experiments.sampling.quench.ase_cg import ASECGQuench
from experiments.sampling.quench.ase_fire import ASEFIREQuench
from experiments.sampling.quench.ase_lbfgs import ASELBFGSQuench
from experiments.sampling.quench.force_fn_calculator import ForceFnCalculator
from experiments.sampling.selection import build_action_inputs
from experiments.sampling.stoppers import SamplingState
from experiments.sampling.structure_store import StructureStore


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _load_structure(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MACE AL sampling entrypoint")
    parser.add_argument("--run_dir", required=True, help="Run directory for outputs")
    parser.add_argument("--config", default=None, help="Sampling config YAML (optional)")
    parser.add_argument("--structure_json", required=True, help="Input structure JSON (numbers/positions/...)")
    parser.add_argument("--steps", type=int, default=200, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true", help="Resume by appending outputs")
    parser.add_argument("--manifest", required=True, help="Model manifest.json path")
    parser.add_argument("--weights", default=None, help="Override weights path (optional)")
    parser.add_argument("--device", default="cuda", help="Device for inference (cuda/cpu)")
    parser.add_argument("--head", default=None, help="Model head override (optional)")
    parser.add_argument("--amp", action="store_true", help="Enable AMP for model inference")
    parser.add_argument(
        "--quench",
        default="none",
        choices=["none", "fire", "cg", "bfgs", "lbfgs"],
        help="Quench mode",
    )
    parser.add_argument("--quench_fmax", type=float, default=0.1, help="Quench fmax (eV/A)")
    parser.add_argument("--quench_steps", type=int, default=200, help="Quench max steps")
    parser.add_argument("--target_basins", type=int, default=None, help="Stop after N unique basins")
    return parser.parse_args()


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_attempts(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    rejected = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("valid") is False:
                rejected += 1
    return total, rejected


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else Path("experiments/sampling_rules/config.yaml")
    raw_config = _load_yaml(config_path)
    config, pipeline_cfg = split_action_pipeline_config(raw_config)

    data = _load_structure(Path(args.structure_json))
    from experiments.sampling.schema import Structure as SamplingStructure

    structure = SamplingStructure(
        positions=np.asarray(data["positions"], dtype=float),
        numbers=np.asarray(data["numbers"], dtype=int),
        tags=np.asarray(data.get("tags")) if "tags" in data else None,
        fixed=np.asarray(data.get("fixed")) if "fixed" in data else None,
        cell=np.asarray(data.get("cell")) if "cell" in data else None,
        pbc=np.asarray(data.get("pbc")) if "pbc" in data else None,
        info=data.get("info", {}),
    )

    if not args.resume:
        for path in list_output_files(pipeline_cfg, run_dir):
            if path.exists():
                raise ValueError("Output exists; rerun with --resume or remove previous outputs")

    store = StructureStore(run_dir / "structures")
    trigger = build_trigger(pipeline_cfg)
    registry = BasinRegistry()
    recorders = build_recorders(
        pipeline_cfg,
        run_dir=run_dir,
        store=store,
        registry=registry,
        trigger=trigger,
    )
    validators = build_validators(pipeline_cfg, structure=structure, action_config=config)

    force_fn = build_force_fn(
        manifest_path=args.manifest,
        weights_path=args.weights,
        device=args.device,
        head=args.head,
        use_amp=args.amp,
    )

    calculator = ForceFnCalculator(force_fn, fixed=structure.fixed, tags=structure.tags)

    actions = [
        RigidTranslateAction(),
        RigidRotateAction(),
        PushPullAction(),
        DihedralTwistAction(),
        JitterAction(),
    ]
    md_cfg = config.get("md") if isinstance(config.get("md"), dict) else None
    if md_cfg and bool(md_cfg.get("enabled", True)):
        md_action = MDAction(calculator)
        if md_action.available:
            actions.append(md_action)
        else:
            print("WARNING: md action enabled but ASE is unavailable; skipping MDAction")

    action_plugins = build_action_plugins(pipeline_cfg, structure=structure, action_config=config)

    quench = None
    if args.quench != "none":
        if args.quench == "fire":
            quench = ASEFIREQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run
        elif args.quench == "cg":
            quench = ASECGQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run
        elif args.quench == "bfgs":
            quench = ASEBFGSQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run
        elif args.quench == "lbfgs":
            quench = ASELBFGSQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run

    basin = HistogramBasin()
    if args.target_basins is not None and not recorder_enabled(pipeline_cfg, "basin_registry"):
        raise ValueError("target_basins requires basin_registry recorder to be enabled")
    pipeline = SamplingPipeline(
        actions=actions,
        quench=quench,
        force_fn=force_fn,
        basin=basin,
        validators=validators,
        action_plugins=action_plugins,
        recorders=recorders,
        rng=np.random.default_rng(args.seed),
        config=config,
    )

    total_steps = 0
    converged = 0
    unconverged = 0
    start = time.monotonic()
    state = SamplingState()
    stoppers = build_stoppers(pipeline_cfg, max_steps=args.steps, target_basins=args.target_basins)
    stop_reason = None
    while True:
        selection_mask, candidates = build_action_inputs(structure)
        record = pipeline.run_one(structure, selection_mask=selection_mask, candidates=candidates)
        total_steps += 1
        state.step_idx = total_steps
        if recorder_enabled(pipeline_cfg, "basin_registry"):
            state.basins = len(registry.basins)
        elif record.basin is not None and record.basin.is_new:
            state.basins += 1
        if record.quench is not None:
            if record.quench.converged:
                converged += 1
            else:
                unconverged += 1
        for stopper in stoppers:
            reason = stopper.on_sample(record, state)
            if reason:
                stop_reason = reason
                break
        if stop_reason:
            break
    elapsed = time.monotonic() - start

    dft_count = _count_jsonl(run_dir / "dft_queue.jsonl")
    basin_count = len(registry.basins)
    attempts_total, attempts_rejected = _count_attempts(run_dir / "steps.jsonl")
    print(
        f"Sampling finished: steps={total_steps}, basins={basin_count}, "
        f"quench_converged={converged}, quench_unconverged={unconverged}, "
        f"dft_candidates={dft_count}, elapsed={elapsed:.1f}s, stop_reason={stop_reason}"
    )
    print(f"- attempts_total: {attempts_total}")
    print(f"- attempts_rejected: {attempts_rejected}")


if __name__ == "__main__":
    main()
