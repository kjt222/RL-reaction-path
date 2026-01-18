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
from experiments.mace_pretrain.selector import build_default_trigger
from experiments.sampling.actions.dihedral_twist import DihedralTwistAction
from experiments.sampling.actions.jitter import JitterAction
from experiments.sampling.actions.push_pull import PushPullAction
from experiments.sampling.actions.rigid_rotate import RigidRotateAction
from experiments.sampling.actions.rigid_translate import RigidTranslateAction
from experiments.sampling.basin import HistogramBasin
from experiments.sampling.graph.registry import BasinRegistry
from experiments.sampling.pipeline import SamplingPipeline
from experiments.sampling.quench.ase_fire import ASEFIREQuench
from experiments.sampling.quench.ase_lbfgs import ASELBFGSQuench
from experiments.sampling.quench.force_fn_calculator import ForceFnCalculator
from experiments.sampling.recorders import ALCandidateRecorder, BasinRegistryRecorder, StepTraceRecorder
from experiments.sampling.selection import build_action_inputs
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
    parser.add_argument("--quench", default="none", choices=["none", "fire", "lbfgs"], help="Quench mode")
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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else Path("experiments/sampling_rules/config.yaml")
    config = _load_yaml(config_path)

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
        for filename in ("steps.jsonl", "basins.jsonl", "dft_queue.jsonl", "structures/index.jsonl"):
            if (run_dir / filename).exists():
                raise ValueError("Output exists; rerun with --resume or remove previous outputs")

    store = StructureStore(run_dir / "structures")
    trigger = build_default_trigger()
    registry = BasinRegistry()
    recorders = [
        StepTraceRecorder(run_dir / "steps.jsonl", structure_store=store),
        BasinRegistryRecorder(run_dir / "basins.jsonl", structure_store=store, registry=registry),
        ALCandidateRecorder(run_dir / "dft_queue.jsonl", trigger_fn=trigger, structure_store=store),
    ]

    actions = [
        RigidTranslateAction(),
        RigidRotateAction(),
        PushPullAction(),
        DihedralTwistAction(),
        JitterAction(),
    ]

    force_fn = build_force_fn(
        manifest_path=args.manifest,
        weights_path=args.weights,
        device=args.device,
        head=args.head,
        use_amp=args.amp,
    )

    quench = None
    if args.quench != "none":
        calculator = ForceFnCalculator(force_fn, fixed=structure.fixed, tags=structure.tags)
        if args.quench == "fire":
            quench = ASEFIREQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run
        elif args.quench == "lbfgs":
            quench = ASELBFGSQuench(calculator, fmax=args.quench_fmax, steps=args.quench_steps).run

    basin = HistogramBasin()
    pipeline = SamplingPipeline(
        actions=actions,
        quench=quench,
        force_fn=force_fn,
        basin=basin,
        validators=[],
        recorders=recorders,
        rng=np.random.default_rng(args.seed),
        config=config,
    )

    total_steps = 0
    converged = 0
    unconverged = 0
    start = time.monotonic()
    for _ in range(int(args.steps)):
        selection_mask, candidates = build_action_inputs(structure)
        record = pipeline.run_one(structure, selection_mask=selection_mask, candidates=candidates)
        total_steps += 1
        if record.quench is not None:
            if record.quench.converged:
                converged += 1
            else:
                unconverged += 1
        if args.target_basins is not None and len(registry.basins) >= int(args.target_basins):
            break
    elapsed = time.monotonic() - start

    dft_count = _count_jsonl(run_dir / "dft_queue.jsonl")
    basin_count = len(registry.basins)
    print(
        f"Sampling finished: steps={total_steps}, basins={basin_count}, "
        f"quench_converged={converged}, quench_unconverged={unconverged}, "
        f"dft_candidates={dft_count}, elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
