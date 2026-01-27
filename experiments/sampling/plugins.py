"""Pipeline plugin builders (recorders/validators/triggers/stoppers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments.mace_pretrain.selector import (
    build_any_trigger,
    build_default_trigger,
    build_max_force_trigger,
    build_topk_force_trigger,
)
from experiments.sampling.graph.registry import BasinRegistry
from experiments.sampling.recorders import (
    ALCandidateRecorder,
    BasinRegistryRecorder,
    StepTraceRecorder,
)
from experiments.sampling.stoppers import MaxStepsStopper, SamplingState, StopperBase, TargetBasinsStopper
from experiments.sampling.structure_store import StructureStore
from experiments.action_quality import noise as noise_mod
from experiments.action_quality import validate as validate_mod


def _as_list(value: Optional[Iterable[Any]]) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _action_ranges_from_config(action_cfg: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not action_cfg:
        return {}
    ranges: Dict[str, Dict[str, float]] = {}
    if isinstance(action_cfg.get("rigid_translate"), dict):
        cfg = action_cfg["rigid_translate"]
        if "min_step" in cfg or "max_step" in cfg:
            ranges["rigid_translate"] = {"min": cfg.get("min_step"), "max": cfg.get("max_step")}
    if isinstance(action_cfg.get("rigid_rotate"), dict):
        cfg = action_cfg["rigid_rotate"]
        if "min_deg" in cfg or "max_deg" in cfg:
            ranges["rigid_rotate"] = {"min": cfg.get("min_deg"), "max": cfg.get("max_deg")}
    if isinstance(action_cfg.get("dihedral_twist"), dict):
        cfg = action_cfg["dihedral_twist"]
        if "min_deg" in cfg or "max_deg" in cfg:
            ranges["dihedral_twist"] = {"min": cfg.get("min_deg"), "max": cfg.get("max_deg")}
    if isinstance(action_cfg.get("push_pull"), dict):
        cfg = action_cfg["push_pull"]
        if "min_delta" in cfg or "max_delta" in cfg:
            ranges["push_pull"] = {"min": cfg.get("min_delta"), "max": cfg.get("max_delta")}
    if isinstance(action_cfg.get("jitter"), dict):
        cfg = action_cfg["jitter"]
        if "sigma" in cfg:
            ranges["jitter"] = {"min": 0.0, "max": cfg.get("sigma")}
    if isinstance(action_cfg.get("md"), dict):
        cfg = action_cfg["md"]
        if "min_temp_K" in cfg or "max_temp_K" in cfg:
            ranges["md"] = {"min": cfg.get("min_temp_K"), "max": cfg.get("max_temp_K")}
    return ranges


def split_action_pipeline_config(config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pipeline_cfg = config.get("pipeline") or {}
    action_cfg = {k: v for k, v in config.items() if k != "pipeline"}
    return action_cfg, pipeline_cfg


def build_trigger(pipeline_cfg: Optional[Dict[str, Any]]):
    if not pipeline_cfg:
        return build_default_trigger()
    trig_cfg = pipeline_cfg.get("trigger")
    if trig_cfg is None:
        return build_default_trigger()
    if isinstance(trig_cfg, str):
        if trig_cfg == "default":
            return build_default_trigger()
        raise ValueError(f"Unknown trigger preset: {trig_cfg}")
    trig_type = trig_cfg.get("type", "default")
    if trig_type == "default":
        return build_default_trigger(
            max_threshold=float(trig_cfg.get("max_threshold", 0.7)),
            topk_threshold=float(trig_cfg.get("topk_threshold", 0.35)),
            k=int(trig_cfg.get("k", 5)),
            source=str(trig_cfg.get("source", "force_pre")),
        )
    if trig_type == "max_force":
        return build_max_force_trigger(
            threshold=float(trig_cfg.get("threshold", 0.7)),
            source=str(trig_cfg.get("source", "force_pre")),
        )
    if trig_type == "topk_force":
        return build_topk_force_trigger(
            threshold=float(trig_cfg.get("threshold", 0.35)),
            k=int(trig_cfg.get("k", 5)),
            source=str(trig_cfg.get("source", "force_pre")),
        )
    if trig_type == "any":
        items = _as_list(trig_cfg.get("items"))
        if not items:
            return build_default_trigger()
        triggers = []
        for item in items:
            if isinstance(item, str) and item == "default":
                triggers.append(build_default_trigger())
                continue
            if not isinstance(item, dict):
                raise ValueError(f"Invalid trigger item: {item}")
            ttype = item.get("type", "default")
            if ttype == "max_force":
                triggers.append(
                    build_max_force_trigger(
                        threshold=float(item.get("threshold", 0.7)),
                        source=str(item.get("source", "force_pre")),
                    )
                )
            elif ttype == "topk_force":
                triggers.append(
                    build_topk_force_trigger(
                        threshold=float(item.get("threshold", 0.35)),
                        k=int(item.get("k", 5)),
                        source=str(item.get("source", "force_pre")),
                    )
                )
            else:
                triggers.append(build_default_trigger())
        return build_any_trigger(triggers)
    raise ValueError(f"Unknown trigger type: {trig_type}")


def list_output_files(pipeline_cfg: Optional[Dict[str, Any]], run_dir: Path) -> List[Path]:
    if not pipeline_cfg or not pipeline_cfg.get("recorders"):
        return [
            run_dir / "steps.jsonl",
            run_dir / "basins.jsonl",
            run_dir / "dft_queue.jsonl",
            run_dir / "structures" / "index.jsonl",
        ]
    paths: List[Path] = []
    for spec in _as_list(pipeline_cfg.get("recorders")):
        if isinstance(spec, str):
            name = spec
            enabled = True
            params = {}
        else:
            name = str(spec.get("name"))
            enabled = bool(spec.get("enabled", True))
            params = spec
        if not enabled:
            continue
        if name == "step_trace":
            paths.append(run_dir / str(params.get("path", "steps.jsonl")))
        elif name == "basin_registry":
            paths.append(run_dir / str(params.get("path", "basins.jsonl")))
        elif name == "dft_queue":
            paths.append(run_dir / str(params.get("path", "dft_queue.jsonl")))
    if paths:
        paths.append(run_dir / "structures" / "index.jsonl")
    return paths


def recorder_enabled(pipeline_cfg: Optional[Dict[str, Any]], name: str) -> bool:
    if not pipeline_cfg or not pipeline_cfg.get("recorders"):
        return name in {"step_trace", "basin_registry", "dft_queue"}
    for spec in _as_list(pipeline_cfg.get("recorders")):
        if isinstance(spec, str):
            if spec == name:
                return True
        else:
            if str(spec.get("name")) == name:
                return bool(spec.get("enabled", True))
    return False


def build_recorders(
    pipeline_cfg: Optional[Dict[str, Any]],
    *,
    run_dir: Path,
    store: StructureStore,
    registry: BasinRegistry,
    trigger,
) -> List:
    if not pipeline_cfg or not pipeline_cfg.get("recorders"):
        return [
            StepTraceRecorder(run_dir / "steps.jsonl", structure_store=store),
            BasinRegistryRecorder(run_dir / "basins.jsonl", structure_store=store, registry=registry),
            ALCandidateRecorder(run_dir / "dft_queue.jsonl", trigger_fn=trigger, structure_store=store),
        ]
    recorders: List = []
    for spec in _as_list(pipeline_cfg.get("recorders")):
        if isinstance(spec, str):
            name = spec
            params = {}
            enabled = True
        else:
            name = str(spec.get("name"))
            params = spec
            enabled = bool(spec.get("enabled", True))
        if not enabled:
            continue
        if name == "step_trace":
            recorders.append(
                StepTraceRecorder(
                    run_dir / str(params.get("path", "steps.jsonl")),
                    include_structures=bool(params.get("include_structures", False)),
                    round_decimals=params.get("round_decimals", 4),
                    structure_store=store,
                )
            )
        elif name == "basin_registry":
            recorders.append(
                BasinRegistryRecorder(
                    run_dir / str(params.get("path", "basins.jsonl")),
                    structure_store=store,
                    registry=registry,
                    round_decimals=params.get("round_decimals", 4),
                )
            )
        elif name == "dft_queue":
            recorders.append(
                ALCandidateRecorder(
                    run_dir / str(params.get("path", "dft_queue.jsonl")),
                    trigger_fn=trigger,
                    stages=tuple(params.get("stages", ("sample", "quench_step"))),
                    round_decimals=params.get("round_decimals", 4),
                    structure_store=store,
                )
            )
        else:
            raise ValueError(f"Unknown recorder name: {name}")
    return recorders


def build_action_plugins(
    pipeline_cfg: Optional[Dict[str, Any]],
    *,
    structure,
    action_config: Optional[Dict[str, Any]] = None,
) -> List:
    plugins: List = []
    if not pipeline_cfg:
        return plugins
    specs = _as_list(pipeline_cfg.get("action_plugins"))
    for spec in specs:
        if isinstance(spec, str):
            name = spec
            params: Dict[str, Any] = {}
        else:
            name = str(spec.get("name"))
            params = spec
        if name == "noise":
            sigma = float(params.get("sigma", 0.0))
            clip = params.get("clip")
            movable_only = bool(params.get("movable_only", True))
            if sigma > 0.0:
                plugins.append(noise_mod.make_position_noise(sigma=sigma, clip=clip, movable_only=movable_only))
        else:
            raise ValueError(f"Unknown action plugin name: {name}")
    return plugins


def build_validators(
    pipeline_cfg: Optional[Dict[str, Any]],
    *,
    structure,
    action_config: Optional[Dict[str, Any]] = None,
) -> List:
    validators: List = []
    if not pipeline_cfg:
        return validators
    specs = _as_list(pipeline_cfg.get("validators"))
    for spec in specs:
        if isinstance(spec, str):
            name = spec
            params = {}
        else:
            name = str(spec.get("name"))
            params = spec
        if name == "min_dist":
            validators.append(
                validate_mod.make_min_dist(
                    min_factor=float(params.get("min_factor", 0.7)),
                    hard_min=float(params.get("hard_min", 0.7)),
                )
            )
        elif name == "fixed":
            if structure.fixed is None:
                raise ValueError("fixed validator requires structure.fixed mask")
            validators.append(
                validate_mod.make_fixed(
                    structure.positions,
                    structure.fixed,
                    tol=float(params.get("tol", 1e-3)),
                )
            )
        elif name == "bond":
            bond_pairs = params.get("bond_pairs") or []
            validators.append(
                validate_mod.make_bond(
                    structure.positions,
                    bond_pairs,
                    max_factor=float(params.get("max_factor", 1.5)),
                    max_abs=params.get("max_abs"),
                )
            )
        elif name == "quality":
            action_ranges = params.get("action_ranges")
            if action_ranges is None:
                action_ranges = _action_ranges_from_config(action_config)
            validators.append(
                validate_mod.make_quality_gate(
                    force_source=str(params.get("force_source", "force_pre")),
                    max_force=params.get("max_force"),
                    min_force=params.get("min_force"),
                    score_threshold=float(params.get("score_threshold", 1.0)),
                    action_ranges=action_ranges,
                )
            )
        else:
            raise ValueError(f"Unknown validator name: {name}")
    return validators


def build_stoppers(
    pipeline_cfg: Optional[Dict[str, Any]],
    *,
    max_steps: Optional[int],
    target_basins: Optional[int],
) -> List[StopperBase]:
    stoppers: List[StopperBase] = []
    specs = _as_list((pipeline_cfg or {}).get("stoppers"))
    for spec in specs:
        if isinstance(spec, str):
            name = spec
            params = {}
        else:
            name = str(spec.get("name"))
            params = spec
        if name == "max_steps":
            stoppers.append(MaxStepsStopper(int(params.get("value", params.get("max_steps", 0)))))
        elif name == "target_basins":
            stoppers.append(TargetBasinsStopper(int(params.get("value", params.get("target", 0)))))
        else:
            raise ValueError(f"Unknown stopper name: {name}")
    if max_steps is not None:
        stoppers.append(MaxStepsStopper(int(max_steps)))
    if target_basins is not None:
        stoppers.append(TargetBasinsStopper(int(target_basins)))
    return stoppers
