"""Render sampling visualization (MP4 + extxyz) from viz_steps.jsonl using OVITO."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.sampling.structure_store import StructureStore

warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")

try:
    from ovito.io import import_file as ovito_import_file
    from ovito.modifiers import ColorByTypeModifier, CreateBondsModifier
    from ovito.vis import TachyonRenderer, TextLabelOverlay, Viewport
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OVITO Python API not found. Install via: conda install conda-forge::ovito"
    ) from exc


def _load_steps(path: Path) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    return steps


def _resolve_structure(ref: Dict[str, Any], run_dir: Path) -> Optional[Atoms]:
    if not ref:
        return None
    path = Path(ref["path"])
    if not path.is_absolute():
        path = run_dir / path
        if not path.exists():
            alt = run_dir / "viz" / Path(ref["path"])
            if alt.exists():
                path = alt
    structure = StructureStore.load_npz(path)
    atoms = Atoms(
        numbers=np.asarray(structure.numbers, dtype=int),
        positions=np.asarray(structure.positions, dtype=float),
        cell=structure.cell,
        pbc=structure.pbc,
    )
    if structure.tags is not None:
        atoms.set_tags(structure.tags)
    if structure.fixed is not None:
        atoms.arrays["fixed"] = np.asarray(structure.fixed, dtype=int)
    return atoms


def _format_trigger(meta: Optional[Dict[str, Any]]) -> Optional[str]:
    if not meta:
        return None
    if meta.get("reason") == "any" and isinstance(meta.get("checks"), list):
        selected = meta.get("selected")
        if selected is None:
            return None
        try:
            meta = meta["checks"][int(selected)]
        except Exception:
            return None
    reason = meta.get("reason")
    value = meta.get("value")
    threshold = meta.get("threshold")
    source = meta.get("source")
    label = None
    if reason == "max_F":
        label = "max_F"
    elif reason == "topk_mean":
        k = meta.get("k")
        label = f"top{k}_mean"
    else:
        label = str(reason) if reason else "trigger"
    if value is None or threshold is None:
        return None
    if source:
        return f"{label}({source})={value:.3f} > {threshold:.3f}"
    return f"{label}={value:.3f} > {threshold:.3f}"


def _attach_info(atoms: Atoms, step: Dict[str, Any]) -> None:
    basin_id = step.get("basin_id") or "NA"
    atoms.info["stage"] = step.get("stage")
    atoms.info["action"] = step.get("action")
    atoms.info["basin_id"] = basin_id
    if step.get("basin_is_new") is not None:
        atoms.info["is_new"] = int(bool(step.get("basin_is_new")))
    if step.get("quench_step") is not None:
        atoms.info["quench_step"] = int(step.get("quench_step"))
    if step.get("triggered"):
        atoms.info["trigger_reason"] = _format_trigger(step.get("trigger"))


def _render_frames_ovito(
    *,
    trajectory_path: Path,
    overlays: List[Dict[str, Any]],
    out_dir: Path,
    size: tuple[int, int],
    renderer_name: str,
    log_every: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ovito_import_file(str(trajectory_path), multiple_frames=True)
    try:
        pipeline.modifiers.append(CreateBondsModifier(mode=CreateBondsModifier.Mode.Auto))
    except Exception:
        # Bonds are optional; keep rendering even if auto-bonds not supported.
        pass
    pipeline.add_to_scene()

    # Element-based coloring (VESTA-like) + smaller atoms.
    try:
        pipeline.modifiers.append(ColorByTypeModifier())
        if hasattr(pipeline.source.data, "particles") and pipeline.source.data.particles is not None:
            pipeline.source.data.particles.vis.radius = 0.35
    except Exception:
        pass

    viewport = Viewport(type=Viewport.Type.ORTHO)
    # Oblique view (not aligned to x/y/z axes).
    viewport.camera_dir = (1.0, 1.0, -1.0)
    viewport.camera_up = (0.0, 0.0, 1.0)
    viewport.zoom_all(size)
    max_lines = max(len(frame["lines"]) for frame in overlays)
    overlay_objs: List[TextLabelOverlay] = []
    base_y = 0.08
    line_gap = 0.03
    for idx in range(max_lines):
        label = TextLabelOverlay(text="", offset_x=0.02, offset_y=base_y + idx * line_gap)
        label.font_size = 0.010
        label.text_color = (0.1, 0.1, 0.1)
        viewport.overlays.append(label)
        overlay_objs.append(label)

    phase_label = TextLabelOverlay(text="", offset_x=0.02, offset_y=0.03)
    phase_label.font_size = 0.016
    viewport.overlays.append(phase_label)

    # Legend (element colors) on the right.
    legend_lines: List[tuple[str, tuple[float, float, float]]] = []
    try:
        data = pipeline.compute(0)
        if "Particle Type" in data.particles:
            types = list(data.particles["Particle Type"].types)
            types.sort(key=lambda t: t.name)
            for t in types:
                legend_lines.append((t.name, tuple(t.color)))
    except Exception:
        legend_lines = []
    legend_objs: List[TextLabelOverlay] = []
    if legend_lines:
        legend_base_x = 0.78
        legend_base_y = 0.02
        legend_gap = 0.02
        for idx, (name, color) in enumerate(legend_lines):
            label = TextLabelOverlay(
                text=f"{name}",
                offset_x=legend_base_x,
                offset_y=legend_base_y + idx * legend_gap,
            )
            label.font_size = 0.010
            label.text_color = color
            viewport.overlays.append(label)
            legend_objs.append(label)

    renderer_name = renderer_name.lower()
    if renderer_name == "opengl":
        try:
            from ovito.vis import OpenGLRenderer

            renderer = OpenGLRenderer()
        except Exception as exc:
            raise RuntimeError("OpenGL renderer not available in this OVITO build.") from exc
    elif renderer_name == "tachyon":
        renderer = TachyonRenderer()
    else:
        raise ValueError(f"Unknown renderer: {renderer_name}")

    total = len(overlays)
    stage_colors = {
        "ACTION": (0.2, 0.4, 0.9),
        "QUENCH": (0.9, 0.5, 0.1),
        "MIN": (0.2, 0.7, 0.2),
    }
    logged_any = False
    bar_width = 30
    for idx, frame in enumerate(overlays):
        lines = frame["lines"]
        stage_label = _stage_label(frame.get("stage"))
        phase_label.text = f"PHASE: {stage_label}"
        phase_label.text_color = stage_colors.get(stage_label, (0.2, 0.2, 0.2))
        for line_idx, label in enumerate(overlay_objs):
            label.text = lines[line_idx] if line_idx < len(lines) else ""
        out_path = out_dir / f"frame_{idx:05d}.png"
        viewport.render_image(
            filename=str(out_path),
            size=size,
            frame=idx,
            renderer=renderer,
        )
        if log_every > 0 and (idx == 0 or (idx + 1) % log_every == 0 or (idx + 1) == total):
            pct = (idx + 1) / total * 100
            filled = int(bar_width * (idx + 1) / total)
            bar = "#" * filled + "-" * (bar_width - filled)
            sys.stdout.write(f"\r[render] {idx + 1}/{total} [{bar}] {pct:.1f}%")
            sys.stdout.flush()
            logged_any = True
    if logged_any:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return out_dir


def _write_mp4(frames_dir: Path, mp4_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; install ffmpeg to render MP4")
    cmd = (
        f'"{ffmpeg}" -y -framerate {fps} -i "{frames_dir}/frame_%05d.png" '
        f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -pix_fmt yuv420p "{mp4_path}"'
    )
    import subprocess

    subprocess.check_call(cmd, shell=True)


def _stage_label(stage: Optional[str]) -> str:
    if stage == "action":
        return "ACTION"
    if stage == "quench_step":
        return "QUENCH"
    if stage == "min":
        return "MIN"
    return str(stage or "NA").upper()


def _format_overlay_lines(step: Dict[str, Any]) -> List[str]:
    basin_id = step.get("basin_id") or "NA"
    lines = [
        f"action: {step.get('action')}",
        f"basin_id: {basin_id}",
    ]
    if step.get("basin_is_new"):
        lines.append("is_new: 1")
    if step.get("quench_step") is not None:
        lines.append(f"quench_step: {step.get('quench_step')}")
    if step.get("triggered"):
        reason = _format_trigger(step.get("trigger"))
        if reason:
            lines.append(f"DFT_TRIGGER: {reason}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Render sampling visualization")
    parser.add_argument("--run_dir", required=True, help="Run directory (contains viz/)")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--fps", type=int, default=12, help="Video FPS")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames")
    parser.add_argument("--width", type=int, default=800, help="Frame width (px)")
    parser.add_argument("--height", type=int, default=600, help="Frame height (px)")
    parser.add_argument(
        "--renderer",
        type=str,
        default="tachyon",
        help="Renderer: tachyon (high quality CPU) or opengl (fast GPU)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Print render progress every N frames (0 to disable).",
    )
    parser.add_argument(
        "--quench_stride",
        type=int,
        default=None,
        help="Only keep every Nth quench_step frame (action/min always kept). Defaults to --stride.",
    )
    parser.add_argument(
        "--skip_movie",
        action="store_true",
        help="Only write extxyz outputs and skip MP4 rendering.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    viz_dir = run_dir / "viz"
    steps_path = viz_dir / "viz_steps.jsonl"
    if not steps_path.exists():
        raise FileNotFoundError(f"Missing viz_steps.jsonl: {steps_path}")

    steps = _load_steps(steps_path)
    quench_stride = args.quench_stride or args.stride
    if quench_stride < 1:
        raise ValueError("quench_stride must be >= 1")
    filtered: List[Dict[str, Any]] = []
    quench_count = 0
    for step in steps:
        if step.get("stage") == "quench_step":
            if quench_count % quench_stride == 0:
                filtered.append(step)
            quench_count += 1
        else:
            filtered.append(step)
    steps = filtered
    if args.max_frames is not None:
        steps = steps[: int(args.max_frames)]

    frames: List[Dict[str, Any]] = []
    atoms_list: List[Atoms] = []
    action_atoms: List[Atoms] = []
    quench_atoms: List[Atoms] = []
    for step in steps:
        ref = step.get("structure_ref")
        atoms = _resolve_structure(ref, run_dir)
        if atoms is None:
            continue
        _attach_info(atoms, step)
        stage = step.get("stage")
        if stage == "action":
            action_atoms.append(atoms.copy())
        elif stage in {"quench_step", "min"}:
            quench_atoms.append(atoms.copy())
        overlay_lines = _format_overlay_lines(step)
        hold = 4 if stage in {"action", "min"} else 1
        for _ in range(hold):
            atoms_list.append(atoms.copy())
            frames.append({"overlay": overlay_lines, "stage": step.get("stage")})

    if not frames:
        raise RuntimeError("No frames to render")

    out_extxyz = viz_dir / "trajectory.extxyz"
    action_path = viz_dir / "trajectory_action.extxyz"
    quench_path = viz_dir / "trajectory_quench.extxyz"
    extxyz_path = out_extxyz
    try:
        ase_write(out_extxyz, atoms_list)
    except PermissionError:
        extxyz_path = viz_dir / "trajectory_render.extxyz"
        ase_write(extxyz_path, atoms_list)
    if action_atoms:
        try:
            ase_write(action_path, action_atoms)
        except PermissionError:
            pass
    if quench_atoms:
        try:
            ase_write(quench_path, quench_atoms)
        except PermissionError:
            pass

    if args.skip_movie:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        overlays = [{"lines": frame["overlay"], "stage": frame.get("stage")} for frame in frames]
        _render_frames_ovito(
            trajectory_path=extxyz_path,
            overlays=overlays,
            out_dir=frames_dir,
            size=(args.width, args.height),
            renderer_name=args.renderer,
            log_every=args.log_every,
        )
        out_mp4 = viz_dir / "movie.mp4"
        _write_mp4(frames_dir, out_mp4, args.fps)


if __name__ == "__main__":
    main()
