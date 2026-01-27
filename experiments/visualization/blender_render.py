import argparse
import math
import os
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


_BOND_SCALE = 1.2
_BOND_MAX = 3.0
_BOND_RADIUS = 0.06
_CELL_RADIUS = 0.25
_CELL_EMISSION = 8.0
_SLAB_ALPHA = 0.35
_SLAB_SCALE = 0.7
_TRIAD_SCALE = 0.08


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Blender render for extxyz trajectory")
    parser.add_argument("--extxyz", required=True, help="Path to trajectory.extxyz")
    parser.add_argument("--out_dir", required=True, help="Output directory for frames/mp4")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames")
    parser.add_argument("--width", type=int, default=1280, help="Render width")
    parser.add_argument("--height", type=int, default=720, help="Render height")
    parser.add_argument("--samples", type=int, default=64, help="Cycles samples")
    parser.add_argument("--radius", type=float, default=0.6, help="Atom radius (uniform)")
    parser.add_argument("--fps", type=int, default=12, help="FPS for optional mp4")
    parser.add_argument("--write_mp4", action="store_true", help="Write movie.mp4 using ffmpeg")
    parser.add_argument("--log_every", type=int, default=20, help="Progress log frequency")
    return parser.parse_args(argv)


def _parse_comment(comment: str):
    fields: dict[str, str] = {}
    lattice = None
    for match in re.finditer(r'(\\w+)=(".*?"|\\S+)', comment):
        key = match.group(1)
        val = match.group(2).strip('"')
        fields[key] = val
    if "Lattice" in fields:
        try:
            lattice = [float(x) for x in fields["Lattice"].split()]
        except ValueError:
            lattice = None
    return fields, lattice


def _parse_properties(comment: str):
    match = re.search(r'Properties=("[^"]+"|[^ ]+)', comment)
    if not match:
        return None
    raw = match.group(1).strip('"')
    parts = raw.split(":")
    if len(parts) < 3 or len(parts) % 3 != 0:
        return None
    props = []
    for idx in range(0, len(parts), 3):
        name = parts[idx]
        kind = parts[idx + 1]
        try:
            count = int(parts[idx + 2])
        except ValueError:
            return None
        props.append((name, kind, count))
    return props


def _iter_extxyz(path: Path, stride: int, max_frames: int | None):
    frame_idx = 0
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError:
                break
            comment = handle.readline().strip()
            props = _parse_properties(comment)
            atoms = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if not parts:
                    continue
                if props:
                    idx = 0
                    sym = None
                    pos = None
                    tag = None
                    for name, _kind, count in props:
                        chunk = parts[idx : idx + count]
                        idx += count
                        if name in {"species", "symbol", "element"}:
                            sym = chunk[0]
                        elif name in {"pos", "positions"} and count >= 3:
                            try:
                                pos = tuple(float(x) for x in chunk[:3])
                            except ValueError:
                                pos = None
                        elif name in {"tags", "tag"}:
                            try:
                                tag = int(float(chunk[0]))
                            except ValueError:
                                tag = None
                    if sym is None and len(parts) >= 4:
                        sym = parts[0]
                    if pos is None and len(parts) >= 4:
                        try:
                            pos = tuple(float(x) for x in parts[1:4])
                        except ValueError:
                            pos = None
                    if sym and pos:
                        atoms.append((sym, pos, tag))
                else:
                    if len(parts) < 4:
                        continue
                    sym = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append((sym, (x, y, z), None))
            if frame_idx % stride == 0:
                fields, lattice = _parse_comment(comment)
                yield atoms, fields, lattice
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_idx += 1


def _count_frames(path: Path, stride: int, max_frames: int | None) -> int:
    total = 0
    frame_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line)
            except ValueError:
                break
            handle.readline()
            for _ in range(natoms):
                handle.readline()
            if frame_idx % stride == 0:
                total += 1
                if max_frames is not None and total >= max_frames:
                    break
            frame_idx += 1
    return total


def _symbol_color(symbol: str):
    # Simple deterministic color map; fallback to hash-based
    palette = {
        "H": (1.0, 1.0, 1.0),
        "C": (0.2, 0.2, 0.2),
        "N": (0.1, 0.2, 0.8),
        "O": (0.8, 0.1, 0.1),
        "F": (0.6, 0.8, 0.1),
        "Cl": (0.2, 0.8, 0.2),
        "Br": (0.6, 0.2, 0.1),
        "I": (0.4, 0.2, 0.6),
        "Cs": (0.2, 0.7, 0.9),
        "Sr": (0.6, 0.9, 0.6),
        "Ta": (0.7, 0.5, 0.4),
        "Sb": (0.6, 0.6, 0.6),
        "Bi": (0.6, 0.4, 0.7),
        "Tl": (0.7, 0.7, 0.2),
        "Sn": (0.5, 0.5, 0.6),
        "Hg": (0.7, 0.7, 0.7),
        "Nb": (0.4, 0.7, 0.4),
        "Pd": (0.7, 0.7, 0.8),
    }
    if symbol in palette:
        return palette[symbol]
    h = abs(hash(symbol)) % 360
    # HSV -> RGB
    c = 0.8
    x = c * (1 - abs((h / 60) % 2 - 1))
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (r, g, b)


def _covalent_radius(symbol: str) -> float:
    radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
        "Fe": 1.24,
        "Se": 1.20,
        "Cu": 1.32,
        "Ni": 1.21,
        "Co": 1.26,
        "Mn": 1.39,
        "Cr": 1.39,
        "V": 1.53,
        "W": 1.62,
        "Mo": 1.54,
        "Pd": 1.39,
        "Pt": 1.36,
        "Au": 1.36,
        "Ag": 1.45,
        "Zn": 1.22,
        "Mg": 1.30,
        "Al": 1.21,
        "Si": 1.11,
        "S": 1.05,
        "P": 1.07,
    }
    return radii.get(symbol, 0.77)


def _setup_scene(width: int, height: int, samples: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    # Avoid OpenImageDenoiser dependency in headless builds.
    scene.cycles.use_denoising = False
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_stamp = True
    scene.render.use_stamp_note = True
    scene.render.use_stamp_labels = False
    scene.render.use_stamp_frame = False
    scene.render.use_stamp_date = False
    scene.render.use_stamp_time = False
    scene.render.use_stamp_render_time = False
    scene.render.use_stamp_camera = False
    scene.render.use_stamp_scene = False
    scene.render.use_stamp_filename = False
    scene.render.use_stamp_lens = False
    scene.render.use_stamp_marker = False
    scene.render.use_stamp_sequencer_strip = False
    scene.render.stamp_font_size = 14
    scene.render.stamp_foreground = (0.05, 0.05, 0.05, 1.0)
    scene.render.stamp_background = (1.0, 1.0, 1.0, 0.35)

    # Prefer GPU if available
    prefs = bpy.context.preferences
    if "cycles" in prefs.addons:
        cy_prefs = prefs.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                cy_prefs.compute_device_type = dev_type
                cy_prefs.get_devices()
                for d in cy_prefs.devices:
                    d.use = True
                scene.cycles.device = "GPU"
                break
            except Exception:
                continue
    # White background
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)
        bg.inputs[1].default_value = 1.0

    # Clear default objects
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_material(symbol: str, alpha: float = 1.0):
    mat = bpy.data.materials.new(name=f"MAT_{symbol}_{alpha:.2f}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        r, g, b = _symbol_color(symbol)
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Alpha"].default_value = alpha
        bsdf.inputs["Roughness"].default_value = 0.4
    if alpha < 1.0:
        mat.blend_method = "BLEND"
        mat.shadow_method = "HASHED"
    return mat


def _add_light(location: Vector):
    light_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    light_data.energy = 800
    light_obj = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    light_obj.location = location
    bpy.context.collection.objects.link(light_obj)


def _make_bond_material():
    mat = bpy.data.materials.new(name="MAT_BOND")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6
    return mat


def _make_cell_material():
    mat = bpy.data.materials.new(name="MAT_CELL")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.2
        if "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = (0.02, 0.02, 0.02, 1.0)
            bsdf.inputs["Emission Strength"].default_value = _CELL_EMISSION
    return mat


def _make_axis_material(name: str, color):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        r, g, b = color
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        if "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = (r, g, b, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 3.0
    return mat


def _add_xyz_triad(cam, ortho_scale: float):
    length = ortho_scale * _TRIAD_SCALE
    radius = length * 0.08
    head_radius = radius * 1.8
    head_length = length * 0.25
    base_length = length - head_length
    colors = {
        "X": (0.9, 0.1, 0.1),
        "Y": (0.1, 0.7, 0.1),
        "Z": (0.1, 0.2, 0.9),
    }
    triad = bpy.data.objects.new("XYZ_Triad", None)
    triad.empty_display_size = 0.0
    bpy.context.collection.objects.link(triad)

    offset_cam = Vector((-ortho_scale * 0.45, -ortho_scale * 0.35, -cam.data.clip_start - 0.5))
    triad.location = cam.matrix_world @ offset_cam

    def add_label(text: str, location: Vector, color):
        curve = bpy.data.curves.new(name=f"Text_{text}", type="FONT")
        curve.body = text
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = length * 0.35
        text_obj = bpy.data.objects.new(f"Label_{text}", curve)
        text_obj.location = location
        text_obj.rotation_euler = cam.rotation_euler
        mat = _make_axis_material(f"MAT_LABEL_{text}", color)
        text_obj.data.materials.append(mat)
        bpy.context.collection.objects.link(text_obj)

    def add_axis(axis_name: str, direction: Vector):
        mat = _make_axis_material(f"MAT_AXIS_{axis_name}", colors[axis_name])
        bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=base_length)
        body = bpy.context.active_object
        body.name = f"Axis_{axis_name}_Body"
        body.data.materials.append(mat)
        bpy.ops.mesh.primitive_cone_add(radius1=head_radius, radius2=0.0, depth=head_length)
        head = bpy.context.active_object
        head.name = f"Axis_{axis_name}_Head"
        head.data.materials.append(mat)

        direction = direction.normalized()
        rot = direction.to_track_quat("Z", "Y")
        body.rotation_mode = "QUATERNION"
        body.rotation_quaternion = rot
        head.rotation_mode = "QUATERNION"
        head.rotation_quaternion = rot
        body.location = triad.location + direction * (base_length * 0.5)
        head.location = triad.location + direction * (base_length + head_length * 0.5)
        label_loc = triad.location + direction * (base_length + head_length * 1.2)
        add_label(axis_name, label_loc, colors[axis_name])

    add_axis("X", Vector((1.0, 0.0, 0.0)))
    add_axis("Y", Vector((0.0, 1.0, 0.0)))
    add_axis("Z", Vector((0.0, 0.0, 1.0)))


def _compute_view(lattice):
    forward = None
    if lattice and len(lattice) == 9:
        a = Vector(lattice[0:3]).normalized()
        b = Vector(lattice[3:6]).normalized()
        c = Vector(lattice[6:9]).normalized()
        forward = (a + b + 0.3 * c).normalized()
    if forward is None or forward.length == 0:
        forward = Vector((1.0, -1.0, 0.6)).normalized()
    return forward


def _create_camera(center: Vector, forward: Vector, ortho_scale: float):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho_scale
    cam.location = center + forward * (ortho_scale * 1.5 + 1.0)
    direction = (center - cam.location).normalized()
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_start = 0.1
    cam.data.clip_end = max(ortho_scale * 10.0, 1000.0)
    return cam


def _format_comment(fields: dict[str, str]):
    # Keep short fields for overlay
    stage = fields.get("stage", "NA")
    action = fields.get("action", "NA")
    basin_id = fields.get("basin_id", "NA")
    quench_step = fields.get("quench_step", "")
    trigger = fields.get("trigger_reason", "")
    lines = [f"stage: {stage}", f"action: {action}", f"basin_id: {basin_id}"]
    if quench_step:
        lines.append(f"quench_step: {quench_step}")
    if trigger:
        lines.append(f"trigger: {trigger}")
    return "\n".join(lines)


def _clear_bonds():
    for obj in list(bpy.data.objects):
        if obj.name.startswith("Bond_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def _add_bonds(atoms, bond_mat):
    positions = [Vector(pos) for _, pos, _ in atoms]
    symbols = [sym for sym, _, _ in atoms]
    n = len(positions)
    for i in range(n):
        ri = _covalent_radius(symbols[i])
        for j in range(i + 1, n):
            rj = _covalent_radius(symbols[j])
            max_dist = min(_BOND_MAX, _BOND_SCALE * (ri + rj))
            dist = (positions[i] - positions[j]).length
            if dist <= max_dist:
                mid = (positions[i] + positions[j]) * 0.5
                bpy.ops.mesh.primitive_cylinder_add(radius=_BOND_RADIUS, depth=dist, location=mid)
                bond_obj = bpy.context.active_object
                bond_obj.name = f"Bond_{i}_{j}"
                bond_obj.data.materials.append(bond_mat)
                direction = (positions[j] - positions[i]).normalized()
                bond_obj.rotation_mode = "QUATERNION"
                bond_obj.rotation_quaternion = direction.to_track_quat("Z", "Y")


def _create_cell_edges(lattice, material, offset: Vector | None = None):
    if not lattice or len(lattice) != 9:
        return
    offset = offset or Vector((0.0, 0.0, 0.0))
    a = Vector(lattice[0:3])
    b = Vector(lattice[3:6])
    c = Vector(lattice[6:9])
    corners = [
        offset + Vector((0.0, 0.0, 0.0)),
        offset + a,
        offset + b,
        offset + c,
        offset + a + b,
        offset + a + c,
        offset + b + c,
        offset + a + b + c,
    ]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for idx, (i, j) in enumerate(edges):
        p0 = corners[i]
        p1 = corners[j]
        mid = (p0 + p1) * 0.5
        dist = (p1 - p0).length
        bpy.ops.mesh.primitive_cylinder_add(radius=_CELL_RADIUS, depth=dist, location=mid)
        edge_obj = bpy.context.active_object
        edge_obj.name = f"CellEdge_{idx}"
        edge_obj.data.materials.append(material)
        direction = (p1 - p0).normalized()
        edge_obj.rotation_mode = "QUATERNION"
        edge_obj.rotation_quaternion = direction.to_track_quat("Z", "Y")


def _lattice_vectors(lattice):
    if not lattice or len(lattice) != 9:
        return None
    a = Vector(lattice[0:3])
    b = Vector(lattice[3:6])
    c = Vector(lattice[6:9])
    return a, b, c


def _wrap_atoms(atoms, lattice, wrap_z: bool = False):
    lattice_vecs = _lattice_vectors(lattice)
    if not lattice_vecs:
        return atoms
    a, b, c = lattice_vecs
    try:
        mat = Matrix((a, b, c)).transposed()
        mat_inv = mat.inverted()
    except Exception:
        return atoms
    wrapped = []
    for sym, pos, tag in atoms:
        r = Vector(pos)
        f = mat_inv @ r
        f[0] = f[0] - math.floor(f[0])
        f[1] = f[1] - math.floor(f[1])
        if wrap_z:
            f[2] = f[2] - math.floor(f[2])
        r2 = mat @ f
        wrapped.append((sym, (r2.x, r2.y, r2.z), tag))
    return wrapped


def _write_mp4(frames_dir: Path, out_mp4: Path, fps: int):
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.check_call(cmd)


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _setup_scene(args.width, args.height, args.samples)

    extxyz_path = Path(args.extxyz)
    total_frames = _count_frames(extxyz_path, args.stride, args.max_frames)
    frame_iter = _iter_extxyz(extxyz_path, args.stride, args.max_frames)
    try:
        first_atoms, first_fields, first_lattice = next(frame_iter)
    except StopIteration:
        raise SystemExit("No frames found in extxyz")

    if first_lattice:
        first_atoms = _wrap_atoms(first_atoms, first_lattice, wrap_z=False)

    # Create atom objects
    symbol_mesh = {}
    symbol_mat = {}
    symbol_mat_slab = {}
    atom_objects = []
    positions = []
    for sym, pos, tag in first_atoms:
        if sym not in symbol_mesh:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=args.radius, segments=24, ring_count=16)
            mesh_obj = bpy.context.active_object
            mesh = mesh_obj.data
            bpy.data.objects.remove(mesh_obj, do_unlink=True)
            symbol_mesh[sym] = mesh
            symbol_mat[sym] = _make_material(sym, alpha=1.0)
            symbol_mat_slab[sym] = _make_material(sym, alpha=_SLAB_ALPHA)
        obj = bpy.data.objects.new(f"Atom_{sym}", symbol_mesh[sym])
        is_slab = tag == 0 if tag is not None else False
        obj.data.materials.append(symbol_mat_slab[sym] if is_slab else symbol_mat[sym])
        obj.location = Vector(pos)
        if is_slab:
            obj.scale = (_SLAB_SCALE, _SLAB_SCALE, _SLAB_SCALE)
        bpy.context.collection.objects.link(obj)
        atom_objects.append(obj)
        positions.append(Vector(pos))

    # Camera + light
    atom_center = sum(positions, Vector((0, 0, 0))) / len(positions)
    lattice_vecs = _lattice_vectors(first_lattice)
    if lattice_vecs:
        a, b, c = lattice_vecs
        cell_center = 0.5 * (a + b + c)
        cell_extent = max(a.length, b.length, c.length)
        cell_offset = Vector((0.0, 0.0, 0.0))
    else:
        cell_extent = 0.0
        cell_offset = Vector((0.0, 0.0, 0.0))
    center = atom_center if cell_extent == 0.0 else cell_center
    max_extent = max((p - center).length for p in positions)
    forward = _compute_view(first_lattice)
    ortho_scale = max(max_extent * 2.8, cell_extent * 1.2, 5.0)
    cam = _create_camera(center, forward, ortho_scale)
    _add_light(center + Vector((0, 0, max_extent * 2.0 + 2.0)))
    bond_mat = _make_bond_material()
    cell_mat = _make_cell_material()
    _create_cell_edges(first_lattice, cell_mat, cell_offset)
    _add_xyz_triad(cam, ortho_scale)
    bpy.context.scene.render.stamp_note_text = _format_comment(first_fields)

    def render_frame(frame_idx, atoms, fields):
        use_atoms = _wrap_atoms(atoms, first_lattice, wrap_z=False)
        for obj, (_, pos, _tag) in zip(atom_objects, use_atoms):
            obj.location = Vector(pos)
        bpy.context.scene.render.stamp_note_text = _format_comment(fields)
        _clear_bonds()
        _add_bonds(use_atoms, bond_mat)
        bpy.context.scene.render.filepath = str(frames_dir / f"frame_{frame_idx:05d}.png")
        bpy.ops.render.render(write_still=True)

    # Render first frame
    rendered = 1
    render_frame(0, first_atoms, first_fields)

    for atoms, fields, _lattice in frame_iter:
        render_frame(rendered, atoms, fields)
        rendered += 1
        if args.log_every > 0 and rendered % args.log_every == 0:
            sys.stdout.write(f"\r[render] {rendered}/{total_frames}")
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    if args.write_mp4:
        out_mp4 = out_dir / "movie.mp4"
        _write_mp4(frames_dir, out_mp4, args.fps)


if __name__ == "__main__":
    main()
