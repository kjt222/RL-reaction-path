"""Convert rMD17-style .npz bundles to Extended XYZ trajectories for MACE."""

import math
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Index 0 is a placeholder so that atomic number 1 maps to "H"
CHEMICAL_SYMBOLS: list[Optional[str]] = [
    None,
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def resolve_data_dir() -> Path:
    """Select the input directory, preferring env override and auto-detected defaults."""
    override = os.getenv("MACE_RMD17_DIR")
    if override:
        override_path = Path(override).expanduser()
        if not override_path.exists():
            raise FileNotFoundError(
                f"Environment variable MACE_RMD17_DIR points to a non-existent directory: {override_path}"
            )
        return override_path

    candidates = [
        Path(r"D:\D\calculate\MLP\training_set\rmd17\npz_data"),
        Path("/mnt/d/D/calculate/MLP/training_set/rmd17/npz_data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate rMD17 data directory. Set the MACE_RMD17_DIR environment variable to the correct path."
    )


DATA_DIR = resolve_data_dir()
OUTPUT_DIR = Path(__file__).resolve().parent / "xyz_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def atomic_number_to_symbol(z_value: int) -> str:
    """Translate atomic number to chemical symbol, raising if unsupported."""
    try:
        symbol = CHEMICAL_SYMBOLS[z_value]
    except IndexError as exc:
        raise ValueError(f"Unsupported atomic number {z_value}") from exc
    if not symbol:
        raise ValueError(f"Unsupported atomic number {z_value}")
    return symbol


def format_lattice(cell_matrix: np.ndarray) -> Optional[str]:
    """Flatten a 3x3 cell matrix into the Extended XYZ Lattice string."""
    if cell_matrix is None:
        return None
    flat = cell_matrix.reshape(-1)
    return " ".join(f"{value:.16f}" for value in flat)


def format_pbc(pbc_flags: np.ndarray) -> Optional[str]:
    """Convert PBC flags into the ASE-style 'T/F T/F T/F' string."""
    if pbc_flags is None:
        return None
    tokens = ["T" if bool(flag) else "F" for flag in np.asarray(pbc_flags).flat[:3]]
    return " ".join(tokens)


def iter_comment_parts(
    properties: str,
    energy: Optional[float],
    lattice: Optional[str],
    pbc_flags: Optional[str],
    virial: Optional[Iterable[float]],
) -> Iterable[str]:
    """Yield comment fragments for the XYZ header line."""
    yield properties
    if energy is not None and math.isfinite(energy):
        yield f"energy={energy:.16f}"
    if lattice is not None:
        yield f'Lattice="{lattice}"'
    if pbc_flags is not None:
        yield f'pbc="{pbc_flags}"'
    if virial is not None:
        virial_values = ",".join(f"{value:.16f}" for value in virial)
        yield f"virial={virial_values}"


def convert_single_npz(npz_path: Path) -> Path:
    """Convert a single .npz file into an Extended XYZ trajectory."""
    key_map = {
        "positions": ["R", "coords"],
        "atomic_numbers": ["z", "nuclear_charges"],
        "energies": ["E", "energies"],
        "forces": ["F", "forces"],
        "virials": ["V", "virials"],
        "cells": ["cell", "cells"],
        "pbc": ["pbc"],
    }

    def resolve_key(npz_data: np.lib.npyio.NpzFile, logical_name: str, required: bool = False):
        """Fetch the first matching key for the logical quantity."""
        for candidate in key_map[logical_name]:
            if candidate in npz_data.files:
                return np.asarray(npz_data[candidate])
        if required:
            raise KeyError(f"{npz_path.name} is missing required keys for '{logical_name}': {key_map[logical_name]}")
        return None

    with np.load(npz_path, allow_pickle=False) as npz_data:
        positions = resolve_key(npz_data, "positions", required=True)
        atomic_numbers = resolve_key(npz_data, "atomic_numbers", required=True)
        energies = resolve_key(npz_data, "energies", required=False)
        forces = resolve_key(npz_data, "forces", required=False)
        virials = resolve_key(npz_data, "virials", required=False)
        cells = resolve_key(npz_data, "cells", required=False)
        pbc = resolve_key(npz_data, "pbc", required=False)

    if positions.ndim != 3:
        raise ValueError(f"Expected positions to have shape (n_frames, n_atoms, 3); got {positions.shape}")

    n_frames, n_atoms, coord_dim = positions.shape
    if coord_dim != 3:
        raise ValueError("Atom coordinates must be 3 dimensional")

    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != n_atoms:
        raise ValueError("Atomic numbers array must be 1D with length equal to number of atoms")

    if forces is not None:
        if forces.shape != positions.shape:
            raise ValueError("Forces array must have the same shape as positions")

    if energies is not None and energies.shape[0] != n_frames:
        raise ValueError("Energy array length must match number of frames")

    if virials is not None and virials.shape[0] != n_frames:
        raise ValueError("Virial array length must match number of frames")

    if cells is not None and cells.shape[0] != n_frames:
        raise ValueError("Cell array length must match number of frames")

    if pbc is not None and pbc.shape[0] != n_frames:
        raise ValueError("PBC array length must match number of frames")

    properties = "Properties=species:S:1:pos:R:3"
    has_forces = forces is not None
    if has_forces:
        properties += ":force:R:3"

    output_path = OUTPUT_DIR / f"{npz_path.stem}.xyz"
    with output_path.open("w", encoding="utf-8") as handle:
        for frame_idx in range(n_frames):
            handle.write(f"{n_atoms}\n")

            lattice_str = format_lattice(cells[frame_idx]) if cells is not None else None
            pbc_str = format_pbc(pbc[frame_idx]) if pbc is not None else None
            virial_values = virials[frame_idx].reshape(-1) if virials is not None else None
            energy_value = float(energies[frame_idx]) if energies is not None else None

            comment = " ".join(
                iter_comment_parts(
                    properties=properties,
                    energy=energy_value,
                    lattice=lattice_str,
                    pbc_flags=pbc_str,
                    virial=virial_values,
                )
            )
            handle.write(comment + "\n")

            for atom_idx, atomic_number in enumerate(atomic_numbers):
                symbol = atomic_number_to_symbol(int(atomic_number))
                x, y, z = positions[frame_idx, atom_idx]
                if has_forces:
                    fx, fy, fz = forces[frame_idx, atom_idx]
                    handle.write(
                        f"{symbol} "
                        f"{x:.16f} {y:.16f} {z:.16f} "
                        f"{fx:.16f} {fy:.16f} {fz:.16f}\n"
                    )
                else:
                    handle.write(f"{symbol} {x:.16f} {y:.16f} {z:.16f}\n")

    return output_path


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}")

    for npz_file in npz_files:
        xyz_path = convert_single_npz(npz_file)
        print(f"Converted {npz_file.name} -> {xyz_path.name}")


if __name__ == "__main__":
    main()
