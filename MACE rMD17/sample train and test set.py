"""Randomly split an XYZ movie into train/validation and test subsets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from ase import io as ase_io


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample disjoint train/validation and test subsets from an XYZ file."
    )
    parser.add_argument(
        "--input_xyz",
        type=Path,
        required=True,
        help="Path to source movie-style XYZ file.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Prefix for output files; '_train.xyz' and '_test.xyz' will be appended.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="Number of frames to sample for train/validation set.",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of frames to sample for test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_all_frames(xyz_path: Path) -> List["ase.Atoms"]:
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Input xyz file does not exist: {xyz_path}")

    logging.info("Reading frames from %s", xyz_path)
    frames: List["ase.Atoms"] = []
    for atoms in ase_io.iread(xyz_path, index=":"):
        energy = atoms.info.get("energy")
        calc_energy = None
        if atoms.calc is not None:
            calc_energy = atoms.calc.results.get("energy")
            if calc_energy is None:
                try:
                    calc_energy = atoms.calc.get_potential_energy()
                except Exception:
                    calc_energy = None
        if energy is None and calc_energy is not None:
            atoms.info["energy"] = float(calc_energy)
        frames.append(atoms.copy())

    logging.info("Total frames read: %d", len(frames))
    return frames


def split_frames(
    frames: Sequence["ase.Atoms"],
    train_size: int,
    test_size: int,
    seed: int,
) -> Tuple[List["ase.Atoms"], List["ase.Atoms"]]:
    total_required = train_size + test_size
    if len(frames) < total_required:
        raise ValueError(
            f"Not enough frames ({len(frames)}) to satisfy train_size + test_size ({total_required})."
        )

    indices = np.arange(len(frames))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size : train_size + test_size]

    train_frames = [frames[i] for i in train_indices]
    test_frames = [frames[i] for i in test_indices]

    logging.info(
        "Selected %d frames for train/val, %d frames for test.",
        len(train_frames),
        len(test_frames),
    )
    return train_frames, test_frames


def write_xyz(frames: Sequence["ase.Atoms"], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing %d frames to %s", len(frames), path)
    with path.open("w", encoding="utf-8") as handle:
        for atoms in frames:
            handle.write(f"{len(atoms)}\n")

            energy = atoms.info.get("energy")
            comment_parts = ["Properties=species:S:1:pos:R:3"]
            if "force" in atoms.arrays:
                comment_parts[0] += ":force:R:3"
            if energy is not None:
                comment_parts.append(f"energy={energy}")
            handle.write(" ".join(comment_parts) + "\n")

            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            forces = atoms.arrays.get("force")

            for i, symbol in enumerate(symbols):
                pos_str = " ".join(f"{value:.16f}" for value in positions[i])
                if forces is not None:
                    force_str = " ".join(f"{value:.16f}" for value in forces[i])
                    handle.write(f"{symbol} {pos_str} {force_str}\n")
                else:
                    handle.write(f"{symbol} {pos_str}\n")


def main() -> None:
    args = parse_args()

    frames = load_all_frames(args.input_xyz)
    train_frames, test_frames = split_frames(
        frames, args.train_size, args.test_size, args.seed
    )

    train_path = args.output_prefix.with_name(
        args.output_prefix.name + "_train.xyz"
    )
    test_path = args.output_prefix.with_name(
        args.output_prefix.name + "_test.xyz"
    )

    write_xyz(train_frames, train_path)
    write_xyz(test_frames, test_path)

    logging.info("Sampling complete. Outputs: %s, %s", train_path, test_path)


if __name__ == "__main__":
    main()
