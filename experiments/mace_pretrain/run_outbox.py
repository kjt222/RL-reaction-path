"""DFT outbox entrypoint (canonicalize + dedup)."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mace_pretrain.dedup import GlobalRMSDDeduper
from experiments.mace_pretrain.outbox import process_dft_queue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process DFT queue")
    parser.add_argument("--run_dir", required=True, help="Run directory containing dft_queue.jsonl")
    parser.add_argument("--rmsd", type=float, default=0.18, help="Dedup RMSD threshold")
    parser.add_argument("--no_canonicalize", action="store_true", help="Disable slab canonicalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    queue_path = run_dir / "dft_queue.jsonl"
    deduper = GlobalRMSDDeduper(rmsd_threshold=args.rmsd)
    process_dft_queue(
        queue_path=queue_path,
        canonicalize=not args.no_canonicalize,
        deduper=deduper,
    )


if __name__ == "__main__":
    main()
