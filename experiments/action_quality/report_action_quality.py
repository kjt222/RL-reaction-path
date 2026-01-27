"""CLI report for action quality diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from experiments.action_quality.action_quality import summarize_run_dir


def _print_summary(result: Dict[str, object]) -> None:
    total = int(result.get("total_steps", 0))
    action_counts = result.get("action_counts", {})
    print(f"steps: {total}")
    print(f"valid_rate: {result.get('valid_rate'):.3f}")
    if "quality_reject_rate" in result:
        print(f"quality_reject_rate: {result.get('quality_reject_rate'):.3f}")
    print(f"quench_converged_rate: {result.get('quench_converged_rate'):.3f}")
    print(f"new_basin_rate: {result.get('new_basin_rate'):.3f}")
    print(f"trigger_rate_max: {result.get('trigger_rate_max'):.3f}")
    print(f"trigger_rate_topk: {result.get('trigger_rate_topk'):.3f}")
    if isinstance(action_counts, dict) and action_counts:
        print("action_counts:")
        for name, count in sorted(action_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {name}: {count}")
    invalid_counts = result.get("invalid_reason_counts", {})
    if isinstance(invalid_counts, dict) and invalid_counts:
        print("invalid_reason_counts:")
        for name, count in sorted(invalid_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sampling action quality")
    parser.add_argument("--run_dir", required=True, help="Sampling run directory")
    parser.add_argument("--out", default=None, help="Write JSON summary to path (optional)")
    parser.add_argument("--source", default="force_pre", help="Force metric source (force_pre/force_min)")
    parser.add_argument("--max_threshold", type=float, default=0.7)
    parser.add_argument("--topk_threshold", type=float, default=0.35)
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    result = summarize_run_dir(
        run_dir,
        max_threshold=args.max_threshold,
        topk_threshold=args.topk_threshold,
        topk=args.topk,
        source=args.source,
    )
    _print_summary(result)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
