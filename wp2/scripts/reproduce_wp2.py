#!/usr/bin/env python3
"""Run the benchmark with externally supplied data and historical exclusion manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the corrected WP2 dataset or full experiment.")
    parser.add_argument("--mode", choices=["dataset-only", "full"], default="dataset-only")
    parser.add_argument("--dataset-root", default=os.environ.get("WP2_DATASET_ROOT"))
    parser.add_argument("--results-root", default=os.environ.get("WP2_RESULTS_ROOT"))
    parser.add_argument("--historical-manifest", required=True, help="External prior viewed-host manifest; not bundled in the code-only release.")
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root:
        raise SystemExit("Provide --dataset-root or set WP2_DATASET_ROOT to the parent of the NCI1 cache.")

    scripts_root = Path(__file__).resolve().parent
    archive_root = scripts_root.parent
    results_root = Path(args.results_root).resolve() if args.results_root else archive_root / "reproduced_results"
    historical_manifest = Path(args.historical_manifest).resolve()
    if not historical_manifest.exists():
        raise SystemExit(f"External historical manifest is missing: {historical_manifest}")

    command = [
        sys.executable,
        str(scripts_root / "run_wp2_benchmark.py"),
        "--dataset", "NCI1",
        "--dataset-root", str(Path(args.dataset_root).resolve()),
        "--results-root", str(results_root),
        "--root-seed", "20260710",
        "--hosts-per-split", "400",
        "--fresh-holdout",
        "--gnn-tune-seeds", "3",
        "--gnn-eval-seeds", "10",
        "--batch-size", "32",
        "--device", "auto",
        "--max-transform-attempts", "12",
        "--pna-on-cpu",
        "--mode", args.mode,
        "--historical-manifest", str(historical_manifest),
    ]
    if args.overwrite_results:
        command.append("--overwrite-results")
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=archive_root, check=True)


if __name__ == "__main__":
    main()
