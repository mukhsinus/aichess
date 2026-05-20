#!/usr/bin/env python
"""
Full Pipeline Runner.

Executes the complete dataset-collection, preparation, augmentation, training,
evaluation, and integration pipeline end-to-end.

Usage:
    python run_pipeline.py                    # run everything
    python run_pipeline.py --skip collect     # skip dataset download
    python run_pipeline.py --only train eval  # run specific phases
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

PHASES = {
    "collect":   ("collect_data.py",     "Phase 1: Collecting datasets"),
    "prepare":   ("prepare_dataset.py",  "Phase 2: Preparing dataset (dedup, normalize, split)"),
    "augment":   ("augment_data.py",     "Phase 3: Augmenting training data"),
    "train":     ("train.py",            "Phase 4: Training YOLOv8s"),
    "eval":      ("validate.py",         "Phase 5: Evaluating model"),
    "integrate": ("integrate.py",        "Phase 6: Integration check"),
}


def run_phase(name: str, script: str, desc: str, extra_args: list[str] | None = None) -> bool:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")

    cmd = [PYTHON, str(TRAINING_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(TRAINING_DIR))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED ({elapsed:.1f}s) -- exit code {result.returncode}")
        return False
    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full chess training pipeline.")
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=list(PHASES),
        help="Phases to skip",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=list(PHASES),
        help="Run only these phases (in order)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing dataset when preparing",
    )
    args = parser.parse_args()

    phases_to_run = list(PHASES) if args.only is None else args.only
    phases_to_run = [p for p in phases_to_run if p not in args.skip]

    print("Pipeline phases to run:", ", ".join(phases_to_run))
    overall_t0 = time.time()

    for phase_name in phases_to_run:
        script, desc = PHASES[phase_name]
        extra = []
        if phase_name == "prepare" and args.force:
            extra.append("--force")
        ok = run_phase(phase_name, script, desc, extra_args=extra or None)
        if not ok:
            print(f"\nPipeline aborted at phase: {phase_name}")
            sys.exit(1)

    total = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete  ({total:.1f}s total)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
