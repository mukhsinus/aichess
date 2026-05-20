#!/usr/bin/env python
"""
Full Pipeline Runner.

Executes the complete dataset collection -> validation -> preparation ->
augmentation -> training -> evaluation -> integration pipeline end-to-end.

Usage:
    python run_pipeline.py                        # run everything
    python run_pipeline.py --skip collect          # skip dataset download
    python run_pipeline.py --only train eval       # run specific phases
    python run_pipeline.py --from prepare          # start from a specific phase
    python run_pipeline.py --force                 # overwrite existing dataset
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

TRAINING_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable

from detect_manual_dataset import check_manual_dataset

PHASES = {
    "collect":   ("collect_data.py",      "Phase 1:  Collecting datasets"),
    "validate":  ("validate_dataset.py",  "Phase 1b: Validating raw data"),
    "prepare":   ("prepare_dataset.py",   "Phase 2:  Preparing dataset (dedup, normalize, split)"),
    "augment":   ("augment_data.py",      "Phase 3:  Augmenting training data"),
    "train":     ("train.py",             "Phase 4:  Training YOLOv8s"),
    "eval":      ("validate.py",          "Phase 5:  Evaluating model"),
    "integrate": ("integrate.py",         "Phase 6:  Integration check"),
}

PHASE_ORDER = list(PHASES.keys())

NON_FATAL_PHASES = {"collect", "validate"}


def run_phase(
    name: str,
    script: str,
    desc: str,
    extra_args: list[str] | None = None,
) -> bool:
    """Run a single pipeline phase. Returns True on success."""
    log.info("")
    log.info("=" * 65)
    log.info("  %s", desc)
    log.info("=" * 65)

    cmd = [PYTHON, str(TRAINING_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    log.info("  Command: %s", " ".join(cmd))
    t0 = time.time()

    try:
        result = subprocess.run(cmd, cwd=str(TRAINING_DIR))
    except Exception as exc:
        elapsed = time.time() - t0
        log.error("  EXCEPTION running %s (%.1fs): %s", name, elapsed, exc)
        return False

    elapsed = time.time() - t0

    if result.returncode != 0:
        log.error("  FAILED (%s) in %.1fs -- exit code %d", name, elapsed, result.returncode)
        return False

    log.info("  Completed '%s' in %.1fs", name, elapsed)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full chess training pipeline.",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[], choices=PHASE_ORDER,
        help="Phases to skip",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, choices=PHASE_ORDER,
        help="Run only these phases (in pipeline order)",
    )
    parser.add_argument(
        "--from", dest="start_from", default=None, choices=PHASE_ORDER,
        help="Start from this phase (skipping all earlier ones)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing dataset when preparing",
    )
    parser.add_argument(
        "--force-collect", action="store_true",
        help="Re-download datasets even if they already exist",
    )
    args = parser.parse_args()

    # Determine which phases to run (in order)
    if args.only is not None:
        phases_to_run = [p for p in PHASE_ORDER if p in args.only]
    elif args.start_from is not None:
        start_idx = PHASE_ORDER.index(args.start_from)
        phases_to_run = PHASE_ORDER[start_idx:]
    else:
        phases_to_run = list(PHASE_ORDER)

    phases_to_run = [p for p in phases_to_run if p not in args.skip]

    # ------------------------------------------------------------------
    # Manual dataset detection: if a valid, ready-to-train dataset is
    # already present in training/dataset/, skip the data-preparation
    # phases and jump straight to training.
    # ------------------------------------------------------------------
    if args.only is None:
        manual = check_manual_dataset()
        if manual["detected"] and manual["valid"]:
            log.info("")
            log.info("Manual dataset detected — entering manual dataset mode")
            log.info("Skipping collection phase")
            log.info("Skipping preparation phase")

            auto_skip = {"collect", "validate", "prepare"}
            if manual["skip_augment"]:
                auto_skip.add("augment")
                log.info("Skipping augmentation phase (augmentations already present)")

            phases_to_run = [p for p in phases_to_run if p not in auto_skip]
            log.info("Starting YOLOv8 training")

    log.info("Pipeline phases: %s", ", ".join(phases_to_run))
    overall_t0 = time.time()
    results: dict[str, bool] = {}

    for phase_name in phases_to_run:
        script, desc = PHASES[phase_name]

        extra: list[str] = []
        if phase_name == "prepare" and args.force:
            extra.append("--force")
        if phase_name == "collect" and args.force_collect:
            extra.append("--force")

        ok = run_phase(phase_name, script, desc, extra_args=extra or None)
        results[phase_name] = ok

        if not ok:
            if phase_name in NON_FATAL_PHASES:
                log.warning(
                    "  Phase '%s' failed but is non-fatal -- continuing pipeline.",
                    phase_name,
                )
            else:
                log.error("  Pipeline ABORTED at phase: %s", phase_name)
                _print_summary(results, time.time() - overall_t0, aborted=phase_name)
                sys.exit(1)

    _print_summary(results, time.time() - overall_t0)


def _print_summary(
    results: dict[str, bool],
    elapsed: float,
    aborted: str | None = None,
) -> None:
    log.info("")
    log.info("=" * 65)
    log.info("  PIPELINE SUMMARY  (%.1fs total)", elapsed)
    log.info("=" * 65)

    for phase, ok in results.items():
        icon = "OK" if ok else "FAIL"
        log.info("    [%4s]  %s", icon, PHASES[phase][1])

    if aborted:
        log.info("")
        log.info("  ** Aborted at: %s **", aborted)
    else:
        log.info("")
        log.info("  Pipeline completed successfully!")
        log.info("  Trained weights: chess.pt (project root)")

    log.info("=" * 65)


if __name__ == "__main__":
    main()
