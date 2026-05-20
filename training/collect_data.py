#!/usr/bin/env python
"""
Phase 1 -- Dataset Collection.

Downloads chess-piece detection datasets from public sources and stores them
under ``training/raw_data/<source_name>/``.

Supported sources (each is attempted independently; failures are non-fatal):
  1. Roboflow Universe  (requires ``roboflow`` package + API key)
  2. Kaggle             (requires ``kaggle`` package + credentials)
  3. HuggingFace Hub    (requires ``huggingface_hub``)
  4. Direct URL bundles (no auth)

Usage:
    python collect_data.py [--sources roboflow kaggle huggingface urls]

Set the following environment variables when needed:
    ROBOFLOW_API_KEY   -- Roboflow API key (free tier is enough)
    KAGGLE_USERNAME    -- Kaggle username
    KAGGLE_KEY         -- Kaggle API key
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests

from config import RAW_DATA_DIR

ALL_SOURCES = ["roboflow", "kaggle", "huggingface", "urls"]


# ---- helpers ---------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_and_extract_zip(url: str, dest: Path, label: str) -> bool:
    """Download a ZIP from *url*, extract to *dest*. Returns True on success."""
    print(f"  [{label}] Downloading {url} ...")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(dest)
        print(f"  [{label}] Extracted to {dest}")
        return True
    except Exception as exc:
        print(f"  [{label}] FAILED: {exc}")
        return False


# ---- Roboflow --------------------------------------------------------------

ROBOFLOW_DATASETS = [
    ("roboflow-100/chess-pieces-mjzgj", 2, "yolov8"),
]


def collect_roboflow(dest_root: Path) -> int:
    """Download public Roboflow datasets. Returns number of datasets fetched."""
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("[roboflow] ROBOFLOW_API_KEY not set -- skipping Roboflow.")
        print("           Get a free key at https://app.roboflow.com/settings/api")
        return 0

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[roboflow] 'roboflow' package not installed. pip install roboflow")
        return 0

    count = 0
    rf = Roboflow(api_key=api_key)
    for slug, version, fmt in ROBOFLOW_DATASETS:
        try:
            workspace, project_name = slug.split("/")
            project = rf.workspace(workspace).project(project_name)
            ds = project.version(version).download(fmt, location=str(dest_root / f"roboflow_{project_name}"))
            print(f"[roboflow] Downloaded {slug} v{version} -> {ds.location}")
            count += 1
        except Exception as exc:
            print(f"[roboflow] Failed to download {slug}: {exc}")
    return count


# ---- Kaggle ----------------------------------------------------------------

KAGGLE_DATASETS = [
    "anshulmehtakaggl/chess-pieces-detection-images-dataset",
]


def collect_kaggle(dest_root: Path) -> int:
    """Download Kaggle datasets. Returns count of successful downloads."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[kaggle] 'kaggle' package not installed. pip install kaggle")
        return 0
    except OSError:
        print("[kaggle] Kaggle credentials not configured.")
        print("         Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME + KAGGLE_KEY.")
        return 0

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    count = 0
    for slug in KAGGLE_DATASETS:
        try:
            dst = dest_root / f"kaggle_{slug.split('/')[-1]}"
            _ensure_dir(dst)
            api.dataset_download_files(slug, path=str(dst), unzip=True)
            print(f"[kaggle] Downloaded {slug} -> {dst}")
            count += 1
        except Exception as exc:
            print(f"[kaggle] Failed to download {slug}: {exc}")
    return count


# ---- HuggingFace -----------------------------------------------------------

HF_DATASETS = [
    "Francesco/chess-pieces",
]


def collect_huggingface(dest_root: Path) -> int:
    """Download HuggingFace datasets. Returns count."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[huggingface] 'datasets' package not installed. pip install datasets")
        return 0

    count = 0
    for name in HF_DATASETS:
        try:
            dst = dest_root / f"hf_{name.replace('/', '_')}"
            _ensure_dir(dst / "images")
            ds = load_dataset(name, split="train")
            for i, sample in enumerate(ds):
                img = sample.get("image")
                if img is not None:
                    img.save(dst / "images" / f"{i:06d}.jpg")
            print(f"[huggingface] Saved {len(ds)} images from {name} -> {dst}")
            count += 1
        except Exception as exc:
            print(f"[huggingface] Failed to load {name}: {exc}")
    return count


# ---- Direct URLs -----------------------------------------------------------

URL_BUNDLES = [
    (
        "https://github.com/erdogant/chess/raw/main/chess/data/chess_dataset.zip",
        "github_chess_dataset",
    ),
]


def collect_urls(dest_root: Path) -> int:
    """Download datasets from direct URLs. Returns count."""
    count = 0
    for url, label in URL_BUNDLES:
        dst = _ensure_dir(dest_root / label)
        if _download_and_extract_zip(url, dst, label):
            count += 1
    return count


# ---- main ------------------------------------------------------------------

COLLECTORS = {
    "roboflow": collect_roboflow,
    "kaggle": collect_kaggle,
    "huggingface": collect_huggingface,
    "urls": collect_urls,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect chess-piece datasets.")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=ALL_SOURCES,
        choices=ALL_SOURCES,
        help="Which sources to attempt (default: all)",
    )
    args = parser.parse_args()

    dest = _ensure_dir(RAW_DATA_DIR)
    print(f"Raw data directory: {dest}\n")

    total = 0
    for source in args.sources:
        print(f"--- Collecting from: {source} ---")
        n = COLLECTORS[source](dest)
        total += n
        print()

    print(f"Done. {total} dataset(s) downloaded to {dest}")
    if total == 0:
        print(
            "\nNo datasets were downloaded.  You can manually place YOLO-format\n"
            "datasets (images + labels) inside training/raw_data/<dataset_name>/\n"
            "and then run prepare_dataset.py."
        )


if __name__ == "__main__":
    main()
