#!/usr/bin/env python
"""
Phases 1-2 -- Dataset Preparation.

Walks every sub-folder of ``training/raw_data/``, ingests image + label pairs,
de-duplicates them by perceptual hash, remaps class IDs to the canonical 12
chess-piece classes, normalises images to a uniform size, and creates the final
train / val / test splits in ``training/dataset/``.

The output layout follows the YOLOv8 convention::

    dataset/
        images/
            train/  val/  test/
        labels/
            train/  val/  test/

Usage:
    python prepare_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import (
    ALIAS_MAP,
    CLASS_NAMES,
    CLASS_TO_ID,
    DATASET_DIR,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    RAW_DATA_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


# ---------------------------------------------------------------------------
# Perceptual hashing for deduplication
# ---------------------------------------------------------------------------

def _dhash(image: np.ndarray, hash_size: int = 16) -> str:
    """Compute a difference-hash of *image* (already loaded by OpenCV)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Class-name resolution
# ---------------------------------------------------------------------------

def _resolve_class_name(raw_name: str) -> str | None:
    """Map *raw_name* to a canonical class name, or None if unknown."""
    if raw_name in CLASS_TO_ID:
        return raw_name
    return ALIAS_MAP.get(raw_name)


def _build_source_class_map(names_file: Path) -> dict[int, int] | None:
    """Read a YOLO ``classes.txt`` / ``obj.names`` and return {src_id: dst_id}.

    Returns None when the file does not exist or contains no mappable classes.
    """
    if not names_file.exists():
        return None
    lines = names_file.read_text().strip().splitlines()
    mapping: dict[int, int] = {}
    for idx, line in enumerate(lines):
        canonical = _resolve_class_name(line.strip())
        if canonical is not None:
            mapping[idx] = CLASS_TO_ID[canonical]
    return mapping if mapping else None


def _try_find_names_file(root: Path) -> Path | None:
    """Search common locations for a class-names file."""
    for candidate in [
        root / "classes.txt",
        root / "obj.names",
        root / "data" / "obj.names",
        root / "_classes.txt",
    ]:
        if candidate.exists():
            return candidate

    # Also check data.yaml / dataset.yaml for 'names:' key
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        yaml_path = root / yaml_name
        if yaml_path.exists():
            return _extract_names_from_yaml(yaml_path)
    return None


def _extract_names_from_yaml(yaml_path: Path) -> Path | None:
    """Parse a YOLO data YAML and write a temporary classes.txt, return path."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return None
    names = data.get("names")
    if not names:
        return None

    if isinstance(names, dict):
        max_idx = max(names.keys())
        ordered = [names.get(i, "unknown") for i in range(max_idx + 1)]
    elif isinstance(names, list):
        ordered = names
    else:
        return None

    tmp = yaml_path.parent / "_classes_from_yaml.txt"
    tmp.write_text("\n".join(ordered))
    return tmp


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _find_label_for_image(img_path: Path) -> Path | None:
    """Heuristic: look for a .txt label next to the image or under ../labels/."""
    txt = img_path.with_suffix(".txt")
    if txt.exists():
        return txt
    labels_dir = img_path.parent.parent / "labels" / img_path.parent.name
    txt = labels_dir / img_path.with_suffix(".txt").name
    if txt.exists():
        return txt
    labels_dir2 = img_path.parent.parent / "labels"
    txt2 = labels_dir2 / img_path.with_suffix(".txt").name
    if txt2.exists():
        return txt2
    return None


def _remap_label(label_path: Path, class_map: dict[int, int]) -> list[str]:
    """Read a YOLO label file, remap class IDs, drop unknown classes."""
    lines_out: list[str] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        src_id = int(parts[0])
        dst_id = class_map.get(src_id)
        if dst_id is None:
            continue
        lines_out.append(f"{dst_id} {' '.join(parts[1:])}")
    return lines_out


def ingest_raw_data() -> list[tuple[Path, list[str]]]:
    """Walk raw_data/, return list of (image_path, remapped_label_lines)."""
    if not RAW_DATA_DIR.exists():
        print(f"No raw data directory found at {RAW_DATA_DIR}")
        return []

    pairs: list[tuple[Path, list[str]]] = []

    for source_dir in sorted(RAW_DATA_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        print(f"Scanning {source_dir.name} ...")

        names_file = _try_find_names_file(source_dir)
        class_map: dict[int, int] | None = None
        if names_file is not None:
            class_map = _build_source_class_map(names_file)
            if class_map:
                print(f"  Class map ({len(class_map)} classes): {class_map}")
            else:
                print(f"  WARNING: found names file but could not map any class.")

        for img_path in sorted(source_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = _find_label_for_image(img_path)
            if label_path is None:
                continue

            if class_map is not None:
                remapped = _remap_label(label_path, class_map)
            else:
                remapped = label_path.read_text().strip().splitlines()

            if remapped:
                pairs.append((img_path, remapped))

    print(f"Total image-label pairs found: {len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(pairs: list[tuple[Path, list[str]]]) -> list[tuple[Path, list[str]]]:
    """Remove near-duplicate images using perceptual hashing."""
    seen: set[str] = set()
    unique: list[tuple[Path, list[str]]] = []
    dups = 0
    for img_path, labels in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h = _dhash(img)
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        unique.append((img_path, labels))
    print(f"Deduplication: {dups} duplicates removed, {len(unique)} unique samples remain.")
    return unique


# ---------------------------------------------------------------------------
# Normalize + write dataset
# ---------------------------------------------------------------------------

def write_dataset(pairs: list[tuple[Path, list[str]]], force: bool = False) -> None:
    """Normalize images, split, and write the final YOLO dataset."""
    if DATASET_DIR.exists():
        if force:
            shutil.rmtree(DATASET_DIR)
        else:
            print(f"Dataset directory already exists: {DATASET_DIR}")
            print("Use --force to overwrite.")
            return

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split_name, split_pairs in splits.items():
        img_dir = DATASET_DIR / "images" / split_name
        lbl_dir = DATASET_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx, (img_path, label_lines) in enumerate(split_pairs):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            out_name = f"{split_name}_{idx:06d}"
            cv2.imwrite(str(img_dir / f"{out_name}.jpg"), img_resized)
            (lbl_dir / f"{out_name}.txt").write_text("\n".join(label_lines) + "\n")

        print(f"  {split_name}: {len(split_pairs)} samples")

    # Write classes.txt alongside the dataset for reference
    (DATASET_DIR / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n")
    print(f"Dataset written to {DATASET_DIR}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO chess-piece dataset.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset/")
    args = parser.parse_args()

    pairs = ingest_raw_data()
    if not pairs:
        print("Nothing to process. Run collect_data.py first, or place data in raw_data/.")
        return
    pairs = deduplicate(pairs)
    write_dataset(pairs, force=args.force)


if __name__ == "__main__":
    main()
