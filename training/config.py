"""
Training Pipeline Configuration.

Central configuration for the dataset collection, preparation, augmentation,
training, evaluation, and integration pipeline.  All paths are relative to
the ``training/`` directory unless stated otherwise.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent

RAW_DATA_DIR = TRAINING_ROOT / "raw_data"
DATASET_DIR = TRAINING_ROOT / "dataset"
AUGMENTED_DIR = TRAINING_ROOT / "augmented"
RUNS_DIR = TRAINING_ROOT / "runs"

DATASET_YAML = TRAINING_ROOT / "dataset.yaml"
FINAL_WEIGHTS = PROJECT_ROOT / "chess.pt"

# ---------------------------------------------------------------------------
# 12 Chess-piece classes (must match config/settings.py PIECE_TO_FEN keys)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "white-pawn",
    "white-knight",
    "white-bishop",
    "white-rook",
    "white-queen",
    "white-king",
    "black-pawn",
    "black-knight",
    "black-bishop",
    "black-rook",
    "black-queen",
    "black-king",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Common alias mappings  (source dataset class name -> canonical name)
# Extend this dict whenever a new dataset uses unexpected labels.
# ---------------------------------------------------------------------------
ALIAS_MAP = {
    # Lowercase / no-prefix variants
    "pawn-w": "white-pawn", "pawn-b": "black-pawn",
    "knight-w": "white-knight", "knight-b": "black-knight",
    "bishop-w": "white-bishop", "bishop-b": "black-bishop",
    "rook-w": "white-rook", "rook-b": "black-rook",
    "queen-w": "white-queen", "queen-b": "black-queen",
    "king-w": "white-king", "king-b": "black-king",
    "White Pawn": "white-pawn", "Black Pawn": "black-pawn",
    "White Knight": "white-knight", "Black Knight": "black-knight",
    "White Bishop": "white-bishop", "Black Bishop": "black-bishop",
    "White Rook": "white-rook", "Black Rook": "black-rook",
    "White Queen": "white-queen", "Black Queen": "black-queen",
    "White King": "white-king", "Black King": "black-king",
    "white_pawn": "white-pawn", "black_pawn": "black-pawn",
    "white_knight": "white-knight", "black_knight": "black-knight",
    "white_bishop": "white-bishop", "black_bishop": "black-bishop",
    "white_rook": "white-rook", "black_rook": "black-rook",
    "white_queen": "white-queen", "black_queen": "black-queen",
    "white_king": "white-king", "black_king": "black-king",
    "w-pawn": "white-pawn", "b-pawn": "black-pawn",
    "w-knight": "white-knight", "b-knight": "black-knight",
    "w-bishop": "white-bishop", "b-bishop": "black-bishop",
    "w-rook": "white-rook", "b-rook": "black-rook",
    "w-queen": "white-queen", "b-queen": "black-queen",
    "w-king": "white-king", "b-king": "black-king",
    "wp": "white-pawn", "bp": "black-pawn",
    "wn": "white-knight", "bn": "black-knight",
    "wb": "white-bishop", "bb": "black-bishop",
    "wr": "white-rook", "br": "black-rook",
    "wq": "white-queen", "bq": "black-queen",
    "wk": "white-king", "bk": "black-king",
    "P": "white-pawn", "p": "black-pawn",
    "N": "white-knight", "n": "black-knight",
    "B": "white-bishop", "b": "black-bishop",
    "R": "white-rook", "r": "black-rook",
    "Q": "white-queen", "q": "black-queen",
    "K": "white-king", "k": "black-king",
}

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
IMG_SIZE = 640
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Dataset split ratios  (must sum to 1.0)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
YOLO_BASE_MODEL = "yolov8s.pt"
EPOCHS = 50
BATCH_SIZE = 16
IMGSZ = 640
WORKERS = 4

# ---------------------------------------------------------------------------
# Augmentation settings
# ---------------------------------------------------------------------------
AUG_PER_IMAGE = 3  # augmented copies per original image
