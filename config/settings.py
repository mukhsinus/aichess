"""
Central configuration for the AI Chess Assistant.
Thesis Section 8 -- System Architecture: all tunable parameters in one place.

Every constant that was previously hardcoded across main.py is collected here.
Import from this module instead of scattering magic numbers through source files.

Environment variable overrides are supported for machine-specific paths
(STOCKFISH_PATH, CAMERA_ID) so the code runs without editing on any machine.
"""

import os

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# int for device index, or a string file path for video playback.
# Override with env var CAMERA_ID (set to a number or file path).
_cam_env = os.environ.get("CAMERA_ID")
if _cam_env is not None:
    try:
        CAMERA_ID = int(_cam_env)
    except ValueError:
        CAMERA_ID = _cam_env
else:
    CAMERA_ID = 1

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# YOLO Model
# ---------------------------------------------------------------------------
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "chess.pt")
DETECTION_CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Display / Warping
# ---------------------------------------------------------------------------
DISPLAY_SIZE = (1280, 720)
BOARD_MARGIN = 100
CROP_OFFSET = 0  # pixels to crop from each side after warping

# ---------------------------------------------------------------------------
# Stockfish Engine
# ---------------------------------------------------------------------------
STOCKFISH_PATH = os.environ.get(
    "STOCKFISH_PATH",
    "C:/Users/Presision/Downloads/stockfish-windows-x86-64/stockfish/stockfish-windows-x86-64.exe",
)
STOCKFISH_TOP_MOVES = 3

# ---------------------------------------------------------------------------
# FEN Stability
# ---------------------------------------------------------------------------
STABILITY_THRESHOLD = 5  # consecutive identical frames before accepting a FEN

# ---------------------------------------------------------------------------
# Speech Synthesis (pyttsx3)
# ---------------------------------------------------------------------------
SPEECH_RATE = 150
SPEECH_VOICE_INDEX = 1  # index into pyttsx3 voices list

# ---------------------------------------------------------------------------
# Board Detection (contour filtering)
# ---------------------------------------------------------------------------
MIN_CONTOUR_AREA = 5000
CONTOUR_APPROX_EPSILON = 0.02  # fraction of perimeter for approxPolyDP
CANNY_THRESHOLD_LOW = 10
CANNY_THRESHOLD_HIGH = 50
GAUSSIAN_BLUR_KERNEL = (5, 5)
GAUSSIAN_BLUR_SIGMA = 1

# ---------------------------------------------------------------------------
# Chess Notation Maps
# ---------------------------------------------------------------------------
COLUMNS = "abcdefgh"
ROWS = "12345678"

PIECE_TO_FEN = {
    "white-pawn": "P", "white-knight": "N", "white-bishop": "B",
    "white-rook": "R", "white-queen": "Q", "white-king": "K",
    "black-pawn": "p", "black-knight": "n", "black-bishop": "b",
    "black-rook": "r", "black-queen": "q", "black-king": "k",
}

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
ARROW_WHITE_COLOR = (255, 0, 0)   # BGR blue
ARROW_BLACK_COLOR = (0, 0, 255)   # BGR red
ARROW_THICKNESS = 2
ARROW_TIP_LENGTH = 0.3
GRID_LINE_COLOR = (255, 255, 255)
GRID_LINE_THICKNESS = 2
