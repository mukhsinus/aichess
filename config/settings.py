"""
Central configuration for the AI Chess Assistant.
Thesis Section 8 -- System Architecture: all tunable parameters in one place.

Every constant that was previously hardcoded across main.py is collected here.
Import from this module instead of scattering magic numbers through source files.

Environment variable overrides are supported for machine-specific paths
(STOCKFISH_PATH, CAMERA_ID) so the code runs without editing on any machine.
"""

import glob
import os

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

YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "chess.pt")
DETECTION_CONFIDENCE_THRESHOLD = 0.6

DISPLAY_SIZE = (1280, 720)
BOARD_MARGIN = 100
CROP_OFFSET = 0

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_stockfish() -> str:
    """Return Stockfish path: env var > auto-detected exe in project root > PATH."""
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env
    import subprocess
    for candidate in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "stockfish*.exe"))):
        try:
            p = subprocess.run(
                [candidate, "quit"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            if p.returncode == 0:
                return candidate
        except Exception:
            continue
    return "stockfish"


STOCKFISH_PATH = _find_stockfish()
STOCKFISH_TOP_MOVES = 3

STABILITY_THRESHOLD = 5

SPEECH_RATE = 150
SPEECH_VOICE_INDEX = 1

MIN_CONTOUR_AREA = 5000
CONTOUR_APPROX_EPSILON = 0.02
CANNY_THRESHOLD_LOW = 10
CANNY_THRESHOLD_HIGH = 50
GAUSSIAN_BLUR_KERNEL = (5, 5)
GAUSSIAN_BLUR_SIGMA = 1

COLUMNS = "abcdefgh"
ROWS = "12345678"

PIECE_TO_FEN = {
    "white-pawn": "P", "white-knight": "N", "white-bishop": "B",
    "white-rook": "R", "white-queen": "Q", "white-king": "K",
    "black-pawn": "p", "black-knight": "n", "black-bishop": "b",
    "black-rook": "r", "black-queen": "q", "black-king": "k",
}

ARROW_WHITE_COLOR = (255, 0, 0)
ARROW_BLACK_COLOR = (0, 0, 255)
ARROW_THICKNESS = 2
ARROW_TIP_LENGTH = 0.3
GRID_LINE_COLOR = (255, 255, 255)
GRID_LINE_THICKNESS = 2
