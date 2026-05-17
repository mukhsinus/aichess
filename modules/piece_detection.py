"""
Piece Detection Module -- Thesis Section 11.

Loads the YOLO model and detects chess pieces on a warped board image.
Each detection is classified into one of 12 piece classes and mapped
to an 8x8 grid position via the reconstruction module.

The function body is extracted verbatim from main.py.
"""

import cvzone
from ultralytics import YOLO

from config.settings import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE_THRESHOLD,
    PIECE_TO_FEN as piece_to_fen,
)
from modules.reconstruction import get_chess_square

model = YOLO(YOLO_MODEL_PATH)
names = model.names


def detect_pieces(img_warped, board_size):
    """
    Detect chess pieces on the warped chessboard using YOLO.
    Returns a list of detected pieces with their grid positions and draws the detections.
    """
    detected_pieces = []
    results = model(img_warped,verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_id = int(box.cls[0])
            class_name = names[class_id]
            conf = float(box.conf[0])
            if conf < DETECTION_CONFIDENCE_THRESHOLD:
                continue
            square, (grid_x, grid_y) = get_chess_square(cx, cy, board_size)
            if grid_x == -1 or grid_y == -1:
                continue
            detected_pieces.append((class_name, (grid_x, grid_y)))
            cvzone.putTextRect(
                img_warped,
                f'{piece_to_fen[class_name]} {square}',
                (max(0, x1), max(35, y1)),
                scale=1,
                thickness=1,
                colorR=(255, 255, 0),
                colorT=(0, 0, 0)
            )
    return detected_pieces
