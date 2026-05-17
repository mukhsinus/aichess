"""
Camera Module -- Thesis Section 9.

Wraps OpenCV VideoCapture to provide frame acquisition from a webcam
or video file.  The public interface (read / release) mirrors OpenCV
so call-sites in main.py require only a one-line init change.
"""

import cv2
from utils.logger import get_logger

logger = get_logger(__name__)


class Camera:
    """Acquire frames from a webcam device index or a video file path."""

    def __init__(self, source, width: int = 1280, height: int = 720):
        """Open *source* (int device id or str file path) at the given resolution."""
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error("Failed to open camera source: %s", source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info("Camera opened  source=%s  %dx%d", source, width, height)

    def read(self):
        """Return (success, frame) exactly like cv2.VideoCapture.read()."""
        return self._cap.read()

    def release(self):
        """Release the underlying capture device."""
        self._cap.release()
        logger.info("Camera released")
