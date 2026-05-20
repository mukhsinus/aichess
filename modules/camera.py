"""
Camera Module -- Thesis Section 9.

Wraps OpenCV VideoCapture to provide frame acquisition from a webcam
or video file.  The public interface (read / release) mirrors OpenCV
so call-sites in main.py require only a one-line init change.

The ``find_best_camera`` helper honours ``CAMERA_ID`` from settings first.
Only when the preferred index fails does it fall back to an automatic scan
that prefers external USB webcams over the built-in laptop camera.
"""

import sys

import cv2
from config.settings import CAMERA_ID
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_CAMERA_INDEX = 5
_IS_WINDOWS = sys.platform == "win32"


def _open_capture(index: int) -> cv2.VideoCapture:
    """Open a VideoCapture with the DirectShow backend on Windows."""
    if _IS_WINDOWS:
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def _test_camera(index: int) -> bool:
    """Return True if *index* can be opened and delivers at least one frame."""
    cap = _open_capture(index)
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def find_best_camera(width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Return a ready-to-use ``cv2.VideoCapture`` set to *width* x *height*.

    Selection strategy
    ------------------
    1. Try ``CAMERA_ID`` from settings **first**.  If it opens and a frame
       can be read, use it immediately — no scan, no fallback.
    2. Only when the preferred index fails, scan 0.._MAX_CAMERA_INDEX.
    3. Among the working indexes found by the scan, prefer non-zero
       (external USB) over index 0 (typically the built-in laptop camera).
    4. If nothing works at all, raise ``RuntimeError``.
    """

    preferred = CAMERA_ID

    if isinstance(preferred, int):
        logger.info("Trying preferred camera index %d", preferred)
        if _test_camera(preferred):
            logger.info("Preferred camera opened successfully")
            cap = _open_capture(preferred)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info("Camera opened  index=%d  %dx%d", preferred, width, height)
            return cap
        logger.warning("Preferred camera failed, starting fallback scan")

    working: list[int] = []

    for idx in range(_MAX_CAMERA_INDEX + 1):
        if _test_camera(idx):
            working.append(idx)
            logger.info("Camera detected at index %d", idx)
        else:
            logger.debug("Index %d not available or returned no frame", idx)

    if not working:
        raise RuntimeError(
            "No working camera found (scanned indexes 0..%d). "
            "Please connect a webcam and try again." % _MAX_CAMERA_INDEX
        )

    logger.info("Working camera indexes: %s", working)

    external = [i for i in working if i != 0]

    if external:
        selected = external[0]
        logger.info("Selected external USB camera at index %d", selected)
    else:
        selected = 0
        logger.warning(
            "No external USB camera detected — falling back to built-in camera (index 0)"
        )

    cap = _open_capture(selected)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    logger.info("Camera opened  index=%d  %dx%d", selected, width, height)
    return cap


class Camera:
    """Acquire frames from a webcam device index or a video file path."""

    def __init__(self, source, width: int = 1280, height: int = 720):
        """Open *source* (int device id or str file path) at the given resolution."""
        if isinstance(source, int):
            self._cap = _open_capture(source)
        else:
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
