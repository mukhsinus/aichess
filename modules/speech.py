"""
Speech Synthesis Module -- Thesis Section 15.

Encapsulates pyttsx3 text-to-speech with a background worker thread
so that calling ``speak(text)`` never blocks the main video loop.
"""

import queue
import threading

import pyttsx3
from utils.logger import get_logger

logger = get_logger(__name__)


class SpeechEngine:
    """Non-blocking TTS powered by pyttsx3 and a daemon worker thread."""

    def __init__(self, rate: int = 150, voice_index: int = 1):
        self._engine = pyttsx3.init()
        voices = self._engine.getProperty("voices")
        self._engine.setProperty("rate", rate)
        if voice_index < len(voices):
            self._engine.setProperty("voice", voices[voice_index].id)
        else:
            logger.warning(
                "Voice index %d not available (%d voices found), using default",
                voice_index,
                len(voices),
            )
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        logger.info("SpeechEngine initialised  rate=%d  voice_index=%d", rate, voice_index)

    def start(self):
        """Launch the background worker that drains the speech queue."""
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("Speech worker thread started")

    def speak(self, text: str):
        """Enqueue *text* for asynchronous speech output."""
        self._queue.put(text)

    def shutdown(self):
        """Signal the worker to stop (blocks until the queue is drained)."""
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _worker(self):
        """Daemon loop: pull text off the queue and speak it."""
        while True:
            text = self._queue.get()
            if text is None:
                break
            self._engine.say(text)
            self._engine.runAndWait()
