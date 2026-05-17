"""
Chess Engine Module -- Thesis Section 13.

Thin wrapper around the ``stockfish`` Python package.  Exposes the same
method names (set_fen_position, get_top_moves, get_evaluation) so that
existing call-sites in main.py require zero changes beyond the import.
"""

from stockfish import Stockfish
from utils.logger import get_logger

logger = get_logger(__name__)


class ChessEngine:
    """Single Stockfish instance with position analysis helpers."""

    def __init__(self, path: str):
        """Create a Stockfish process at *path*."""
        self._sf = Stockfish(path=path)
        logger.info("Stockfish engine ready  path=%s", path)

    def set_fen_position(self, fen: str):
        """Load a FEN string into the engine."""
        self._sf.set_fen_position(fen)

    def get_top_moves(self, n: int = 3) -> list:
        """Return the top *n* moves as a list of dicts (Move, Centipawn, …)."""
        return self._sf.get_top_moves(n)

    def get_evaluation(self) -> dict:
        """Return the current position evaluation (type + value)."""
        return self._sf.get_evaluation()
