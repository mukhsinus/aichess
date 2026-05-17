"""
Board Reconstruction Module -- Thesis Section 12.

Converts object-detection results into a chess board representation:
  * pixel coordinates  ->  algebraic square notation  (get_chess_square)
  * detected pieces    ->  FEN string                 (create_fen_from_detections)
  * algebraic square   ->  pixel coordinates          (square_to_pixel)

All function bodies are extracted verbatim from main.py.
"""

from config.settings import COLUMNS, ROWS, PIECE_TO_FEN as piece_to_fen


def get_chess_square(x, y, board_size):
    """
    Convert pixel coordinates (x, y) in the warped image to chess notation.
    Returns (square notation, grid indices).
    """
    square_size = board_size // 8
    grid_x = x // square_size
    grid_y = y // square_size
    if not (0 <= grid_x < 8 and 0 <= grid_y < 8):
        return "Out of Bounds", (-1, -1)
    col = COLUMNS[grid_x]
    row = ROWS[7 - grid_y]
    return f"{col}{row}", (grid_x, grid_y)


def create_fen_from_detections(piece_positions, current_turn='w'):
    """
    Convert detected pieces (and their grid positions) into a FEN string.
    The board is built as an 8x8 matrix (row 0 = top).
    Castling rights are determined dynamically by checking if kings and rooks are in their starting positions.
    """
    # Build an empty board (row 0 = top, row 7 = bottom)
    board = [['' for _ in range(8)] for _ in range(8)]
    for piece, pos in piece_positions:
        grid_x, grid_y = pos
        if 0 <= grid_x < 8 and 0 <= grid_y < 8:
            board[grid_y][grid_x] = piece_to_fen.get(piece, '')

    # Create FEN rows from the board
    fen_rows = []
    for row in board:
        empty_count = 0
        row_fen = ''
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    row_fen += str(empty_count)
                    empty_count = 0
                row_fen += cell
        if empty_count > 0:
            row_fen += str(empty_count)
        fen_rows.append(row_fen)
    position = '/'.join(fen_rows)

    # Dynamically determine castling rights:
    # For white, the king should be on e1 (grid position (4,7))
    # For black, the king should be on e8 (grid position (4,0))
    castling = ""
    # White castling rights:
    if board[7][4] == 'K':
        if board[7][0] == 'R':  # Rook on a1
            castling += "Q"
        if board[7][7] == 'R':  # Rook on h1
            castling += "K"
    # Black castling rights:
    if board[0][4] == 'k':
        if board[0][0] == 'r':  # Rook on a8
            castling += "q"
        if board[0][7] == 'r':  # Rook on h8
            castling += "k"
    if castling == "":
        castling = "-"

    return f"{position} {current_turn} {castling} - 0 1"


def square_to_pixel(square, board_size):
    """
    Convert a square in algebraic notation (e.g., "e4") to pixel coordinates (center)
    on the warped chessboard image.
    """
    square_size = board_size / 8
    col = COLUMNS.index(square[0])
    row = 8 - int(square[1])
    x = int(col * square_size + square_size / 2)
    y = int(row * square_size + square_size / 2)
    return (x, y)
