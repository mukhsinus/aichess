import cv2
import numpy as np
import cvzone
import chess

from config.settings import STOCKFISH_PATH, SPEECH_RATE, SPEECH_VOICE_INDEX
from modules.camera import Camera
from modules.engine import ChessEngine
from modules.speech import SpeechEngine
from modules.board_detection import (
    find_chessboard_corners, warp_image, crop_inner_squares, draw_chess_grid,
)
from modules.piece_detection import detect_pieces
from modules.reconstruction import create_fen_from_detections, square_to_pixel

# ---------------------------------------------------------------------------
# Suppress YOLO Logging Messages
# ---------------------------------------------------------------------------
# logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Configuration and Initialization
# ---------------------------------------------------------------------------
CAMERA_ID = 1 #checkVidChess.mp4   chessvid2.mp4  testCheck.mp4 ../chessvid2.mp4
WIDTH, HEIGHT = 1280, 720
CROP_OFFSET = 0  # Pixels to crop from each side after warping||change back to 30

# Initialize Stockfish (path from config/settings.py, overridable via STOCKFISH_PATH env var)
stockfish = ChessEngine(STOCKFISH_PATH)
stockfishBlack = ChessEngine(STOCKFISH_PATH)

# Use a chess.Board for debugging; FENs will be built manually.
chess_board = chess.Board()
chess_board_black = chess.Board()
prev_board_white = chess.Board()
prev_board_black = chess.Board()
move_history = []
current_fen_candidate = None

cap = Camera(CAMERA_ID, WIDTH, HEIGHT)
speaker = SpeechEngine(rate=SPEECH_RATE, voice_index=SPEECH_VOICE_INDEX)

# ---------------------------------------------------------------------------
# Main Loop with Automatic Board Detection, Cropping, and Rotation
# ---------------------------------------------------------------------------
def main():
    last_stable_fen = None  # Last confirmed FEN (after stability check)
    stable_fen = ""  # FEN candidate that is being confirmed
    fen_counter = 0
    STABILITY_THRESHOLD = 5  # Frames required for stability
    best_moves_white = []  # Top moves for White
    best_moves_black = []  # Top moves for Black

    # Board detection mode flag
    board_detection_mode = True
    board_corners = None

    # Rotation state: 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
    rotation_state = 0


    global prev_board_white,prev_board_black, move_history, chess_board_black,current_fen_candidate,chess_board

    speaker.start()

    while True:
        success, img = cap.read()
        if not success:
            break

        img_display = img.copy()
        imgBlank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)  # Blank image for debugging

        if board_detection_mode:
            # Detect chessboard corners automatically
            corners = find_chessboard_corners(img)
            if corners.size != 0:
                cv2.drawContours(img_display, [corners], -1, (0, 255, 0), 10)
                cvzone.putTextRect(
                    img_display,
                    "Chessboard detected! Press 'C' to confirm or 'R' to retry",
                    (50, 50),
                    scale=1,
                    thickness=2,
                    colorR=(0, 255, 0)
                )
                cv2.imshow("Chessboard Detection", img_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    board_corners = corners
                    board_detection_mode = False
                    rotation_state = 0  # Reset rotation
                    cv2.destroyWindow("Chessboard Detection")
                elif key == ord('q'):
                    break
            else:
                cvzone.putTextRect(
                    img_display,
                    "No chessboard detected. Adjust camera position.",
                    (50, 50),
                    scale=1,
                    thickness=2,
                    colorR=(0, 0, 255)
                )
                cv2.imshow("Chessboard Detection", img_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # Warp the image using detected board corners
            img_warped, matrix, board_size = warp_image(img, board_corners.reshape(4, 2))
            # Apply rotation if needed
            if rotation_state == 1:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_state == 2:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_180)
            elif rotation_state == 3:
                img_warped = cv2.rotate(img_warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Crop out the extra border so only the inner squares remain
            img_warped_cropped, new_board_size = crop_inner_squares(img_warped, board_size, offset=CROP_OFFSET)

            # Draw grid lines on the cropped image and detect pieces there
            img_warped_cropped = draw_chess_grid(img_warped_cropped, new_board_size)
            detected_pieces = detect_pieces(img_warped_cropped, new_board_size)

            # Create a FEN candidate from detections
            current_fen_candidate = create_fen_from_detections(detected_pieces, current_turn='w')
            current_fen_candidate_black = create_fen_from_detections(detected_pieces, current_turn='b')

            # turn_char = 'w' if chess_board.fen().split(' ')[0] == chess_board_black.fen().split(' ')[0] else 'b'
            # current_fen_candidate = create_fen_from_detections(detected_pieces, current_turn=turn_char)
            # current_fen_candidate_black = create_fen_from_detections(detected_pieces, current_turn='b')


            # Stability check for FEN detection over consecutive frames
            if current_fen_candidate.split(' ')[0] == stable_fen.split(' ')[0]:
                fen_counter += 1
            else:
                stable_fen = current_fen_candidate
                fen_counter = 1

            if fen_counter >= STABILITY_THRESHOLD and stable_fen != last_stable_fen:
                last_stable_fen = stable_fen

                try:
                    stable_fen_black = f"{stable_fen.split(' ')[0]} b {stable_fen.split(' ')[2]} - 0 1"
                    temp_board_black = chess.Board(stable_fen_black)
                    temp_board = chess.Board(stable_fen)

                    if temp_board.is_check() or temp_board_black.is_check():
                        speaker.speak("check")
                        print("check")

                    if temp_board.is_checkmate() or temp_board_black.is_checkmate():
                        speaker.speak("mate")
                        print("mate")
                        # break

                    if temp_board.is_valid():
                        chess_board.set_fen(stable_fen)
                        # added
                        # chess_board_black.set_fen(current_fen_candidate_black)
                        # if prev_board_black.is_check():
                        #     speak("check")
                        #     print("check")

                        print("\nUpdated Chess Board:")
                        print(chess_board)
                        # print(temp_board)
                        print("Stable FEN:", stable_fen)
                        fen_white = f"{stable_fen.split(' ')[0]} w {stable_fen.split(' ')[2]} - 0 1"
                        fen_black = f"{stable_fen.split(' ')[0]} b {stable_fen.split(' ')[2]} - 0 1"
                        # if chess_board.is_check():
                        #     speak("check")
                        #     print("check")
                        try:

                            stockfish.set_fen_position(fen_white)
                            best_moves_white = stockfish.get_top_moves(3)

                            new_board = chess.Board(fen_white)

                            # find actual move from prev_board
                            played = None
                            for m in prev_board_white.pseudo_legal_moves:
                                tb = prev_board_white.copy()
                                tb.push(m)
                                # if tb.is_check():
                                #     speak("Check")
                                #     print("check white")
                                if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                    played = m
                                    break
                            if played:
                                san = prev_board_white.san(played)
                                turn = 'White' if prev_board_white.turn == chess.WHITE else 'Black'
                                stockfish.set_fen_position(fen_white)
                                eval_info = stockfish.get_evaluation()
                                cp = eval_info.get('value')
                                move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                                # if prev_board_white.is_checkmate():
                                #     speak("Checkmate White")
                                # elif prev_board_white.is_check():
                                #     speak("Check White")
                                # speak('working')
                            prev_board_white = new_board
                            chess_board.set_fen(fen_white)
                            last_stable_fen = stable_fen
                            # print("Move History:")
                            # for h in move_history:
                            #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")

                        except Exception as e:
                            print("Error getting top moves for White:", e)
                            best_moves_white = []
                        try:
                            stockfish.set_fen_position(fen_black)
                            best_moves_black = stockfish.get_top_moves(3)

                            new_board = chess.Board(fen_black)

                            # find actual move from prev_board_black
                            black_check_fen = ""
                            played = None
                            for m in prev_board_black.pseudo_legal_moves:
                                tb = prev_board_black.copy()
                                tb.push(m)

                                if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                    played = m
                                    black_check_fen = tb.board_fen()
                                    # if tb.is_into_check():
                                    #     speak("Check")
                                    #     print("check black")
                                    break
                            if played:
                                fen_black = f"{fen_black.split(' ')[0]} b {fen_black.split(' ')[2]} - 0 1"

                                san = prev_board_black.san(played)
                                turn = 'White' if prev_board_black.turn == chess.WHITE else 'Black'
                                stockfish.set_fen_position(fen_black)
                                chess_board_black.set_fen(black_check_fen)
                                # if chess_board.is_check():
                                #     speak("check")
                                #     print("check")
                                eval_info = stockfish.get_evaluation()
                                cp = eval_info.get('value')
                                move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                                # if prev_board_black.is_checkmate():
                                #     speak("Checkmate Black")
                                # elif prev_board_black.is_check():
                                #     speak("Check Black")

                            prev_board_black = new_board
                            chess_board.set_fen(fen_black)
                            # test
                            # chess_board_black.set_fen(fen_black)
                            last_stable_fen = stable_fen
                            # if chess_board_black.is_check():
                            #     speak("check")
                            #     print("check")

                            # print("Move History:")
                            # for h in move_history:
                            #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")
                        except Exception as e:
                            print("Error getting top moves for Black:", e)
                            best_moves_black = []
                        print("Top moves for White:", best_moves_white)
                        print("Top moves for Black:", best_moves_black)

                    else:
                        print("white board not working brev temp board below")
                        # stable_fen = stable_fen_black
                        # temp_board.set_fen(stable_fen_black)
                        print(temp_board)

                        if temp_board_black.is_valid():
                            print('good black board brev')
                            chess_board_black.set_fen(stable_fen_black)
                            # added
                            # chess_board_black.set_fen(current_fen_candidate_black)
                            # if prev_board_black.is_check():
                            #     speak("check")
                            #     print("check")

                            print("\nUpdated Chess Board:")
                            print(chess_board_black)
                            # print(temp_board)
                            print("Stable FEN:", stable_fen_black)
                            fen_white = f"{stable_fen_black.split(' ')[0]} w {stable_fen_black.split(' ')[2]} - 0 1"
                            fen_black = f"{stable_fen_black.split(' ')[0]} b {stable_fen_black.split(' ')[2]} - 0 1"
                            # if chess_board.is_check():
                            #     speak("check")
                            #     print("check")

                            try:
                                stockfishBlack.set_fen_position(fen_black)
                                best_moves_black = stockfishBlack.get_top_moves(3)

                                new_board = chess.Board(fen_black)

                                # find actual move from prev_board_black
                                black_check_fen = ""
                                played = None
                                for m in prev_board_black.pseudo_legal_moves:
                                    tb = prev_board_black.copy()
                                    tb.push(m)

                                    if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                        played = m
                                        # black_check_fen = tb.board_fen()
                                        # if tb.is_into_check():
                                        #     speak("Check")
                                        #     print("check black")
                                        break
                                if played:
                                    fen_black = f"{fen_black.split(' ')[0]} b {fen_black.split(' ')[2]} - 0 1"

                                    san = prev_board_black.san(played)
                                    turn = 'White' if prev_board_black.turn == chess.WHITE else 'Black'
                                    stockfishBlack.set_fen_position(fen_black)
                                    chess_board_black.set_fen(black_check_fen)
                                    # if chess_board.is_check():
                                    #     speak("check")
                                    #     print("check")
                                    eval_info = stockfishBlack.get_evaluation()
                                    cp = eval_info.get('value')
                                    move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                                    # if prev_board_black.is_checkmate():
                                    #     speak("Checkmate Black")
                                    # elif prev_board_black.is_check():
                                    #     speak("Check Black")

                                prev_board_black = new_board
                                chess_board_black.set_fen(stable_fen_black)
                                # test
                                # chess_board_black.set_fen(fen_black)
                                last_stable_fen = stable_fen
                                # if chess_board_black.is_check():
                                #     speak("check")
                                #     print("check")

                                # print("Move History:")
                                # for h in move_history:
                                #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")
                            except Exception as e:
                                print("Error getting top moves for Black:", e)
                                best_moves_black = []

                            try:

                                stockfishBlack.set_fen_position(fen_white)
                                best_moves_white = stockfishBlack.get_top_moves(3)

                                new_board = chess.Board(fen_black)

                                # find actual move from prev_board
                                played = None
                                for m in prev_board_white.pseudo_legal_moves:
                                    tb = prev_board_white.copy()
                                    tb.push(m)
                                    # if tb.is_check():
                                    #     speak("Check")
                                    #     print("check white")
                                    if tb.board_fen().split(' ')[0] == new_board.board_fen().split(' ')[0]:
                                        played = m
                                        break
                                if played:
                                    san = prev_board_white.san(played)
                                    turn = 'White' if prev_board_white.turn == chess.WHITE else 'Black'
                                    stockfishBlack.set_fen_position(fen_white)
                                    eval_info = stockfishBlack.get_evaluation()
                                    cp = eval_info.get('value')
                                    move_history.append({'turn': turn, 'move': san, 'evaluation_cp': cp})
                                    # if prev_board_white.is_checkmate():
                                    #     speak("Checkmate White")
                                    # elif prev_board_white.is_check():
                                    #     speak("Check White")
                                    # speak('working')
                                prev_board_white = new_board
                                chess_board_black.set_fen(fen_white)
                                last_stable_fen = stable_fen
                                # print("Move History:")
                                # for h in move_history:
                                #     print(f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp")

                            except Exception as e:
                                print("Error getting top moves for White:", e)
                                best_moves_white = []

                            print("Top moves for White:", best_moves_white)
                            print("Top moves for Black:", best_moves_black)


                    # else:
                    #     print("Detected FEN is not valid:", stable_fen)
                    #     best_moves_white = best_moves_black = []
                    #     # else_fen = chess_board.fen()
                    #     # fen_black = f"{else_fen.split(' ')[0]} b {else_fen.split(' ')[2]} - 0 1"
                    #     # temp_board.set_fen(fen_black)
                    #     # fen_black = f"{stable_fen.split(' ')[0]} b {stable_fen.split(' ')[2]} - 0 1"
                    #     # stable_fen = fen_black
                    #     # chess_board.set_fen(fen_black)
                    #     stable_fen = stable_fen_black
                    #     # chess_board = chess_board_black.copy()
                    #
                    #     print(f"else fen {stable_fen_black}")

                except Exception as e:
                    print("Error processing FEN:", stable_fen, e)
                    best_moves_white = best_moves_black = []

                # print move history
                print("Move History:")
                for h in move_history:
                    print(f"{h['turn']} {h['move']} ⇒ {h['evaluation_cp']} cp")

            y0 = 30
            dy = 30
            for i, h in enumerate(move_history):  # show last 10 moves
                text = f"{h['turn']} {h['move']} => {h['evaluation_cp']} cp"
                cv2.putText(
                    imgBlank,
                    text,
                    (10, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            # Draw arrows for top moves on the cropped image
            for move_info in best_moves_white:
                move = move_info.get("Move", None)
                score = move_info.get("Centipawn", None)

                if move and len(move) >= 4:
                    start_square = move[:2]
                    end_square = move[2:4]
                    start_px = square_to_pixel(start_square, new_board_size)
                    end_px = square_to_pixel(end_square, new_board_size)
                    cv2.arrowedLine(img_warped_cropped, start_px, end_px, (255, 0, 0), 2, tipLength=0.3)
                    cv2.putText(img_warped_cropped, f"{score}", (end_px[0], end_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for move_info in best_moves_black:
                move = move_info.get("Move", None)
                score = move_info.get("Centipawn", None)

                if move and len(move) >= 4:
                    start_square = move[:2]
                    end_square = move[2:4]
                    start_px = square_to_pixel(start_square, new_board_size)
                    end_px = square_to_pixel(end_square, new_board_size)
                    cv2.arrowedLine(img_warped_cropped, start_px, end_px, (0, 0, 255), 2, tipLength=0.3)
                    cv2.putText(img_warped_cropped, f"{score}", (end_px[0], end_px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # Display best moves as text on the original image
            display_text = ""
            if best_moves_white:
                white_moves = ", ".join([m.get("Move", "") for m in best_moves_white])
                display_text += f"White: {white_moves}  "
            if best_moves_black:
                black_moves = ", ".join([m.get("Move", "") for m in best_moves_black])
                display_text += f"Black: {black_moves}"
            if display_text:
                cv2.putText(img, display_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display rotation and recalibration instructions
            rotation_text = f"Current Rotation: {rotation_state * 90}°"
            cvzone.putTextRect(
                img_warped_cropped,
                rotation_text,
                (10, 30),
                scale=1,
                thickness=2,
                colorR=(0, 0, 0)
            )
            cvzone.putTextRect(
                img,
                "Press 'R' to rotate board | 'D' to detect new board",
                (50, HEIGHT - 50),
                scale=1,
                thickness=2,
                colorR=(0, 0, 0)
            )
            # cv2.imshow("Warped Chessboard", img_warped_cropped)
            # cv2.imshow("Original Image", img)
            # played_moves = [ f"{h['turn']} {h['move']} ⇒ {h['evaluation_cp']} cp" for h in move_history]
            # cv2.putText(imgBlank,f"{played_moves}"+"\n",(0,0),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



            cvzone.putTextRect(
                imgBlank,
                "Move History",
                (700, 50),
                scale=5,
                thickness=2,
                colorR=(0, 255, 0)
            )

            # imgStack = cvzone.stackImages([img_warped_cropped,imgBlank,img],2,0.7)
            # cv2.imshow("Stacked Image", imgStack)

            imgStack = cvzone.stackImages([img_warped_cropped, imgBlank, img], 2, 1.5)
            #
            # Create a resizable named window
            cv2.namedWindow("Stacked Image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Stacked Image", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

            cv2.imshow("Stacked Image", imgStack)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Rotate the warped image (90° per press)
                rotation_state = (rotation_state + 1) % 4
                # Reset FEN stability on rotation change
                fen_counter = 0
                last_stable_fen = None
                stable_fen = ""
                best_moves_white = []
                best_moves_black = []
            elif key == ord('d'):
                # Switch back to board detection mode
                board_detection_mode = True
                # cv2.destroyWindow("Warped Chessboard")
                # cv2.destroyWindow("Original Image")
                cv2.destroyWindow("Stacked Image")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

