import cv2
import numpy as np
import pickle
import cvzone
from cvzone.HandTrackingModule import HandDetector

cam_id = 1
width, height = 1280, 720

board_corners_file = "chessboard_corners.p"
with open(board_corners_file, 'rb') as f:
    board_corners = pickle.load(f)

cap = cv2.VideoCapture(cam_id)
cap.set(3, width)
cap.set(4, height)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.5)

columns = "abcdefgh"
rows = "12345678"


def warp_image(img, points):
    """Warp the detected chessboard to a top-down view."""
    display_width, display_height = 1280, 720
    board_size = min(display_width, display_height) - 100

    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarped = cv2.warpPerspective(img, matrix, (int(board_size), int(board_size)))

    return imgWarped, matrix, int(board_size)


def get_chess_square(x, y, board_size):
    """Map pixel positions to chess notation."""
    square_size = board_size // 8
    grid_x = x // square_size
    grid_y = y // square_size

    if not (0 <= grid_x < 8) or not (0 <= grid_y < 8):
        return "Out of Bounds", (-1, -1)

    col = columns[grid_x]
    row = rows[7 - grid_y]
    return f"{col}{row}", (grid_x, grid_y)


def draw_chess_grid(img, board_size):
    """Draws grid lines on the warped chessboard."""
    square_size = board_size // 8
    for i in range(1, 8):
        cv2.line(img, (i * square_size, 0), (i * square_size, board_size), (255, 255, 255), 2)
        cv2.line(img, (0, i * square_size), (board_size, i * square_size), (255, 255, 255), 2)
    return img


while True:
    success, img = cap.read()
    if not success:
        break

    imgWarped, matrix, board_size = warp_image(img, board_corners)
    imgWarped = draw_chess_grid(imgWarped, board_size)

    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        index_finger = hand["lmList"][8][:2]

        point = np.array([[index_finger]], dtype=np.float32)
        warped_point = cv2.perspectiveTransform(point, matrix)[0][0]

        square, (grid_x, grid_y) = get_chess_square(int(warped_point[0]), int(warped_point[1]), board_size)

        if grid_x != -1 and grid_y != -1:
            square_size = board_size // 8
            top_left = (grid_x * square_size, grid_y * square_size)
            bottom_right = ((grid_x + 1) * square_size, (grid_y + 1) * square_size)
            cv2.rectangle(imgWarped, top_left, bottom_right, (255, 255, 0), -1)

            imgWarped = draw_chess_grid(imgWarped, board_size)

        cv2.putText(imgWarped, square, (int(warped_point[0]), int(warped_point[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Warped Chessboard", imgWarped)
    cv2.imshow("Original Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
