"""
Chessboard Detection Module -- Thesis Section 10.

Detects and isolates the chessboard region from a camera frame,
applies perspective correction to produce a normalised top-down view,
and draws the 8x8 grid overlay.

All functions are extracted verbatim from the original main.py to
preserve existing OpenCV behaviour.
"""

import cv2
import numpy as np

from config.settings import DISPLAY_SIZE, BOARD_MARGIN


# ------------------------------------------------------------------
# Corner / contour helpers
# ------------------------------------------------------------------

def find_chessboard_corners(img):
    """
    Automatically detect chessboard corners from the image.
    Returns the four corner points of the largest rectangular contour.
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = rectContour(contours)
    if len(rectCon) == 0:
        return np.array([])
    biggestContour = getCornerPoints(rectCon[0])
    if biggestContour.size != 0:
        biggestContour = reorder(biggestContour)
        return biggestContour
    return np.array([])


def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000: #1800
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon


def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left
    return myPointsNew


# ------------------------------------------------------------------
# Perspective transform & grid
# ------------------------------------------------------------------

def warp_image(img, points, display_size=DISPLAY_SIZE, margin=BOARD_MARGIN):
    """
    Warp the chessboard to a top-down view.
    Returns the warped image, the transformation matrix, and the board size.
    """
    board_size = min(display_size) - margin
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, matrix, (board_size, board_size))
    return img_warped, matrix, board_size


def crop_inner_squares(img_warped, board_size, offset=0):
    """
    Crop the warped image to remove the border and retain only the inner playable squares.
    Returns the cropped image and the new board size.
    """
    cropped = img_warped[offset:board_size - offset, offset:board_size - offset]
    new_board_size = board_size - 2 * offset
    return cropped, new_board_size


def draw_chess_grid(img, board_size):
    """
    Draw grid lines on the warped chessboard.
    """
    square_size = board_size // 8
    for i in range(1, 8):
        cv2.line(img, (i * square_size, 0), (i * square_size, board_size), (255, 255, 255), 2)
        cv2.line(img, (0, i * square_size), (board_size, i * square_size), (255, 255, 255), 2)
    return img
