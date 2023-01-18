"""This script provides a function to extract contours of white lines in an image using Hough lines
algorithm and returns the image with the contours and the corner points of the rectangle(s) in the
image. The script also provides helper functions for finding the corner points of the rectangle(s)
and for finding the point of intersection of two lines. The provided functions use type hinting for
better readability and self-explanation."""

import cv2
import numpy as np


def extract_contours(
    image: np.ndarray, lower_threshold: int, upper_threshold: int, show_contours: bool=False
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Extracts the contours of the white lines in an image using Hough lines algorithm, and returns the image with the contours and the corner points of the rectangle(s) in the image.

    Parameters:
        image (np.ndarray): Input image.
        lower_threshold (int): Lower threshold for thresholding.
        upper_threshold (int): Upper threshold for thresholding.

    Returns:
        Tuple[np.ndarray, list[np.ndarray]]: The image with the contours and the corner points of the rectangle(s) in the image.
    """
    # Convert the image to grayscale, apply thresholds to convert image to binary image and find contours.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _ret, thresh = cv2.threshold(gray, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Find the rectangle corner points: try a naive and a hough-based approach
    rectangles, rectangles_hough = [], []
    for contour in contours:
        if len(contour) >= 4:
            rect = cv2.minAreaRect(contour)
            rectangles.append(cv2.boxPoints(rect))
            # Find lines in the contour
            points = contour.reshape(-1, 2).astype(np.uint8)
            lines = cv2.HoughLinesP(points, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                # Find the corner points of the rectangle
                corners = find_rectangle_corners(lines)
                if len(corners) == 4:
                    rectangles.append(corners)
    if show_contours:
        cv2.imshow("detected contours", image)

    return image, rectangles, rectangles_hough


def find_rectangle_corners(lines: np.ndarray) -> list[tuple[int, int]]:
    """
    Finds the corner points of a rectangle in an image using the intersection
    of the lines found using the Hough lines algorithm.

    Parameters:
        lines (np.ndarray): The lines found using the Hough lines algorithm.

    Returns:
        list[tuple[int, int]]: The corner points of the rectangle in the form of a
        list of tuples of the coordinates (x, y).
    """
    corners = []
    for line1 in lines:
        for line2 in lines:
            if not np.array_equal(line1, line2):
                corner = intersection(line1[0], line2[0])
                if corner not in corners:
                    corners.append(corner)
    return corners


def intersection(
    line1: tuple[int, int, int, int], line2: tuple[int, int, int, int]
) -> tuple[int, int]:
    """
    Finds the point of intersection of two lines.

    Parameters:
        line1 (tuple[int, int, int, int]): The first line.
        line2 (tuple[int, int, int, int]): The second line.

    Returns:
        tuple[int, int]: The point of intersection of the two lines in the form of a tuple of the coordinates (x, y).
    """
    xdiff = (line1[0] - line1[2], line2[0] - line2[2])
    ydiff = (line1[1] - line1[3], line2[1] - line2[3])
    div = determinant(xdiff, ydiff)
    if div == 0:
        return None

    d = (determinant((line1[0], line1[1]), ydiff), determinant((line2[0], line2[1]), xdiff))
    x = determinant(d, xdiff) / div
    y = determinant(d, ydiff) / div
    return (int(x), int(y))


def determinant(a: tuple[int, int], b: tuple[int, int]) -> int:
    """
    Finds the determinant of two vectors.

    Parameters:
        a (tuple[int, int]): The first vector.
        b (tuple[int, int]): The second vector.

    Returns:
        int: The determinant of the two vectors.
    """
    return a[0] * b[1] - a[1] * b[0]
