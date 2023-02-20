import cv2
import numpy as np

from utilities.timing import timer

from .config import TIME_FUNCTIONS


@timer(enabled=TIME_FUNCTIONS)
def canny_method(
    image: np.ndarray,
    threshold: tuple[int, int],
    boundaries: tuple[tuple, tuple],
    show_intermediate_steps: bool = False,
):
    # create NumPy arrays from the boundaries
    lower = np.array(boundaries[0], dtype="uint8")
    upper = np.array(boundaries[1], dtype="uint8")
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold[0], threshold[1])
    kernel = np.ones((5, 5), "uint8")
    dilated = cv2.dilate(edges, kernel)
    if show_intermediate_steps:
        cv2.imshow("output", gray)
        cv2.imshow("canny", edges)
    return dilated


@timer(enabled=TIME_FUNCTIONS)
def houghlines_method(
    image: np.ndarray,
    rho: int = 1,
    theta: float = np.pi / 180,
    threshold=50,
    min_line_length: int = 100,
    max_line_gap: int = 1,
):
    # Create black image to draw on
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(
        image,
        rho,
        theta,
        threshold,
        np.array([]),
        min_line_length,
        max_line_gap,
    )
    # Output "lines" is an array containing endpoints of detected line segments
    points = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
    return line_image, points


@timer(enabled=TIME_FUNCTIONS)
def find_contours_harris(
    image: np.ndarray,
    block_size: int = 9,
    k_size: int = 3,
    k: float = 0.01,
    show_intermediate_steps: bool = False,
):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(
        src=image, blockSize=block_size, ksize=k_size, k=k
    )
    corners_norm = np.empty(corners.shape, dtype=np.float32)
    cv2.normalize(
        corners, corners_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    corners_norm_scaled = cv2.convertScaleAbs(corners_norm)
    _, thresh = cv2.threshold(
        corners_norm_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    corners_norm_scaled_dilated = cv2.dilate(thresh, None, iterations=3)
    contours, hierarchy = cv2.findContours(
        corners_norm_scaled_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if show_intermediate_steps:
        cv2.imshow("corners", corners)
        cv2.imshow("corners norm scaled", corners_norm_scaled)
        cv2.imshow("thres", thresh)
        cv2.imshow("corners_dilated", corners_norm_scaled_dilated)
    return corners_norm_scaled_dilated, contours, hierarchy


@timer(enabled=TIME_FUNCTIONS)
def auto_canny(image, sigma=0.33):
    """
    Detects lines automatically, but doesn't perform very well currently.
    """
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # Return the edged image
    return edged


@timer(enabled=TIME_FUNCTIONS)
def find_corner_pts(contour_object, output_res: tuple[int, int] = None):
    centers = []
    for _, contour in enumerate(contour_object):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    if output_res is not None:
        image = np.empty(output_res)
        for center in centers:
            cv2.circle(
                image, center=center, radius=3, color=(255, 0, 0), thickness=-1
            )

    return image if output_res else None, centers


@timer(enabled=TIME_FUNCTIONS)
def get_corners_pts(
    corner_pts: list,
    output_res: tuple[int, int] = None,
    show_intermediate_steps: bool = False,
):
    # We can make use of an idosyncracy of tennis court perspectives:
    # The top two corners have, b/c of perspective shift, neither the largest nor the smallest X
    # But they are, assuming your thresholding for corners is good enough, the two smallest Y.
    # Deducting from the x value, we can then infer which one is the top LEFT and which the top RIGHT
    center_array = np.array(corner_pts)
    two_top_pts = np.array(
        [
            center_array[
                center_array[:, 1].argsort()[0]
            ],  # lowest Y val # top
            center_array[
                center_array[:, 1].argsort()[1]
            ],  # second lowest Y val # top
        ]
    )
    two_bottom_pts = np.array(
        [
            center_array[
                center_array[:, 0].argsort()[0]
            ],  # lowest X val # left
            center_array[
                center_array[:, 0].argsort()[-1]
            ],  # highest X val # right
        ]
    )
    two_top_pts_sorted = two_top_pts[two_top_pts[:, 0].argsort()]
    two_bottom_pts_sorted = two_bottom_pts[(-two_bottom_pts[:, 0]).argsort()]
    corners_list = np.concatenate(
        [two_top_pts_sorted, two_bottom_pts_sorted], axis=0
    )
    if output_res is not None:
        image = np.empty(output_res)
        for point in corners_list:
            cv2.circle(
                image,
                center=point,
                radius=3,
                color=(255, 255, 255),
                thickness=3,
            )
        if show_intermediate_steps:
            cv2.imshow("Found corners", image)
    return image if output_res else None, corners_list
