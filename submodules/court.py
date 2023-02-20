import cv2
import numpy as np

from utilities.timing import timer

from .config import TIME_FUNCTIONS
from .find_corners import (
    canny_method,
    find_contours_harris,
    find_corner_pts,
    get_corners_pts,
    houghlines_method,
)


@timer(enabled=TIME_FUNCTIONS)
def create_avg_frame(
    video_capture_object,
    start_frame: int = 0,
    end_frame: int = 100,
    n_th_frame: int = 10,
) -> np.ndarray:
    frames = []
    for frame_no in range(start_frame, end_frame, n_th_frame):
        video_capture_object.set(1, frame_no)
        success, frame_in = video_capture_object.read()
        if not success:
            raise ValueError(f" Couldn't load frame {frame_no} !")
        frames.append(frame_in)
    return np.mean(frames, axis=0).astype(np.uint8)


@timer(enabled=TIME_FUNCTIONS)
def draw_minimap_to_frame(
    image: np.ndarray,
    image_minimap: np.ndarray,
    alpha: float = 0.15,
    border_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    # Get a clean copies and add black border around the minimap image
    images_combined = image.copy()  # deepcopy(image)
    frame_out_2d_bordered = image_minimap.copy()  # deepcopy(image_minimap)
    frame_out_2d_bordered = cv2.copyMakeBorder(
        frame_out_2d_bordered,
        5,
        5,
        5,
        5,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    # Get the shape of minimap, define the region of interest on image and blend minimap with it
    rows, cols, _ = frame_out_2d_bordered.shape
    roi = images_combined[0:rows, 0:cols]
    result = cv2.addWeighted(roi, alpha, frame_out_2d_bordered, 1 - alpha, 0)
    images_combined[0:rows, 0:cols] = result
    return images_combined


@timer(enabled=TIME_FUNCTIONS)
def draw_court_portrait(
    height: int = 1000,
    pad: float = 0.22,
    line_color: tuple[int, int, int] = (255, 255, 255),
    court_color_1: tuple[int, int, int] = (192, 158, 128),
    court_color_2: tuple[int, int, int] = (153, 112, 80),
):
    width = int(height * (1 - pad) * 2 * (1 + pad))
    court = np.zeros((width, height, 3)).astype(np.uint8)
    # Outline
    cv2.rectangle(court, (0, 0), (height, width), court_color_1, -1)
    # baseline
    x1, y1 = int(height * pad), int(width * pad)
    x2, y2 = int(height * (1 - pad)), int(width * pad)
    x3, y3 = int(height * (1 - pad)), int(width * (1 - pad))
    x4, y4 = int(height * pad), int(width * (1 - pad))
    cv2.rectangle(court, (x1, y1), (x3, y3), court_color_2, -1)
    cv2.line(court, (x1, y1), (x2, y2), line_color, 2)
    cv2.line(court, (x2, y2), (x3, y3), line_color, 2)
    cv2.line(court, (x3, y3), (x4, y4), line_color, 2)
    cv2.line(court, (x4, y4), (x1, y1), line_color, 2)
    x_ratio = (x2 - x1) / 10.97
    y_ratio = (y3 - y2) / 23.78
    # doubles sidelines
    xc_1, yc_1 = int(x1 + x_ratio * 1.372), y1
    xc_2, yc_2 = int(x2 - x_ratio * 1.372), y2
    xc_3, yc_3 = int(x3 - x_ratio * 1.372), y3
    xc_4, yc_4 = int(x4 + x_ratio * 1.372), y4
    cv2.line(court, (xc_1, yc_1), (xc_4, yc_4), line_color, 2)
    cv2.line(court, (xc_2, yc_2), (xc_3, yc_3), line_color, 2)
    # service lane
    xs_1, ys_1 = xc_1, int(y1 + 5.50 * y_ratio)
    xs_2, ys_2 = xc_2, int(y2 + 5.50 * y_ratio)
    xs_3, ys_3 = xc_3, int(y3 - 5.50 * y_ratio)
    xs_4, ys_4 = xc_4, int(y4 - 5.50 * y_ratio)
    cv2.line(court, (xs_1, ys_1), (xs_2, ys_2), line_color, 2)
    cv2.line(court, (xs_3, ys_3), (xs_4, ys_4), line_color, 2)
    # net
    xnet_1, ynet_1 = x1, int((y4 - y1) / 2 + y1)
    xnet_2, ynet_2 = x2, int((y4 - y1) / 2 + y1)
    cv2.line(court, (xnet_1, ynet_1), (xnet_2, ynet_2), line_color, 2)
    # center service line
    xv_1, yv_1 = int((x2 - x1) / 2 + x1), ys_1
    xv_2, yv_2 = int((x2 - x1) / 2 + x1), ys_3
    cv2.line(court, (xv_1, yv_1), (xv_2, yv_2), line_color, 2)
    # central mark
    xm = int((x2 - x1) / 2 + x1)
    ym_1 = y1
    ym_2 = int(y1 + 10)
    ym_3 = int(y4 - 10)
    ym_4 = y4
    cv2.line(court, (xm, ym_1), (xm, ym_2), line_color, 2)
    cv2.line(court, (xm, ym_3), (xm, ym_4), line_color, 2)
    return court


@timer(enabled=TIME_FUNCTIONS)
def get_court_corners_2d(height: int = 1000, pad: float = 0.22):
    # compute court court dimensions accorgind to image dimension
    width = int(height * (1 - pad) * 2 * (1 + pad))
    corner_coords_1 = int(height * pad), int(width * pad)
    corner_coords_2 = int(height * (1 - pad)), int(width * pad)
    corner_coords_3 = int(height * (1 - pad)), int(width * (1 - pad))
    corner_coords_4 = int(height * pad), int(width * (1 - pad))

    return corner_coords_1, corner_coords_2, corner_coords_3, corner_coords_4


@timer(enabled=TIME_FUNCTIONS)
def draw_court_lines(
    image: np.ndarray,
    court_corners: tuple[tuple, tuple, tuple, tuple],
    *args,
    **kwargs,
) -> np.ndarray:
    """DOC"""
    (
        corner_coords_1,
        corner_coords_2,
        corner_coords_3,
        corner_coords_4,
    ) = court_corners
    cv2.line(image, corner_coords_1, corner_coords_2, *args, **kwargs)
    cv2.line(image, corner_coords_2, corner_coords_3, *args, **kwargs)
    cv2.line(image, corner_coords_3, corner_coords_4, *args, **kwargs)
    cv2.line(image, corner_coords_4, corner_coords_1, *args, **kwargs)
    return image


@timer(enabled=TIME_FUNCTIONS)
def draw_court_corners_2d(
    image: np.ndarray,
    court_corners: tuple[tuple, tuple, tuple, tuple],
    *args,
    **kwargs,
):
    for cc in court_corners:
        cv2.circle(img=image, center=cc, *args, **kwargs)
    return image


@timer(enabled=TIME_FUNCTIONS)
def get_court_corners(
    mode: str = "mouse_input", image: np.ndarray = None
) -> tuple[tuple, tuple, tuple, tuple]:
    if mode == "mouse_input":
        assert image is not None, "No image provided!"
        (
            corner_coords_1,
            corner_coords_2,
            corner_coords_3,
            corner_coords_4,
        ) = return_xy_for_mouseclick(image=image)
    elif mode == "automatic":
        assert image is not None, "No image provided!"
        (
            corner_coords_1,
            corner_coords_2,
            corner_coords_3,
            corner_coords_4,
        ) = detect_corners_auto(image=image, show_intermediate_steps=False)
    elif mode == "else":
        # Ground truth coordinates for video_cut.mp4
        corner_coords_1 = (574, 300)
        corner_coords_2 = (1338, 300)
        corner_coords_3 = (1570, 857)
        corner_coords_4 = (358, 857)
    return corner_coords_1, corner_coords_2, corner_coords_3, corner_coords_4


@timer(enabled=TIME_FUNCTIONS)
def detect_corners_auto(
    image: np.ndarray,
    threshold_canny: tuple[int, int] = (150, 200),
    boundaries: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (190, 190, 190),
        (255, 255, 255),
    ),
    rho: int = 1,
    theta: float = np.pi / 180,
    threshold_votes: int = 50,
    min_line_length: int = 200,
    max_line_gap: int = 1,
    show_intermediate_steps: bool = False,
):
    image = image.astype(np.uint8)
    canny_dilated = canny_method(
        image=image,
        threshold=threshold_canny,
        boundaries=boundaries,
        show_intermediate_steps=show_intermediate_steps,
    )
    houghlines, _ = houghlines_method(
        image=canny_dilated,
        rho=rho,
        theta=theta,
        threshold=threshold_votes,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    image, contours, _ = find_contours_harris(
        image=houghlines, show_intermediate_steps=show_intermediate_steps
    )
    image, corner_points = find_corner_pts(
        contour_object=contours, output_res=np.shape(image)
    )
    image, corner_list = get_corners_pts(
        corner_pts=corner_points, output_res=np.shape(image)
    )
    if show_intermediate_steps:
        cv2.imshow("image_in", image)
        cv2.imshow("canny_dilated", canny_dilated)
        cv2.imshow("houghlines", houghlines)
    return (
        tuple(point.tolist()) for point in corner_list
    )  # to be consist with the other methods, return tuples


def return_xy_for_mouseclick(image):
    """
    Show an image and accept four mouse clicks to retrieve the coordinates of four points.

    Parameters:
        image (np.ndarray): The image on which the clicks will be performed.
    Returns:
        clicks[:4] (list): A list of four tuples containing the x, y coordinates of the
        four clicks in clockwise order.
    """
    global clicks
    clicks = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        image,
        "Clockwise order: top left, top right, bottom right, bottom left",
        (10, 30),
        font,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    base_image = image.copy()  # deepcopy(image)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_callback, image)
    while True:
        cv2.imshow("image", image)
        if len(clicks) == 5:  # if 5 clicks are given quit
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):  # allow to quit with Q
            break
        if (
            cv2.waitKey(1) & 0xFF == 8
        ):  # if user presses backspace, he can redo the last click
            if clicks:
                clicks.pop()
                # Clear the last circle
                image = base_image.copy()  # deepcopy(base_image)
                for coords in clicks:
                    x, y = coords
                    plus_size = 10
                    cv2.line(
                        image,
                        (x - plus_size, y),
                        (x + plus_size, y),
                        (0, 0, 255),
                        2,
                    )
                    cv2.line(
                        image,
                        (x, y - plus_size),
                        (x, y + plus_size),
                        (0, 0, 255),
                        2,
                    )
                cv2.imshow("image", image)
    cv2.destroyAllWindows()
    return clicks[:4]


def mouse_callback(event, x, y, flags, image):
    """
    A mouse callback function for OpenCV to handle left button mouse clicks.
    This function is called every time a left button mouse event occurs (i.e. when the left mouse button is pressed and released)

    Parameters:
        event (int): The event that occurred.
        x (int): The x-coordinate of the event.
        y (int): The y-coordinate of the event.
        flags (int): The flags for the event.
        image (np.ndarray): The image on which the event occurred.
    """
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) == 5:
            cv2.destroyAllWindows()
        clicks.append((x, y))
        plus_size = 10
        cv2.line(image, (x - plus_size, y), (x + plus_size, y), (0, 0, 255), 2)
        cv2.line(image, (x, y - plus_size), (x, y + plus_size), (0, 0, 255), 2)
        cv2.imshow("image", image)
