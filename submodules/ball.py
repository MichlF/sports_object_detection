import math
from collections import deque
from itertools import islice

import cv2
import numpy as np
import pandas as pd

from utilities.timing import timer
from utilities.trajectory_ops import detect_direction_angle

from .config import MODEL_RESOLUTION, TIME_FUNCTIONS


@timer(enabled=TIME_FUNCTIONS)
def get_ball_position_tracknet(
    images: np.ndarray,
    tracknet_object,
    output_res: tuple[int, int],
    n_classes: int = 256,
) -> np.ndarray:
    """
    Use the TrackNet model to predict the position of the ball in the given images.

    Parameters:
        images (np.ndarray): A numpy array of shape (height, width, channels) or (height, width, channels*num_images)
        tracknet_object (object): A trained TrackNet object
        output_res (tuple): The desired output resolution for the heatmap of the ball position.
        n_classes (int): The number of classes in the TrackNet object's output.
    Returns:
        circles (np.ndarray): A numpy array of shape (1, num_circles, 3) containing the x, y and radius of the circles
        representing the ball position in the images.
    """
    images = np.rollaxis(a=images, axis=2, start=0)
    model_tracknet_pred = tracknet_object.predict(
        np.array([images]), verbose=0
    )[0]
    # Tracknet returns (net_output_height, model_output_width, n_classes)
    # Thus, reshape image to (net_output_height, model_output_width, n_classes(depth))
    model_tracknet_pred = model_tracknet_pred.reshape(
        (MODEL_RESOLUTION[0], MODEL_RESOLUTION[1], n_classes)
    ).argmax(axis=2)
    # Reshape to original image resolution (as uint8, necessary for cv2) to get ball heatmap
    heatmap = cv2.resize(
        src=model_tracknet_pred.astype(np.uint8), dsize=output_res
    )
    # Turn gaussian blob/s into binary circular object and find ball/s in heatmap with 2<=radius<=7
    # Return only the first circle without radius
    _, heatmap = cv2.threshold(
        src=heatmap, thresh=127, maxval=255, type=cv2.THRESH_BINARY
    )
    try:
        circles = cv2.HoughCircles(
            image=heatmap,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=1,
            param1=50,
            param2=2,
            minRadius=2,
            maxRadius=7,
        )[0][0][:2]
    except Exception as e:
        print(
            " TrackNet couldn't detect the tennis ball. This is expected"
            " behavior when the ball is not in frame.",
            e,
        )
        circles = None
    return circles


@timer(enabled=TIME_FUNCTIONS)
def draw_ball(
    image: np.ndarray,
    xy: np.ndarray,
    trajectory_deque: deque = None,
    annotation_type: str = "points",
    direction_threshold: int = 20,
    draw_len_trajectory: int = 1,
    *args,
    **kwargs,
) -> tuple[np.ndarray, deque]:
    """Draws ball onto an image including its trajectory"""
    if xy is not None:  # ball was detected
        if trajectory_deque is not None:
            # Append new ball coordinates to deque
            trajectory_deque.appendleft(xy)
    else:  # no ball was detected
        if trajectory_deque is not None:
            trajectory_deque.appendleft(None)
    # Actually draw balls or directional vectors based on flag
    if annotation_type == "location" or annotation_type == "both":
        if trajectory_deque:
            for ball in islice(trajectory_deque, draw_len_trajectory):
                if ball is not None:
                    try:
                        cv2.circle(
                            img=image,
                            center=ball.astype(np.uint32),
                            *args,
                            **kwargs,
                        )
                    except Exception as e:
                        print(" Couldn't draw ball, see x,y: ", ball, e)
    if annotation_type == "direction" or annotation_type == "both":
        direction_list = detect_direction_angle(
            positions_vector=list(
                islice(trajectory_deque, draw_len_trajectory)
            )
        )
        arrow_length = 25
        # Correct thickness values below 0 because arrowedLine does not accept them.
        if kwargs["thickness"] is not None and kwargs["thickness"] <= 0:
            kwargs["thickness"] = 2
        # When direction and ball position are plotted, reduce arrow thickness to prevent
        # occlusion of the ball by the arrow. Also, remove radius from kwargs
        if annotation_type == "both":
            kwargs["thickness"] = int(kwargs["radius"] / 2)
        kwargs.pop("radius", None)
        for idx in range(len(direction_list)):
            if not math.isnan(direction_list[idx]):
                # Calculate end_point using the vector direction and arrow length
                start_point = (
                    int(trajectory_deque[idx][0]),
                    int(trajectory_deque[idx][1]),
                )
                end_point = (
                    int(
                        trajectory_deque[idx][0]
                        + arrow_length
                        * math.cos(math.radians(direction_list[idx]))
                    ),
                    int(
                        trajectory_deque[idx][1]
                        + arrow_length
                        * math.sin(math.radians(direction_list[idx]))
                    ),
                )
                if (
                    (
                        -direction_threshold
                        <= direction_list[idx]
                        <= direction_threshold
                    )
                    or (
                        180 - direction_threshold <= direction_list[idx] <= 180
                    )
                    or (
                        -180
                        <= direction_list[idx]
                        <= -180 + direction_threshold
                    )
                ):
                    color = (255, 255, 255)
                elif direction_list[idx] > 0:
                    color = (0, 0, 255)
                elif direction_list[idx] < 0:
                    color = (255, 255, 0)
                else:
                    color = (0, 0, 0)
                # Now provide and override any already existing color value (i.e. through **kwargs)
                kwargs["color"] = color
                cv2.arrowedLine(
                    img=image,
                    pt1=start_point,
                    pt2=end_point,
                    tipLength=0.8,
                    *args,
                    **kwargs,
                )
    if annotation_type not in ["both", "direction", "location"]:
        raise KeyError(
            f" {annotation_type} is an unknown annotation type for drawing the"
            " ball."
        )
    return image, trajectory_deque


@timer(enabled=TIME_FUNCTIONS)
def draw_ball_2d(
    image: np.ndarray,
    homography_mat: np.ndarray,
    ball_coords: np.ndarray,
    adjust_height: int = None,
    *args,
    **kwargs,
) -> np.ndarray:
    """doc"""
    if ball_coords is not None:
        try:
            ball_center = ball_coords.reshape(-1, 1, 2).copy()
            # Quick hack: lower position by half of a players height
            if adjust_height is not None:
                # Note: deque is fixed to one memory location. Thus, any change to it will
                # be retained on the global scale.
                ball_center[0][0][1] = ball_center[0][0][1] + adjust_height
            ball_center = np.squeeze(
                cv2.perspectiveTransform(ball_center, homography_mat)
            ).astype(np.uint32)
            cv2.circle(img=image, center=ball_center, *args, **kwargs)
            cv2.circle(
                img=image,
                center=ball_center,
                color=(0, 0, 0),
                radius=1,
                thickness=-1,
            )
        except Exception as e:
            print(
                " 2D transformation for ball failed. This is expected behavior"
                " when the ball is not in frame.",
                e,
            )
    return image


@timer(enabled=TIME_FUNCTIONS)
def bounce_analysis(
    trajectory_deque: deque = None,
    window_size: int = 5,
    frame_count_debug=0,
    image_debug=None,
):
    direction_list = detect_direction_angle(positions_vector=trajectory_deque)
    bounces = detect_bounce_average_window(
        directions_vectors=direction_list, window_size=window_size
    )
    print("Sum of bounces: ", sum(bounces))
    # cv2.imshow("Test", image_debug)
    return 1 if sum(bounces) > 0 else 0


def translate_values(value: float) -> float:
    """
    translate_values will translate the given value from 0 to 180 range to 180 to 0 range.
    :param value : float value which needs to be translated
    :return : translated float value
    """
    return 180 % float(value) if float(value) > 90.0 else float(value)


def detect_bounce_average_window(
    directions_vectors: list[float], window_size: int = 5, threshold: int = 20
) -> list[int]:
    """
    detect_bounce_average_window function detects the bouncing of an object by calculating the average of a window of values in the given direction vectors.
    :param directions_vectors : list of float values representing the direction of an object
    :param window_size : int value representing the size of the window for which average will be calculated. Default is 5.
    :param threshold : int value representing the threshold for detecting bouncing. Default is 20.
    :return : list of integers representing if there is a bounce at that index or not
    """
    if window_size > len(directions_vectors):
        raise ValueError(
            "Window size cannot be larger than the length of"
            " direction_vectors !"
        )
    if window_size is None:
        window_size = len(directions_vectors)
    bounces = []
    for i in range(len(directions_vectors)):
        if i < window_size - 1:
            bounces.append(0)
        else:
            window = [
                translate_values(abs(x))
                for x in directions_vectors[i - window_size + 1 : i + 1]
            ]
            avg = sum(window) / window_size
            print(window, avg)
            if avg < threshold or avg > (180 - threshold):
                bounces.append(1)
            else:
                bounces.append(0)
    return bounces


def detect_bounce_near_zero(
    direction_vectors: list[float],
    near_zero_count: list[int],
    threshold: int = 20,
) -> list[int]:
    """
    detect_bounce_near_zero will detect if direction vector is near zero based on threshold value.
    :param direction_vectors : list of float values representing direction
    :param near_zero_count : list of int values to store near zero count
    :param threshold : int values as threshold to detect near zero
    :return : list of int where 1 represents near zero and 0 represents not near zero
    """
    near_zero_count = []

    for _, direction in enumerate(direction_vectors):
        if abs(direction) < threshold or abs(direction) > (180 - threshold):
            near_zero_count.append(1)
        else:
            near_zero_count.append(0)
    return near_zero_count


def find_middle_of_series(arr: list[int], min_series_length: int) -> int:
    """
    Given a list of integers, this function finds the middle index of a series of consecutive 1's in the list that is at least
    min_series_length long. If no such series is found, it returns -1.
    :param arr: a list of integers
    :param min_series_length: minimum number of consecutive 1's in the series
    :return: the middle index of a series of consecutive 1's that is at least min_series_length long, or -1 if not found
    """
    start_index = -1
    end_index = -1
    zero_detected = False
    for idx in range(len(arr)):
        if arr[idx] == 1:
            if start_index == -1:
                start_index = idx
            end_index = idx
        else:
            if zero_detected:
                if (
                    start_index != -1
                    and end_index - start_index + 1 >= min_series_length
                ):
                    break
                else:
                    start_index = -1
                    end_index = -1
                    zero_detected = False
            else:
                if start_index == -1:
                    start_index = idx
                end_index = idx
                if idx != 0:
                    if arr[idx - 1] == 0:
                        start_index = -1
                        end_index = -1
                        zero_detected = False
                    else:
                        zero_detected = True
    if start_index != -1 and end_index - start_index + 1 >= min_series_length:
        return (start_index + end_index) // 2
    return -1


def store_data(list_of_arrays):
    data = []
    for i, array in enumerate(list_of_arrays):
        if array is not None and len(array) == 2:
            data.append([i, array[0], array[1]])
    df = pd.DataFrame(data, columns=["frame_no", "x", "y"])
    df.set_index("frame_no", inplace=True)
    return df
