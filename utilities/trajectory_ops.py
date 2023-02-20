import math

import cv2
import numpy as np

from submodules.config import TIME_FUNCTIONS
from utilities.image_ops import pixels_to_real
from utilities.timing import timer


@timer(enabled=TIME_FUNCTIONS)
def draw_directional_vectors(
    img, vectors, speeds, minmax_speed: tuple, thickness: int = 2
):
    for idx, vector in enumerate(vectors):
        color_value = int(
            255
            * (speeds[idx] - minmax_speed[0])
            / (minmax_speed[1] - minmax_speed[0])
        )
        color_map = cv2.applyColorMap(
            (color_value, color_value, color_value), cv2.COLORMAP_JET
        )
        cv2.arrowedLine(img, vector[0], vector[1], color_map, thickness)
    return img


@timer(enabled=TIME_FUNCTIONS)
def vector_speed(
    coord_frame_1: tuple[int, int],
    coord_frame_2: tuple[int, int],
    fps: int = 60,
    real_world_units: bool = False,
) -> float:
    """
    Calculates the speed of a movement from coord_frame_1 to coord_frame_2 in km/h.

    This function calculates the Euclidean distance between two points represented by
    tuples and then divides it by time (1 second) to get the speed of the movement.

    Args:
    coord_frame_1 (tuple): A tuple representing the starting point of the movement.
    coord_frame_2 (tuple): A tuple representing the ending point of the movement.
    fps (int): frames per second of the video.
    real_world_units (bool): if True, returns the speed in real-world units.

    Returns:
    float: The speed of the movement in km/h.

    Note:
    The function assumes that the coordinates are in pixels, If the coordinates are in
    real-world units, you should adjust the calculation accordingly.
    """
    euclidean_distance = np.linalg.norm(
        np.array(coord_frame_1) - np.array(coord_frame_2)
    )
    speed = euclidean_distance / (1 / fps)

    if real_world_units:
        speed = pixels_to_real(speed)
        return speed * 3600  # Converting from m/s to km/h
    else:
        return speed


@timer(enabled=TIME_FUNCTIONS)
def vector_speed_change(vector: np.ndarray) -> float:
    """
    Calculates the speed of change of a given vector.
    Speed of change is the magnitude of the derivative of the given vector.

    Parameters:
    vector (np.ndarray): The vector for which the speed of change is to be calculated.

    Returns:
    float: The speed of change of the given vector.
    """
    derivative = np.gradient(vector)
    speed = np.linalg.norm(derivative)
    return speed


# @timer(enabled=TIME_FUNCTIONS)
# def detect_trajectory_change(
#     vector_samples: list[np.ndarray, ...],
#     speed_threshold: float,
#     direction_threshold: float,
#     change_span: int = None,
# ) -> tuple[bool, bool]:
#     """
#     Given a list of vector samples, it checks if the direction or speed of the samples
#     has changed in the latest change_span elements.

#     Parameters:
#     vector_samples (list[np.ndarray]): A list of vector samples.
#     speed_threshold (float): A threshold for checking if the speed of the samples has changed.
#     direction_threshold (float): A threshold for checking if the direction of the samples has changed.
#     change_span (int): The number of latest elements to check if direction or speed has changed.

#     Returns:
#     tuple[bool, bool]: A tuple of two boolean values. The first value is True if the direction of the samples
#     has changed in the latest change_span elements, the second value is True if the speed of the samples has
#     changed in the latest change_span elements.
#     """
#     if change_span is None:
#         change_span = len(vector_samples)
#     speeds = []
#     for sample in vector_samples:
#         try:
#             speeds.append(vector_speed_change(sample))
#         except:
#             speeds.append(None)
#     # speeds = [vector_speed_change(sample) for sample in vector_samples]
#     speed_change_list = detect_speed_change(speeds, speed_threshold)
#     direction_change_list = detect_direction_change(
#         vector_samples, direction_threshold
#     )
#     direction_changed = any(direction_change_list[-change_span:])
#     speed_changed = any(speed_change_list[-change_span:])
#     return direction_changed, speed_changed


@timer(enabled=TIME_FUNCTIONS)
def detect_distance_change(
    samples: list[np.ndarray, ...], direction_threshold: float
) -> list[bool]:
    """
    Given a list of samples, it checks if the distance of the samples has changed.

    Parameters:
    samples (list[np.ndarray]): A list of samples.
    direction_threshold (float): A threshold for checking if the distance of the
    samples has changed.

    Returns:
    list[bool]: A list of boolean values. Each value is True if the distance of the
    corresponding sample in the input list has changed.
    """
    direction_change_list = []
    for i in range(1, len(samples)):
        # only if two consecutive ball positions are obtained
        if samples[i] is None or samples[i - 1] is None:
            direction_change_list.append(np.nan)
            continue
        diff = np.subtract(samples[i], samples[i - 1])
        dist = np.linalg.norm(diff)
        direction_change_list.append(dist)
        if dist > direction_threshold:
            direction_change_list.append(True)
        else:
            direction_change_list.append(False)
    return direction_change_list


@timer(enabled=TIME_FUNCTIONS)
def detect_direction_angle(positions_vector: list[np.ndarray]) -> list[float]:
    direction_list = []
    for i in range(len(positions_vector) - 1):
        # Only if two consecutive ball trackings were successfull.
        if positions_vector[i] is None or positions_vector[i + 1] is None:
            direction_list.append(np.nan)
            continue
        direction_vector = np.subtract(
            positions_vector[i], positions_vector[i + 1]
        )
        direction_angle = math.degrees(
            math.atan2(direction_vector[1], direction_vector[0])
        )
        direction_list.append(direction_angle)
    return direction_list


@timer(enabled=TIME_FUNCTIONS)
def detect_speed_change(
    speeds: list[float], speed_threshold: float
) -> list[bool]:
    """
    Given a list of speeds, it checks if the speed of the samples has changed.

    Parameters:
    speeds (List[float]): A list of speeds.
    speed_threshold (float): A threshold for checking if the speed of the samples has changed.

    Returns:
    List[bool]: A list of boolean values. Each value is True if the speed of the
    corresponding sample in the input list has changed.
    """
    speed_change_list = []
    for i in range(1, len(speeds)):
        # only if two consecutive ball positions are obtained
        if speeds[i] is None or speeds[i - 1] is None:
            speed_change_list.append(False)
            continue
        if abs(speeds[i] - speeds[i - 1]) > speed_threshold:
            speed_change_list.append(True)
        else:
            speed_change_list.append(False)
    return speed_change_list


@timer(enabled=TIME_FUNCTIONS)
def distance_to_vertical_centerline(
    pt: tuple[int, int],
    pt_top: tuple[int, int] = None,
    pt_bottom: tuple[int, int] = None,
    resolution_wh: tuple[int, int] = None,
) -> np.ndarray:
    if pt_top is not None and pt_bottom is not None:
        pt_top = np.array(pt_top) if isinstance(pt_top, tuple) else pt_top
        pt_bottom = (
            np.array(pt_bottom) if isinstance(pt_bottom, tuple) else pt_bottom
        )
    elif resolution_wh:
        pt_top = np.array([resolution_wh[1] / 2, 0])
        pt_bottom = np.array([resolution_wh[1] / 2, resolution_wh[0]])
    else:
        raise ValueError(
            " At least one of pt_top, pt_bottom or resolution_wh must be"
            " provided !"
        )
    pt = np.array(pt) if isinstance(pt, tuple) else pt

    return np.cross(pt_bottom - pt_top, pt_top - pt) / np.linalg.norm(
        pt_bottom - pt_top
    )
