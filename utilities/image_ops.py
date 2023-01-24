import numpy as np


def image_crop(
    image: np.ndarray,
    x1: int = None,
    y1: int = None,
    x2: int = None,
    y2: int = None,
    width: int = None,
    height: int = None,
    tensor_box_xywh: list[
        int,
    ] = None,
    tensor_box_xyxy: list[
        int,
    ] = None,
) -> np.ndarray:
    """
    Crops a rectangle out of a given image.
    Reminder: image style
    # x1,y1 --w---
    # |          |
    # h          |
    # |          |
    # --------x2,y2
    Parameters:
        image (np.ndarray): The image to be cropped
        x1 (int): The x coordinate of the top-left corner of the crop rectangle
        y1 (int): The y coordinate of the top-left corner of the crop rectangle
        x2 (int): The x coordinate of the bottom-right corner of the crop rectangle
        y2 (int): The y coordinate of the bottom-right corner of the crop rectangle
        width (int): The width of the crop rectangle
        height (int): The height of the crop rectangle
        tensor_box_xywh (list[int]): A list of four integers representing the x, y, width, and height of the crop rectangle in the format [x, y, width, height]
        tensor_box_xyxy (list[int]): A list of four integers representing the x, y, x, y of the crop rectangle in the format [x1, y1, x2, y2]
    Returns:
        np.ndarray: The cropped image
    """
    if tensor_box_xywh is not None:
        x1 = int(tensor_box_xywh[0])
        y1 = int(tensor_box_xywh[1])
        width = int(tensor_box_xywh[2])
        height = int(tensor_box_xywh[3])
    elif tensor_box_xyxy is not None:
        x1 = int(tensor_box_xyxy[0])
        y1 = int(tensor_box_xyxy[1])
        x2 = int(tensor_box_xyxy[2])
        y2 = int(tensor_box_xyxy[3])
    if x2 and y2:
        return image[y1:y2, x1:x2]
    else:
        x1 = int(x1 - width / 2)
        y1 = int(y1 - height / 2)
        return image[y1 : y1 + height, x1 : x1 + width]
