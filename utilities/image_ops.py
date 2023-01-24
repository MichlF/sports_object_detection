"""Module containing functions to crop an image"""
import numpy as np

# x1,y1 --w---
# |          |
# h          |
# |          |
# --------x2,y2
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
    """Crops a rectangle out of a given image. Accepts coordinates from a Tensor box from YOLO."""
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
