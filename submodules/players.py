import cv2
import numpy as np

from utilities.timing import timer
from utilities.trajectory_ops import distance_to_vertical_centerline

from .config import TIME_FUNCTIONS


@timer(enabled=TIME_FUNCTIONS)
def get_player_boxes_yolo(
    image: np.ndarray,
    model_yolo_object,
    yolo_conf: float,
    apply_nms=False,
    nms_thresh=0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_yolo_results = model_yolo_object(
        source=image, conf=yolo_conf, verbose=False
    )  # , device="cpu")#, device=torch.device('cuda:0'))
    # Results is a generator containing a result class for each analyzed image/frame
    # Each result class contains a box for each detected object in boxes
    # Each box has a value for the xyxy / xywh coordinates, confidence scores (conf) and class id (cls).
    player_boxes, player_confs, player_centroids, player_centergrounds = (
        [],
        [],
        [],
        [],
    )
    for (
        results_frame
    ) in model_yolo_results:  # assumes only a single frame is analyzed
        # Remove any non-person objects
        person_selector = results_frame.boxes.cls == 0  # 0 is Person
        player_confs = np.array(
            results_frame.boxes.conf[person_selector].cpu()
        )
        player_boxes = np.array(
            results_frame.boxes.xyxy[person_selector].cpu()
        ).astype(np.uint32)
        player_boxes_wh = np.array(
            results_frame.boxes.xywh[person_selector].cpu()
        ).astype(np.uint32)
        player_centroids = np.array(
            results_frame.boxes.xywh[person_selector][:, :2].cpu()
        ).astype(np.uint32)
        for _, rectangle in enumerate(player_boxes):
            # Get bottom point (half the width of bottom boundary box line)
            player_centergrounds.append(
                ((rectangle[0] + rectangle[2]) / 2, rectangle[3])
            )
        # YOLO's xywh method returns boundary boxes relative to the centroid,
        # but we actually want it relative to top-left point for DeepSort.
        for idx, _ in enumerate(player_boxes_wh):
            player_boxes_wh[idx][:2] = player_boxes[idx][:2]
        player_centergrounds = np.array(player_centergrounds).astype(np.uint32)
    # Remove overlapping boundary boxes using non-maximum suppression
    # Note: Since we previously already selected on only class person, we don't need to be worried about
    # overlaps between boxes of different classes.
    if apply_nms:
        keep = non_max_suppression(
            boxes=player_boxes, confs=player_confs, threshold=nms_thresh
        )
        player_boxes = player_boxes[keep]
        player_boxes_wh = player_boxes_wh[keep]
        player_confs = player_confs[keep]
        player_centroids = player_centroids[keep]
        player_centergrounds = player_centergrounds[keep]
    return (
        player_boxes,
        player_boxes_wh,
        player_confs,
        player_centroids,
        player_centergrounds,
    )


def non_max_suppression(
    boxes: np.ndarray, confs: np.ndarray, threshold: np.ndarray
):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Parameters:
        boxes (np.ndarray): The bounding boxes coordinates in the format of (x1, y1, x2, y2)
        confs (np.ndarray): The confidence scores of the bounding boxes
        threshold (float): The threshold for the IoU, boxes with IoU greater than this threshold will be removed.
    Returns:
        keep (list): A list of indices of the bounding boxes that passed the NMS.
    """
    # Confidence scores and bounding box coordinates
    x1, y1, x2, y2 = boxes.T
    bb_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # Compute the IoU between bounding boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (bb_area[i] + bb_area[order[1:]] - intersection)
        # Remove overlapping bounding boxes with an IoU threshold
        index = np.where(iou <= threshold)[0]
        order = order[index + 1]
    return keep


@timer(enabled=TIME_FUNCTIONS)
def track_player_deep_sort(
    image: np.ndarray,
    tracker_object,
    player_xywh: np.ndarray,
    player_confs: np.ndarray,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    bb_update, track_id, bboxes, centergrounds = [], [], [], []
    for idx, bb_player in enumerate(player_xywh):
        bb_update.append(
            (list(bb_player), player_confs[idx], "0")
        )  # player is always class id 0 (person)
    tracks = tracker_object.update_tracks(bb_update, frame=image)
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_ltrb(orig=True, orig_strict=False).astype(np.uint32)
        centerground = np.array([np.mean([bbox[0], bbox[2]]), bbox[3]]).astype(
            np.uint32
        )
        track_id.append(track.track_id)
        bboxes.append(bbox)
        centergrounds.append(centerground)
    return track_id, bboxes, centergrounds


@timer(enabled=TIME_FUNCTIONS)
def draw_players(
    image: np.ndarray, xyxy: np.ndarray, *args, **kwargs
) -> np.ndarray:
    """Draws colored boundary box for each given player onto an image,
    adds player ids and classification confidence"""
    for box_coords in xyxy:
        x1, y1, x2, y2 = box_coords
        cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), *args, **kwargs)
    return image


@timer(enabled=TIME_FUNCTIONS)
def draw_players_2d(
    image: np.ndarray,
    homography_mat: np.ndarray,
    player_locations: np.ndarray,
    *args,
    **kwargs,
) -> np.ndarray:
    """doc"""
    for player in player_locations:
        center = cv2.perspectiveTransform(
            player.reshape(-1, 1, 2).astype(np.float32), homography_mat
        )
        try:
            cv2.circle(
                img=image,
                center=np.squeeze(center).astype(np.uint32),
                *args,
                **kwargs,
            )
            cv2.circle(
                img=image,
                center=np.squeeze(center).astype(np.uint32),
                radius=3,
                color=(0, 0, 0),
                thickness=-1,
            )
        except:
            print(
                " Could not 2D draw a player, see xy: ",
                np.squeeze(center).astype(np.uint32),
            )
    return image


@timer(enabled=TIME_FUNCTIONS)
def draw_players_ID(
    image: np.ndarray,
    xyxy: np.ndarray,
    ids: list[str],
    draw_bboxes: bool = False,
    font_scale: float = 1,
    *args,
    **kwargs,
):
    for idx, player in enumerate(xyxy):
        cv2.putText(
            img=image,
            text=f"ID: {ids[idx]}",
            org=(int(player[0]), int(player[1]) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            *args,
            **kwargs,
        )
        if draw_bboxes:
            cv2.rectangle(
                img=image,
                pt1=(int(player[0]), int(player[1])),
                pt2=(int(player[2]), int(player[3])),
                *args,
                **kwargs,
            )

    return image


def extract_players_close_to_baseline(
    image: np.ndarray,
    bboxes: np.ndarray,
    show_result: bool = False,
    *args,
    **kwargs,
) -> np.ndarray:
    players_to_keep = []
    for _, player_xy in enumerate(bboxes):
        players_to_keep.append(
            np.abs(
                distance_to_vertical_centerline(
                    pt=player_xy, resolution_wh=image.shape[:2]
                )
            )
        )
    # Pick the two shortest distances and plot results if flagged
    keep = np.argsort(players_to_keep)[:2]
    if show_result:
        results_image = image.copy()
        for kept in bboxes[keep]:
            cv2.circle(
                img=results_image,
                center=kept,
                radius=10,
                color=(0, 0, 255),
                thickness=-1,
                *args,
                **kwargs,
            )
        cv2.line(
            img=results_image,
            pt1=np.array([image.shape[:2][1] / 2, 0]).astype(np.uint32),
            pt2=np.array([image.shape[:2][1] / 2, image.shape[:2][0]]).astype(
                np.uint32
            ),
            color=(255, 255, 255),
            thickness=10,
        )
        cv2.imshow("Players close to vertical center line", results_image)
    return keep
