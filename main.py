import time
from collections import deque

import cv2
import matplotlib
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from ultralytics import YOLO

from models.TrackNet.track_net import track_net
from submodules.ball import (
    bounce_analysis,
    draw_ball,
    draw_ball_2d,
    get_ball_position_tracknet,
    store_data,
)
from submodules.config import *
from submodules.court import (
    create_avg_frame,
    draw_court_corners_2d,
    draw_court_lines,
    draw_court_portrait,
    draw_minimap_to_frame,
    get_court_corners,
    get_court_corners_2d,
)
from submodules.object_tracking import PlayerPathTracker
from submodules.players import (
    draw_players,
    draw_players_2d,
    draw_players_ID,
    extract_players_close_to_baseline,
    get_player_boxes_yolo,
    track_player_deep_sort,
)
from utilities.gpu import init_gpu_tpu
from utilities.image_ops import real_to_pixels

# Force activate TkAgg backend otherwise cv2 images may not load (mostly in debug mode)
matplotlib.use("TkAgg", force=True)
print("Switched to:", matplotlib.get_backend())

# Activate GPU support (necessary for some TF versions)
init_gpu_tpu()


def process_video():
    frames_list, ball_list = [], []

    # Load input video, get info: fps, resolution, total frame count
    path_in_video = PATH_IN_VIDEO / VIDEO_IN_NAME
    if not path_in_video.is_file():
        raise ValueError("Couldn't load the video !")
    vid_in = cv2.VideoCapture(str(path_in_video))
    # fps = int(vid_in.get(cv2.CAP_PROP_FPS))
    vid_in_resolution_wh = (
        int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    try:
        vid_in_total_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
    except:  # if codec doesn't have the frame count container
        raise ValueError("Cannot get total frame count !")

    # Define output video parameters
    vid_out = cv2.VideoWriter(
        str(PATH_OUT_VIDEO / ("processed_" + VIDEO_IN_NAME)),
        cv2.VideoWriter_fourcc(*VIDEO_OUT_CODEC),
        VIDEO_OUT_FPS,
        vid_in_resolution_wh,
    )

    # Setting model parameters & loading model objects
    model_resolution_wh = MODEL_RESOLUTION
    # TrackNetV2 takes the current + the last two images as input
    frame_in_n_2, frame_in_n_1, frame_in = None, None, None
    model_yolo = YOLO(YOLO_MODEL)
    model_tracknet = track_net(
        n_classes=TRACKNET_CLASSES,
        input_n_images=TRACKNET_IMAGES,
        input_height=model_resolution_wh[1],
        input_width=model_resolution_wh[0],
        return_summary=False,
    )
    model_tracknet.compile(
        loss="categorical_crossentropy",
        optimizer="adadelta",
        metrics=["accuracy"],
    )
    model_tracknet.load_weights(TRACKNET_WEIGHTS)

    # Ball and player tracking
    ball_trajectory = deque(maxlen=BALL_TRAJECTORY_LEN)
    for _ in range(0, BALL_TRAJECTORY_LEN):
        ball_trajectory.appendleft(None)
    pp_tracking = PlayerPathTracker(maxlen=OBJECT_TRACKER_LEN)
    tracker = DeepSort(
        max_age=DEEPSORT_MAX_AGE,
        n_init=DEEPSORT_N_INIT,
        nms_max_overlap=DEEPSORT_NMS_MAX_OVERLAP,
        max_cosine_distance=DEEPSORT_COSINE_DISTANCE,
        max_iou_distance=DEEPSORT_MAX_IOU_DISTANCE,
    )

    # Generate avg frame (washeding out moving objects, i.e. players) for automatic line detection
    frame_corner = create_avg_frame(
        video_capture_object=vid_in,
        start_frame=0,
        end_frame=400,
        n_th_frame=20,
    )

    # Instantiate list objects
    frames_list, ball_list = [], []

    # Start processing the video
    for frame_count in tqdm(range(vid_in_total_frames)):
        # tqdm.write("") # instead of print
        # Update previous images
        frame_in_n_2 = frame_in_n_1
        frame_in_n_1 = frame_in

        # 1. Return video frame-by-frame and create copy to draw on it
        vid_in.set(1, frame_count)
        success, frame_in = vid_in.read()
        # cv2.imwrite(str(PATH_OUT_FRAME / f"frame_{frame_count}.jpg"), frame_in)
        frame_out = frame_in.copy()
        if not success:
            raise ValueError(f"Couldn't load frame {frame_count} !")

        # Frame 1: create minimap, get court corners, transformation matrix and average player height in px
        if frame_count == 0:
            court_corners = get_court_corners(
                image=frame_corner, mode="automatic"
            )
            court_corners_2d = get_court_corners_2d(
                height=int(vid_in_resolution_wh[1] / MINIMAP_SCALING)
            )
            image_court_2d = draw_court_portrait(
                height=int(vid_in_resolution_wh[1] / MINIMAP_SCALING)
            )
            homography_matrix = cv2.getPerspectiveTransform(
                np.float32(court_corners), np.float32(court_corners_2d)
            )
            player_height_px = real_to_pixels(
                distance_real=AVG_PLAYER_HEIGHT,
                reference_len_px=int(
                    np.linalg.norm(
                        np.array(court_corners[2]) - np.array(court_corners[3])
                    )
                ),
                reference_len_world=BASELINE_LEN,
            )

        # 2. Get player boundary boxes (YOLO) and ensure tracking (DeepSort)
        (
            player_boxes,
            player_boxes_wh,
            player_confs,
            player_centroids,
            player_centergrounds,
        ) = get_player_boxes_yolo(
            image=frame_in,
            model_yolo_object=model_yolo,
            yolo_conf=YOLO_CONFIDENCE,
            apply_nms=True,
        )
        # Feed the tracker only the two persons closest to the vertical midline on any given frame,
        # since they are most likely the players.
        idx_to_keep = extract_players_close_to_baseline(
            image=frame_in, bboxes=player_centroids
        )
        (
            player_ids,
            player_boxes,
            player_centergrounds,
        ) = track_player_deep_sort(
            image=frame_in,
            tracker_object=tracker,
            player_xywh=player_boxes_wh[idx_to_keep],
            player_confs=player_confs[idx_to_keep],
        )
        # A hack for now: remove player IDs above 2 because they are not players
        player_ids = [
            player_id for player_id in player_ids if int(player_id) < 3
        ]
        player_boxes = [
            player_id
            for i, player_id in zip(player_ids, player_boxes)
            if int(i) < 3
        ]
        player_centergrounds = [
            player_id
            for i, player_id in zip(player_ids, player_centergrounds)
            if int(i) < 3
        ]
        pp_tracking.update_path(
            frame_no=frame_count,
            player_positions=player_centergrounds,
            player_ids=player_ids,
        )

        # 3. Get current frame ball location based on last three images (TrackNet)
        frame_in = cv2.resize(
            src=frame_in,
            dsize=(model_resolution_wh[0], model_resolution_wh[1]),
        ).astype(np.float32)
        # TrackNet cannot predict the ball location on the first two frames
        ball = None
        # start = time.time()
        if frame_count > 1:
            last_three_images = np.concatenate(
                (frame_in_n_2, frame_in_n_1, frame_in), axis=2
            )
            ball = get_ball_position_tracknet(
                images=last_three_images,
                tracknet_object=model_tracknet,
                output_res=vid_in_resolution_wh,
            )
        # print("Time elapsed:", 1000*(time.time()-start))

        # 4. Draw objects onto the frame
        frame_out = draw_court_lines(
            image=frame_out,
            court_corners=court_corners,
            color=COLOR_COURTLINES,
            thickness=THICKNESS_COURTLINES,
        )
        frame_out = pp_tracking.draw_player_paths(
            image=frame_out,
            color=COLOR_PATHS,
            radius=RADIUS_PATHS,
            thickness=THICKNESS_PATHS,
        )
        frame_out = draw_players(
            image=frame_out,
            xyxy=player_boxes,
            color=COLOR_PLAYERS,
            thickness=THICKNESS_PLAYERS,
        )
        frame_out = draw_players_ID(
            image=frame_out,
            xyxy=player_boxes,
            ids=player_ids,
            draw_bboxes=False,
            color=COLOR_PLAYERS_ID,
            thickness=THICKNESS_PLAYERS_ID,
        )
        frame_out, ball_trajectory = draw_ball(
            image=frame_out,
            xy=ball,
            trajectory_deque=ball_trajectory,
            annotation_type="direction",  # "location" for x,y "direction" for vectors or "both"
            color=COLOR_BALL,
            radius=RADIUS_BALL,
            thickness=THICKNESS_BALL,
            draw_len_trajectory=DRAW_BALL_TRAJECTORY_LEN,
        )

        # 5. Draw objects onto 2D minimap
        frame_out_2d = image_court_2d.copy()
        frame_out_2d = draw_court_corners_2d(
            image=frame_out_2d,
            court_corners=court_corners_2d,
            color=COLOR_COURTLINES_2D,
            radius=RADIUS_COURTLINES_2D,
            thickness=THICKNESS_COURTLINES_2D,
        )
        frame_out_2d = pp_tracking.draw_object_paths_2d(
            image=frame_out_2d,
            homography_mat=homography_matrix,
            color=COLOR_PATHS_2D,
            radius=RADIUS_PATHS_2D,
            thickness=THICKNESS_PATHS_2D,
        )
        frame_out_2d = draw_players_2d(
            image=frame_out_2d,
            homography_mat=homography_matrix,
            player_locations=player_centergrounds,
            color=COLOR_PLAYERS_2D,
            radius=RADIUS_PLAYERS_2D,
            thickness=THICKNESS_PLAYERS_2D,
        )
        frame_out_2d = draw_ball_2d(
            image=frame_out_2d,
            homography_mat=homography_matrix,
            ball_coords=ball,
            adjust_height=int(player_height_px / 2),
            color=COLOR_BALL_2D,
            radius=RADIUS_BALL_2D,
            thickness=THICKNESS_BALL_2D,
        )
        frames_combined = draw_minimap_to_frame(
            image=frame_out,
            image_minimap=frame_out_2d,
            border_color=COLOR_BORDER_2D,
        )

        #! TRYOUTING
        if frame_count != 0 and frame_count % OBJECT_TRACKER_LEN == 0:
            # Every OBJECT_TRACKER_LEN we expand the dataframes for ball and player locations
            player_paths_to_store = pp_tracking.retrieve_batch(
                current_frame=frame_count, batch_size=OBJECT_TRACKER_LEN
            )
            pp_tracking.to_dataframe(data=player_paths_to_store)
            ball_list.extend(list(ball_trajectory))

        # # Estimate a bounce every ball_trajectory_len - half the window size frames
        # window_size = 7
        # # if frame_count % len(ball_trajectory)-int(window_size/2):
        # #     bounce_analysis(ball_trajectory, window_size, frame_count)
        # frames_list.append(
        #     bounce_analysis(
        #         trajectory_deque=ball_trajectory,
        #         window_size=window_size,
        #         frame_count_debug=frame_count,
        #         image_debug=frame_out,
        #     )
        # )
        # 5. Write the frame to the output video
        vid_out.write(frames_combined)
        # print("Frames_list: ", frames_list)

    # Last frame processed: release videocaptures and store statistics
    vid_in.release()
    vid_out.release()

    pp_tracking.player_locations.to_csv(str(PATH_OUT_STATS / "players.csv"))
    store_data(list_of_arrays=ball_list).to_csv(
        str(PATH_OUT_STATS / "ball.csv")
    )


def main():
    process_video()


if __name__ == "__main__":
    main()
