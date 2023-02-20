from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from utilities.timing import timer

from .config import TIME_FUNCTIONS


@dataclass
class SimpleCentroidTracker:
    object_paths: dict = field(default_factory=lambda: {})
    current_position: dict = field(default_factory=lambda: {})
    next_id: int = 0
    maxlen: int = 30

    def __post_init__(self):
        self.prev_positions = deque(maxlen=self.maxlen)

    @timer(enabled=TIME_FUNCTIONS)
    def update(self, player_position: np.ndarray):
        if not self.prev_positions:
            for position in player_position:
                self.prev_positions.append(position)
                self.current_position[self.next_id] = position
                self.object_paths[self.next_id] = [position]
                self.next_id += 1
        else:
            distances = []
            for position in player_position:
                distances.append(
                    [
                        np.linalg.norm(position - prev_centroid)
                        for prev_centroid in self.prev_positions
                    ]
                )
            for i, position in enumerate(player_position):
                match = np.argmin(distances[i])
                if match not in self.current_position:
                    self.current_position[match] = position
                    self.object_paths[match] = [position]
                else:
                    self.current_position[match] = position
                    self.object_paths[match].append(position)
        self.prev_positions.extend(player_position)

    @timer(enabled=TIME_FUNCTIONS)
    def draw_player_paths(
        self,
        image: np.ndarray,
        radius: int = 4,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = -1,
        *args,
        **kwargs,
    ):
        # ObjectID isn't implemented yet
        for (objectID, path) in self.object_paths.items():
            for i in range(max(1, len(path) - self.maxlen), len(path)):
                if path[i - 1] is None or path[i] is None:
                    continue
                try:
                    cv2.circle(
                        img=image,
                        center=(path[i][0], path[i][1]),
                        radius=radius,
                        color=color,
                        thickness=thickness,
                        *args,
                        **kwargs,
                    )
                except Exception as e:
                    print(
                        " Couldn't draw path, see x,y: ",
                        (path[i][0], path[i][1]),
                        e,
                    )
        return image

    @timer(enabled=TIME_FUNCTIONS)
    def draw_object_paths_2d(
        self,
        image: np.ndarray,
        homography_mat,
        radius: int = 4,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = -1,
        *args,
        **kwargs,
    ):
        # ObjectID isn't implemented yet
        for (objectID, path) in self.object_paths.items():
            for i in range(max(1, len(path) - self.maxlen), len(path)):
                if path[i - 1] is None or path[i] is None:
                    continue
                ball_center = np.squeeze(
                    cv2.perspectiveTransform(
                        path[i].reshape(-1, 1, 2).astype(np.float32),
                        homography_mat,
                    )
                ).astype(np.uint32)
                try:
                    cv2.circle(
                        img=image,
                        center=ball_center,
                        radius=radius,
                        color=color,
                        thickness=thickness,
                        *args,
                        **kwargs,
                    )
                except Exception as e:
                    print(" Couldn't draw 2D path, see x,y: ", ball_center, e)
        return image


# @dataclass
# class Tracker:
#     object_paths: dict = field(default_factory=lambda: {})
#     maxlen: int = 30
#     frame_counts: list
#     ball_locations: pd.DataFrame
#     player_locations: pd.DataFrame

#     @timer(enabled=TIME_FUNCTIONS)
#     def update(self, player_positions: list[np.ndarray], player_ids: list[str]):
#         # Missing location for a given player id? Add a None
#         for player_id in self.object_paths.keys():
#             if player_id not in player_ids:
#                 self.object_paths[player_id].append(None)
#         # For each player ID add the path. If it doesn't exist, create it.
#         for idx in range(len(player_positions)):
#             if player_ids[idx] not in self.object_paths:
#                 self.object_paths[player_ids[idx]] = deque(maxlen=self.maxlen)
#             self.object_paths[player_ids[idx]].append(player_positions[idx])

#     @timer(enabled=TIME_FUNCTIONS)
#     def draw_player_paths(
#         self,
#         image: np.ndarray,
#         radius: int = 4,
#         color: tuple[int, int, int] = (0, 0, 0),
#         thickness: int = -1,
#         *args,
#         **kwargs
#     ):
#         for player_id, positions in self.object_paths.items():
#             for position in positions:
#                 if position is not None:
#                     try:
#                         cv2.circle(
#                             image, tuple(position), radius, color, thickness, *args, **kwargs
#                         )
#                     except Exception as e:
#                         print(" Couldn't draw path, see x,y: ", tuple(position), e)
#         return image

#     @timer(enabled=TIME_FUNCTIONS)
#     def draw_object_paths_2d(
#         self,
#         image: np.ndarray,
#         homography_mat: np.ndarray,
#         radius: int = 4,
#         color: tuple[int, int, int] = (0, 0, 0),
#         thickness: int = -1,
#         *args,
#         **kwargs
#     ):
#         for player_id, positions in self.object_paths.items():
#             for position in positions:
#                 if position is not None:
#                     ball_center = np.squeeze(
#                         cv2.perspectiveTransform(
#                             position.reshape(-1, 1, 2).astype(np.float32), homography_mat
#                         )
#                     ).astype(np.uint32)
#                     try:
#                         cv2.circle(
#                             image, tuple(ball_center), radius, color, thickness, *args, **kwargs
#                         )
#                     except Exception as e:
#                         print(" Couldn't draw 2D path, see x,y: ", tuple(ball_center), e)
#         return image

#     def store_tracking(
#         self,
#         frame_no: int,
#         ball_location: tuple(int, int) = None,
#         s_path_player: bool = True,
#     ):
#         self.frame_counts.append(frame_no)
#         # Append to a list before writing to file since concatenating or appending in pandas is slow
#         if ball_location is not None:
#             self.ball_locations.append(ball_location)
#         if s_path_player is not None:
#             self.player_locations.append()

#     def write_tracking(
#         self,
#         path_base: str,
#         s_path_ball: bool = True,
#         s_path_player: bool = True,
#     ):
#         if s_path_ball:
#             pd.DataFrame()
#         if s_path_player:
#             pd.DataFrame()


@dataclass
class PlayerPathTracker:
    object_paths: dict = field(default_factory=lambda: {})
    maxlen: int = 30
    player_locations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["frame_no"])
    )

    @timer(enabled=TIME_FUNCTIONS)
    def update_path(
        self,
        frame_no: int,
        player_positions: list[np.ndarray],
        player_ids: list[str],
    ):
        # Missing location for a given player id? Add a None
        for player_id in self.object_paths.keys():
            if player_id not in player_ids:
                self.object_paths[player_id].append((frame_no, None))
        # For each player ID add the path. If it doesn't exist, create it.
        for idx in range(len(player_positions)):
            if player_ids[idx] not in self.object_paths:
                self.object_paths[player_ids[idx]] = deque(maxlen=self.maxlen)
            self.object_paths[player_ids[idx]].append(
                (frame_no, player_positions[idx])
            )

    @timer(enabled=TIME_FUNCTIONS)
    def draw_player_paths(
        self,
        image: np.ndarray,
        radius: int = 4,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = -1,
        *args,
        **kwargs,
    ):
        for player_id, positions in self.object_paths.items():
            for position in positions:
                if position[1:][0] is not None:
                    try:
                        cv2.circle(
                            image,
                            tuple(position[1:][0]),
                            radius,
                            color,
                            thickness,
                            *args,
                            **kwargs,
                        )
                        cv2.circle(
                            image,
                            tuple(position[1:][0]),
                            radius=int(radius / 2),
                            color=(0, 0, 0),
                            thickness=thickness,
                        )
                    except Exception as e:
                        print(
                            " Couldn't draw path, see x,y: ",
                            tuple(position),
                            e,
                        )
        return image

    @timer(enabled=TIME_FUNCTIONS)
    def draw_object_paths_2d(
        self,
        image: np.ndarray,
        homography_mat: np.ndarray,
        radius: int = 4,
        color: tuple[int, int, int] = (0, 0, 0),
        thickness: int = -1,
        *args,
        **kwargs,
    ):
        for player_id, positions in self.object_paths.items():
            for position in positions:
                if position[1:][0] is not None:
                    ball_center = np.squeeze(
                        cv2.perspectiveTransform(
                            position[1:][0]
                            .reshape(-1, 1, 2)
                            .astype(np.float32),
                            homography_mat,
                        )
                    ).astype(np.uint32)
                    try:
                        cv2.circle(
                            image,
                            tuple(ball_center),
                            radius,
                            color,
                            thickness,
                            *args,
                            **kwargs,
                        )
                        cv2.circle(
                            image,
                            tuple(ball_center),
                            int(radius / 2),
                            (0, 0, 0),
                            thickness,
                            *args,
                            **kwargs,
                        )
                    except Exception as e:
                        print(
                            " Couldn't draw 2D path, see x,y: ",
                            tuple(ball_center),
                            e,
                        )
        return image

    def retrieve_batch(self, current_frame: int, batch_size: int = None):
        data = {}
        if not batch_size:  # search the entire object
            for player_id, player_path in self.object_paths.items():
                for f, position in player_path:
                    if f not in data:
                        data[f] = {player_id: position}
                    else:
                        data[f][player_id] = position
        else:  # search only a given batch
            start_frame = current_frame + 1 - batch_size
            for frame_no in range(start_frame, current_frame + 1):
                data[frame_no] = {
                    player_id: None for player_id in self.object_paths.keys()
                }
                for player_id, player_path in self.object_paths.items():
                    for f, position in player_path:
                        if f == frame_no:
                            data[frame_no][player_id] = position
                            break
        return data

    def to_dataframe(self, data: dict):
        for frame_no, positions in data.items():
            x_values = []
            y_values = []
            for player_id, position in positions.items():
                if position is None:
                    x_values.append(np.nan)
                    y_values.append(np.nan)
                else:
                    x_values.append(position[0])
                    y_values.append(position[1])
            x_cols = {
                f"{player_id}_x": x_val
                for player_id, x_val in zip(positions.keys(), x_values)
            }
            y_cols = {
                f"{player_id}_y": y_val
                for player_id, y_val in zip(positions.keys(), y_values)
            }
            row = {"frame_no": frame_no, **x_cols, **y_cols}
            self.player_locations = pd.concat(
                [self.player_locations, pd.DataFrame([row])], ignore_index=True
            )

        return self.player_locations
