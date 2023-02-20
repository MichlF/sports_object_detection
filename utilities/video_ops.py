"""Module containing functions to read a video"""
from pathlib import Path

import cv2
import imageio
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def read_video(
    video_path: str,
    show_frames: bool = True,
    frame_by_frame: bool = False,
    save_frame: bool = False,
    *args,
    **kwargs,
):
    """
    Reads in and shows a video (continuously or frame-by-frame). Allows storing each frame as image.
    Args:
        video_path (str): The path of the video to be read.
        show_frames (bool): Whether or not to show the frames of the video. Default is True.
        frame_by_frame (bool): Whether to show the frames one by one, pausing after each. Default is False.
        save_frame (bool): Whether to save each frame of the video as an image. Default is False.
        *args: Additional arguments to be passed to `cv2.imwrite` when saving frames.
        **kwargs: Additional keyword arguments to be passed to `cv2.imwrite` when saving frames.
    """
    video = cv2.VideoCapture(video_path)
    if save_frame:
        output_folder = Path("output_frames")
        output_folder.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    # Read in the video, show each frame, if defined frame-by-frame and if defined save each frame
    while True:
        success, frame = video.read()
        if not success:
            print("Couldn't load the video.")
            break
        if show_frames:
            cv2.imshow("Frame", frame)
        if save_frame:
            cv2.imwrite(
                f"output_frames/frame{frame_count}.jpg", frame, *args, **kwargs
            )
            frame_count += 1
        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise KeyError("You have stopped the simulation!")
        elif frame_by_frame and cv2.waitKey(0) & 0xFF == ord(
            " "
        ):  # frame by frame with other key
            continue
    # Release the video capture object and close all windows
    video.release()
    cv2.destroyAllWindows()


def resize_moviepy(
    input_path: str,
    output_path: str,
    resolution_height: int = 640,
    resolution_width: int = 640,
    *args,
    **kwargs,
):
    """
    Resize a video using moviepy.
    Args:
        input_path (str): The path of the input video.
        output_path (str): The path of the output video.
        resolution_height (int): The height of the output video. Default is 640.
        resolution_width (int): The width of the output video. Default is 640.
        *args: Additional arguments to be passed to `video.resize`
        **kwargs: Additional keyword arguments to be passed to `video.resize`
    """
    video = VideoFileClip(input_path)
    resized_video = video.resize(
        height=resolution_height, width=resolution_width, *args, **kwargs
    )
    resized_video.write_videofile(output_path)


def resize_opencv(
    input_path: str,
    output_path: str,
    resolution_height: int = 640,
    resolution_width: int = 640,
    *args,
    **kwargs,
):
    """
    Resize a video using OpenCV
    Args:
        input_path (str): The path of the input video.
        output_path (str): The path of the output video.
        resolution_height (int): The height of the output video. Default is 640.
        resolution_width (int): The width of the output video. Default is 640.
        *args: Additional arguments to be passed to `cv2.resize`
        **kwargs: Additional keyword arguments to be passed to `cv2.resize`
    """
    cap = cv2.VideoCapture(input_path)
    # Get the frames per second (fps) and codec info from the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(
        output_path, fourcc, fps, (resolution_height, resolution_width)
    )
    # Start the loop with a progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count)):
        # Read the current frame from the video
        success, frame = cap.read()
        if not success:
            print("Couln't read the video.")
            break
        # Resize the frame to 640x640 pixels and write to output
        frame = cv2.resize(
            frame, (resolution_height, resolution_width), *args, **kwargs
        )
        out.write(frame)
    # Release the video capture and output objects
    cap.release()
    out.release()


def mp4_to_gif_opencv(
    path_to_mp4, path_to_gif, start_frame, duration_seconds, fps
):
    vid = cv2.VideoCapture(path_to_mp4)
    frames = []
    vid.set(cv2.CAP_PROP_POS_MSEC, start_frame * 1000)
    vid.set(cv2.CAP_PROP_FPS, fps)
    print("Starting to write animated gif...")
    while True:
        success, image = vid.read()
        if not success:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if vid.get(cv2.CAP_PROP_POS_MSEC) > duration_seconds * 1000:
            print(vid.get(cv2.CAP_PROP_POS_MSEC))
            break
    vid.release()
    imageio.mimsave(path_to_gif, frames, fps=fps)
    print("Animated gif successfully written.")
