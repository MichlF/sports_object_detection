"""Module containing functions to read a video"""
import cv2
from pathlib import Path
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
    """Reads in and shows a video (continuously or frame-by-frame). Allows storing each frame as image."""
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
            cv2.imwrite(f"output_frames/frame{frame_count}.jpg", frame, *args, **kwargs)
            frame_count += 1
        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(0) & 0xFF == ord("q"):  # exit with q
            cv2.destroyAllWindows()
            raise KeyError("You have stopped the simulation!")
        elif frame_by_frame and cv2.waitKey(0) & 0xFF == ord(" "):  # frame by frame with other key
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
    video = VideoFileClip(input_path)
    resized_video = video.resize(height=resolution_height, width=resolution_width, *args, **kwargs)
    resized_video.write_videofile(output_path)


def resize_opencv(
    input_path: str,
    output_path: str,
    resolution_height: int = 640,
    resolution_width: int = 640,
    *args,
    **kwargs,
):
    # Open the video
    cap = cv2.VideoCapture(input_path)
    # Get the frames per second (fps) and codec info from the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(output_path, fourcc, fps, (resolution_height, resolution_width))

    # Start the loop with a progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count)):
        # Read the current frame from the video
        success, frame = cap.read()
        if not success:
            print("Couln't read the video.")
            break
        # Resize the frame to 640x640 pixels and write to output
        frame = cv2.resize(frame, (resolution_height, resolution_width), *args, **kwargs)
        out.write(frame)
    # Release the video capture and output objects
    cap.release()
    out.release()
