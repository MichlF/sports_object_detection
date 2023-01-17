"""Module containing using different packages for resizing videos. Tested with .mp4 movie format, should work with any usual format."""
import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def resize_moviepy(input_path: str, output_path: str, resolution_height: int = 640, resolution_width: int = 640, *args, **kwargs):
    video = VideoFileClip(input_path)
    resized_video = video.resize(height=resolution_height, width=resolution_width, *args, **kwargs)
    resized_video.write_videofile(output_path)

def resize_opencv(input_path: str, output_path: str, resolution_height: int = 640,  resolution_width: int = 640, *args, **kwargs):
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