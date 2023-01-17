import cv2
from pathlib import Path

def read_video(video_path: str, show_frames: bool = True, frame_by_frame: bool = False, save_frame: bool = False, *args, **kwargs):
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
        if cv2.waitKey(0) & 0xFF == ord('q'): # exit with q
            cv2.destroyAllWindows()
            raise KeyError("You have stopped the simulation!")
        elif frame_by_frame and cv2.waitKey(0) & 0xFF == ord(' '): # frame by frame with other key
            continue
    # Release the video capture object and close all windows
    video.release()
    cv2.destroyAllWindows()