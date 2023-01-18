# Import the model
# We load the video
# Split it into individual images/frames
# for each frame
# turn grayscale if necessary
# Resize if necessary
# Turn into tensor if possible (speeds up processing)?
# Feed into the model and return prediction
### Should we restrict the output classes by person, tennis racket, sports ball?

# Imports
import cv2
import numpy as np
from utilities import image_crop, edge_detection
from ultralytics import YOLO


model_yolo = YOLO("yolov8n.pt")
input_path = "C:\\Users\\miche\\Documents\\GitHub\\sports_object_detection\\input"
output_path = "C:\\Users\\miche\\Documents\\GitHub\\sports_object_detection\\output"
video_name = "set_lowq.mp4"
grayscale = False
show_image = False
resize = False
save_image = True

video = cv2.VideoCapture(input_path + "/" + video_name)

frame_count = 0
while True:
    success, frame = video.read()
    if not success:
        print("Couldn't load the video.")
        break
    frame_count += 1
    output_image = f"{output_path}/frame{frame_count}.jpg"

    # Preprocessing
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if resize:
        frame = cv2.resize(frame, (640, 640))
    if save_image:
        cv2.imwrite(output_image, frame)

    # YOLO needs (image no, height, width, color channels) format. Note color channels must be 3
    # However, in the current state of the Python package, it does not read ndarrays for some reason
    image = cv2.imread(output_image)
    results = model_yolo(source=image, conf=0.45)
    # Results is a generator containing a result class for each analyzed image/frame
    # Each result class contains a box for each detected object in boxes
    # Each box has a value for the xyxy / xywh coordinates, confidence scores (conf) and class id (cls).
    for result_frame in results:
        print(result_frame.boxes.conf)
        print(result_frame.boxes.cls)
        for idx, object_tensorbox in enumerate(result_frame.boxes.xyxy):
            detected_object = image_crop.image_crop(image=image, tensor_box_xyxy=object_tensorbox)
            cv2.imshow(f"Object {idx}", detected_object)
    image, rectangles, rectangles_hough = edge_detection.extract_contours(
        image=frame, lower_threshold=200, upper_threshold=200, show_contours=True
    )
    # Debug
    if frame_count > 5:
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()
