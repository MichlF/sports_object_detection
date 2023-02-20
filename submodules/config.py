# Imports
from pathlib import Path

# Which video?
VIDEO_IN_NAME = "video_cut.mp4"

# Ball trajectory and player path lengths
DRAW_BALL_TRAJECTORY_LEN = 20
BALL_TRAJECTORY_LEN = 150
OBJECT_TRACKER_LEN = 150

# Styling
# Regular frame for video
COLOR_PLAYERS = (0, 255, 0)
COLOR_PLAYERS_ID = (0, 255, 0)
COLOR_BALL = (79, 255, 223)
COLOR_COURTLINES = (255, 0, 0)
COLOR_PATHS = (0, 0, 200)
THICKNESS_PLAYERS = 4
THICKNESS_PLAYERS_ID = 2
THICKNESS_BALL = -1
THICKNESS_COURTLINES = 4
THICKNESS_PATHS = -1
RADIUS_BALL = 5
RADIUS_COURTLINES = 4
RADIUS_PATHS = 4
# 2D Minimap
COLOR_PLAYERS_2D = (0, 255, 0)
COLOR_BALL_2D = (79, 255, 223)
COLOR_COURTLINES_2D = (255, 0, 0)
COLOR_PATHS_2D = (0, 0, 200)
COLOR_BORDER_2D = (0, 0, 0)
THICKNESS_PLAYERS_2D = -1
THICKNESS_BALL_2D = -1
THICKNESS_COURTLINES_2D = -1
THICKNESS_PATHS_2D = -1
RADIUS_PLAYERS_2D = 8
RADIUS_BALL_2D = 5
RADIUS_COURTLINES_2D = 4
RADIUS_PATHS_2D = 3

# Real world measurements (in meters)
BASELINE_LEN = 10.97
AVG_PLAYER_HEIGHT = 1.885

# Model parameters
DEEPSORT_MAX_AGE = 30
DEEPSORT_MAX_IOU_DISTANCE = 0.7
DEEPSORT_N_INIT = 2
DEEPSORT_NMS_MAX_OVERLAP = 1.0
DEEPSORT_COSINE_DISTANCE = 0.5
MODEL_RESOLUTION = (640, 640)
# Note maybe (640,360) instead of (640,640) b/c 16-9 aspect ratio, but (detection)
# performance is actually worse possibly because objects are even smaller
TRACKNET_CLASSES = 256
TRACKNET_IMAGES = 3
TRACKNET_WEIGHTS = "models/TrackNet/model.3"
YOLO_CONFIDENCE = 0.38
YOLO_MODEL = "models/YOLO/yolov8n.pt"

# Time functions
TIME_FUNCTIONS = False

# Video output params
VIDEO_OUT_FPS = 60
VIDEO_OUT_CODEC = "mp4v"
MINIMAP_SCALING = 3.5

# Define paths
PATH_IN_VIDEO = Path("input")
PATH_OUT_VIDEO = Path("output\\videos")
PATH_OUT_FRAME = Path("output\\frames")
PATH_OUT_STATS = Path("output\\statistics")

# Create folder structures
PATH_OUT_VIDEO.mkdir(parents=True, exist_ok=True)
PATH_OUT_FRAME.mkdir(parents=True, exist_ok=True)
