import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Labelbox Configuration
LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")
LABELBOX_PROJECT_ID = os.getenv("LABELBOX_PROJECT_ID")

# YOLO Configuration
YOLO_CONFIG = {
    'device': os.getenv('YOLO_DEVICE', '0'),  # Use GPU 0 by default, 'cpu' for CPU-only
    'batch': int(os.getenv('YOLO_BATCH_SIZE', '8')),
    'epochs': int(os.getenv('YOLO_EPOCHS', '50')),
    'imgsz': int(os.getenv('YOLO_IMG_SIZE', '1080')),
    'conf_threshold': float(os.getenv('YOLO_CONF_THRESHOLD', '0.25')),
    'iou_threshold': float(os.getenv('YOLO_IOU_THRESHOLD', '0.45')),
    'pretrained_model': os.getenv('YOLO_PRETRAINED_MODEL', 'yolo11n.pt'),
    'max_det': int(os.getenv('YOLO_MAX_DET', '10'))  # Valorant-optimized: max 10 players per frame
}

# Directory Configuration
BASE_DIR = Path(__file__).parent
WORKFLOW_DATA_DIR = BASE_DIR / "workflow_data"
DOWNLOADS_DIR = WORKFLOW_DATA_DIR / "downloads"
DATASET_DIR = WORKFLOW_DATA_DIR / "dataset"
MODELS_DIR = WORKFLOW_DATA_DIR / "models"
INFERENCE_DIR = WORKFLOW_DATA_DIR / "inference"
LOGS_DIR = WORKFLOW_DATA_DIR / "logs"
ARCHIVE_DIR = WORKFLOW_DATA_DIR / "archive"

# Logging Configuration
LOG_FILE = LOGS_DIR / "workflow.log"

# Class Mapping (for your Valorant project)
CLASS_MAPPING = {
    'agent': 0,
    'ability': 1,
    'ultimate': 2,
    'round_info': 3,
    'kill': 4,
    'map': 5,
    'player': 6,
    'weapon': 7,
    'ability_model': 8,
    'ultimate_model': 9,
    'spike': 10
}

# Reverse mapping for inference
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Labelbox Export Parameters
EXPORT_PARAMS = {
    "attachments": True,
    "metadata_fields": True,
    "data_row_details": True,
    "project_details": True,
    "label_details": True,
    "performance_details": True,
    "labels": True
}

# Labelbox Status Configuration
LABELBOX_STATUS = {
    'TO_REVIEW': 'LABEL_TO_REVIEW',
    'COMPLETE': 'LABEL_COMPLETE',
    'SKIPPED': 'LABEL_SKIPPED'
}

# Object tracking configuration - using ByteTrack instead of custom tracking
BYTETRACK_CONFIG = {
    # ByteTrack specific parameters
    'track_thresh': 0.25,       # High confidence threshold for first association (lowered to create tracks)
    'track_low_thresh': 0.1,    # Low confidence threshold for second association
    'new_track_thresh': 0.25,   # Threshold for initializing new tracks (should match track_thresh)
    'track_buffer': 30,         # Number of frames to keep lost tracks
    'match_thresh': 0.8,        # Matching threshold for association
    'distance_threshold': 200,  # Distance threshold for track association (pixels)
    'mot20': False,             # MOT20 specific settings
    
    # Valorant-specific constraints
    'max_tracks': 10,           # Maximum number of simultaneous tracks (10 players max)
    'max_detections_per_frame': 10,  # Maximum detections per frame to process
    'valorant_optimized': True,  # Flag to enable Valorant-specific optimizations
    
    # Frame rate dependent settings (adjust based on your video FPS)
    'frame_rate': 30,           # Expected frame rate of your videos
    
    # Use ByteTrack for both training and inference consistency
    'use_ultralytics_tracker': True,
    'tracker_config': 'bytetrack.yaml'  # Ultralytics tracker configuration
}

def ensure_directories():
    """Create all necessary directories"""
    directories = [
        WORKFLOW_DATA_DIR,
        DOWNLOADS_DIR,
        DATASET_DIR,
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "train" / "labels",
        DATASET_DIR / "val" / "images",
        DATASET_DIR / "val" / "labels",
        MODELS_DIR,
        INFERENCE_DIR,
        INFERENCE_DIR / "videos",
        LOGS_DIR,
        ARCHIVE_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
ensure_directories()