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

# Directory Configuration - NEW ORGANIZED STRUCTURE
BASE_DIR = Path(__file__).parent.parent

# Root data directory
DATA_ROOT = BASE_DIR / "project_data"

# Organized by purpose and workflow
WORKFLOWS_DIR = DATA_ROOT / "workflows"      # Individual workflow runs
MODELS_DIR = DATA_ROOT / "models"            # Trained models (permanent)  
CONFIGS_DIR = DATA_ROOT / "configs"          # Configuration files
TEMP_DIR = DATA_ROOT / "temp"                # Temporary files (safe to delete)
LOGS_DIR = DATA_ROOT / "logs"                # Centralized logging
ARCHIVE_DIR = DATA_ROOT / "archive"          # Completed workflows

# Temporary files structure (organized by type)
TEMP_DOWNLOADS = TEMP_DIR / "downloads"      # Raw downloads
TEMP_PROCESSING = TEMP_DIR / "processing"    # Processing workspace
TEMP_INFERENCE = TEMP_DIR / "inference"      # Inference workspace

# Models directory structure (permanent storage)
MODELS_ACTIVE = MODELS_DIR / "active"        # Currently used models
MODELS_VERSIONS = MODELS_DIR / "versions"    # Model version history
MODELS_PRETRAINED = MODELS_DIR / "pretrained"  # Downloaded pretrained models

# Configs directory structure
TRACKER_CONFIGS_DIR = CONFIGS_DIR / "trackers"   # Tracking configurations
DATASET_CONFIGS_DIR = CONFIGS_DIR / "datasets"   # Dataset configurations
MODEL_CONFIGS_DIR = CONFIGS_DIR / "models"       # Model configurations

# Legacy directory aliases for backward compatibility (DEPRECATED)
WORKFLOW_DATA_DIR = DATA_ROOT  # Will be removed in future version
DOWNLOADS_DIR = TEMP_DOWNLOADS  # Will be removed in future version
DATASET_DIR = None  # Now generated per workflow
INFERENCE_DIR = None  # Now generated per workflow

# Logging Configuration  
LOG_FILE = LOGS_DIR / "main.log"
DATA_ROW_LOG = LOGS_DIR / "data_rows.log"
ERROR_LOG = LOGS_DIR / "errors.log"

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

# Tracking Configuration - Multiple tracking methods supported
TRACKING_CONFIG = {
    # Default tracking method: "sightline", "bytetrack", or "botsort"
    'default_method': 'sightline',
    
    # General tracking parameters
    'conf_threshold': 0.25,     # Confidence threshold for detections
    'iou_threshold': 0.45,      # IoU threshold for NMS
}

# Custom Sightline Tracker Configuration (formerly BYTETRACK_CONFIG)
SIGHTLINE_TRACKER_CONFIG = {
    # Two-stage tracking parameters (inspired by ByteTrack)
    'track_thresh': 0.25,       # High confidence threshold for first association
    'track_low_thresh': 0.1,    # Low confidence threshold for second association
    'new_track_thresh': 0.25,   # Threshold for initializing new tracks
    'track_buffer': 30,         # Number of frames to keep lost tracks
    'match_thresh': 0.8,        # Matching threshold for association
    'distance_threshold': 200,  # Distance threshold for track association (pixels)
    
    # Valorant-specific constraints
    'max_tracks': 10,           # Maximum number of simultaneous tracks (10 players max)
    'max_detections_per_frame': 10,  # Maximum detections per frame to process
    'valorant_optimized': True,  # Flag to enable Valorant-specific optimizations
    
    # Frame rate dependent settings
    'frame_rate': 30,           # Expected frame rate of your videos
}

# Official Tracker Configurations (using Ultralytics built-in trackers)
OFFICIAL_TRACKER_CONFIG = {
    'bytetrack': {
        'tracker_config': 'bytetrack.yaml',
        'valorant_config': 'valorant_bytetrack.yaml',
        'description': 'Official ByteTrack implementation via Ultralytics'
    },
    'botsort': {
        'tracker_config': 'botsort.yaml', 
        'valorant_config': 'valorant_botsort.yaml', 
        'description': 'Official BoT-SORT implementation via Ultralytics'
    }
}

# Valorant-specific tracker configurations
VALORANT_TRACKER_CONFIGS_DIR = TRACKER_CONFIGS_DIR

# Valorant ByteTrack configuration (ultralytics format)
VALORANT_BYTETRACK_CONFIG = {
    'tracker_type': 'bytetrack',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'fuse_score': False,
    # Valorant constraint: max 10 players (handled in post-processing)
}

# Valorant BoT-SORT configuration (ultralytics format)
VALORANT_BOTSORT_CONFIG = {
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,  
    'new_track_thresh': 0.6,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': True,
    'fuse_score': False,
    'gmc_method': 'sparseOptFlow',
    # Valorant constraint: max 10 players (handled in post-processing)
}

# Legacy config name for backward compatibility
BYTETRACK_CONFIG = SIGHTLINE_TRACKER_CONFIG  # Deprecated: use SIGHTLINE_TRACKER_CONFIG

def get_workflow_directories(workflow_id: str) -> dict:
    """
    Get organized directory structure for a specific workflow
    
    Args:
        workflow_id: Unique workflow identifier (e.g., "training_20250122_143022")
        
    Returns:
        Dictionary with all workflow-specific paths
    """
    from datetime import datetime
    
    # Create timestamp-based directory name for organization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    workflow_name = f"{workflow_id}_{timestamp}"
    
    workflow_root = WORKFLOWS_DIR / workflow_name
    
    return {
        # Workflow root
        'root': workflow_root,
        'workflow_id': workflow_id,
        'timestamp': timestamp,
        
        # Input data (raw from Labelbox)
        'inputs': {
            'root': workflow_root / "inputs",
            'videos': workflow_root / "inputs" / "videos",
            'annotations': workflow_root / "inputs" / "annotations"
        },
        
        # Training data (if training workflow)
        'training': {
            'root': workflow_root / "training",
            'dataset': workflow_root / "training" / "dataset",
            'train_images': workflow_root / "training" / "dataset" / "train" / "images",
            'train_labels': workflow_root / "training" / "dataset" / "train" / "labels", 
            'val_images': workflow_root / "training" / "dataset" / "val" / "images",
            'val_labels': workflow_root / "training" / "dataset" / "val" / "labels",
            'configs': workflow_root / "training" / "configs",
            'checkpoints': workflow_root / "training" / "checkpoints",
            'results': workflow_root / "training" / "results"
        },
        
        # Inference data (if inference workflow)
        'inference': {
            'root': workflow_root / "inference",
            'videos': workflow_root / "inference" / "videos",
            'annotations': workflow_root / "inference" / "annotations",
            'results': workflow_root / "inference" / "results"
        },
        
        # Tracking method specific outputs
        'tracking': {
            'sightline': workflow_root / "tracking" / "sightline",
            'bytetrack': workflow_root / "tracking" / "bytetrack", 
            'botsort': workflow_root / "tracking" / "botsort",
            'comparisons': workflow_root / "tracking" / "comparisons"
        },
        
        # Temporary processing files
        'temp': {
            'root': workflow_root / "temp",
            'frames': workflow_root / "temp" / "frames",
            'processing': workflow_root / "temp" / "processing"
        },
        
        # Output files
        'outputs': {
            'root': workflow_root / "outputs",
            'models': workflow_root / "outputs" / "models",
            'videos': workflow_root / "outputs" / "videos", 
            'annotations': workflow_root / "outputs" / "annotations",
            'reports': workflow_root / "outputs" / "reports"
        },
        
        # Logs specific to this workflow
        'logs': {
            'root': workflow_root / "logs",
            'main': workflow_root / "logs" / f"{workflow_name}.log",
            'data_rows': workflow_root / "logs" / "data_rows.log",
            'errors': workflow_root / "logs" / "errors.log"
        }
    }

def get_tracking_file_paths(workflow_dirs: dict, tracking_method: str, video_name: str) -> dict:
    """
    Get standardized file paths for tracking outputs
    
    Args:
        workflow_dirs: Workflow directories from get_workflow_directories()
        tracking_method: Tracking method ("sightline", "bytetrack", "botsort")
        video_name: Base video name (without extension)
        
    Returns:
        Dictionary with standardized tracking file paths
    """
    tracking_dir = workflow_dirs['tracking'][tracking_method.lower()]
    
    return {
        'annotations_json': tracking_dir / f"{video_name}_annotations.json",
        'video_output': tracking_dir / f"{video_name}_tracked.mp4", 
        'metadata': tracking_dir / f"{video_name}_metadata.json",
        'stats': tracking_dir / f"{video_name}_stats.json"
    }

def ensure_directories():
    """Create all necessary base directories"""
    base_directories = [
        # Core directories
        DATA_ROOT,
        WORKFLOWS_DIR,
        MODELS_DIR,
        CONFIGS_DIR,
        TEMP_DIR,
        LOGS_DIR,
        ARCHIVE_DIR,
        
        # Temporary directories
        TEMP_DOWNLOADS,
        TEMP_PROCESSING,
        TEMP_INFERENCE,
        
        # Model directories  
        MODELS_ACTIVE,
        MODELS_VERSIONS,
        MODELS_PRETRAINED,
        
        # Config directories
        TRACKER_CONFIGS_DIR,
        DATASET_CONFIGS_DIR,
        MODEL_CONFIGS_DIR,
    ]
    
    for directory in base_directories:
        directory.mkdir(parents=True, exist_ok=True)

def ensure_workflow_directories(workflow_dirs: dict):
    """Create all directories for a specific workflow"""
    directories_to_create = []
    
    # Flatten the nested directory structure
    def collect_paths(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith('_dirs') or key in ['root']:
                    continue
                collect_paths(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(obj, Path):
            directories_to_create.append(obj)
    
    collect_paths(workflow_dirs)
    
    # Create all directories
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

# Legacy support - will be deprecated
VALORANT_TRACKER_CONFIGS_DIR = TRACKER_CONFIGS_DIR