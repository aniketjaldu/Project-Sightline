import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import cv2

from ultralytics import YOLO
from core.config import (
    CLASS_MAPPING, LABELBOX_PROJECT_ID,
    YOLO_CONFIG, ensure_directories, get_workflow_directories
)
from pipeline.sightline_tracker import SightlineTracker
from pipeline.tracking_manager import TrackingManager
from utils.logger import setup_logger, log_data_row_action, safe_write_file

class VideoCapture:
    """Enhanced video capture context manager for proper resource management"""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.cap = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        return self.cap
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

class DataProcessor:
    """Class to process Labelbox data and convert to YOLO format"""
    
    def __init__(self, tracking_method: str = "botsort"):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
        self.tracking_method = tracking_method.lower()
        
        # Initialize tracking manager with the specified method
        self.tracking_manager = TrackingManager(tracking_method=self.tracking_method)
        
        # Keep Sightline tracker for backward compatibility and extraction methods
        self.sightline_tracker = SightlineTracker()
        
        # Track current workflow directories for consistency
        self.current_workflow_dirs = None
    
# Old custom tracking methods removed - now using ByteTrack integration
    
# Old enhanced tracking method removed - now using ByteTrack
        
    def convert_bbox_to_yolo(self, bbox: Dict[str, float], img_width: int, img_height: int) -> List[float]:
        """
        Convert bounding box to YOLO format
        
        Args:
            bbox: Bounding box with 'left', 'top', 'width', 'height'
            img_width: Image width
            img_height: Image height
            
        Returns:
            YOLO format bbox [x_center, y_center, width, height] normalized
        """
        x = bbox['left']
        y = bbox['top']
        w = bbox['width']
        h = bbox['height']
        
        # Normalize coordinates
        x_center = (x + w/2) / img_width
        y_center = (y + h/2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        return [x_center, y_center, w_norm, h_norm]
    
    def extract_frame(self, video_path: Path, frame_num: int, output_path: Path) -> bool:
        """
        Extract a specific frame from video with enhanced error handling and resource management
        
        Args:
            video_path: Path to video file
            frame_num: Frame number to extract
            output_path: Path to save the extracted frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add detailed logging for debugging
            self.logger.debug(f"Extracting frame {frame_num} from {video_path.name} to {output_path}")
            
            # Use context manager for proper resource cleanup
            with VideoCapture(video_path) as cap:
                # Get total frames for validation
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_num >= total_frames:
                    self.logger.error(f"Frame {frame_num} is beyond video length ({total_frames} frames)")
                    return False
                    
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Create output directory if it doesn't exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Check if frame extraction was successful before writing
                    if cv2.imwrite(str(output_path), frame):
                        self.logger.debug(f"Successfully extracted frame {frame_num} to {output_path}")
                        return True
                    else:
                        self.logger.error(f"Failed to write frame {frame_num} to {output_path}")
                        return False
                else:
                    self.logger.error(f"Could not read frame {frame_num} from video")
                    return False
                    
        except ValueError as e:
            self.logger.error(f"ValueError extracting frame {frame_num}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error extracting frame {frame_num}: {str(e)}")
            return False
            
    def process_labelbox_data(self, json_path: Path, video_path: Path, dataset_type: str = "train", workflow_dirs: dict = None) -> bool:
        """
        Process Labelbox annotations and convert to YOLO format with enhanced error handling
        
        Args:
            json_path: Path to Labelbox JSON file
            video_path: Path to video file
            dataset_type: Dataset type ("train" or "val")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not json_path.exists():
                self.logger.error(f"JSON file not found: {json_path}")
                return False
                
            if not video_path.exists():
                self.logger.error(f"Video file not found: {video_path}")
                return False
            
            # Load JSON data with proper error handling
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self.logger.error(f"Error reading JSON file {json_path}: {str(e)}")
                return False
            
            # Validate required data structure
            required_keys = ['data_row', 'media_attributes', 'projects']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Missing required key '{key}' in JSON data")
                    return False
                    
            data_row_id = data['data_row']['id']
            self.logger.info(f"Processing data row: {data_row_id}")
            
            # Set workflow directories for consistent file paths
            if workflow_dirs:
                self.current_workflow_dirs = workflow_dirs
                self.logger.info(f"Using provided workflow directories (workflow ID: {workflow_dirs.get('workflow_id', 'unknown')})")
            else:
                # Fallback - create new workflow (should not happen in normal flow)
                from core.config import get_workflow_directories
                self.current_workflow_dirs = get_workflow_directories(f"training_{data_row_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                self.logger.warning("No workflow directories provided, created new ones")
            
            log_data_row_action(data_row_id, "PROCESSING_STARTED")
            
            # Extract frame annotations
            frame_annotations = {}
            project_data = data['projects'].get(LABELBOX_PROJECT_ID, {})
            labels = project_data.get('labels', [])
            
            for label in labels:
                annotations = label.get('annotations', {})
                frames = annotations.get('frames', {})
                
                # Handle frames as dictionary (key = frame number as string, value = frame data)
                for frame_num_str, frame_data in frames.items():
                    frame_num = int(frame_num_str)  # Convert string key to integer
                    frame_annotations[frame_num] = frame_data.get('objects', [])

            total_frames = len(frame_annotations)
            
            self.logger.info(f"Processing {total_frames} frames for data row: {data_row_id}")
            
            # Get image dimensions
            img_width = data['media_attributes']['width']
            img_height = data['media_attributes']['height']
            
            # STEP 1 & 2: Apply specified tracking method to training data
            self.logger.info(f"Step 1-2: Processing training data with {self.tracking_method.upper()} tracker...")
            
            # Use specified tracking method for training consistency
            tracked_detections = self.tracking_manager.process_training_annotations(data)
            
            # STEP 3: Generate frames and YOLO annotation files based on tracked detections
            self.logger.info("Step 3: Generating frames and YOLO annotations...")
            
            # Use the current workflow directories passed from main workflow
            # Don't create a new workflow - use the existing one to maintain consistency
            workflow_dirs = self.current_workflow_dirs
            
            # Use workflow training directories
            if dataset_type == 'train':
                images_dir = workflow_dirs['training']['train_images']
                labels_dir = workflow_dirs['training']['train_labels']
            else:  # val
                images_dir = workflow_dirs['training']['val_images']
                labels_dir = workflow_dirs['training']['val_labels']
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Group tracked detections by frame for efficient processing
            detections_by_frame = {}
            for detection in tracked_detections:
                frame_num = detection['frame_number']
                if frame_num not in detections_by_frame:
                    detections_by_frame[frame_num] = []
                detections_by_frame[frame_num].append(detection)
            
            processed_frames = 0
            successful_frames = 0
            frames_to_process = sorted(detections_by_frame.keys())
            
            
            for frame_num in frames_to_process:
                processed_frames += 1
                
                # Show progress
                if processed_frames % 10 == 0:
                    progress = (processed_frames / len(frames_to_process)) * 100
                    self.logger.info(f"Progress: {processed_frames}/{len(frames_to_process)} frames ({progress:.1f}%)")
                
                # Extract frame from video
                frame_filename = f"{data_row_id}_frame_{frame_num}.jpg"
                frame_path = images_dir / frame_filename
                    
                
                if not self.extract_frame(video_path, frame_num, frame_path):
                    self.logger.warning(f"Failed to extract frame {frame_num}, skipping")
                    continue
                
                
                # Create YOLO annotation file
                label_filename = f"{data_row_id}_frame_{frame_num}.txt"
                label_path = labels_dir / label_filename
                
                annotations = []
                frame_detections = detections_by_frame[frame_num]
                
                # Convert each detection to YOLO format
                for detection in frame_detections:
                    class_name = detection['class_name']
                    
                    # Skip if class not in mapping
                    if class_name not in CLASS_MAPPING:
                        continue
                        
                    class_idx = CLASS_MAPPING[class_name]
                    
                    # Use normalized bbox from Sightline tracker
                    x_center, y_center, width, height = detection['bbox_normalized']
                    
                    # Add annotation (YOLO format: class_id x_center y_center width height)
                    annotation_line = f"{class_idx} {x_center} {y_center} {width} {height}"
                    annotations.append(annotation_line)
                
                # Write YOLO annotation file using safe write
                annotation_content = '\n'.join(annotations)
                if safe_write_file(label_path, annotation_content):
                    successful_frames += 1
                else:
                    self.logger.warning(f"Failed to write annotation file: {label_path}")
                    # Remove the corresponding image file if annotation failed
                    if frame_path.exists():
                        frame_path.unlink()
            
            # STEP 4: Log tracking summary
            num_unique_tracks = len(set(d['track_id'] for d in tracked_detections))
            
            self.logger.info(f"Sightline tracking summary:")
            self.logger.info(f"  - Tracked detections: {len(tracked_detections)}")
            self.logger.info(f"  - Unique tracks created: {num_unique_tracks}")
            self.logger.info(f"  - Frames with annotations: {successful_frames}")
            
            self.logger.info(f"Completed processing: {successful_frames}/{len(frames_to_process)} frames successful")
            log_data_row_action(
                data_row_id, 
                "PROCESSING_COMPLETE", 
                f"Successful frames: {successful_frames}, Tracks: {num_unique_tracks}"
            )
            
            # Store workflow_dirs for other methods to use
            self.current_workflow_dirs = workflow_dirs
            
            return successful_frames > 0  # Return True only if at least some frames were processed successfully
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            if 'data_row_id' in locals():
                log_data_row_action(data_row_id, "PROCESSING_ERROR", str(e))
            return False
    
    def create_dataset_yaml(self, workflow_dirs: dict) -> Optional[Path]:
        """
        Create YOLO dataset YAML configuration file with enhanced error handling
        
        Args:
            workflow_dirs: Workflow directories structure
        
        Returns:
            Path to created YAML file, None if failed
        """
        
        try:
            # Reverse the class mapping for YAML
            class_names = {v: k for k, v in CLASS_MAPPING.items()}
            
            dataset_config = {
                'path': str(workflow_dirs['training']['dataset'].absolute()),
                'train': 'train/images',
                'val': 'val/images',
                'names': class_names
            }
            
            yaml_path = workflow_dirs['training']['dataset'] / 'dataset.yaml'
            
            # Write YAML manually to avoid dependency - build content first
            yaml_content = f"path: {dataset_config['path']}\n"
            yaml_content += f"train: {dataset_config['train']}\n"
            yaml_content += f"val: {dataset_config['val']}\n"
            yaml_content += "names:\n"
            for idx, name in class_names.items():
                yaml_content += f"  {idx}: {name}\n"
            
            # Use safe write operation
            if safe_write_file(yaml_path, yaml_content):
                self.logger.info(f"Dataset YAML created: {yaml_path}")
                return yaml_path
            else:
                raise IOError(f"Failed to write YAML file: {yaml_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating dataset YAML: {str(e)}")
            raise

    def split_train_validation(self, workflow_dirs: dict, train_ratio: float = 0.8) -> bool:
        """
        Split training data into train/validation sets with enhanced error handling
        
        Args:
            workflow_dirs: Workflow directories structure
            train_ratio: Ratio of data to use for training
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import random
            import shutil
            
            train_images_dir = workflow_dirs['training']['train_images']
            train_labels_dir = workflow_dirs['training']['train_labels']
            val_images_dir = workflow_dirs['training']['val_images']
            val_labels_dir = workflow_dirs['training']['val_labels']
            
            # Create validation directories
            val_images_dir.mkdir(parents=True, exist_ok=True)
            val_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all image files
            image_files = list(train_images_dir.glob('*.jpg'))
            
            if not image_files:
                self.logger.warning("No image files found to split")
                return True
            
            # Shuffle and split
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_ratio)
            
            val_images = image_files[split_idx:]
            
            # Move validation files with better error handling
            moved_files = 0
            for img_path in val_images:
                try:
                    # Move image
                    val_img_path = val_images_dir / img_path.name
                    shutil.move(str(img_path), str(val_img_path))
                    
                    # Move corresponding label file
                    label_name = img_path.stem + '.txt'
                    label_path = train_labels_dir / label_name
                    if label_path.exists():
                        val_label_path = val_labels_dir / label_name
                        shutil.move(str(label_path), str(val_label_path))
                    
                    moved_files += 1
                    
                except Exception as move_error:
                    self.logger.warning(f"Failed to move {img_path.name}: {str(move_error)}")
            
            total_images = len(image_files)
            train_count = total_images - moved_files
            val_count = moved_files
            
            self.logger.info(f"Train/validation split completed:")
            self.logger.info(f"  Training images: {train_count}")
            self.logger.info(f"  Validation images: {val_count}")
            self.logger.info(f"  Split ratio: {train_count/total_images:.2f}/{val_count/total_images:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in train/validation split: {str(e)}")
            return False 