import json
import cv2
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from config import (
    CLASS_MAPPING, DATASET_DIR, LABELBOX_PROJECT_ID, 
    ensure_directories, BYTETRACK_CONFIG
)
from bytetrack_integration import ByteTrackProcessor
from utils.logger import setup_logger, log_data_row_action

class DataProcessor:
    """Class to process Labelbox data and convert to YOLO format"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
        self.bytetrack_processor = ByteTrackProcessor()
    
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
        Extract a specific frame from video
        
        Args:
            video_path: Path to video file
            frame_num: Frame number to extract
            output_path: Path to save the extracted frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return False
                
            # Get total frames for validation
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_num >= total_frames:
                self.logger.error(f"Frame {frame_num} is beyond video length ({total_frames} frames)")
                cap.release()
                return False
                
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Create output directory if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), frame)
                cap.release()
                return True
            else:
                self.logger.error(f"Could not read frame {frame_num}")
                cap.release()
                return False
                
        except Exception as e:
            self.logger.error(f"Error extracting frame {frame_num}: {str(e)}")
            if 'cap' in locals():
                cap.release()
            return False
    
    def process_labelbox_data(self, json_path: Path, video_path: Path, dataset_type: str = "train") -> bool:
        """
        Process Labelbox JSON data and convert to YOLO format with enhanced tracking
        
        Args:
            json_path: Path to Labelbox JSON file
            video_path: Path to corresponding video file
            dataset_type: Type of dataset ('train' or 'val')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing Labelbox data: {json_path}")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            data_row_id = data['data_row']['id']
            log_data_row_action(data_row_id, "PROCESSING_STARTED")
            
            # Get project labels
            if 'projects' not in data or LABELBOX_PROJECT_ID not in data['projects']:
                self.logger.error(f"No project data found for project ID: {LABELBOX_PROJECT_ID}")
                log_data_row_action(data_row_id, "PROCESSING_ERROR", "No project data found")
                return False
            
            project_data = data['projects'][LABELBOX_PROJECT_ID]
            if 'labels' not in project_data or not project_data['labels']:
                self.logger.warning(f"No labels found for data row: {data_row_id}")
                log_data_row_action(data_row_id, "PROCESSING_NO_LABELS")
                return True  # Not an error, just no labels
            
            # Get frame annotations
            label = project_data['labels'][0]  # Assuming single label
            if 'annotations' not in label or 'frames' not in label['annotations']:
                self.logger.warning(f"No frame annotations found for data row: {data_row_id}")
                log_data_row_action(data_row_id, "PROCESSING_NO_FRAME_ANNOTATIONS")
                return True
            
            frame_annotations = label['annotations']['frames']
            total_frames = len(frame_annotations)
            
            self.logger.info(f"Processing {total_frames} frames for data row: {data_row_id}")
            
            # Get image dimensions
            img_width = data['media_attributes']['width']
            img_height = data['media_attributes']['height']
            
            # STEP 1 & 2: Apply ByteTrack processing to training data
            self.logger.info("Step 1-2: Processing training data with ByteTrack...")
            
            # Use ByteTrack processor for consistent tracking
            tracked_detections = self.bytetrack_processor.process_training_annotations(data)
            
            # STEP 3: Generate frames and YOLO annotation files based on tracked detections
            self.logger.info("Step 3: Generating frames and YOLO annotations...")
            
            # Set up output directories
            images_dir = DATASET_DIR / dataset_type / 'images'
            labels_dir = DATASET_DIR / dataset_type / 'labels'
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
                    
                    # Use normalized bbox from ByteTrack processor
                    x_center, y_center, width, height = detection['bbox_normalized']
                    
                    # Add annotation (YOLO format: class_id x_center y_center width height)
                    annotation_line = f"{class_idx} {x_center} {y_center} {width} {height}"
                    annotations.append(annotation_line)
                
                # Write YOLO annotation file
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
                
                successful_frames += 1
            
            # STEP 4: Log tracking summary
            num_unique_tracks = len(set(d['track_id'] for d in tracked_detections))
            
            self.logger.info(f"ByteTrack processing summary:")
            self.logger.info(f"  - Tracked detections: {len(tracked_detections)}")
            self.logger.info(f"  - Unique tracks created: {num_unique_tracks}")
            self.logger.info(f"  - Frames with annotations: {successful_frames}")
            
            self.logger.info(f"Completed processing: {successful_frames}/{len(frames_to_process)} frames successful")
            log_data_row_action(
                data_row_id, 
                "PROCESSING_COMPLETE", 
                f"Successful frames: {successful_frames}, Tracks: {num_unique_tracks}"
            )
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            if 'data_row_id' in locals():
                log_data_row_action(data_row_id, "PROCESSING_ERROR", str(e))
            return False
    
    def create_dataset_yaml(self) -> Path:
        """
        Create YOLO dataset YAML configuration file
        
        Returns:
            Path to the created YAML file
        """
        try:
            # Reverse the class mapping for YAML
            class_names = {v: k for k, v in CLASS_MAPPING.items()}
            
            dataset_config = {
                'path': str(DATASET_DIR.absolute()),
                'train': 'train/images',
                'val': 'val/images',
                'names': class_names
            }
            
            yaml_path = DATASET_DIR / 'dataset.yaml'
            
            # Write YAML manually to avoid dependency
            with open(yaml_path, 'w') as f:
                f.write(f"path: {dataset_config['path']}\n")
                f.write(f"train: {dataset_config['train']}\n")
                f.write(f"val: {dataset_config['val']}\n")
                f.write("names:\n")
                for idx, name in class_names.items():
                    f.write(f"  {idx}: {name}\n")
            
            self.logger.info(f"Dataset YAML created: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            self.logger.error(f"Error creating dataset YAML: {str(e)}")
            raise
    
    def split_dataset(self, train_ratio: float = 0.8) -> bool:
        """
        Split dataset into train and validation sets
        
        Args:
            train_ratio: Ratio of data to use for training
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import random
            import shutil
            
            train_images_dir = DATASET_DIR / 'train' / 'images'
            train_labels_dir = DATASET_DIR / 'train' / 'labels'
            val_images_dir = DATASET_DIR / 'val' / 'images'
            val_labels_dir = DATASET_DIR / 'val' / 'labels'
            
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
            
            # Move validation files
            for img_path in val_images:
                # Move image
                val_img_path = val_images_dir / img_path.name
                shutil.move(str(img_path), str(val_img_path))
                
                # Move corresponding label if it exists
                label_name = img_path.stem + '.txt'
                label_path = train_labels_dir / label_name
                if label_path.exists():
                    val_label_path = val_labels_dir / label_name
                    shutil.move(str(label_path), str(val_label_path))
            
            self.logger.info(f"Dataset split: {len(image_files) - len(val_images)} train, {len(val_images)} val")
            return True
            
        except Exception as e:
            self.logger.error(f"Error splitting dataset: {str(e)}")
            return False 