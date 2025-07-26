"""
Comprehensive Tracking Manager

This module provides a unified interface for different tracking methods:
- Official ByteTrack (via Ultralytics)
- Official BoT-SORT (via Ultralytics) 
- Custom Sightline Tracker
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from ultralytics import YOLO

from core.config import TRACKING_CONFIG
from pipeline.sightline_tracker import SightlineTracker
from utils.logger import setup_logger, safe_write_json

class TrackingManager:
    """
    Unified tracking manager that handles different tracking methods
    """
    
    def __init__(self, tracking_method: str = "sightline"):
        """
        Initialize tracking manager
        
        Args:
            tracking_method: One of "sightline", "bytetrack", "botsort"
        """
        self.logger = setup_logger(self.__class__.__name__)
        self.tracking_method = tracking_method.lower()
        self.config = TRACKING_CONFIG
        
        # Validate tracking method
        valid_methods = ["sightline", "bytetrack", "botsort"]
        if self.tracking_method not in valid_methods:
            raise ValueError(f"Invalid tracking method: {self.tracking_method}. Must be one of {valid_methods}")
        
        # Initialize tracker based on method
        if self.tracking_method == "sightline":
            self.tracker = SightlineTracker()
            self.logger.info("Initialized Sightline custom tracker")
        else:
            # For official trackers, we'll initialize them per inference run
            self.tracker = None
            self.logger.info(f"Configured for official {self.tracking_method.upper()} tracker")
            
        # Ensure custom tracker configs exist
        self._ensure_valorant_tracker_configs()
    
    def _get_standardized_paths(self, output_video_path: Optional[Path], video_name: str) -> Dict[str, Optional[Path]]:
        """
        Get standardized file paths for consistent naming across all tracking methods
        
        Args:
            output_video_path: Path to output video (optional)
            video_name: Base name for files
            
        Returns:
            Dictionary with standardized paths
        """
        paths = {
            'annotations': None,
            'video': output_video_path
        }
        
        if output_video_path:
            # Standardized naming: {video_name}_tracked_{method}_annotations.json
            paths['annotations'] = output_video_path.parent / f"{video_name}_tracked_{self.tracking_method}_annotations.json"
        
        return paths
    
    def _ensure_output_directories(self, output_video_path: Optional[Path]) -> bool:
        """
        Ensure all necessary output directories exist
        
        Args:
            output_video_path: Path to output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_video_path:
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error creating output directories: {str(e)}")
            return False
    
    def _save_annotations_standardized(self, annotations: Dict[str, Any], annotations_path: Path) -> bool:
        """
        Save annotations using standardized format and error handling
        
        Args:
            annotations: Annotations dictionary
            annotations_path: Path to save annotations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add tracking method metadata to annotations
            annotations['metadata'] = {
                'tracking_method': self.tracking_method,
                'generated_by': f'TrackingManager-{self.tracking_method}',
                'format_version': '1.0'
            }
            
            if safe_write_json(annotations_path, annotations):
                self.logger.info(f"Annotations saved to: {annotations_path}")
                return True
            else:
                self.logger.error(f"Failed to save annotations to: {annotations_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            return False
    
    def _log_tracking_completion(self, data_row_id: Optional[str], frame_count: int, success: bool, error: str = None):
        """
        Standardized logging for tracking completion across all methods
        
        Args:
            data_row_id: Data row ID for logging
            frame_count: Number of frames processed
            success: Whether tracking was successful
            error: Error message if unsuccessful
        """
        if not data_row_id:
            return
            
        try:
            from utils.logger import log_data_row_action
            
            if success:
                log_data_row_action(
                    data_row_id, 
                    f"TRACKING_COMPLETE_{self.tracking_method.upper()}",
                    f"Method: {self.tracking_method}, Frames: {frame_count}"
                )
            else:
                log_data_row_action(
                    data_row_id, 
                    f"TRACKING_ERROR_{self.tracking_method.upper()}",
                    f"Method: {self.tracking_method}, Error: {error or 'Unknown'}"
                )
        except Exception as e:
            self.logger.warning(f"Error logging tracking completion: {str(e)}")
    
    def track_video_with_yolo(
        self, 
        model: YOLO, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run tracking on video using official YOLO trackers (ByteTrack, BoT-SORT)
        
        Args:
            model: YOLO model instance
            video_path: Path to input video
            output_video_path: Path to save annotated video (optional)
            data_row_id: Data row ID for logging
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with tracking results and annotations
        """
        try:
            self.logger.info(f"Starting {self.tracking_method.upper()} inference on video: {video_path}")
            
            # Set confidence threshold
            if conf_threshold is None:
                conf_threshold = self.config.get('conf_threshold', 0.25)
            
            # Ensure output directories exist
            if not self._ensure_output_directories(output_video_path):
                raise RuntimeError("Failed to create output directories")
            
            # Get standardized paths
            paths = self._get_standardized_paths(output_video_path, video_path.stem)
            
            # Determine tracker config file
            if self.tracking_method == "bytetrack":
                from core.config import TRACKER_CONFIGS_DIR
                custom_config_path = TRACKER_CONFIGS_DIR / "valorant_bytetrack.yaml"
                if custom_config_path.exists():
                    tracker_config = str(custom_config_path)
                    self.logger.info(f"Using Valorant-optimized ByteTrack config: {tracker_config}")
                else:
                    tracker_config = "bytetrack.yaml"
                    self.logger.warning(f"Valorant ByteTrack config not found, falling back to default")
            elif self.tracking_method == "botsort":
                from core.config import TRACKER_CONFIGS_DIR
                custom_config_path = TRACKER_CONFIGS_DIR / "valorant_botsort.yaml"
                if custom_config_path.exists():
                    tracker_config = str(custom_config_path)
                    self.logger.info(f"Using Valorant-optimized BoT-SORT config: {tracker_config}")
                else:
                    tracker_config = "botsort.yaml"
                    self.logger.warning(f"Valorant BoT-SORT config not found, falling back to default")
            else:
                raise ValueError(f"Official tracking not supported for method: {self.tracking_method}")
            
            # Open video to get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            self.logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Run YOLO tracking
            results = model.track(
                source=str(video_path),
                tracker=tracker_config,
                conf=conf_threshold,
                save=bool(output_video_path),
                project=str(output_video_path.parent) if output_video_path else None,
                name=output_video_path.stem if output_video_path else None,
                stream=True,
                verbose=False
            )
            
            # Process results with standardized format
            all_annotations = {
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames,
                    'tracking_method': self.tracking_method
                },
                'frames': {}
            }
            
            frame_count = 0
            for result in results:
                if result.boxes is not None:
                    # Check if we have any valid detections with track IDs
                    if result.boxes.id is not None and len(result.boxes.id) > 0:
                        frame_annotations = self._extract_yolo_tracking_annotations(result, frame_count)
                        if frame_annotations:
                            all_annotations['frames'][frame_count] = frame_annotations
                    else:
                        # If no track IDs, try to extract detections without tracking
                        frame_annotations = self._extract_frame_detections_with_tracking(result, frame_count)
                        if frame_annotations:
                            all_annotations['frames'][frame_count] = frame_annotations
                frame_count += 1
            
            # Apply Valorant constraint: limit to top 10 unique tracks globally
            all_annotations = self._limit_unique_tracks_for_valorant(all_annotations, max_tracks=10)
            
            self.logger.info(f"{self.tracking_method.upper()} inference completed: {frame_count} frames processed")
            
            # Save annotations using standardized method
            annotations_saved = False
            if paths['annotations']:
                annotations_saved = self._save_annotations_standardized(all_annotations, paths['annotations'])
            
            result_dict = {
                'success': True,
                'tracking_method': self.tracking_method,
                'total_frames': frame_count,
                'annotations': all_annotations,
                'output_video': output_video_path,
                'annotations_file': paths['annotations'] if annotations_saved else None
            }
            
            # Standardized logging
            self._log_tracking_completion(data_row_id, frame_count, True)
            
            return result_dict
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error during {self.tracking_method.upper()} inference: {error_msg}")
            self._log_tracking_completion(data_row_id, 0, False, error_msg)
            return {'success': False, 'tracking_method': self.tracking_method, 'error': error_msg}
    
    def track_video_with_custom_method(
        self, 
        model: YOLO, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run tracking on video using custom Sightline tracker
        
        Args:
            model: YOLO model instance
            video_path: Path to input video
            output_video_path: Path to save annotated video (optional)
            data_row_id: Data row ID for logging
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with tracking results and annotations
        """
        try:
            self.logger.info(f"Starting {self.tracking_method.upper()} inference on video: {video_path}")
            
            # Set confidence threshold
            if conf_threshold is None:
                conf_threshold = self.config.get('conf_threshold', 0.25)
            
            # Ensure output directories exist
            if not self._ensure_output_directories(output_video_path):
                raise RuntimeError("Failed to create output directories")
            
            # Get standardized paths
            paths = self._get_standardized_paths(output_video_path, video_path.stem)
            
            # Open video file to get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Initialize video writer if output path is specified
            writer = None
            if output_video_path:
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
            
            # Storage for all detections (to be processed by custom tracker)
            all_detections = []
            
            # Process video frame by frame
            frame_count = 0
            
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        progress = (frame_count / total_frames) * 100
                        self.logger.info(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
                    
                    # Run YOLO detection only (no tracking)
                    results = model.predict(
                        frame,
                        conf=conf_threshold,
                        verbose=False
                    )
                    
                    # Extract detections from this frame
                    frame_detections = self._extract_frame_detections(results[0], frame_count)
                    all_detections.extend(frame_detections)
                    
                    frame_count += 1
                        
            except KeyboardInterrupt:
                self.logger.info("Inference stopped by user")
            finally:
                cap.release()
                
            # Now apply custom Sightline tracker to all detections
            self.logger.info("Applying Sightline custom tracking to detections...")
            tracked_detections = self.tracker.process_inference_detections(all_detections)
            
            # Convert to standardized annotation format
            all_annotations = {
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': frame_count,
                    'tracking_method': self.tracking_method
                },
                'frames': {}
            }
            
            # Group tracked detections by frame
            for detection in tracked_detections:
                frame_num = detection['frame_number']
                if frame_num not in all_annotations['frames']:
                    all_annotations['frames'][frame_num] = []
                
                # Convert to standardized annotation format with track_id
                from core.config import CLASS_MAPPING
                annotation = {
                    'frame': frame_num,
                    'class_id': CLASS_MAPPING.get(detection['class_name'], 0),
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'track_id': detection['track_id'],
                    'bounding_box': {
                        'left': detection['bbox_pixel'][0],
                        'top': detection['bbox_pixel'][1],
                        'width': detection['bbox_pixel'][2],
                        'height': detection['bbox_pixel'][3]
                    },
                    'center_x': detection['center_x'],
                    'center_y': detection['center_y']
                }
                
                all_annotations['frames'][frame_num].append(annotation)
            
            # Apply Valorant constraint: limit to top 10 unique tracks globally
            all_annotations = self._limit_unique_tracks_for_valorant(all_annotations, max_tracks=10)
            
            # Generate annotated video if writer was initialized
            if writer:
                self.logger.info("Generating annotated video...")
                cap = cv2.VideoCapture(str(video_path))
                frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Draw annotations for this frame
                    if frame_idx in all_annotations['frames']:
                        for annotation in all_annotations['frames'][frame_idx]:
                            self._draw_annotation(frame, annotation)
                    
                    writer.write(frame)
                    frame_idx += 1
                
                cap.release()
                writer.release()
            
            self.logger.info(f"{self.tracking_method.upper()} inference completed: {frame_count} frames processed")
            
            # Save annotations using standardized method
            annotations_saved = False
            if paths['annotations']:
                annotations_saved = self._save_annotations_standardized(all_annotations, paths['annotations'])
            
            result_dict = {
                'success': True,
                'tracking_method': self.tracking_method,
                'total_frames': frame_count,
                'annotations': all_annotations,
                'output_video': output_video_path,
                'annotations_file': paths['annotations'] if annotations_saved else None
            }
            
            # Standardized logging
            self._log_tracking_completion(data_row_id, frame_count, True)
            
            return result_dict
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error during {self.tracking_method.upper()} inference: {error_msg}")
            self._log_tracking_completion(data_row_id, 0, False, error_msg)
            return {'success': False, 'tracking_method': self.tracking_method, 'error': error_msg}
    
    def run_inference(
        self, 
        model: YOLO, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference with the configured tracking method using standardized interface
        
        Args:
            model: YOLO model instance
            video_path: Path to input video
            output_video_path: Path to save annotated video (optional)
            data_row_id: Data row ID for logging
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary with tracking results and annotations
        """
        if self.tracking_method == "sightline":
            return self.track_video_with_custom_method(
                model, video_path, output_video_path, data_row_id, conf_threshold
            )
        else:
            return self.track_video_with_yolo(
                model, video_path, output_video_path, data_row_id, conf_threshold
            )
    
    def process_training_annotations(self, labelbox_data: Dict) -> List[Dict]:
        """
        Process training annotations with tracking
        
        Uses the actual configured tracking method for training consistency.
        For official trackers, simulates the training process using inference logic.
        
        Args:
            labelbox_data: Raw Labelbox annotation data
            
        Returns:
            List of detections with track IDs assigned
        """
        if self.tracking_method == "sightline":
            return self.tracker.process_training_annotations(labelbox_data)
        else:
            # For official trackers (ByteTrack/BoT-SORT), simulate training processing
            # by extracting detections and running them through tracking logic
            self.logger.info(f"Processing training data with simulated {self.tracking_method.upper()} tracking")
            
            # Extract detections from Labelbox format (same logic as Sightline)
            temp_sightline = SightlineTracker()
            detections = temp_sightline._extract_labelbox_detections(labelbox_data)
            
            if not detections:
                return []
            
            # Get image dimensions for normalization
            img_width = labelbox_data.get('media_attributes', {}).get('width', 1920)
            img_height = labelbox_data.get('media_attributes', {}).get('height', 1080)
            
            # Process detections using simulated tracking (same as inference)
            tracked_detections = self._simulate_official_tracker_training(detections)
            
            # Add normalized bounding boxes
            for detection in tracked_detections:
                bbox_pixel = detection['bbox_pixel']
                x, y, w, h = bbox_pixel
                
                # Convert to normalized YOLO format (x_center, y_center, width, height)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                detection['bbox_normalized'] = [x_center, y_center, width, height]
            
            self.logger.info(f"Training {self.tracking_method.upper()}: {len(detections)} detections -> {len(tracked_detections)} tracked")
            
            return tracked_detections
    
    def _extract_yolo_tracking_annotations(self, result, frame_num: int) -> List[Dict[str, Any]]:
        """Extract tracking annotations from YOLO tracking results for a single frame"""
        annotations = []
        
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence, class, and track ID
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                track_id = int(boxes.id[i].cpu().numpy())
                
                # Get class name
                from core.config import REVERSE_CLASS_MAPPING
                class_name = REVERSE_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Calculate bounding box dimensions
                left = float(x1)
                top = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                # Convert to annotation format
                annotation = {
                    'frame': frame_num,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'track_id': track_id,
                    'bounding_box': {
                        'left': left,
                        'top': top,
                        'width': width,
                        'height': height
                    },
                    'center_x': left + width / 2,
                    'center_y': top + height / 2
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def _extract_frame_detections(self, result, frame_num: int) -> List[Dict]:
        """Extract detections from YOLO result for custom tracking"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence and class
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                from core.config import REVERSE_CLASS_MAPPING
                class_name = REVERSE_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Calculate bounding box dimensions
                left = float(x1)
                top = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                detection = {
                    'frame_number': frame_num,
                    'class_name': class_name.lower(),
                    'confidence': confidence,
                    'bbox_pixel': [left, top, width, height],
                    'center_x': left + width / 2,
                    'center_y': top + height / 2
                }
                
                detections.append(detection)
        
        return detections

    def _extract_frame_detections_with_tracking(self, result, frame_num: int) -> List[Dict[str, Any]]:
        """Extract detections from YOLO result and assign temporary track IDs for tracking methods"""
        annotations = []
        
        if result.boxes is not None:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence and class
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                from core.config import REVERSE_CLASS_MAPPING
                class_name = REVERSE_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Calculate bounding box dimensions
                left = float(x1)
                top = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                # Assign a temporary track ID (frame-based for uniqueness)
                temp_track_id = frame_num * 1000 + i
                
                # Convert to annotation format
                annotation = {
                    'frame': frame_num,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'track_id': temp_track_id,
                    'bounding_box': {
                        'left': left,
                        'top': top,
                        'width': width,
                        'height': height
                    },
                    'center_x': left + width / 2,
                    'center_y': top + height / 2
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def _draw_annotation(self, frame: np.ndarray, annotation: Dict) -> None:
        """Draw annotation on frame"""
        bbox = annotation['bounding_box']
        x1 = int(bbox['left'])
        y1 = int(bbox['top'])
        x2 = int(bbox['left'] + bbox['width'])
        y2 = int(bbox['top'] + bbox['height'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with track ID
        label = f"{annotation['class_name']} ({annotation['track_id']}) {annotation['confidence']:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    def _save_annotations(self, annotations: Dict, output_path: Path) -> None:
        """
        Legacy save method - redirects to standardized method for consistency
        """
        self._save_annotations_standardized(annotations, output_path)

    def _simulate_official_tracker_training(self, detections: List[Dict]) -> List[Dict]:
        """
        Simulate official tracker behavior for training data processing
        
        This applies similar logic to what ByteTrack/BoT-SORT would use during inference,
        but adapted for processing training annotations.
        
        Args:
            detections: List of detection dictionaries from Labelbox
            
        Returns:
            List of detections with track IDs assigned
        """
        # Sort detections by frame number
        detections.sort(key=lambda x: x['frame_number'])
        
        # Group detections by frame
        frame_detections = {}
        for det in detections:
            frame = det['frame_number']
            if frame not in frame_detections:
                frame_detections[frame] = []
            frame_detections[frame].append(det)
        
        # Initialize tracking variables
        active_tracks = {}
        next_track_id = 0
        tracked_detections = []
        
        # Valorant constraint: Maximum 10 tracks
        MAX_TRACKS = 10
        
        # Process each frame
        for frame_num in sorted(frame_detections.keys()):
            frame_dets = frame_detections[frame_num]
            
            # Valorant optimization: Limit detections per frame to top 10 by confidence
            frame_dets = sorted(frame_dets, key=lambda x: x['confidence'], reverse=True)[:10]
            
            # Age existing tracks
            tracks_to_remove = []
            for track_id in active_tracks:
                active_tracks[track_id]['age'] += 1
                # Remove tracks that haven't been updated for too long (30 frames similar to Sightline)
                if active_tracks[track_id]['age'] > 30:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del active_tracks[track_id]
            
            # Associate detections to existing tracks using simple distance-based matching
            unmatched_detections = []
            for detection in frame_dets:
                best_match = None
                best_distance = float('inf')
                
                # Find closest existing track
                for track_id, track_info in active_tracks.items():
                    distance = self._calculate_distance(
                        (detection['center_x'], detection['center_y']),
                        track_info['last_center']
                    )
                    
                    # Use distance threshold similar to Sightline
                    if distance < 200 and distance < best_distance:
                        best_distance = distance
                        best_match = track_id
                
                if best_match is not None:
                    # Update existing track
                    active_tracks[best_match]['last_center'] = (detection['center_x'], detection['center_y'])
                    active_tracks[best_match]['age'] = 0
                    tracked_detection = {**detection, 'track_id': best_match}
                    tracked_detections.append(tracked_detection)
                else:
                    unmatched_detections.append(detection)
            
            # Create new tracks for unmatched detections (up to limit)
            for detection in unmatched_detections:
                if len(active_tracks) < MAX_TRACKS:
                    # Create new track
                    track_id = next_track_id
                    next_track_id += 1
                    
                    active_tracks[track_id] = {
                        'last_center': (detection['center_x'], detection['center_y']),
                        'age': 0
                    }
                    
                    tracked_detection = {**detection, 'track_id': track_id}
                    tracked_detections.append(tracked_detection)
                else:
                    # Remove oldest track to make room for new one
                    oldest_track = min(active_tracks.keys(), 
                                     key=lambda tid: active_tracks[tid]['age'])
                    del active_tracks[oldest_track]
                    
                    # Create new track
                    track_id = next_track_id
                    next_track_id += 1
                    
                    active_tracks[track_id] = {
                        'last_center': (detection['center_x'], detection['center_y']),
                        'age': 0
                    }
                    
                    tracked_detection = {**detection, 'track_id': track_id}
                    tracked_detections.append(tracked_detection)
                    
                    self.logger.debug(f"Replaced oldest track {oldest_track} with new track {track_id}")
        
        return tracked_detections
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        import math
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) 

    def _limit_unique_tracks_for_valorant(self, annotations: Dict, max_tracks: int) -> Dict:
        """
        Post-processes YOLO tracking results to limit the number of unique tracks
        to a maximum of max_tracks, prioritizing higher confidence tracks.
        """
        # Collect all unique track_ids from all frames
        all_track_ids = set()
        for frame_num in annotations['frames']:
            for annotation in annotations['frames'][frame_num]:
                all_track_ids.add(annotation['track_id'])
        
        if len(all_track_ids) <= max_tracks:
            return annotations
        
        # Sort track_ids by average confidence (descending)
        track_confidences = []
        for track_id in all_track_ids:
            avg_confidence = self._get_track_average_confidence(annotations, track_id)
            track_confidences.append((track_id, avg_confidence))
        
        # Sort by confidence descending
        track_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # Select the top 'max_tracks' track_ids
        limited_track_ids = set([track_id for track_id, _ in track_confidences[:max_tracks]])
        
        # Filter annotations to only include limited track_ids
        filtered_annotations = {
            'video_info': annotations['video_info'],
            'frames': {}
        }
        
        for frame_num in annotations['frames']:
            filtered_frame_annotations = []
            for annotation in annotations['frames'][frame_num]:
                if annotation['track_id'] in limited_track_ids:
                    filtered_frame_annotations.append(annotation)
            
            if filtered_frame_annotations:
                filtered_annotations['frames'][frame_num] = filtered_frame_annotations
        
        self.logger.info(f"Limited tracks from {len(all_track_ids)} to {len(limited_track_ids)} for Valorant constraints")
        
        return filtered_annotations
    
    def _get_track_average_confidence(self, annotations: Dict, track_id: int) -> float:
        """Helper to get the average confidence of a specific track across all frames."""
        confidences = []
        for frame_num in annotations['frames']:
            for annotation in annotations['frames'][frame_num]:
                if annotation['track_id'] == track_id:
                    confidences.append(annotation['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0 

    def _ensure_valorant_tracker_configs(self) -> None:
        """
        Ensures that Valorant-specific tracker config files exist using standardized file operations
        """
        from core.config import TRACKER_CONFIGS_DIR, VALORANT_BYTETRACK_CONFIG, VALORANT_BOTSORT_CONFIG
        from utils.logger import safe_write_file
        
        # Create configs directory if it doesn't exist
        try:
            TRACKER_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating tracker configs directory: {str(e)}")
            return
        
        # ByteTrack config - use config values in official ultralytics format
        bytetrack_config_path = TRACKER_CONFIGS_DIR / "valorant_bytetrack.yaml"
        if not bytetrack_config_path.exists():
            bytetrack_yaml = f"""# Valorant-optimized ByteTrack configuration
                            # For documentation and examples see https://docs.ultralytics.com/modes/track/

                            tracker_type: {VALORANT_BYTETRACK_CONFIG['tracker_type']} # tracker type, ['botsort', 'bytetrack']
                            track_high_thresh: {VALORANT_BYTETRACK_CONFIG['track_high_thresh']} # threshold for the first association
                            track_low_thresh: {VALORANT_BYTETRACK_CONFIG['track_low_thresh']} # threshold for the second association
                            new_track_thresh: {VALORANT_BYTETRACK_CONFIG['new_track_thresh']} # threshold for init new track if the detection does not match any tracks
                            track_buffer: {VALORANT_BYTETRACK_CONFIG['track_buffer']} # buffer to calculate the time when to remove tracks
                            match_thresh: {VALORANT_BYTETRACK_CONFIG['match_thresh']} # threshold for matching tracks
                            fuse_score: {str(VALORANT_BYTETRACK_CONFIG['fuse_score']).title()} # Whether to fuse confidence scores with the iou distances before matching
                            """
            if safe_write_file(bytetrack_config_path, bytetrack_yaml):
                self.logger.info(f"Created Valorant ByteTrack config: {bytetrack_config_path}")
            else:
                self.logger.error(f"Failed to create ByteTrack config: {bytetrack_config_path}")
        
        # BoT-SORT config - use config values in official ultralytics format
        botsort_config_path = TRACKER_CONFIGS_DIR / "valorant_botsort.yaml"
        if not botsort_config_path.exists():
            botsort_yaml = f"""# Valorant-optimized BoT-SORT configuration
                            # For documentation and examples see https://docs.ultralytics.com/modes/track/
                            # For BoT-SORT source code see https://github.com/NirAharon/BoT-SORT

                            tracker_type: {VALORANT_BOTSORT_CONFIG['tracker_type']} # tracker type, ['botsort', 'bytetrack']
                            track_high_thresh: {VALORANT_BOTSORT_CONFIG['track_high_thresh']} # threshold for the first association
                            track_low_thresh: {VALORANT_BOTSORT_CONFIG['track_low_thresh']} # threshold for the second association
                            new_track_thresh: {VALORANT_BOTSORT_CONFIG['new_track_thresh']} # threshold for init new track if the detection does not match any tracks
                            track_buffer: {VALORANT_BOTSORT_CONFIG['track_buffer']} # buffer to calculate the time when to remove tracks
                            match_thresh: {VALORANT_BOTSORT_CONFIG['match_thresh']} # threshold for matching tracks
                            fuse_score: {str(VALORANT_BOTSORT_CONFIG['fuse_score']).title()} # Whether to fuse confidence scores with the iou distances before matching

                            # BoT-SORT settings
                            gmc_method: {VALORANT_BOTSORT_CONFIG['gmc_method']} # method of global motion compensation
                            # ReID model related thresh
                            proximity_thresh: {VALORANT_BOTSORT_CONFIG['proximity_thresh']} # minimum IoU for valid match with ReID
                            appearance_thresh: {VALORANT_BOTSORT_CONFIG['appearance_thresh']} # minimum appearance similarity for ReID
                            with_reid: {str(VALORANT_BOTSORT_CONFIG['with_reid']).title()} # Whether to use ReID
                            model: auto # uses native features if detector is YOLO else yolo11n-cls.pt
                            """
            if safe_write_file(botsort_config_path, botsort_yaml):
                self.logger.info(f"Created Valorant BoT-SORT config: {botsort_config_path}")
            else:
                self.logger.error(f"Failed to create BoT-SORT config: {botsort_config_path}") 