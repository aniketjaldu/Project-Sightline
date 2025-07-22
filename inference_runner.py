import cv2
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
from config import (
    YOLO_CONFIG, CLASS_MAPPING, REVERSE_CLASS_MAPPING, INFERENCE_DIR, 
    ensure_directories, BYTETRACK_CONFIG
)
from bytetrack_integration import ByteTrackProcessor
from utils.logger import setup_logger, log_data_row_action

class YOLOInferenceRunner:
    """Class to handle YOLO inference on videos"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
    
    def run_bytetrack_inference_on_video(
        self, 
        model_path: Path, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO inference with custom ByteTrack tracking on a video file
        This uses the same ByteTrack processor as training for bidirectional consistency
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_video_path: Path to save the annotated output video (optional)
            data_row_id: Data row ID for logging purposes
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing inference results and annotations with tracking
        """
        try:
            if data_row_id:
                log_data_row_action(data_row_id, "BYTETRACK_INFERENCE_STARTED")
            
            self.logger.info(f"Starting custom ByteTrack inference on video: {video_path}")
            
            # Load the model
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None

            model = YOLO(str(model_path))
            
            # Set confidence threshold
            if conf_threshold is None:
                conf_threshold = BYTETRACK_CONFIG['track_low_thresh']
            
            # Open video file to get properties
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return None

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Initialize custom ByteTrack processor (same as training)
            bytetrack_processor = ByteTrackProcessor()
            
            # Initialize video writer if output path is specified
            writer = None
            if output_video_path:
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
            
            # Storage for all detections (to be processed by ByteTrack)
            all_detections = []
            
            # Process video frame by frame with YOLO detection only (no tracking)
            frame_count = 0
            start_time = time.time()
            
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        frames_remaining = total_frames - frame_count
                        eta = (elapsed_time / max(frame_count, 1)) * frames_remaining
                        fps_current = frame_count / max(elapsed_time, 0.001)
                        progress = (frame_count / total_frames) * 100
                        self.logger.info(
                            f"Processing frame {frame_count}/{total_frames} "
                            f"({progress:.1f}%) | FPS: {fps_current:.1f} | ETA: {eta:.1f}s"
                        )
                    
                    # Run YOLO detection only (no tracking)
                    results = model.predict(
                        frame,
                        conf=conf_threshold,
                        max_det=YOLO_CONFIG['max_det'],  # Valorant optimization: limit detections per frame
                        verbose=False
                    )
                    
                    # Extract detections from this frame
                    frame_detections = self._extract_frame_annotations(results[0], frame_count)
                    
                    # Convert to format expected by ByteTrack processor
                    for detection in frame_detections:
                        bbox = detection['bounding_box']
                        all_detections.append({
                            'frame_number': frame_count,
                            'class_name': detection['class_name'],
                            'confidence': detection['confidence'],
                            'bbox_pixel': [bbox['left'], bbox['top'], bbox['width'], bbox['height']],
                            'center_x': bbox['left'] + bbox['width'] / 2,
                            'center_y': bbox['top'] + bbox['height'] / 2
                        })
                    
                    frame_count += 1
                        
            except KeyboardInterrupt:
                self.logger.info("Inference stopped by user")
            finally:
                cap.release()
            
            # Now apply our custom ByteTrack processor to all detections
            self.logger.info("Applying custom ByteTrack tracking to detections...")
            tracked_detections = bytetrack_processor.process_inference_detections(all_detections)
            
            # Convert tracked detections back to annotation format
            all_annotations = {
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': frame_count,
                    'data_row_id': data_row_id
                },
                'frames': {}
            }
            
            # Group tracked detections by frame
            for detection in tracked_detections:
                frame_num = detection['frame_number']
                if frame_num not in all_annotations['frames']:
                    all_annotations['frames'][frame_num] = []
                
                # Convert back to annotation format with track_id
                annotation = {
                    'frame': frame_num,
                    'class_id': CLASS_MAPPING.get(detection['class_name'], 0),
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'track_id': detection['track_id'],  # This comes from our custom ByteTrack processor
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
            
            # Generate annotated video if requested
            if writer and output_video_path:
                self.logger.info("Generating annotated video...")
                cap = cv2.VideoCapture(str(video_path))
                frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Draw annotations on frame
                    if frame_idx in all_annotations['frames']:
                        for annotation in all_annotations['frames'][frame_idx]:
                            bbox = annotation['bounding_box']
                            track_id = annotation['track_id']
                            class_name = annotation['class_name']
                            confidence = annotation['confidence']
                            
                            # Draw bounding box
                            left = int(bbox['left'])
                            top = int(bbox['top'])
                            right = int(left + bbox['width'])
                            bottom = int(top + bbox['height'])
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Draw label with track ID
                            label = f"{class_name} #{track_id} ({confidence:.2f})"
                            cv2.putText(frame, label, (left, top - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    writer.write(frame)
                    frame_idx += 1
                
                cap.release()
                writer.release()
            
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / max(elapsed_time, 0.001)
            
            self.logger.info(f"Custom ByteTrack inference completed: {frame_count} frames processed")
            self.logger.info(f"Average FPS: {avg_fps:.1f}")
            
            if output_video_path and output_video_path.exists():
                self.logger.info(f"Tracked video saved to: {output_video_path}")
            
            # Save annotations to file
            annotations_path = self._save_annotations(all_annotations, f"{video_path.stem}_bytetrack")
            
            if data_row_id:
                log_data_row_action(
                    data_row_id, 
                    "BYTETRACK_INFERENCE_COMPLETE", 
                    f"Frames: {frame_count}, Annotations: {annotations_path}"
                )
            
            return {
                'annotations': all_annotations,
                'annotations_file': annotations_path,
                'output_video': output_video_path,
                'stats': {
                    'frames_processed': frame_count,
                    'avg_fps': avg_fps,
                    'total_time': elapsed_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during custom ByteTrack inference: {str(e)}")
            if data_row_id:
                log_data_row_action(data_row_id, "BYTETRACK_INFERENCE_ERROR", str(e))
            return None
        
    def run_inference_on_video(
        self, 
        model_path: Path, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO inference on a video file
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_video_path: Path to save the annotated output video (optional)
            data_row_id: Data row ID for logging purposes
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing inference results and annotations
        """
        try:
            if data_row_id:
                log_data_row_action(data_row_id, "INFERENCE_STARTED")
            
            self.logger.info(f"Starting inference on video: {video_path}")
            
            # Load the model
            if not model_path.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            model = YOLO(str(model_path))
            
            # Set confidence threshold
            if conf_threshold is None:
                conf_threshold = YOLO_CONFIG['conf_threshold']
            
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video file: {video_path}")
                return None
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Initialize video writer if output path is specified
            writer = None
            if output_video_path:
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_video_path),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width, height)
                )
            
            # Storage for all annotations
            all_annotations = {
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames,
                    'data_row_id': data_row_id
                },
                'frames': {}
            }
            
            # Process frames
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Show progress
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        frames_remaining = total_frames - frame_count
                        eta = (elapsed_time / max(frame_count, 1)) * frames_remaining
                        fps_current = frame_count / max(elapsed_time, 0.001)
                        progress = (frame_count / total_frames) * 100
                        self.logger.info(
                            f"Processing frame {frame_count}/{total_frames} "
                            f"({progress:.1f}%) | FPS: {fps_current:.1f} | ETA: {eta:.1f}s"
                        )
                    
                    # Run inference
                    results = model(frame, conf=conf_threshold, max_det=YOLO_CONFIG['max_det'])[0]
                    
                    # Extract annotations for this frame
                    frame_annotations = self._extract_frame_annotations(results, frame_count)
                    if frame_annotations:
                        all_annotations['frames'][frame_count] = frame_annotations
                    
                    # Draw boxes on frame if output video is requested
                    if writer:
                        annotated_frame = results.plot()
                        writer.write(annotated_frame)
                    
                    frame_count += 1
                        
            except KeyboardInterrupt:
                self.logger.info("Inference stopped by user")
            finally:
                # Cleanup
                cap.release()
                if writer:
                    writer.release()
            
            elapsed_time = time.time() - start_time
            avg_fps = frame_count / max(elapsed_time, 0.001)
            
            self.logger.info(f"Inference completed: {frame_count} frames processed")
            self.logger.info(f"Average FPS: {avg_fps:.1f}")
            
            if output_video_path and output_video_path.exists():
                self.logger.info(f"Annotated video saved to: {output_video_path}")
            
            # Save annotations to file
            annotations_path = self._save_annotations(all_annotations, video_path.stem)
            
            if data_row_id:
                log_data_row_action(
                    data_row_id, 
                    "INFERENCE_COMPLETE", 
                    f"Frames: {frame_count}, Annotations: {annotations_path}"
                )
            
            return {
                'annotations': all_annotations,
                'annotations_file': annotations_path,
                'output_video': output_video_path,
                'stats': {
                    'frames_processed': frame_count,
                    'avg_fps': avg_fps,
                    'total_time': elapsed_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            if data_row_id:
                log_data_row_action(data_row_id, "INFERENCE_ERROR", str(e))
            return None
    
    def _extract_tracking_annotations(self, result, frame_num: int) -> List[Dict[str, Any]]:
        """Extract tracking annotations from YOLO+ByteTrack results for a single frame"""
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
                class_name = REVERSE_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Calculate bounding box dimensions
                left = float(x1)
                top = float(y1)
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                # Convert to Labelbox format (left, top, width, height)
                annotation = {
                    'frame': frame_num,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'track_id': track_id,  # Added track ID from ByteTrack
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
    
    def _extract_frame_annotations(self, results, frame_num: int) -> List[Dict[str, Any]]:
        """Extract annotations from YOLO results for a single frame"""
        annotations = []
        
        if results.boxes is not None:
            boxes = results.boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (xyxy format)
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Get confidence and class
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Get class name
                class_name = REVERSE_CLASS_MAPPING.get(class_id, f"class_{class_id}")
                
                # Convert to Labelbox format (left, top, width, height)
                annotation = {
                    'frame': frame_num,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bounding_box': {
                        'left': float(x1),
                        'top': float(y1),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    }
                }
                
                annotations.append(annotation)
        
        return annotations
    
    def _save_annotations(self, annotations: Dict[str, Any], video_name: str) -> Path:
        """Save annotations to JSON file"""
        try:
            annotations_path = INFERENCE_DIR / f"{video_name}_annotations.json"
            
            import json
            with open(annotations_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            self.logger.info(f"Annotations saved to: {annotations_path}")
            return annotations_path
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            raise
    
    def run_inference_on_multiple_videos(
        self, 
        model_path: Path, 
        video_paths: List[Path],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run inference on multiple videos
        
        Args:
            model_path: Path to the trained YOLO model
            video_paths: List of video file paths
            output_dir: Directory to save output videos (optional)
            
        Returns:
            Dictionary with results for each video
        """
        results = {}
        
        for i, video_path in enumerate(video_paths):
            self.logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path.name}")
            
            output_video_path = None
            if output_dir:
                output_video_path = output_dir / f"{video_path.stem}_inference.mp4"
            
            video_result = self.run_inference_on_video(
                model_path=model_path,
                video_path=video_path,
                output_video_path=output_video_path
            )
            
            results[str(video_path)] = video_result
        
        return results
    
    def extract_keyframes_for_annotation(
        self, 
        annotations: Dict[str, Any], 
        video_path: Path,
        output_dir: Optional[Path] = None,
        min_confidence: float = 0.5
    ) -> List[Path]:
        """
        Extract keyframes with high-confidence detections for manual review
        
        Args:
            annotations: Annotations dictionary from inference
            video_path: Path to the original video
            output_dir: Directory to save keyframes
            min_confidence: Minimum confidence threshold for frame extraction
            
        Returns:
            List of paths to extracted keyframes
        """
        try:
            if output_dir is None:
                output_dir = INFERENCE_DIR / "keyframes"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return []
            
            extracted_frames = []
            
            # Find frames with high-confidence detections
            high_conf_frames = []
            for frame_num, frame_data in annotations['frames'].items():
                max_conf = max(obj['confidence'] for obj in frame_data)
                if max_conf >= min_confidence:
                    high_conf_frames.append((int(frame_num), max_conf))
            
            # Sort by confidence
            high_conf_frames.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top frames
            for frame_num, confidence in high_conf_frames[:10]:  # Top 10 frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    frame_path = output_dir / f"{video_path.stem}_frame_{frame_num}_{confidence:.2f}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(frame_path)
            
            cap.release()
            
            self.logger.info(f"Extracted {len(extracted_frames)} keyframes to {output_dir}")
            return extracted_frames
            
        except Exception as e:
            self.logger.error(f"Error extracting keyframes: {str(e)}")
            return [] 