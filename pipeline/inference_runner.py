import cv2
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
from core.config import (
    YOLO_CONFIG, CLASS_MAPPING, REVERSE_CLASS_MAPPING, INFERENCE_DIR, 
    ensure_directories, TRACKING_CONFIG
)
from pipeline.tracking_manager import TrackingManager
from utils.logger import setup_logger, log_data_row_action, safe_write_json

class YOLOInferenceRunner:
    """Class to handle YOLO inference on videos"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
    
    def run_tracking_inference_on_video(
        self, 
        model_path: Path, 
        video_path: Path, 
        output_video_path: Optional[Path] = None,
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        tracking_method: str = "sightline"
    ) -> Optional[Dict[str, Any]]:
        """
        Run YOLO inference with configurable tracking on a video file
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_video_path: Path to save the annotated output video (optional)
            data_row_id: Data row ID for logging purposes
            conf_threshold: Confidence threshold for detections
            tracking_method: Tracking method to use ("sightline", "bytetrack", "botsort")
            
        Returns:
            Dictionary containing inference results and annotations with tracking
        """
        try:
            self.logger.info(f"Starting {tracking_method.upper()} inference on video: {video_path}")
            
            # Validate inputs
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Load YOLO model
            model = YOLO(str(model_path))
            
            # Set default confidence threshold
            if conf_threshold is None:
                conf_threshold = TRACKING_CONFIG.get('conf_threshold', 0.25)
            
            # Initialize tracking manager with the specified method
            tracking_manager = TrackingManager(tracking_method=tracking_method)
            
            # Run inference with tracking using standardized interface
            result = tracking_manager.run_inference(
                model=model,
                video_path=video_path,
                output_video_path=output_video_path,
                data_row_id=data_row_id,
                conf_threshold=conf_threshold
            )
                
            # Enhance result with additional metadata
            if result and result.get('success'):
                result['model_path'] = str(model_path)
                result['video_path'] = str(video_path)
                result['conf_threshold'] = conf_threshold
                
                # Log summary
                total_frames = result.get('total_frames', 0)
                annotations = result.get('annotations', {})
                total_detections = sum(len(frame_annotations) for frame_annotations in annotations.get('frames', {}).values())
                
                self.logger.info(f"Inference completed successfully:")
                self.logger.info(f"  - Tracking method: {tracking_method.upper()}")
                self.logger.info(f"  - Total frames: {total_frames}")
                self.logger.info(f"  - Total detections: {total_detections}")
                if output_video_path:
                    self.logger.info(f"  - Output video: {output_video_path}")
                if result.get('annotations_file'):
                    self.logger.info(f"  - Annotations: {result['annotations_file']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during {tracking_method.upper()} inference: {str(e)}")
            if data_row_id:
                log_data_row_action(data_row_id, f"{tracking_method.upper()}_INFERENCE_ERROR", str(e))
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
        Legacy method - redirects to tracking inference with default sightline method
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_video_path: Path to save the annotated output video (optional)
            data_row_id: Data row ID for logging purposes
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary containing inference results and annotations
        """
        return self.run_tracking_inference_on_video(
            model_path=model_path,
            video_path=video_path,
            output_video_path=output_video_path,
            data_row_id=data_row_id,
            conf_threshold=conf_threshold,
            tracking_method="sightline"
        )
    
    def run_batch_inference(
        self,
        model_path: Path,
        video_dir: Path,
        output_dir: Optional[Path] = None,
        video_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference on multiple videos in a directory
        
        Args:
            model_path: Path to the trained YOLO model
            video_dir: Directory containing video files
            output_dir: Directory to save output videos (optional)
            video_extensions: List of video file extensions to process
            
        Returns:
            Dictionary with results for each video
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        results = {}
        
        try:
            if not video_dir.exists():
                self.logger.error(f"Video directory not found: {video_dir}")
                return results
            
            # Find all video files
            video_files = []
            for ext in video_extensions:
                video_files.extend(video_dir.glob(f"*{ext}"))
                video_files.extend(video_dir.glob(f"*{ext.upper()}"))
            
            if not video_files:
                self.logger.warning(f"No video files found in {video_dir}")
                return results
            
            self.logger.info(f"Found {len(video_files)} video files to process")
            
            for i, video_path in enumerate(video_files):
                self.logger.info(f"Processing video {i+1}/{len(video_files)}: {video_path.name}")
                
                output_video_path = None
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_video_path = output_dir / f"{video_path.stem}_inference{video_path.suffix}"
                
                video_result = self.run_inference_on_video(
                    model_path=model_path,
                    video_path=video_path,
                    output_video_path=output_video_path
                )
                
                results[str(video_path)] = video_result
                
                # Log progress
                if video_result and video_result.get('stats'):
                    stats = video_result['stats']
                    self.logger.info(f"Completed: {stats['frames_processed']} frames, {stats['avg_fps']:.1f} fps")
            
            successful = sum(1 for result in results.values() if result is not None)
            self.logger.info(f"Batch inference completed: {successful}/{len(video_files)} videos processed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch inference: {str(e)}")
            return results
    
    def _save_annotations(self, annotations: Dict[str, Any], video_name: str) -> Optional[Path]:
        """Save annotations to JSON file using safe write operations"""
        try:
            annotations_path = INFERENCE_DIR / f"{video_name}_annotations.json"
            
            if safe_write_json(annotations_path, annotations):
                self.logger.info(f"Annotations saved to: {annotations_path}")
                return annotations_path
            else:
                self.logger.error(f"Failed to save annotations to: {annotations_path}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error saving annotations: {str(e)}")
            return None
    
    def run_inference_on_multiple_videos(
        self, 
        model_path: Path, 
        video_paths: List[Path],
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run inference on multiple videos (legacy method for backward compatibility)
        
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
                output_dir.mkdir(parents=True, exist_ok=True)
                output_video_path = output_dir / f"{video_path.stem}_inference.mp4"
            
            video_result = self.run_inference_on_video(
                model_path=model_path,
                video_path=video_path,
                output_video_path=output_video_path
            )
            
            results[str(video_path)] = video_result
        
        return results