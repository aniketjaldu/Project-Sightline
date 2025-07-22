"""
Enhanced Inference Runner

This module provides a unified interface for running inference with different tracking methods.
It wraps the TrackingManager to provide easy access to official ByteTrack, BoT-SORT, and custom Sightline tracking.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from ultralytics import YOLO

from core.config import TRACKING_CONFIG, ensure_directories, INFERENCE_DIR
from pipeline.tracking_manager import TrackingManager
from utils.logger import setup_logger, safe_write_json

class EnhancedInferenceRunner:
    """Enhanced inference runner with support for multiple tracking methods and comparison"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
    
    def run_inference_with_tracking(
        self, 
        model_path: Union[Path, str], 
        video_path: Union[Path, str], 
        output_video_path: Optional[Union[Path, str]] = None,
        tracking_method: str = "sightline",
        data_row_id: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        save_annotations: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference with the specified tracking method using standardized interface
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_video_path: Path to save annotated video (optional)
            tracking_method: Tracking method to use ("sightline", "bytetrack", "botsort")
            data_row_id: Data row ID for logging purposes
            conf_threshold: Confidence threshold for detections
            save_annotations: Whether to save annotations to JSON file
            
        Returns:
            Dictionary containing inference results and annotations with tracking
        """
        # Convert paths to Path objects
        model_path = Path(model_path)
        video_path = Path(video_path)
        if output_video_path:
            output_video_path = Path(output_video_path)
        
        # Validate inputs
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate tracking method
        valid_methods = ["sightline", "bytetrack", "botsort"]
        if tracking_method not in valid_methods:
            raise ValueError(f"Invalid tracking method: {tracking_method}. Must be one of {valid_methods}")
        
        self.logger.info(f"Starting {tracking_method.upper()} inference")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"Video: {video_path}")
        
        try:
            # Load YOLO model
            model = YOLO(str(model_path))
            
            # Set default confidence threshold
            if conf_threshold is None:
                conf_threshold = TRACKING_CONFIG.get('conf_threshold', 0.25)
            
            # Initialize tracking manager with standardized interface
            tracking_manager = TrackingManager(tracking_method=tracking_method)
            
            # Run inference with tracking using standardized method
            result = tracking_manager.run_inference(
                model=model,
                video_path=video_path,
                output_video_path=output_video_path,
                data_row_id=data_row_id,
                conf_threshold=conf_threshold
            )
            
            # Add tracking method info to result if successful
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
            else:
                self.logger.error(f"Inference failed for {tracking_method.upper()}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during {tracking_method.upper()} inference: {str(e)}")
            return {'success': False, 'tracking_method': tracking_method, 'error': str(e)}

    def compare_tracking_methods(
        self, 
        model_path: Union[Path, str], 
        video_path: Union[Path, str], 
        output_dir: Union[Path, str],
        methods: List[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple tracking methods on the same video using standardized interface
        
        Args:
            model_path: Path to the trained YOLO model
            video_path: Path to input video
            output_dir: Directory to save comparison results
            methods: List of tracking methods to compare (default: all available)
            **kwargs: Additional arguments for inference
            
        Returns:
            Dictionary with results for each tracking method
        """
        if methods is None:
            methods = ["sightline", "bytetrack", "botsort"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = Path(video_path)
        comparison_results = {}
        
        self.logger.info(f"Comparing tracking methods: {methods}")
        self.logger.info(f"Video: {video_path}")
        self.logger.info(f"Output directory: {output_dir}")
        
        for method in methods:
            self.logger.info(f"\n--- Running {method.upper()} tracking ---")
            
            # Use standardized naming for comparison output videos
            output_video_path = output_dir / f"{video_path.stem}_tracked_{method}.mp4"
            
            result = self.run_inference_with_tracking(
                model_path=model_path,
                video_path=video_path,
                output_video_path=output_video_path,
                tracking_method=method,
                **kwargs
            )
            
            comparison_results[method] = result
        
        # Generate comparison summary with standardized file naming
        summary_path = output_dir / "tracking_comparison_summary.json"
        self._save_comparison_summary(comparison_results, summary_path)
        
        self.logger.info(f"\nComparison completed. Results saved to: {output_dir}")
        self.logger.info(f"Summary: {summary_path}")
        
        return comparison_results
    
    def _save_comparison_summary(self, results: Dict[str, Dict[str, Any]], output_path: Path) -> None:
        """
        Save comparison summary using standardized file operations
        
        Args:
            results: Dictionary with results for each tracking method
            output_path: Path to save summary
        """
        try:
            summary = {
                "comparison_timestamp": time.time(),
                "methods_compared": list(results.keys()),
                "summary": {}
            }
            
            for method, result in results.items():
                if result.get('success'):
                    annotations = result.get('annotations', {})
                    total_frames = result.get('total_frames', 0)
                    total_detections = sum(len(frame_annotations) for frame_annotations in annotations.get('frames', {}).values())
                    
                    # Calculate unique tracks
                    unique_tracks = set()
                    for frame_annotations in annotations.get('frames', {}).values():
                        for annotation in frame_annotations:
                            if 'track_id' in annotation:
                                unique_tracks.add(annotation['track_id'])
                    
                    summary["summary"][method] = {
                        "success": True,
                        "total_frames": total_frames,
                        "total_detections": total_detections,
                        "unique_tracks": len(unique_tracks),
                        "average_detections_per_frame": total_detections / max(total_frames, 1),
                        "output_video": str(result.get('output_video', '')),
                        "annotations_file": str(result.get('annotations_file', ''))
                    }
                else:
                    summary["summary"][method] = {
                        "success": False,
                        "error": result.get('error', 'Unknown error')
                    }
            
            # Use safe file writing for consistency
            if safe_write_json(output_path, summary):
                self.logger.info(f"Comparison summary saved to: {output_path}")
            else:
                self.logger.error(f"Failed to save comparison summary to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving comparison summary: {str(e)}") 