import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from core.config import (
    DOWNLOADS_DIR, INFERENCE_DIR, MODELS_DIR, DATASET_DIR,
    LOGS_DIR, ensure_directories, WORKFLOW_DATA_DIR, ARCHIVE_DIR, BASE_DIR
)
from utils.logger import setup_logger, log_data_row_action, safe_write_json

class DataManager:
    """Class to handle data management, cleanup, and logging"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
    
    def safe_file_operation(self, operation_func, *args, **kwargs) -> bool:
        """
        Wrapper for safe file operations with error handling and rollback
        
        Args:
            operation_func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = operation_func(*args, **kwargs)
            return result if isinstance(result, bool) else True
        except (PermissionError, OSError) as e:
            self.logger.error(f"File system error in {operation_func.__name__}: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in {operation_func.__name__}: {str(e)}")
            return False
        
    def cleanup_training_data(self, data_row_ids: List[str], keep_models: bool = True) -> bool:
        """
        Clean up training data files after training completion with enhanced error handling
        
        Improved cleanup logic:
        - Removes downloaded files (JSON annotations and videos)
        - Removes extracted keyframes and YOLO annotation files
        - Removes cache files (can be regenerated)
        - Keeps training artifacts (plots, weights) for visualization
        - Keeps dataset.yaml file (needed for future training)
        - Keeps logs (timestamped)
        - Keeps annotated videos (for manual deletion)
        
        Args:
            data_row_ids: List of data row IDs to clean up
            keep_models: Whether to keep trained models
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info(f"Starting improved cleanup for {len(data_row_ids)} data rows")
            
            cleaned_files = []
            failed_operations = []
            
            for data_row_id in data_row_ids:
                log_data_row_action(data_row_id, "CLEANUP_STARTED")
                
                # Clean up downloaded files (JSON annotations and videos)
                download_files = [
                    DOWNLOADS_DIR / f"{data_row_id}.mp4",
                    DOWNLOADS_DIR / f"{data_row_id}.json"
                ]
                
                for file_path in download_files:
                    if file_path.exists():
                        def remove_file():
                            file_path.unlink()
                            return True
                            
                        if self.safe_file_operation(remove_file):
                            cleaned_files.append(str(file_path))
                            self.logger.info(f"Deleted: {file_path}")
                        else:
                            failed_operations.append(str(file_path))
                
                # Clean up dataset files (extracted keyframes and YOLO annotations)
                dataset_patterns = [
                    DATASET_DIR / "train" / "images" / f"{data_row_id}_frame_*.jpg",
                    DATASET_DIR / "train" / "labels" / f"{data_row_id}_frame_*.txt",
                    DATASET_DIR / "val" / "images" / f"{data_row_id}_frame_*.jpg",
                    DATASET_DIR / "val" / "labels" / f"{data_row_id}_frame_*.txt"
                ]
                
                for pattern_path in dataset_patterns:
                    # Use glob to find matching files
                    try:
                        matching_files = list(pattern_path.parent.glob(pattern_path.name))
                        for file_path in matching_files:
                            def remove_pattern_file():
                                file_path.unlink()
                                return True
                                
                            if self.safe_file_operation(remove_pattern_file):
                                cleaned_files.append(str(file_path))
                            else:
                                failed_operations.append(str(file_path))
                    except Exception as e:
                        self.logger.warning(f"Error processing pattern {pattern_path}: {str(e)}")
                
                log_data_row_action(data_row_id, "CLEANUP_COMPLETE")
            
            # Clean up cache files (can be regenerated, not essential for transfer learning)
            cache_files = [
                DATASET_DIR / "train" / "labels.cache",
                DATASET_DIR / "val" / "labels.cache"
            ]
            
            for cache_file in cache_files:
                if cache_file.exists():
                    def remove_cache():
                        cache_file.unlink()
                        return True
                        
                    if self.safe_file_operation(remove_cache):
                        cleaned_files.append(str(cache_file))
                        self.logger.info(f"Deleted cache file: {cache_file}")
                    else:
                        failed_operations.append(str(cache_file))
            
            # Optionally clean up models
            if not keep_models:
                try:
                    model_files = list(MODELS_DIR.glob("*.pt"))
                    for model_file in model_files:
                        def remove_model():
                            model_file.unlink()
                            return True
                            
                        if self.safe_file_operation(remove_model):
                            cleaned_files.append(str(model_file))
                            self.logger.info(f"Deleted model: {model_file}")
                        else:
                            failed_operations.append(str(model_file))
                except Exception as e:
                    self.logger.warning(f"Error cleaning up models: {str(e)}")
            
            # Log what we're keeping for future training
            kept_files = []
            dataset_yaml = DATASET_DIR / "dataset.yaml"
            if dataset_yaml.exists():
                kept_files.append(str(dataset_yaml))
            
            try:
                model_dirs = list(MODELS_DIR.glob("*/"))
                for model_dir in model_dirs:
                    if (model_dir / "weights").exists():
                        kept_files.append(f"{model_dir}/weights/ (training artifacts)")
            except Exception as e:
                self.logger.warning(f"Error checking model directories: {str(e)}")
            
            if kept_files:
                self.logger.info(f"Kept for future training: {kept_files}")
            
            # Report results
            if failed_operations:
                self.logger.warning(f"Failed to clean {len(failed_operations)} files: {failed_operations[:5]}...")
            
            self.logger.info(f"Improved cleanup completed: {len(cleaned_files)} files deleted, {len(failed_operations)} failed")
            self._log_cleanup_summary(data_row_ids, cleaned_files, failed_operations)
            
            return len(failed_operations) == 0  # Success only if no failures
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False
    
    def move_inference_videos(self, video_paths: List[Path], destination_dir: Optional[Path] = None) -> List[Path]:
        """
        Move inference videos to a dedicated directory with enhanced error handling
        
        Args:
            video_paths: List of video file paths to move
            destination_dir: Destination directory (defaults to INFERENCE_DIR)
            
        Returns:
            List of new video paths after moving
        """
        try:
            if destination_dir is None:
                # Use temp inference directory since INFERENCE_DIR is now workflow-specific
                from core.config import TEMP_INFERENCE
                destination_dir = TEMP_INFERENCE / "videos"
            
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            moved_videos = []
            failed_moves = []
            
            for video_path in video_paths:
                if not video_path.exists():
                    self.logger.warning(f"Video file not found: {video_path}")
                    continue
                
                # Create new path with collision handling
                base_name = video_path.stem
                suffix = video_path.suffix
                counter = 1
                new_path = destination_dir / f"{base_name}{suffix}"
                
                while new_path.exists():
                    new_path = destination_dir / f"{base_name}_{counter}{suffix}"
                    counter += 1
                
                def move_video():
                    shutil.move(str(video_path), str(new_path))
                    return True
                
                if self.safe_file_operation(move_video):
                    moved_videos.append(new_path)
                    self.logger.info(f"Moved video: {video_path} -> {new_path}")
                else:
                    failed_moves.append(video_path)
                    self.logger.error(f"Failed to move video: {video_path}")
            
            if failed_moves:
                self.logger.warning(f"Failed to move {len(failed_moves)} videos")
            
            self.logger.info(f"Moved {len(moved_videos)} videos to {destination_dir}")
            return moved_videos
            
        except Exception as e:
            self.logger.error(f"Error moving videos: {str(e)}")
            return []
    
    def archive_completed_workflow(self, workflow_id: str, data_row_ids: List[str]) -> bool:
        """
        Archive a completed workflow with all related files using atomic operations
        
        Args:
            workflow_id: Unique workflow identifier
            data_row_ids: List of data row IDs processed in this workflow
            
        Returns:
            True if archival successful, False otherwise
        """
        try:
            archive_dir = LOGS_DIR / "archives" / workflow_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Create workflow summary
            workflow_summary = {
                'workflow_id': workflow_id,
                'timestamp': datetime.now().isoformat(),
                'data_row_ids': data_row_ids,
                'status': 'completed',
                'files_archived': []
            }
            
            # Copy log files with error handling
            log_files = [
                LOGS_DIR / "workflow.log",
                LOGS_DIR / "data_row_actions.log"
            ]
            
            for log_file in log_files:
                if log_file.exists():
                    def copy_log():
                        archive_log = archive_dir / log_file.name
                        shutil.copy2(str(log_file), str(archive_log))
                        return archive_log
                        
                    archive_log = self.safe_file_operation(copy_log)
                    if archive_log:
                        workflow_summary['files_archived'].append(str(archive_log))
            
            # Copy inference results with error handling
            try:
                # Check both temp inference directory and workflow directories for annotation files
                from core.config import TEMP_INFERENCE, WORKFLOWS_DIR
                inference_files = []
                if TEMP_INFERENCE and TEMP_INFERENCE.exists():
                    inference_files.extend(list(TEMP_INFERENCE.glob("*_annotations.json")))
                
                # Also check workflow directories
                if WORKFLOWS_DIR.exists():
                    for workflow_dir in WORKFLOWS_DIR.glob("*"):
                        if workflow_dir.is_dir():
                            inference_results_dir = workflow_dir / "inference" / "results"
                            if inference_results_dir.exists():
                                inference_files.extend(list(inference_results_dir.glob("*_annotations.json")))
                
                for inf_file in inference_files:
                    def copy_inference():
                        archive_inf = archive_dir / inf_file.name
                        shutil.copy2(str(inf_file), str(archive_inf))
                        return archive_inf
                        
                    archive_inf = self.safe_file_operation(copy_inference)
                    if archive_inf:
                        workflow_summary['files_archived'].append(str(archive_inf))
            except Exception as e:
                self.logger.warning(f"Error copying inference files: {str(e)}")
            
            # Save workflow summary using safe write
            summary_path = archive_dir / "workflow_summary.json"
            if safe_write_json(summary_path, workflow_summary):
                self.logger.info(f"Workflow archived: {archive_dir}")
                return True
            else:
                self.logger.error(f"Failed to save workflow summary: {summary_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error archiving workflow: {str(e)}")
            return False
    
    def _log_cleanup_summary(self, data_row_ids: List[str], cleaned_files: List[str], failed_files: List[str] = None):
        """Log cleanup summary with enhanced information"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'action': 'cleanup',
                'data_row_ids': data_row_ids,
                'files_deleted': len(cleaned_files),
                'files_failed': len(failed_files) if failed_files else 0,
                'deleted_files': cleaned_files[:100],  # Limit to first 100 for readability
                'failed_files': failed_files[:100] if failed_files else []
            }
            
            if len(cleaned_files) > 100:
                summary['deleted_files_truncated'] = f"... and {len(cleaned_files) - 100} more"
            
            summary_path = LOGS_DIR / f"cleanup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            safe_write_json(summary_path, summary)
            
        except Exception as e:
            self.logger.error(f"Error logging cleanup summary: {str(e)}")
    
    def get_tracking_annotation_files(self, tracking_method: Optional[str] = None) -> List[Path]:
        """
        Get all tracking annotation files, optionally filtered by tracking method
        
        Args:
            tracking_method: Optional method to filter by ("sightline", "bytetrack", "botsort")
            
        Returns:
            List of annotation file paths
        """
        try:
            annotation_files = []
            
            # Search in inference directory for standardized annotation files
            if INFERENCE_DIR.exists():
                if tracking_method:
                    # Look for method-specific files: *_tracked_{method}_annotations.json
                    pattern = f"*_tracked_{tracking_method.lower()}_annotations.json"
                    annotation_files.extend(INFERENCE_DIR.glob(pattern))
                    # Also look for legacy naming patterns
                    legacy_pattern = f"*_{tracking_method.lower()}_annotations.json"
                    annotation_files.extend(INFERENCE_DIR.glob(legacy_pattern))
                else:
                    # Get all tracking annotation files
                    patterns = [
                        "*_tracked_*_annotations.json",  # New standardized format
                        "*_sightline_annotations.json",  # Legacy sightline
                        "*_bytetrack_annotations.json",  # Legacy bytetrack
                        "*_botsort_annotations.json"     # Legacy botsort
                    ]
                    for pattern in patterns:
                        annotation_files.extend(INFERENCE_DIR.glob(pattern))
            
            # Remove duplicates
            annotation_files = list(set(annotation_files))
            
            self.logger.info(f"Found {len(annotation_files)} tracking annotation files" + 
                           (f" for {tracking_method}" if tracking_method else ""))
            
            return annotation_files
            
        except Exception as e:
            self.logger.error(f"Error getting tracking annotation files: {str(e)}")
            return []
    
    def get_tracking_video_files(self, tracking_method: Optional[str] = None) -> List[Path]:
        """
        Get all tracking video files, optionally filtered by tracking method
        
        Args:
            tracking_method: Optional method to filter by ("sightline", "bytetrack", "botsort")
            
        Returns:
            List of video file paths
        """
        try:
            video_files = []
            
            # Search in inference directory for tracking videos
            if INFERENCE_DIR.exists():
                if tracking_method:
                    # Look for method-specific videos: *_tracked_{method}.mp4
                    pattern = f"*_tracked_{tracking_method.lower()}.mp4"
                    video_files.extend(INFERENCE_DIR.glob(pattern))
                    # Also look for legacy naming
                    legacy_patterns = [
                        f"*_{tracking_method.lower()}.mp4",
                        f"*_annotated.mp4"  # Generic annotated videos
                    ]
                    for pattern in legacy_patterns:
                        video_files.extend(INFERENCE_DIR.glob(pattern))
                else:
                    # Get all tracking videos
                    patterns = [
                        "*_tracked_*.mp4",     # New standardized format
                        "*_annotated.mp4",     # Generic annotated
                        "*_sightline.mp4",     # Legacy formats
                        "*_bytetrack.mp4",
                        "*_botsort.mp4"
                    ]
                    for pattern in patterns:
                        video_files.extend(INFERENCE_DIR.glob(pattern))
            
            # Remove duplicates
            video_files = list(set(video_files))
            
            self.logger.info(f"Found {len(video_files)} tracking video files" + 
                           (f" for {tracking_method}" if tracking_method else ""))
            
            return video_files
            
        except Exception as e:
            self.logger.error(f"Error getting tracking video files: {str(e)}")
            return []

    def cleanup_tracking_files(self, tracking_method: Optional[str] = None, max_age_days: int = 7) -> Dict[str, int]:
        """
        Clean up old tracking files for specific method or all methods
        
        Args:
            tracking_method: Optional method to clean up ("sightline", "bytetrack", "botsort")
            max_age_days: Maximum age in days before files are cleaned up
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cleanup_stats = {
                'annotations_removed': 0,
                'videos_removed': 0,
                'total_size_freed': 0,
                'errors': 0
            }
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up annotation files
            annotation_files = self.get_tracking_annotation_files(tracking_method)
            for file_path in annotation_files:
                try:
                    if file_path.stat().st_mtime < cutoff_time.timestamp():
                        file_size = file_path.stat().st_size
                        if self.safe_file_operation(lambda: file_path.unlink()):
                            cleanup_stats['annotations_removed'] += 1
                            cleanup_stats['total_size_freed'] += file_size
                            self.logger.debug(f"Removed old annotation file: {file_path}")
                        else:
                            cleanup_stats['errors'] += 1
                except Exception as e:
                    self.logger.warning(f"Error processing annotation file {file_path}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            # Clean up video files
            video_files = self.get_tracking_video_files(tracking_method)
            for file_path in video_files:
                try:
                    if file_path.stat().st_mtime < cutoff_time.timestamp():
                        file_size = file_path.stat().st_size
                        if self.safe_file_operation(lambda: file_path.unlink()):
                            cleanup_stats['videos_removed'] += 1
                            cleanup_stats['total_size_freed'] += file_size
                            self.logger.debug(f"Removed old video file: {file_path}")
                        else:
                            cleanup_stats['errors'] += 1
                except Exception as e:
                    self.logger.warning(f"Error processing video file {file_path}: {str(e)}")
                    cleanup_stats['errors'] += 1
            
            method_str = f" for {tracking_method}" if tracking_method else ""
            self.logger.info(f"Tracking file cleanup{method_str} completed:")
            self.logger.info(f"  - Annotations removed: {cleanup_stats['annotations_removed']}")
            self.logger.info(f"  - Videos removed: {cleanup_stats['videos_removed']}")
            self.logger.info(f"  - Size freed: {cleanup_stats['total_size_freed'] / (1024*1024):.1f} MB")
            if cleanup_stats['errors'] > 0:
                self.logger.warning(f"  - Errors encountered: {cleanup_stats['errors']}")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error during tracking file cleanup: {str(e)}")
            return {'annotations_removed': 0, 'videos_removed': 0, 'total_size_freed': 0, 'errors': 1}
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics for workflow directories
        
        Returns:
            Dictionary with storage usage information
        """
        try:
            def get_dir_size(directory: Path) -> int:
                """Get total size of directory in bytes"""
                if not directory.exists():
                    return 0
                return sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            
            def format_bytes(bytes_val: int) -> str:
                """Format bytes to human readable format"""
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.1f}{unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.1f}TB"
            
            # Get directory sizes
            directories = {
                'workflow_data': WORKFLOW_DATA_DIR,
                'downloads': DOWNLOADS_DIR,
                'dataset': DATASET_DIR,
                'models': MODELS_DIR,
                'inference': TEMP_INFERENCE,
                'logs': LOGS_DIR,
                'archive': ARCHIVE_DIR,
                'tracker_configs': Path(BASE_DIR) / "tracker_configs"  # Add tracker configs
            }
            
            usage_stats = {}
            total_size = 0
            
            for name, directory in directories.items():
                size_bytes = get_dir_size(directory)
                total_size += size_bytes
                
                # Get file counts by type for inference directory
                if name == 'inference' and directory.exists():
                    annotation_files = len(self.get_tracking_annotation_files())
                    video_files = len(self.get_tracking_video_files())
                    
                    # Break down by tracking method
                    method_breakdown = {}
                    for method in ['sightline', 'bytetrack', 'botsort']:
                        method_annotations = len(self.get_tracking_annotation_files(method))
                        method_videos = len(self.get_tracking_video_files(method))
                        if method_annotations > 0 or method_videos > 0:
                            method_breakdown[method] = {
                                'annotations': method_annotations,
                                'videos': method_videos
                            }
                    
                    usage_stats[name] = {
                        'size_bytes': size_bytes,
                        'size_formatted': format_bytes(size_bytes),
                        'file_count': len(list(directory.rglob('*'))),
                        'tracking_files': {
                            'total_annotations': annotation_files,
                            'total_videos': video_files,
                            'by_method': method_breakdown
                        }
                    }
                else:
                    usage_stats[name] = {
                        'size_bytes': size_bytes,
                        'size_formatted': format_bytes(size_bytes),
                        'file_count': len(list(directory.rglob('*'))) if directory.exists() else 0
                    }
            
            # Add total statistics
            usage_stats['total'] = {
                'size_bytes': total_size,
                'size_formatted': format_bytes(total_size)
            }
            
            return usage_stats
            
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {str(e)}")
            return {}
    
    def cleanup_old_files(self, days_old: int = 7) -> bool:
        """
        Clean up files older than specified days with enhanced error handling
        
        Args:
            days_old: Number of days after which files should be considered old
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            cleaned_files = []
            failed_files = []
            
            # Directories to clean
            # Directories to clean
            cleanup_dirs = [DOWNLOADS_DIR]
            
            # Add inference keyframes if it exists
            from core.config import TEMP_INFERENCE
            if TEMP_INFERENCE.exists():
                keyframes_dir = TEMP_INFERENCE / "keyframes"
                if keyframes_dir.exists():
                    cleanup_dirs.append(keyframes_dir)
            
            for cleanup_dir in cleanup_dirs:
                if not cleanup_dir.exists():
                    continue
                
                try:
                    for file_path in cleanup_dir.rglob('*'):
                        if file_path.is_file():
                            try:
                                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                                if file_time < cutoff_time:
                                    def remove_old_file():
                                        file_path.unlink()
                                        return True
                                        
                                    if self.safe_file_operation(remove_old_file):
                                        cleaned_files.append(str(file_path))
                                    else:
                                        failed_files.append(str(file_path))
                            except Exception as e:
                                self.logger.warning(f"Error processing file {file_path}: {str(e)}")
                                failed_files.append(str(file_path))
                except Exception as e:
                    self.logger.warning(f"Error processing directory {cleanup_dir}: {str(e)}")
            
            self.logger.info(f"Cleaned up {len(cleaned_files)} old files (older than {days_old} days)")
            if failed_files:
                self.logger.warning(f"Failed to clean {len(failed_files)} old files")
                
            return len(failed_files) == 0
            
        except Exception as e:
            self.logger.error(f"Error cleaning old files: {str(e)}")
            return False
    
    def export_workflow_report(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Export a comprehensive workflow report with enhanced error handling
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report, None if failed
        """
        try:
            if output_path is None:
                output_path = LOGS_DIR / f"workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Collect workflow statistics
            report = {
                'generated_at': datetime.now().isoformat(),
                'storage_usage': self.get_storage_usage(),
                'log_files': [],
                'data_row_actions': []
            }
            
            # Collect log files with error handling
            try:
                for log_file in LOGS_DIR.glob('*.log'):
                    try:
                        report['log_files'].append({
                            'file': str(log_file),
                            'size': log_file.stat().st_size,
                            'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing log file {log_file}: {str(e)}")
            except Exception as e:
                self.logger.warning(f"Error collecting log files: {str(e)}")
            
            # Read data row actions if available
            data_row_log = LOGS_DIR / "data_row_actions.log"
            if data_row_log.exists():
                try:
                    with open(data_row_log, 'r', encoding='utf-8') as f:
                        report['data_row_actions'] = f.readlines()
                except Exception as e:
                    self.logger.warning(f"Error reading data row log: {str(e)}")
                    report['data_row_actions'] = []
            
            # Save report using safe write
            if safe_write_json(output_path, report):
                self.logger.info(f"Workflow report exported: {output_path}")
                return output_path
            else:
                self.logger.error(f"Failed to save workflow report: {output_path}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error exporting workflow report: {str(e)}")
            return None 

    def cleanup_workflow_data(self, workflow_dirs: dict, keep_outputs: bool = False) -> bool:
        """
        Clean up all data for a specific workflow (downloads, dataset, outputs, temp, etc.)
        Args:
            workflow_dirs: Directory structure for the workflow (from get_workflow_directories)
            keep_outputs: If True, keep the outputs directory (models, videos, annotations, reports)
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info(f"Starting cleanup for workflow: {workflow_dirs.get('workflow_id', workflow_dirs.get('root'))}")
            cleaned_files = []
            failed_files = []
            # Clean up inputs (videos, annotations)
            for sub in ['videos', 'annotations']:
                input_dir = workflow_dirs['inputs'][sub]
                if input_dir.exists():
                    for file_path in input_dir.glob('*'):
                        try:
                            file_path.unlink()
                            cleaned_files.append(str(file_path))
                        except Exception as e:
                            self.logger.warning(f"Failed to delete {file_path}: {e}")
                            failed_files.append(str(file_path))
            # Clean up training dataset (images, labels)
            for split in ['train', 'val']:
                for sub in ['images', 'labels']:
                    data_dir = workflow_dirs['training'][f'{split}_{sub}']
                    if data_dir.exists():
                        for file_path in data_dir.glob('*'):
                            try:
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {file_path}: {e}")
                                failed_files.append(str(file_path))
            # Clean up dataset.yaml
            dataset_yaml = workflow_dirs['training']['dataset'] / 'dataset.yaml'
            if dataset_yaml.exists():
                try:
                    dataset_yaml.unlink()
                    cleaned_files.append(str(dataset_yaml))
                except Exception as e:
                    self.logger.warning(f"Failed to delete {dataset_yaml}: {e}")
                    failed_files.append(str(dataset_yaml))
            # Clean up temp files
            temp_root = workflow_dirs['temp']['root']
            if temp_root.exists():
                for file_path in temp_root.rglob('*'):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_files.append(str(file_path))
                        elif file_path.is_dir():
                            file_path.rmdir()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")
                        failed_files.append(str(file_path))
                try:
                    temp_root.rmdir()
                except Exception:
                    pass
            # Clean up outputs if not keeping
            if not keep_outputs:
                outputs_root = workflow_dirs['outputs']['root']
                if outputs_root.exists():
                    for file_path in outputs_root.rglob('*'):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                cleaned_files.append(str(file_path))
                            elif file_path.is_dir():
                                file_path.rmdir()
                        except Exception as e:
                            self.logger.warning(f"Failed to delete {file_path}: {e}")
                            failed_files.append(str(file_path))
                    try:
                        outputs_root.rmdir()
                    except Exception:
                        pass
            self.logger.info(f"Workflow cleanup complete: {len(cleaned_files)} files deleted, {len(failed_files)} failed.")
            return len(failed_files) == 0
        except Exception as e:
            self.logger.error(f"Error during workflow cleanup: {e}")
            return False 