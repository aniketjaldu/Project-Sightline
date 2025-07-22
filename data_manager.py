import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import (
    DOWNLOADS_DIR, INFERENCE_DIR, MODELS_DIR, DATASET_DIR,
    LOGS_DIR, ensure_directories
)
from utils.logger import setup_logger, log_data_row_action

class DataManager:
    """Class to handle data management, cleanup, and logging"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
        
    def cleanup_training_data(self, data_row_ids: List[str], keep_models: bool = True) -> bool:
        """
        Clean up training data files after training completion
        
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
            
            for data_row_id in data_row_ids:
                log_data_row_action(data_row_id, "CLEANUP_STARTED")
                
                # Clean up downloaded files (JSON annotations and videos)
                download_files = [
                    DOWNLOADS_DIR / f"{data_row_id}.mp4",
                    DOWNLOADS_DIR / f"{data_row_id}.json"
                ]
                
                for file_path in download_files:
                    if file_path.exists():
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                        self.logger.info(f"Deleted: {file_path}")
                
                # Clean up dataset files (extracted keyframes and YOLO annotations)
                dataset_patterns = [
                    DATASET_DIR / "train" / "images" / f"{data_row_id}_frame_*.jpg",
                    DATASET_DIR / "train" / "labels" / f"{data_row_id}_frame_*.txt",
                    DATASET_DIR / "val" / "images" / f"{data_row_id}_frame_*.jpg",
                    DATASET_DIR / "val" / "labels" / f"{data_row_id}_frame_*.txt"
                ]
                
                for pattern_path in dataset_patterns:
                    # Use glob to find matching files
                    matching_files = list(pattern_path.parent.glob(pattern_path.name))
                    for file_path in matching_files:
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                
                log_data_row_action(data_row_id, "CLEANUP_COMPLETE")
            
            # Clean up cache files (can be regenerated, not essential for transfer learning)
            cache_files = [
                DATASET_DIR / "train" / "labels.cache",
                DATASET_DIR / "val" / "labels.cache"
            ]
            
            for cache_file in cache_files:
                if cache_file.exists():
                    cache_file.unlink()
                    cleaned_files.append(str(cache_file))
                    self.logger.info(f"Deleted cache file: {cache_file}")
            
            # Optionally clean up models
            if not keep_models:
                model_files = list(MODELS_DIR.glob("*.pt"))
                for model_file in model_files:
                    model_file.unlink()
                    cleaned_files.append(str(model_file))
                    self.logger.info(f"Deleted model: {model_file}")
            
            # Log what we're keeping for future training
            kept_files = []
            dataset_yaml = DATASET_DIR / "dataset.yaml"
            if dataset_yaml.exists():
                kept_files.append(str(dataset_yaml))
            
            model_dirs = list(MODELS_DIR.glob("*/"))
            for model_dir in model_dirs:
                if (model_dir / "weights").exists():
                    kept_files.append(f"{model_dir}/weights/ (training artifacts)")
            
            if kept_files:
                self.logger.info(f"Kept for future training: {kept_files}")
            
            self.logger.info(f"Improved cleanup completed: {len(cleaned_files)} files deleted")
            self._log_cleanup_summary(data_row_ids, cleaned_files)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return False
    
    def move_inference_videos(self, video_paths: List[Path], destination_dir: Optional[Path] = None) -> List[Path]:
        """
        Move inference videos to a dedicated directory
        
        Args:
            video_paths: List of video file paths to move
            destination_dir: Destination directory (defaults to INFERENCE_DIR)
            
        Returns:
            List of new video paths after moving
        """
        try:
            if destination_dir is None:
                destination_dir = INFERENCE_DIR / "videos"
            
            destination_dir.mkdir(parents=True, exist_ok=True)
            
            moved_videos = []
            
            for video_path in video_paths:
                if not video_path.exists():
                    self.logger.warning(f"Video file not found: {video_path}")
                    continue
                
                # Create new path
                new_path = destination_dir / video_path.name
                
                # Move the file
                shutil.move(str(video_path), str(new_path))
                moved_videos.append(new_path)
                
                self.logger.info(f"Moved video: {video_path} -> {new_path}")
            
            self.logger.info(f"Moved {len(moved_videos)} videos to {destination_dir}")
            return moved_videos
            
        except Exception as e:
            self.logger.error(f"Error moving videos: {str(e)}")
            return []
    
    def archive_completed_workflow(self, workflow_id: str, data_row_ids: List[str]) -> bool:
        """
        Archive a completed workflow with all related files
        
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
            
            # Copy log files
            log_files = [
                LOGS_DIR / "workflow.log",
                LOGS_DIR / "data_row_actions.log"
            ]
            
            for log_file in log_files:
                if log_file.exists():
                    archive_log = archive_dir / log_file.name
                    shutil.copy2(str(log_file), str(archive_log))
                    workflow_summary['files_archived'].append(str(archive_log))
            
            # Copy inference results
            inference_files = list(INFERENCE_DIR.glob("*_annotations.json"))
            for inf_file in inference_files:
                archive_inf = archive_dir / inf_file.name
                shutil.copy2(str(inf_file), str(archive_inf))
                workflow_summary['files_archived'].append(str(archive_inf))
            
            # Save workflow summary
            summary_path = archive_dir / "workflow_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(workflow_summary, f, indent=2)
            
            self.logger.info(f"Workflow archived: {archive_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error archiving workflow: {str(e)}")
            return False
    
    def _log_cleanup_summary(self, data_row_ids: List[str], cleaned_files: List[str]):
        """Log cleanup summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'action': 'cleanup',
            'data_row_ids': data_row_ids,
            'files_deleted': len(cleaned_files),
            'deleted_files': cleaned_files
        }
        
        summary_path = LOGS_DIR / f"cleanup_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
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
                        return f"{bytes_val:.1f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.1f} TB"
            
            usage = {}
            directories = {
                'downloads': DOWNLOADS_DIR,
                'dataset': DATASET_DIR,
                'models': MODELS_DIR,
                'inference': INFERENCE_DIR,
                'logs': LOGS_DIR
            }
            
            total_size = 0
            for name, directory in directories.items():
                size_bytes = get_dir_size(directory)
                usage[name] = {
                    'bytes': size_bytes,
                    'formatted': format_bytes(size_bytes),
                    'path': str(directory)
                }
                total_size += size_bytes
            
            usage['total'] = {
                'bytes': total_size,
                'formatted': format_bytes(total_size)
            }
            
            return usage
            
        except Exception as e:
            self.logger.error(f"Error getting storage usage: {str(e)}")
            return {}
    
    def cleanup_old_files(self, days_old: int = 7) -> bool:
        """
        Clean up files older than specified days
        
        Args:
            days_old: Number of days after which files should be considered old
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.now() - timedelta(days=days_old)
            cleaned_files = []
            
            # Directories to clean
            cleanup_dirs = [DOWNLOADS_DIR, INFERENCE_DIR / "keyframes"]
            
            for cleanup_dir in cleanup_dirs:
                if not cleanup_dir.exists():
                    continue
                
                for file_path in cleanup_dir.rglob('*'):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            cleaned_files.append(str(file_path))
            
            self.logger.info(f"Cleaned up {len(cleaned_files)} old files (older than {days_old} days)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning old files: {str(e)}")
            return False
    
    def export_workflow_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Export a comprehensive workflow report
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
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
            
            # Collect log files
            for log_file in LOGS_DIR.glob('*.log'):
                report['log_files'].append({
                    'file': str(log_file),
                    'size': log_file.stat().st_size,
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
            
            # Read data row actions if available
            data_row_log = LOGS_DIR / "data_row_actions.log"
            if data_row_log.exists():
                with open(data_row_log, 'r') as f:
                    report['data_row_actions'] = f.readlines()
            
            # Save report
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Workflow report exported: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting workflow report: {str(e)}")
            raise 