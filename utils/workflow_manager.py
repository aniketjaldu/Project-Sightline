"""
Workflow Directory Manager

This module provides utilities for managing the new organized directory structure.
It handles workflow creation, file organization, and cleanup operations.

Author: AI Assistant
"""

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from core.config import (
    get_workflow_directories, ensure_workflow_directories, get_tracking_file_paths,
    WORKFLOWS_DIR, MODELS_ACTIVE, MODELS_VERSIONS, ARCHIVE_DIR, TEMP_DIR
)
from utils.logger import setup_logger, safe_write_json, safe_write_file

class WorkflowManager:
    """
    Manages organized workflow directories and file operations
    """
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def create_workflow(self, workflow_type: str, workflow_id: str = None) -> Dict[str, Any]:
        """
        Create a new organized workflow with proper directory structure
        
        Args:
            workflow_type: Type of workflow ("training", "inference", "comparison")
            workflow_id: Optional custom workflow ID
            
        Returns:
            Dictionary with workflow info and directory structure
        """
        try:
            # Generate workflow ID if not provided
            if workflow_id is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                workflow_id = f"{workflow_type}_{timestamp}"
            
            # Get organized directory structure
            workflow_dirs = get_workflow_directories(workflow_id)
            
            # Create all necessary directories
            ensure_workflow_directories(workflow_dirs)
            
            # Create workflow manifest
            manifest = {
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'created_at': datetime.now().isoformat(),
                'status': 'initialized',
                'directories': {
                    key: str(path) for key, path in workflow_dirs.items() 
                    if isinstance(path, Path)
                },
                'metadata': {
                    'version': '2.0',
                    'structure': 'organized'
                }
            }
            
            # Save workflow manifest
            manifest_path = workflow_dirs['root'] / 'workflow.json'
            if safe_write_json(manifest_path, manifest):
                self.logger.info(f"Created {workflow_type} workflow: {workflow_id}")
                self.logger.info(f"Workflow directory: {workflow_dirs['root']}")
            else:
                raise IOError("Failed to save workflow manifest")
            
            return {
                'success': True,
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'directories': workflow_dirs,
                'manifest_path': manifest_path
            }
            
        except Exception as e:
            self.logger.error(f"Error creating workflow: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_workflow_info(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about an existing workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Workflow information dictionary or None if not found
        """
        try:
            # Search for workflow in workflows directory
            for workflow_dir in WORKFLOWS_DIR.glob(f"*{workflow_id}*"):
                manifest_path = workflow_dir / 'workflow.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting workflow info: {str(e)}")
            return None
    
    def save_workflow_file(self, workflow_dirs: Dict[str, Any], file_type: str, 
                          content: Union[str, dict], filename: str) -> Optional[Path]:
        """
        Save a file in the appropriate workflow directory
        
        Args:
            workflow_dirs: Workflow directories structure
            file_type: Type of file ('input', 'output', 'temp', 'log', 'tracking')
            content: File content (string or dict for JSON)
            filename: Name of the file
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            # Determine target directory based on file type
            if file_type == 'input_video':
                target_dir = workflow_dirs['inputs']['videos']
            elif file_type == 'input_annotation':
                target_dir = workflow_dirs['inputs']['annotations']
            elif file_type == 'output_model':
                target_dir = workflow_dirs['outputs']['models']
            elif file_type == 'output_video':
                target_dir = workflow_dirs['outputs']['videos']
            elif file_type == 'output_annotation':
                target_dir = workflow_dirs['outputs']['annotations']
            elif file_type == 'temp':
                target_dir = workflow_dirs['temp']['processing']
            elif file_type == 'log':
                target_dir = workflow_dirs['logs']['root']
            elif file_type.startswith('tracking_'):
                method = file_type.split('_')[1]  # e.g., 'tracking_sightline'
                target_dir = workflow_dirs['tracking'][method]
            else:
                raise ValueError(f"Unknown file type: {file_type}")
            
            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = target_dir / filename
            
            if isinstance(content, dict):
                success = safe_write_json(file_path, content)
            else:
                success = safe_write_file(file_path, str(content))
            
            if success:
                self.logger.debug(f"Saved {file_type} file: {file_path}")
                return file_path
            else:
                self.logger.error(f"Failed to save {file_type} file: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving workflow file: {str(e)}")
            return None
    
    def organize_tracking_outputs(self, workflow_dirs: Dict[str, Any], tracking_method: str,
                                video_name: str, annotations: dict, video_path: Path = None) -> Dict[str, Path]:
        """
        Save tracking outputs in organized structure
        
        Args:
            workflow_dirs: Workflow directories structure
            tracking_method: Tracking method used
            video_name: Base video name
            annotations: Tracking annotations
            video_path: Optional path to annotated video
            
        Returns:
            Dictionary with paths to saved files
        """
        try:
            # Get standardized file paths
            file_paths = get_tracking_file_paths(workflow_dirs, tracking_method, video_name)
            saved_files = {}
            
            # Save annotations
            if safe_write_json(file_paths['annotations_json'], annotations):
                saved_files['annotations'] = file_paths['annotations_json']
            
            # Save video if provided
            if video_path and video_path.exists():
                try:
                    shutil.copy2(video_path, file_paths['video_output'])
                    saved_files['video'] = file_paths['video_output']
                except Exception as e:
                    self.logger.warning(f"Failed to copy video: {str(e)}")
            
            # Create metadata
            metadata = {
                'tracking_method': tracking_method,
                'video_name': video_name,
                'created_at': datetime.now().isoformat(),
                'total_frames': annotations.get('video_info', {}).get('total_frames', 0),
                'total_detections': sum(
                    len(frame_data) for frame_data in annotations.get('frames', {}).values()
                )
            }
            
            if safe_write_json(file_paths['metadata'], metadata):
                saved_files['metadata'] = file_paths['metadata']
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error organizing tracking outputs: {str(e)}")
            return {}
    
    def promote_model(self, workflow_dirs: Dict[str, Any], model_name: str, 
                     source_path: Path) -> Optional[Path]:
        """
        Promote a trained model from workflow to active models
        
        Args:
            workflow_dirs: Workflow directories structure
            model_name: Name for the model
            source_path: Path to trained model file
            
        Returns:
            Path to promoted model or None if failed
        """
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Source model not found: {source_path}")
            
            # Create active model path
            active_model_path = MODELS_ACTIVE / f"{model_name}.pt"
            
            # Create version path for backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_path = MODELS_VERSIONS / model_name / f"{model_name}_{timestamp}.pt"
            version_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy to active location
            shutil.copy2(source_path, active_model_path)
            
            # Create version backup
            shutil.copy2(source_path, version_path)
            
            # Update model info
            model_info = {
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'source_workflow': workflow_dirs['workflow_id'],
                'source_path': str(source_path),
                'active_path': str(active_model_path),
                'version_path': str(version_path)
            }
            
            info_path = MODELS_ACTIVE / f"{model_name}_info.json"
            safe_write_json(info_path, model_info)
            
            self.logger.info(f"Promoted model {model_name} to active models")
            return active_model_path
            
        except Exception as e:
            self.logger.error(f"Error promoting model: {str(e)}")
            return None
    
    def cleanup_workflow_temp(self, workflow_dirs: Dict[str, Any]) -> bool:
        """
        Clean up temporary files from a workflow
        
        Args:
            workflow_dirs: Workflow directories structure
            
        Returns:
            True if successful, False otherwise
        """
        try:
            temp_dir = workflow_dirs['temp']['root']
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up temporary files: {temp_dir}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up workflow temp: {str(e)}")
            return False
    
    def archive_workflow(self, workflow_id: str) -> bool:
        """
        Archive a completed workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find workflow directory
            workflow_dir = None
            for dir_path in WORKFLOWS_DIR.glob(f"*{workflow_id}*"):
                if dir_path.is_dir():
                    workflow_dir = dir_path
                    break
            
            if not workflow_dir:
                raise FileNotFoundError(f"Workflow not found: {workflow_id}")
            
            # Create archive location
            archive_path = ARCHIVE_DIR / workflow_dir.name
            
            # Move workflow to archive
            shutil.move(str(workflow_dir), str(archive_path))
            
            self.logger.info(f"Archived workflow: {workflow_id} -> {archive_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error archiving workflow: {str(e)}")
            return False
    
    def cleanup_old_workflows(self, max_age_days: int = 30) -> Dict[str, int]:
        """
        Clean up workflows older than specified days
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            stats = {'archived': 0, 'errors': 0}
            
            for workflow_dir in WORKFLOWS_DIR.iterdir():
                if not workflow_dir.is_dir():
                    continue
                
                # Check modification time
                if workflow_dir.stat().st_mtime < cutoff_time.timestamp():
                    try:
                        # Archive old workflow
                        if self.archive_workflow(workflow_dir.name):
                            stats['archived'] += 1
                        else:
                            stats['errors'] += 1
                    except Exception as e:
                        self.logger.warning(f"Error archiving {workflow_dir.name}: {str(e)}")
                        stats['errors'] += 1
            
            self.logger.info(f"Cleanup completed: {stats['archived']} workflows archived, {stats['errors']} errors")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error during workflow cleanup: {str(e)}")
            return {'archived': 0, 'errors': 1}
    
    def cleanup_global_temp(self) -> bool:
        """
        Clean up global temporary directory
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if TEMP_DIR.exists():
                # Remove all contents but keep the directory
                for item in TEMP_DIR.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                
                self.logger.info(f"Cleaned up global temporary directory: {TEMP_DIR}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning up global temp: {str(e)}")
            return False
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Get summary of all workflows
        
        Returns:
            Summary dictionary
        """
        try:
            summary = {
                'active_workflows': 0,
                'archived_workflows': 0,
                'total_models': 0,
                'temp_size_mb': 0,
                'workflows': []
            }
            
            # Count active workflows
            if WORKFLOWS_DIR.exists():
                active_dirs = [d for d in WORKFLOWS_DIR.iterdir() if d.is_dir()]
                summary['active_workflows'] = len(active_dirs)
                
                for workflow_dir in active_dirs[:10]:  # Limit to 10 recent
                    manifest_path = workflow_dir / 'workflow.json'
                    if manifest_path.exists():
                        try:
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                                summary['workflows'].append({
                                    'id': manifest.get('workflow_id', workflow_dir.name),
                                    'type': manifest.get('workflow_type', 'unknown'),
                                    'created': manifest.get('created_at', 'unknown'),
                                    'status': manifest.get('status', 'unknown')
                                })
                        except:
                            pass
            
            # Count archived workflows
            if ARCHIVE_DIR.exists():
                archived_dirs = [d for d in ARCHIVE_DIR.iterdir() if d.is_dir()]
                summary['archived_workflows'] = len(archived_dirs)
            
            # Count models
            if MODELS_ACTIVE.exists():
                models = [f for f in MODELS_ACTIVE.iterdir() if f.suffix == '.pt']
                summary['total_models'] = len(models)
            
            # Calculate temp size
            if TEMP_DIR.exists():
                temp_size = sum(f.stat().st_size for f in TEMP_DIR.rglob('*') if f.is_file())
                summary['temp_size_mb'] = round(temp_size / (1024 * 1024), 1)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting workflow summary: {str(e)}")
            return {'error': str(e)} 