import labelbox as lb
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
from core.config import (
    LABELBOX_API_KEY, LABELBOX_PROJECT_ID, DOWNLOADS_DIR, 
    EXPORT_PARAMS, ensure_directories
)
from utils.logger import setup_logger, log_data_row_action, safe_write_json
import tempfile
import shutil

class LabelboxDataFetcher:
    """Class to handle downloading data from Labelbox"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
        
        if not LABELBOX_API_KEY:
            raise ValueError("LABELBOX_API_KEY environment variable is required")
        
        self.client = lb.Client(api_key=LABELBOX_API_KEY)
        self.project = self.client.get_project(LABELBOX_PROJECT_ID) if LABELBOX_PROJECT_ID else None
        
        self.logger.info("Labelbox client initialized successfully")
    
    def fetch_data_row_by_id(self, data_row_id: str, include_labels: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific data row by ID with enhanced error handling
        
        Args:
            data_row_id: The ID of the data row to fetch
            include_labels: Whether to include label information
            
        Returns:
            Data row information as dictionary, None if failed
        """
        try:
            if not self.project:
                self.logger.error("No project configured. Set LABELBOX_PROJECT_ID in environment.")
                return None
            
            self.logger.info(f"Fetching data row: {data_row_id}")
            log_data_row_action(data_row_id, "FETCH_STARTED")
            
            # Create export parameters
            export_params = EXPORT_PARAMS.copy()
            if not include_labels:
                export_params['labels'] = False
                export_params['label_details'] = False
                export_params['performance_details'] = False
            
            # Create export task with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    export_task = self.project.export_v2(
                        params=export_params,
                        filters={
                            'data_row_ids': [data_row_id]
                        }
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    self.logger.warning(f"Export attempt {attempt + 1} failed: {str(e)}, retrying...")
                    
            # Wait for completion with timeout
            export_task.wait_till_done(timeout_seconds=300)
            
            if export_task.has_errors():
                error_stream = export_task.get_buffered_stream(stream_type=lb.StreamType.ERRORS)
                errors = list(error_stream)
                self.logger.error(f"Export errors for {data_row_id}: {errors}")
                log_data_row_action(data_row_id, "FETCH_ERROR", f"Export errors: {errors}")
                return None
            
            # Get the result
            result_stream = export_task.get_buffered_stream(stream_type=lb.StreamType.RESULT)
            results = list(result_stream)
            
            if not results:
                self.logger.warning(f"No data found for data row: {data_row_id}")
                log_data_row_action(data_row_id, "FETCH_NO_DATA")
                return None
            
            data_row_data = results[0].json
            self.logger.info(f"Successfully fetched data row: {data_row_id}")
            log_data_row_action(data_row_id, "FETCH_SUCCESS")
            
            return data_row_data
            
        except Exception as e:
            self.logger.error(f"Error fetching data row {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "FETCH_ERROR", str(e))
            return None
    
    def download_video(self, data_row_data: Dict[str, Any], output_dir: Path = None) -> Optional[Path]:
        """
        Download video file from data row data with enhanced error handling and atomic operations
        
        Args:
            data_row_data: Data row information from Labelbox
            output_dir: Directory to save the video (defaults to DOWNLOADS_DIR)
            
        Returns:
            Path to the downloaded video file
        """
        if output_dir is None:
            output_dir = DOWNLOADS_DIR
        
        try:
            data_row_id = data_row_data['data_row']['id']
            video_url = data_row_data['data_row']['row_data']
            
            self.logger.info(f"Downloading video for data row: {data_row_id}")
            log_data_row_action(data_row_id, "DOWNLOAD_STARTED")
            
            # Get filename from URL or use data row ID
            filename = f"{data_row_id}.mp4"
            if 'external_id' in data_row_data['data_row']:
                external_id = data_row_data['data_row']['external_id']
                if external_id and external_id.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    filename = external_id
            
            video_path = output_dir / filename
            
            # Check if file already exists and is valid
            if video_path.exists():
                try:
                    # Basic validation - check if file is not empty
                    if video_path.stat().st_size > 0:
                        self.logger.info(f"Video already exists and appears valid: {video_path}")
                        log_data_row_action(data_row_id, "DOWNLOAD_SKIPPED", f"File exists: {video_path}")
                        return video_path
                except Exception:
                    # File exists but is invalid, will redownload
                    pass
            
            # Download with proper error handling and atomic operation
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Windows-compatible atomic download operation
            temp_path = None
            try:
                # Create temporary file without auto-deletion to handle Windows file locking
                temp_file = tempfile.NamedTemporaryFile(
                    dir=output_dir, 
                    delete=False, 
                    suffix='.tmp'
                )
                temp_path = Path(temp_file.name)
                
                # Download in chunks with progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                try:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            temp_file.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress for large files
                            if total_size > 0 and downloaded_size % (1024*1024) == 0:  # Every MB
                                progress = (downloaded_size / total_size) * 100
                                self.logger.debug(f"Download progress: {progress:.1f}%")
                    
                    # Ensure all data is written and close file handle (critical for Windows)
                    temp_file.flush()
                    import os
                    os.fsync(temp_file.fileno())  # Force write to disk
                    
                finally:
                    # Always close the file handle before attempting move (Windows requirement)
                    temp_file.close()
                
                # Validate downloaded file
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise ValueError("Downloaded file is empty or doesn't exist")
                
                # Windows-compatible atomic move operation
                try:
                    # If target exists, remove it first (Windows requirement)
                    if video_path.exists():
                        video_path.unlink()
                    
                    # Atomic move to final location
                    temp_path.replace(video_path)
                    temp_path = None  # Mark as successfully moved
                    
                except OSError as move_error:
                    # On Windows, sometimes we need to retry the move operation
                    import time
                    self.logger.warning(f"First move attempt failed, retrying: {move_error}")
                    time.sleep(0.1)  # Brief pause
                    
                    try:
                        if video_path.exists():
                            video_path.unlink()
                        temp_path.rename(video_path)  # Alternative move method
                        temp_path = None  # Mark as successfully moved
                    except Exception as retry_error:
                        self.logger.error(f"Failed to move temp file after retry: {retry_error}")
                        raise
                
                self.logger.info(f"Video downloaded successfully: {video_path} ({downloaded_size:,} bytes)")
                log_data_row_action(data_row_id, "DOWNLOAD_SUCCESS", f"Path: {video_path}, Size: {downloaded_size:,} bytes")
                
                return video_path
                
            except Exception as e:
                # Clean up temp file on error (only if it wasn't successfully moved)
                if temp_path and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception as cleanup_error:
                        self.logger.warning(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
                raise
            
        except requests.RequestException as e:
            self.logger.error(f"Network error downloading video for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "DOWNLOAD_ERROR", f"Network error: {str(e)}")
            return None
        except Exception as e:
            data_row_id = data_row_data.get('data_row', {}).get('id', 'unknown')
            self.logger.error(f"Error downloading video for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "DOWNLOAD_ERROR", str(e))
            return None
    
    def save_data_row_json(self, data_row_data: Dict[str, Any], output_dir: Path = None) -> Optional[Path]:
        """
        Save data row JSON to file with enhanced error handling and validation
        
        Args:
            data_row_data: Data row information from Labelbox
            output_dir: Directory to save the JSON (defaults to DOWNLOADS_DIR)
            
        Returns:
            Path to the saved JSON file
        """
        if output_dir is None:
            output_dir = DOWNLOADS_DIR
        
        try:
            data_row_id = data_row_data['data_row']['id']
            json_path = output_dir / f"{data_row_id}.json"
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use safe JSON write with validation
            if safe_write_json(json_path, data_row_data):
                self.logger.info(f"Data row JSON saved: {json_path}")
                log_data_row_action(data_row_id, "JSON_SAVED", f"Path: {json_path}")
                return json_path
            else:
                raise IOError(f"Failed to write JSON file: {json_path}")
            
        except Exception as e:
            data_row_id = data_row_data.get('data_row', {}).get('id', 'unknown')
            self.logger.error(f"Error saving JSON for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "JSON_SAVE_ERROR", str(e))
            return None
    
    def fetch_and_download_complete(self, data_row_id: str, include_labels: bool = True) -> Dict[str, Path]:
        """
        Complete fetch and download process for a data row with enhanced error handling
        
        Args:
            data_row_id: The ID of the data row to fetch
            include_labels: Whether to include labels in the export
            
        Returns:
            Dictionary with paths to downloaded files
        """
        result = {}
        
        try:
            # Fetch data row
            data_row_data = self.fetch_data_row_by_id(data_row_id, include_labels)
            if not data_row_data:
                self.logger.error(f"Failed to fetch data row: {data_row_id}")
                return result
            
            # Download video
            video_path = self.download_video(data_row_data)
            if video_path:
                result['video'] = video_path
            else:
                self.logger.error(f"Failed to download video for: {data_row_id}")
            
            # Save JSON
            json_path = self.save_data_row_json(data_row_data)
            if json_path:
                result['json'] = json_path
            else:
                self.logger.error(f"Failed to save JSON for: {data_row_id}")
            
            # Validate we got both files
            if 'video' in result and 'json' in result:
                log_data_row_action(data_row_id, "COMPLETE_DOWNLOAD_SUCCESS", f"Files: video={result['video'].name}, json={result['json'].name}")
            else:
                log_data_row_action(data_row_id, "COMPLETE_DOWNLOAD_PARTIAL", f"Retrieved: {list(result.keys())}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in complete download for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "COMPLETE_DOWNLOAD_ERROR", str(e))
            return result 