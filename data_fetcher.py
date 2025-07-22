import labelbox as lb
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
from config import (
    LABELBOX_API_KEY, LABELBOX_PROJECT_ID, DOWNLOADS_DIR, 
    EXPORT_PARAMS, ensure_directories
)
from utils.logger import setup_logger, log_data_row_action

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
        Fetch a specific data row by ID
        
        Args:
            data_row_id: The ID of the data row to fetch
            include_labels: Whether to include labels in the export
            
        Returns:
            Dictionary containing the data row information
        """
        try:
            self.logger.info(f"Fetching data row: {data_row_id}")
            log_data_row_action(data_row_id, "FETCH_STARTED")
            
            # Set up export parameters
            export_params = EXPORT_PARAMS.copy()
            export_params["labels"] = include_labels
            
            # Get data row first to verify it exists
            try:
                data_row = self.client.get_data_row(data_row_id)
            except Exception as e:
                self.logger.error(f"Could not find data row {data_row_id}: {str(e)}")
                log_data_row_action(data_row_id, "FETCH_ERROR", f"Data row not found: {str(e)}")
                return None
            
            # Export from project instead of by global key
            if self.project:
                export_task = self.project.export(
                    params=export_params,
                    filters={"data_row_ids": [data_row_id]}
                )
            else:
                # Fallback to direct data row export
                export_task = lb.DataRow.export(
                    client=self.client,
                    global_keys=[data_row_id],
                    params=export_params
                )
            
            # Wait for export to complete
            export_task.wait_till_done()
            
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
        Download video file from data row data
        
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
            
            # Download the video
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Video downloaded successfully: {video_path}")
            log_data_row_action(data_row_id, "DOWNLOAD_SUCCESS", f"Path: {video_path}")
            
            return video_path
            
        except Exception as e:
            data_row_id = data_row_data.get('data_row', {}).get('id', 'unknown')
            self.logger.error(f"Error downloading video for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "DOWNLOAD_ERROR", str(e))
            return None
    
    def save_data_row_json(self, data_row_data: Dict[str, Any], output_dir: Path = None) -> Optional[Path]:
        """
        Save data row JSON to file
        
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
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(data_row_data, f, indent=2)
            
            self.logger.info(f"Data row JSON saved: {json_path}")
            log_data_row_action(data_row_id, "JSON_SAVED", f"Path: {json_path}")
            
            return json_path
            
        except Exception as e:
            data_row_id = data_row_data.get('data_row', {}).get('id', 'unknown')
            self.logger.error(f"Error saving JSON for {data_row_id}: {str(e)}")
            log_data_row_action(data_row_id, "JSON_SAVE_ERROR", str(e))
            return None
    
    def fetch_and_download_complete(self, data_row_id: str, include_labels: bool = True) -> Dict[str, Path]:
        """
        Complete fetch and download process for a data row
        
        Args:
            data_row_id: The ID of the data row to fetch
            include_labels: Whether to include labels in the export
            
        Returns:
            Dictionary with paths to downloaded files
        """
        result = {}
        
        # Fetch data row
        data_row_data = self.fetch_data_row_by_id(data_row_id, include_labels)
        if not data_row_data:
            return result
        
        # Download video
        video_path = self.download_video(data_row_data)
        if video_path:
            result['video'] = video_path
        
        # Save JSON
        json_path = self.save_data_row_json(data_row_data)
        if json_path:
            result['json'] = json_path
        
        return result 