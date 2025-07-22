import logging
import sys
from datetime import datetime
from pathlib import Path
from core.config import LOG_FILE, LOGS_DIR

def setup_logger(name: str, log_level: str = "INFO"):
    """Set up a logger with both file and console handlers"""
    
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    # File handler with better error handling
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def safe_write_file(file_path: Path, content: str, encoding: str = 'utf-8') -> bool:
    """
    Safely write content to a file using atomic operations
    
    Args:
        file_path: Path to write to
        content: Content to write
        encoding: File encoding
        
    Returns:
        True if successful, False otherwise
    """
    import tempfile
    import os
    
    temp_path = None
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary file without auto-deletion for Windows compatibility
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            dir=file_path.parent, 
            delete=False, 
            encoding=encoding,
            suffix='.tmp'
        )
        temp_path = Path(temp_file.name)
        
        try:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
        finally:
            # Always close the file handle before attempting move (Windows requirement)
            temp_file.close()
        
        # Windows-compatible atomic move operation
        try:
            # If target exists, remove it first (Windows requirement)  
            if file_path.exists():
                file_path.unlink()
            
            # Atomic move operation
            temp_path.replace(file_path)
            temp_path = None  # Mark as successfully moved
            
        except OSError as move_error:
            # On Windows, sometimes we need to retry the move operation
            import time
            time.sleep(0.1)  # Brief pause
            
            try:
                if file_path.exists():
                    file_path.unlink()
                temp_path.rename(file_path)  # Alternative move method
                temp_path = None  # Mark as successfully moved
            except Exception as retry_error:
                print(f"Failed to move temp file after retry: {retry_error}")
                raise
        
        return True
        
    except Exception as e:
        # Clean up temp file if it exists and wasn't successfully moved
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_error:
                print(f"Failed to clean up temp file {temp_path}: {cleanup_error}")
        print(f"Error writing file {file_path}: {e}")
        return False

def safe_write_json(file_path: Path, data: dict, indent: int = 2) -> bool:
    """
    Safely write JSON data to file with validation
    
    Args:
        file_path: Path to write to
        data: Data to write as JSON
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    import json
    
    try:
        # Validate JSON serializability first
        json_content = json.dumps(data, indent=indent, default=str)
        return safe_write_file(file_path, json_content)
    except (TypeError, ValueError) as e:
        print(f"Error serializing JSON for {file_path}: {e}")
        return False

def log_data_row_action(data_row_id: str, action: str, details: str = ""):
    """Log data row action with enhanced error handling"""
    try:
        data_row_log = LOGS_DIR / "data_row_actions.log"
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} - {data_row_id} - {action} - {details}\n"
        
        # Ensure logs directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write for log entries to prevent corruption
        with open(data_row_log, 'a', encoding='utf-8') as f:
            f.write(log_entry)
            f.flush()
            
    except Exception as e:
        print(f"Warning: Could not log data row action: {e}")
        # Don't raise - logging failures shouldn't break the main workflow 