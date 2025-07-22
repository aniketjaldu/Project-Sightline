import logging
import sys
from datetime import datetime
from pathlib import Path
from config import LOG_FILE, LOGS_DIR

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
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_data_row_action(data_row_id: str, action: str, details: str = ""):
    """Log actions related to specific data rows"""
    logger = setup_logger("DATA_ROW_TRACKER")
    timestamp = datetime.now().isoformat()
    log_message = f"DataRow: {data_row_id} | Action: {action}"
    if details:
        log_message += f" | Details: {details}"
    logger.info(log_message)
    
    # Also write to a separate data row log file
    data_row_log = LOGS_DIR / "data_row_actions.log"
    with open(data_row_log, 'a') as f:
        f.write(f"{timestamp} | {log_message}\n") 