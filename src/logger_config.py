import logging
from datetime import datetime
import os

def setup_logger(log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = f"{datetime.now().strftime('%Y-%m-%d')}_analysis.log"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('analysis_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

logger = setup_logger()

def log_analysis(file_number, analysis_type, special_note=''):
    logger.info(f"File: {file_number}, Type: {analysis_type}, Note: {special_note}")

def log_error(file_number, error_message):
    logger.error(f"File: {file_number}, Error: {error_message}")
    
log_error(1, "a")