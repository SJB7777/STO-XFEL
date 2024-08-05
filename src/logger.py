import os
import json
import logging
from datetime import datetime

def format_run_scan_number(run_scan_list):
    return ', '.join([f'({x}, {y})' for x, y in run_scan_list])

class AppLogger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(AppLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: str, log_dir: str = "logs"):
        if not hasattr(self, 'logger'):  # Ensure initialization only happens once
            self.log_dir = log_dir
            self.logger = self._setup_logger(name)
            self.logger.info("Start Logging")  # Log info message on first creation
            self.results = {}

    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_analysis.log"
        
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        log_dir = os.path.join(self.log_dir, str(datetime.now().date()))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path)

        self.log_path = log_path
        
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger

    def add_metadata(self, metadata:dict) -> None:
        self.logger.info(f"{metadata}")

    def add_result(self, key, value) -> None:
        self.results[key] = value
        self.logger.info(f"Result added: {key} = {value}")

    def save_to_json(self, filename: str) -> None:
        data = {
            "metadata": self.metadata,
            "results": self.results
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Record saved to {filename}")
        
    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)


