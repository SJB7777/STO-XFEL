import os
import json
import logging
import yaml
from datetime import datetime

def format_run_scan_number(run_scan_list):
    return ', '.join([f'({x}, {y})' for x, y in run_scan_list])

class Logger:

    def __init__(self, name: str, log_dir:str = "logs"):
        self.log_dir = log_dir
        self.logger = self._setup_logger(name)
        self.logger.info("Starting analysis")
        self.metadata = {}
        self.results = {}

    def _setup_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        
        log_file = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_analysis.log"
        log_path = os.path.join(self.log_dir, log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        else:
            print("what?")
        return logger

    def add_result(self, key, value):
        self.results[key] = value
        self.logger.info(f"Result added: {key} = {value}")

    def save_to_json(self, filename):
        data = {
            "metadata": self.metadata,
            "results": self.results
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        self.logger.info(f"Record saved to {filename}")
        
        
    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

def run_analysis(name: str):
    record = Logger(name)

    # 분석 로직을 여기에 추가
    record.logger.info("Analysis logic executed")
    record.add_result("analysis_status", "completed")

    record.save_to_json("analysis_record.json")
    record.logger.info("Analysis completed")

    return record

# 분석 실행
# analysis_record = run_analysis()
