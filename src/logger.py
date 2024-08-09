import os
import logging
from datetime import datetime


class AppLogger:
    """Basic Logging Class For Main Processing"""
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

    def add_metadata(self, metadata: dict) -> None:
        """Logging Metadata that is type dict"""
        self.logger.info(str(metadata))

    def debug(self, message: str) -> None:
        "Logging Debug Messages"
        self.logger.debug(message)

    def info(self, message: str) -> None:
        "Logging Regular Info"
        self.logger.info(message)

    def warning(self, message: str) -> None:
        "Logging Warning But Not Error"
        self.logger.warning(message)

    def error(self, message: str) -> None:
        "Logging Error"
        self.logger.error(message)

    def exception(self, message: str) -> None:
        "Logging Exception"
        self.logger.exception(message)


if __name__ == "__main__":
    logger = AppLogger("TestLogger")

    logger.debug("This is a debug message.")

    logger.info("This is an info message.")

    logger.warning("This is a warning message.")

    logger.error("This is an error message.")

    try:
        raise ValueError("This is a test exception.")
    except ValueError as e:
        logger.exception(f"An exception occurred: {e}")

    metadata = {"key1": "value1", "key2": "value2"}
    logger.add_metadata(metadata)

    logger2 = AppLogger("TestLogger")
    assert logger is logger2, "Singleton pattern is not working correctly."

    print("All tests passed.")
