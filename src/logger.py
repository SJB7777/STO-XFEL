import loguru
from loguru._logger import Logger


def setup_logger() -> Logger:
    format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
    log_file: str = "logs\\{time:YYYY-MM-DD}\\{time:YYYYMMDD_HHmmss}.log"
    loguru.logger.add(log_file, format=format, rotation="500 MB", compression="zip")

    return loguru.logger


if __name__ == "__main__":
    logger = setup_logger()
    logger.debug("This is a debug message.")

    logger.info("This is an info message.")

    logger.warning("This is a warning message.")

    logger.error("This is an error message.")

    try:
        raise ValueError("This is a test exception.")
    except ValueError as e:
        logger.exception(f"An exception occurred: {e}")

    metadata = {"key1": "value1", "key2": "value2"}
    logger.info(metadata)

    print("All tests passed.")
