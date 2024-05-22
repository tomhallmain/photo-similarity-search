import logging
from logging.handlers import RotatingFileHandler
import os

from config import config

def get_logger(log_app_name, log_level='INFO'):
    # Configure logging
    log_level = os.getenv('LOG_LEVEL', log_level)
    log_level = getattr(logging, log_level.upper())

    file_handler = RotatingFileHandler(f"{log_app_name}_{config.unique_id}.log", maxBytes=10485760, backupCount=10)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logger = logging.getLogger(log_app_name)
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_level

