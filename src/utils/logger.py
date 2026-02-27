import logging
import os
import yaml
import logging.config
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance configured from logging.yaml.
    Ensures log directory exists and avoids duplicate handlers.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():  # Prevent duplicate handlers
        config_path = Path("config/logging.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Ensure log directory exists
        log_file = config["handlers"]["file"]["filename"]
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Configure logging
        logging.config.dictConfig(config)

    return logger