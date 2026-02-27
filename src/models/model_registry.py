import os
import joblib
from src.utils.logger import get_logger

logger = get_logger(__name__)
def save_model(model, path, logger):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")