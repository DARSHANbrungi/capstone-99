import os
import sys
import joblib
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using joblib.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a file using joblib.
    """
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)

