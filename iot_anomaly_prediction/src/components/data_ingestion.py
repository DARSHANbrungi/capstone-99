import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

# Add project root to Python path to allow for package-like imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@dataclass
class DataIngestionConfig:
    # This path should point to your complete, raw dataset.
    raw_data_path: str = os.path.join('data', 'raw', 'final_combined_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion component initialized.")

    def initiate_data_ingestion(self):
        """
        This method locates the raw data file and returns its path.
        """
        logging.info("Starting data ingestion process.")
        try:
            raw_path = self.ingestion_config.raw_data_path
            
            if not os.path.exists(raw_path):
                error_msg = f"Raw data file not found at: {raw_path}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            logging.info(f"Data ingestion complete. Located raw data at: {raw_path}")
            return raw_path

        except Exception as e:
            raise CustomException(e, sys)

