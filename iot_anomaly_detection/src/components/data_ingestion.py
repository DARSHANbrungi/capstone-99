import os
import sys
from dataclasses import dataclass

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    # This path should point to your complete, raw dataset.
    # Update the filename if it's different.
    raw_data_path: str = os.path.join('data', 'raw', 'final_combined_dataset.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Data Ingestion component initialized for UNSUPERVISED learning.")

    def initiate_data_ingestion(self):
        """
        This method checks for the raw data file and returns its path.
        """
        logging.info("Data ingestion process started.")
        try:
            raw_path = self.ingestion_config.raw_data_path
            
            # Check if the raw data file exists
            if not os.path.exists(raw_path):
                error_msg = f"Raw data file not found at the specified path: {raw_path}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            logging.info(f"Data ingestion complete. Located raw data at: {raw_path}")
            
            # Return the single path to the entire raw dataset
            return raw_path

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise CustomException(e, sys)