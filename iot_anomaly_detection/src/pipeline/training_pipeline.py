import os
import sys

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_training_pipeline():
    logging.info("--- Starting Training Pipeline ---")
    
    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    raw_data_path = ingestion.initiate_data_ingestion()
    
    # Step 2: Data Transformation
    transformer = DataTransformation()
    data_pca, proxy_target = transformer.initiate_data_transformation(raw_data_path)
    
    # Step 3: Model Training
    trainer = ModelTrainer()
    trainer.initiate_model_training(data_pca, proxy_target)
    
    logging.info("--- Training Pipeline Finished Successfully ---")

if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        logging.error(f"The training pipeline failed catastrophically.")
        raise CustomException(e, sys)