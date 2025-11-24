import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add the project root to the system path to allow for package-like imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import setup_logging
from src.utils.utils import save_object

@dataclass
class UnsupervisedModelTrainerConfig:
    # Path where the trained unsupervised model will be saved
    trained_model_file_path: str = os.path.join("saved_models", "unsupervised_model.pkl")

class UnsupervisedModelTrainer:
    def __init__(self):
        self.model_trainer_config = UnsupervisedModelTrainerConfig()
        logging.info("Unsupervised Model Trainer component initialized.")

    def initiate_model_training(self, train_array, test_array, y_train_true, y_test_true):
        """
        This method trains an unsupervised Isolation Forest model and evaluates its performance.
        """
        try:
            logging.info("Preparing data for unsupervised training.")
            # In unsupervised learning, we only use the features (X) for training.
            X_train = train_array
            X_test = test_array

            # The true labels (y) are kept aside for evaluation only.
            # We need to convert our original labels (0 for normal, 3 for issue)
            # to match Isolation Forest's output (1 for normal, -1 for anomaly).
            y_test_true_mapped = y_test_true.copy()
            y_test_true_mapped[y_test_true_mapped == 0] = 1  # Normal is 1
            y_test_true_mapped[y_test_true_mapped == 3] = -1 # Anomaly is -1
            
            # --- Train the Isolation Forest Model ---
            logging.info("Training Isolation Forest model.")
            # 'contamination' is the expected proportion of anomalies in the data.
            # We know from our data analysis that about 19% of our data is anomalous (type 3).
            iso_forest = IsolationForest(n_estimators=100, contamination=0.19, random_state=42, n_jobs=-1)
            iso_forest.fit(X_train)
            
            # --- Evaluate the Model ---
            logging.info("Evaluating the model on the test set.")
            y_pred_test = iso_forest.predict(X_test)

            accuracy = accuracy_score(y_test_true_mapped, y_pred_test)
            
            print("\n--- Unsupervised Model Performance Report ---")
            print(f"Model: Isolation Forest")
            print(f"  - Accuracy: {accuracy:.4f}")
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test_true_mapped, y_pred_test))
            print("\nClassification Report:")
            # Note: In the report, '1' represents Normal points, '-1' represents Anomalies.
            print(classification_report(y_test_true_mapped, y_pred_test))
            print("-------------------------------------------\n")

            # --- Save the Model ---
            logging.info("Saving the trained Isolation Forest model.")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=iso_forest
            )
            logging.info(f"Unsupervised model saved to {self.model_trainer_config.trained_model_file_path}")

            return accuracy

        except Exception as e:
            logging.error(f"Error in unsupervised model training: {e}")
            raise CustomException(e, sys)

# This block allows you to test this script independently
if __name__ == "__main__":
    setup_logging()
    
    # Import the previous pipeline components here
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    # Run the full pipeline up to this point
    ingestion = DataIngestion()
    train_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Use the unsupervised data transformation script
    transformer = DataTransformation()
    train_arr, test_arr, y_train, y_test, _ = transformer.initiate_data_transformation(train_data_path, test_data_path)

    # Initiate model training
    trainer = UnsupervisedModelTrainer()
    final_score = trainer.initiate_model_training(train_arr, test_arr, y_train, y_test)
    
    print("\n--- Unsupervised Model Training Completed ---")
    print(f"The final accuracy of the model is: {final_score:.4f}")
    print("---------------------------------------------")
