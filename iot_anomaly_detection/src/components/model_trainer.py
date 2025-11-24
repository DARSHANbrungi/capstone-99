import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from joblib import dump

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("saved_models", "detector_model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_data_pca, proxy_target):
        """
        Trains the Isolation Forest model, evaluates its performance against
        a proxy target, and saves the trained model artifact.
        """
        try:
            logging.info("Starting model training with Isolation Forest.")
            
            # Initialize the Isolation Forest model with optimized parameters
            iso_forest = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
            
            # Train the model on the PCA-transformed data
            iso_forest.fit(train_data_pca)
            logging.info("Model training complete.")
            
            # --- Performance Evaluation Step ---
            logging.info("Evaluating model performance against the proxy target.")
            
            # Get predictions from the trained model
            # Note: predict() returns -1 for outliers and 1 for inliers
            predictions = iso_forest.predict(train_data_pca)
            
            # Convert predictions to our standard format (1 for outlier, 0 for inlier)
            predictions_formatted = [1 if x == -1 else 0 for x in predictions]

            # Calculate the percentage of data flagged as an anomaly
            anomaly_percentage = (sum(predictions_formatted) / len(predictions_formatted)) * 100
            logging.info(f"The model identified {anomaly_percentage:.2f}% of the data as anomalies.")
            
            # Generate and log the full classification report
            report = classification_report(proxy_target, predictions_formatted, target_names=['Normal', 'Anomaly'])
            logging.info(f"\n--- Classification Report vs. Proxy Target ---\n{report}\n-------------------------------------------------")
            
            # --- Save the Trained Model ---
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=iso_forest
            )
            logging.info(f"Trained Isolation Forest model saved to: {self.model_trainer_config.trained_model_file_path}")
            
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)