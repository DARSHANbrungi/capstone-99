import os
import sys
import pandas as pd
from joblib import load

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        try:
            # Construct the full paths to the saved model and preprocessor
            self.model_path = os.path.join(project_root, "saved_models", "detector_model.pkl")
            self.preprocessor_path = os.path.join(project_root, "saved_models", "preprocessor.pkl")
            
            # Load the model and preprocessor objects
            self.model = load(self.model_path)
            self.preprocessor = load(self.preprocessor_path)
            logging.info("Model and preprocessor loaded successfully for prediction.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df):
        """
        Takes a pandas DataFrame of raw features and returns a prediction.
        """
        try:
            # --- Data Transformation ---
            # Select only the numerical features for scaling and PCA
            numerical_features = features_df.select_dtypes(include=['number'])
            
            # Apply the saved preprocessor (scaler + PCA)
            data_scaled = self.preprocessor['scaler'].transform(numerical_features)
            data_pca = self.preprocessor['pca'].transform(data_scaled)
            
            # --- Model Prediction ---
            # Get the raw prediction from Isolation Forest (-1 for anomaly, 1 for normal)
            prediction = self.model.predict(data_pca)
            
            # Get the anomaly score
            # decision_function returns the anomaly score. Lower is more anomalous.
            # We invert it so that higher scores mean more anomalous.
            score = -self.model.decision_function(data_pca)

            # Convert to our standard format (1 for anomaly, 0 for normal)
            final_prediction = 1 if prediction[0] == -1 else 0
            
            return final_prediction, score[0]

        except Exception as e:
            raise CustomException(e, sys)

