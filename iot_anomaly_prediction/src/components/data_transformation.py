import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

@dataclass
class DataTransformationConfig:
    scaler_obj_file_path: str = os.path.join('saved_models', 'predictor_scaler.pkl')
    # Configuration for sequence creation
    lookback_period: int = 60
    prediction_horizon_minutes: int = 5
    latency_col_name: str = 'Latency_ms'

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        logging.info("Data Transformation component initialized.")

    def initiate_data_transformation(self, data_path):
        try:
            logging.info("Starting data transformation.")
            df = pd.read_csv(data_path)

            # --- 1. Chronological Sorting ---
            df = df.sort_values(by='Timestamp').reset_index(drop=True)
            logging.info("Data sorted by Timestamp.")

            # --- 2. Feature Selection and Scaling ---
            df_numeric = df.select_dtypes(include=np.number).drop(columns=['Timestamp'])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_numeric)
            df_scaled = pd.DataFrame(X_scaled, columns=df_numeric.columns)
            logging.info("Numerical features scaled with StandardScaler.")

            # Save the scaler object for future predictions
            save_object(self.config.scaler_obj_file_path, scaler)

            # --- 3. Sequence and Target Creation ---
            logging.info("Creating time-series sequences and future targets.")
            time_diffs = df['Timestamp'].diff().mean()
            data_frequency_hz = 1 / time_diffs if time_diffs > 0 else 1.0
            
            horizon_steps = int(self.config.prediction_horizon_minutes * 60 * data_frequency_hz)
            logging.info(f"Prediction horizon set to {horizon_steps} steps ({self.config.prediction_horizon_minutes} mins).")

            X, y = [], []
            anomaly_threshold = df_scaled[self.config.latency_col_name].quantile(0.95)
            future_max_latency = df_scaled[self.config.latency_col_name].rolling(window=horizon_steps, min_periods=1).max().shift(-horizon_steps)
            future_target = (future_max_latency > anomaly_threshold).astype(int)

            for i in range(len(df_scaled) - self.config.lookback_period - horizon_steps):
                X.append(df_scaled.iloc[i:(i + self.config.lookback_period)].values)
                y.append(future_target.iloc[i + self.config.lookback_period])
            
            X_sequences, y_sequences = np.array(X), np.array(y)
            logging.info(f"Created {len(X_sequences)} sequences.")
            
            return X_sequences, y_sequences

        except Exception as e:
            raise CustomException(e, sys)

