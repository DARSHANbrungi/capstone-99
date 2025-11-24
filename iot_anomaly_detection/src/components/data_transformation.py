import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
from src.utils.utils import save_object # Ensure you have this utility function

@dataclass
class DataTransformationConfig:
    # This file will store both the scaler and PCA model
    preprocessor_obj_file_path: str = os.path.join('saved_models', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a simple preprocessing pipeline that only scales the data.
        PCA will be handled separately but as part of the same overall object.
        """
        try:
            scaling_pipeline = ColumnTransformer(
                transformers=[('scaler', StandardScaler(), slice(0, None))],
                remainder='passthrough'
            )
            return scaling_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, data_path):
        """
        Loads data, applies scaling and PCA, saves the combined preprocessor,
        and returns the final data and proxy target.
        """
        try:
            df = pd.read_csv(data_path)
            logging.info(f"Loaded data from {data_path} for transformation.")

            # Drop redundant/unnecessary features first
            if 'Published_Payload' in df.columns:
                df.drop(columns=['Published_Payload'], inplace=True, errors='ignore')
            
            # Create the proxy target for performance evaluation
            proxy_target = (df['Latency_ms'] > df['Latency_ms'].quantile(0.95)).astype(int)
            logging.info("Created proxy target based on high latency.")

            numerical_features = df.select_dtypes(include=np.number)
            
            # Get the scaling pipeline and apply it
            scaling_pipeline = self.get_data_transformer_object()
            data_scaled = scaling_pipeline.fit_transform(numerical_features)
            logging.info("Data scaling complete.")

            # Apply PCA
            pca = PCA(n_components=0.95)
            data_pca = pca.fit_transform(data_scaled)
            logging.info(f"PCA complete. Retained {pca.n_components_} components.")

            # Save the full preprocessor (scaler + PCA)
            full_preprocessor_obj = {'scaler': scaling_pipeline, 'pca': pca}
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=full_preprocessor_obj
            )
            logging.info(f"Preprocessor (Scaler+PCA) saved to {self.data_transformation_config.preprocessor_obj_file_path}")

            return data_pca, proxy_target

        except Exception as e:
            raise CustomException(e, sys)