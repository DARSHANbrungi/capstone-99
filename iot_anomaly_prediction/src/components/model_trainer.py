import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass

# Add project root to Python path to allow for package-like imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.exception import CustomException
from src.logger import logging
from src.models.predictor_model import GRUPredictor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("saved_models", "predictor_model.pth")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.001

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        logging.info("Model Trainer component initialized.")
        logging.info(f"Training will be performed on device: {self.config.device}")

    def initiate_model_training(self, X_sequences, y_sequences):
        try:
            logging.info("Splitting data into training and testing sets.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.2, random_state=42, stratify=y_sequences
            )

            # --- Data Setup for PyTorch ---
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=self.config.batch_size * 2)

            # --- Model Initialization ---
            input_dim = X_sequences.shape[2]
            model = GRUPredictor(input_dim=input_dim).to(self.config.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

            # --- Training Loop ---
            logging.info("Starting model training.")
            for epoch in range(self.config.num_epochs):
                model.train()
                total_loss = 0
                for features, labels in train_loader:
                    features, labels = features.to(self.config.device), labels.to(self.config.device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_loader)
                logging.info(f'Epoch [{epoch+1}/{self.config.num_epochs}], Average Training Loss: {avg_loss:.6f}')
            
            logging.info("Model training complete.")

            # --- Evaluation ---
            logging.info("Evaluating model performance on the test set.")
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.config.device)
                    outputs = model(features)
                    preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # --- Detailed Performance Logging ---
            logging.info("--- Detailed Performance Metrics ---")
            
            # Calculate individual metrics
            accuracy = accuracy_score(all_labels, all_preds)
            # We specify pos_label=1 to get the score for the "Future Anomaly" class
            precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
            
            # Log each metric
            logging.info(f"Overall Accuracy: {accuracy*100:.2f}%")
            logging.info(f"Anomaly Precision: {precision*100:.2f}% (How many predicted anomalies were real)")
            logging.info(f"Anomaly Recall: {recall*100:.2f}% (How many real anomalies were caught)")
            logging.info(f"Anomaly F1-Score: {f1*100:.2f}% (Harmonic mean of Precision and Recall)")

            # Log the full classification report for a complete view
            report = classification_report(all_labels, all_preds, target_names=['Future Normal', 'Future Anomaly'])
            logging.info(f"\n--- Full Classification Report ---\n{report}\n----------------------------------")

            # --- Save the Trained Model ---
            torch.save(model.state_dict(), self.config.trained_model_file_path)
            logging.info(f"Trained predictor model saved to: {self.config.trained_model_file_path}")

        except Exception as e:
            raise CustomException(e, sys)
