import os
import sys
import pickle
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# After installing the project with 'pip install -e .', we can use direct imports.
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains and evaluates a dictionary of models, returning a detailed report.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)

            logging.info(f"Evaluating model: {model_name}")
            y_test_pred = model.predict(X_test)

            # Calculate evaluation metrics
            test_model_accuracy = accuracy_score(y_test, y_test_pred)
            test_model_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Store both metrics in the report
            report[model_name] = {
                'accuracy': test_model_accuracy,
                'f1_score': test_model_f1
            }

        return report

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)

