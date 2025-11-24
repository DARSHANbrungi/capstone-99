import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import requests
import time
import random
from flask import Flask, jsonify
from flask_cors import CORS

# --- 1. ADD THESE IMPORTS ---
from prometheus_client import Gauge, make_wsgi_app, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware
# ------------------------------

from joblib import load
from collections import deque

# MODIFIED: Go up one level to the 'CAPSTONE PROJECT' root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Also keep the current dir

try:
    from src.logger import logging
    from src.models.predictor_model import GRUPredictor
    print("Successfully imported custom modules.")
    logging.info("Successfully imported custom modules.")
except ImportError as e:
    print(f"Failed to import custom modules. Error: {e}")
    logging.error(f"Failed to import custom modules: {e}")
    sys.exit(1)

app = Flask(__name__)
# Match the name from your log
app.name = 'app' 
CORS(app)

# --- 2. ADD PROMETHEUS METRICS ---
PREDICTION_RESULT_GAUGE = Gauge('prediction_result', 'The 0 or 1 result from the prediction model')
PREDICTION_CONFIDENCE_GAUGE = Gauge('prediction_confidence_score', 'The confidence score from the prediction model')
# ---------------------------------

# --- 3. ADD A /metrics ENDPOINT ---
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app(REGISTRY)
})
# ----------------------------------


# --- Configuration ---
LOOKBACK_PERIOD = 60  # How many seconds of data to query
PROMETHEUS_URL = "http://localhost:9090"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
logging.info(f"Using device: {device}")
print(f"Connecting to Prometheus at: {PROMETHEUS_URL}")

# This is the list of *all* features your forecasting model expects
# We will get this directly from the scaler
MODEL_FEATURE_NAMES = [] 

# --- Load Model and Scaler ---
try:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    SCALER_PATH = os.path.join(current_dir, 'saved_models', 'predictor_scaler.pkl')
    scaler = load(SCALER_PATH)
    print("Predictor scaler loaded successfully.")
    logging.info("Predictor scaler loaded successfully.")
    
    MODEL_PATH = os.path.join(current_dir, 'saved_models', 'predictor_model.pth')
    
    # Get the feature names *from the scaler*
    # This is the source of truth for our data
    MODEL_FEATURE_NAMES = scaler.feature_names_in_
    input_dim = len(MODEL_FEATURE_NAMES)
    
    model = GRUPredictor(input_dim=input_dim).to(device)
    
    # --- FIX FOR FUTUREWARNING ---
    # Set weights_only=True for security and to remove the warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    # -----------------------------
    
    model.eval()
    print(f"Predictor model loaded successfully with {input_dim} features.")
    logging.info("Predictor model loaded successfully.")
    
except Exception as e:
    print(f"CRITICAL: Failed to load model or scaler: {e}")
    logging.error(f"Failed to load model or scaler: {e}")
    scaler = None
    model = None


# --- NEW PROMETHEUS HELPER FUNCTION ---
def get_prometheus_range_data():
    """
    Queries Prometheus for the last 60 seconds of all metrics.
    Returns:
        - A pandas DataFrame with timestamps as the index
          and features as the columns.
    """
    
    # 1. Define the time window
    end_time = time.time()
    start_time = end_time - LOOKBACK_PERIOD  # 60 seconds ago
    step = '1s' # Get one data point per second

    # 2. Get the labels for the active instance
    try:
        query = 'cpu_utilization_percent{job="iot_subscriber"}'
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        response.raise_for_status()
        results = response.json()['data']['result']
        if not results:
            raise Exception("No data for 'cpu_utilization_percent' found.")
        
        labels_to_query = results[0]['metric']
        labels_to_query.pop('__name__', None)
        label_query = "{" + ",".join([f'{k}="{v}"' for k, v in labels_to_query.items()]) + "}"
    
    except Exception as e:
        print(f"ERROR: Failed to query Prometheus for active sensor. {e}")
        return None

    # 3. Query ALL metrics over the time range
    full_query = label_query
    print(f"Querying {LOOKBACK_PERIOD}s of data for: {full_query}")
    
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range", 
            params={
                'query': full_query,
                'start': start_time,
                'end': end_time,
                'step': step
            }
        )
        response.raise_for_status()
        results = response.json()['data']['result']
        
        # 4. --- Pivot the JSON data into a DataFrame ---
        data_by_metric = {}
        for metric_data in results:
            metric_name_prom = metric_data['metric']['__name__']
            
            # Find the "pretty" name (e.g., 'cpu_utilization_percent' -> 'CPU_Utilization_Percent')
            metric_name = None
            for key in MODEL_FEATURE_NAMES: # Use keys from scaler
                 if key.lower().replace('%', 'percent').replace('-', '_') == metric_name_prom:
                    metric_name = key
                    break
            
            if metric_name:
                # Get all [timestamp, value] pairs
                df = pd.DataFrame(metric_data['values'], columns=['timestamp', metric_name])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df[metric_name] = pd.to_numeric(df[metric_name])
                data_by_metric[metric_name] = df

        if not data_by_metric:
            raise Exception("No data returned from range query.")

        # 5. Combine all metrics into one big DataFrame
        final_df = pd.concat(data_by_metric.values(), axis=1)
        
        # --- THIS IS THE FIX ---
        # 6. Add any missing columns that the scaler expects
        
        # --- NEW: Generate Communication_Issue_Type based on distribution ---
        # Total rows = 212250 + 39231 + 4 = 251485
        # P(0) = 212250 / 251485 = 0.84398
        # P(3) = 39231 / 251485 = 0.15599
        # P(5) = 4 / 251485 = 0.0000159 
        # We use these weights for a random choice:
        weights = [0.84398, 0.15599, 0.00002] # Approximate, sums to 1
        
        # Generate *one* issue type for this 60-second window
        generated_issue_type = random.choices([0, 3, 5], weights=weights, k=1)[0]
        print(f"Simulating Communication_Issue_Type: {generated_issue_type}")
        # -----------------------------------------------------------------

        for col in MODEL_FEATURE_NAMES:
            # --- MODIFIED SECTION ---
            if col == 'Communication_Issue_Type':
                # Add the simulated issue type
                final_df[col] = generated_issue_type
            elif col not in final_df.columns:
                # Fill other missing metrics with 0
                print(f"Warning: Metric '{col}' not found in Prometheus. Filling with 0.")
                final_df[col] = 0.0
            # --- END MODIFIED SECTION ---
        # -----------------------
        
        # 7. Re-order columns to *exactly* match the model's training
        final_df = final_df[MODEL_FEATURE_NAMES]
        
        # 8. Fill any gaps
        final_df = final_df.ffill().bfill().fillna(0) 
        
        if final_df.empty:
            raise Exception("Created DataFrame is empty after processing.")

        return final_df

    except Exception as e:
        print(f"Error querying/pivoting range data: {e}")
        return None

# --- MODIFIED API ENDPOINT ---
@app.route('/api/get_prediction', methods=['GET'])
def get_prediction():
    
    if model is None or scaler is None:
        return jsonify({"error": "Prediction model is not available."}), 500
        
    try:
        # 1. Get the last 60s of data directly from Prometheus
        history_df = get_prometheus_range_data()
        
        if history_df is None or history_df.empty:
             return jsonify({"error": "Failed to retrieve historical data from Prometheus."}), 500

        # 2. Get the features in the exact order the scaler expects
        # (This is already done by get_prometheus_range_data, but we double-check)
        sequence_df = history_df[MODEL_FEATURE_NAMES]
        sequence_np = sequence_df.values
        
        # 3. Scale the entire sequence
        sequence_scaled = scaler.transform(sequence_np)
        
        # 4. Convert to tensor for the model
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # 5. Make prediction
        prediction = 0
        confidence = 0.0
        with torch.no_grad():
            output = model(sequence_tensor)
            confidence = torch.sigmoid(output).item()
            prediction = 1 if confidence > 0.5 else 0

        # --- 4. SET THE GAUGE VALUES! ---
        confidence_percent = confidence * 100.0
        PREDICTION_RESULT_GAUGE.set(prediction)
        PREDICTION_CONFIDENCE_GAUGE.set(confidence_percent)
        # ----------------------------------

        # 6. Get the *very last row* of data to show as KPIs
        current_kpi_series = sequence_df.iloc[-1]
        
        # --- RP_RSSI HACK REMOVED ---
        
        # 7. Format the response
        response_data = {
            "prediction": prediction,
            "confidence": float(confidence),
            "current_kpis": {
                "Latency_ms": float(current_kpi_series['Latency_ms']),
                "CPU_Utilization_Percent": float(current_kpi_series['CPU_Utilization_Percent']),
                "TCP_Retransmissions": int(current_kpi_series['TCP_Retransmissions']),
                # --- THIS IS THE CHANGED LINE ---
                "Jitter_ms": float(current_kpi_series['Jitter_ms'])
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in prediction API endpoint: {e}")
        print(f"ERROR in API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Script execution has reached the main block.")
    logging.info("Starting Flask API server for PREDICTION...")
    
    # Run on port 5002
    app.run(host="0.0.0.0", port=5002, debug=True)