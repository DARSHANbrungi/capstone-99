import os
import sys
import pandas as pd
import requests  # <-- ADDED
import time      # <-- ADDED
from flask import Flask, jsonify
from flask_cors import CORS

# --- 1. ADD THESE IMPORTS ---
from prometheus_client import Gauge, make_wsgi_app, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware
# ------------------------------

# Add the project root to the system path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Robustly import custom modules and provide clear error messages
try:
    from src.pipeline.prediction_pipeline import PredictPipeline
    from src.logger import logging
    print("Successfully imported custom modules.")
except ImportError as e:
    print(f"Failed to import custom modules. Error: {e}")
    print("Please ensure you have run 'pip install -e .' in your environment.")
    sys.exit(1)

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- 2. ADD PROMETHEUS METRICS ---
# Create gauges to hold the scores.
DETECTION_ANOMALY_GAUGE = Gauge('detection_is_anomaly', '1 if an anomaly is detected, 0 otherwise')
DETECTION_SCORE_GAUGE = Gauge('detection_anomaly_score', 'The raw anomaly score from the detection model')
# ---------------------------------

# --- 3. ADD A /metrics ENDPOINT ---
# This makes Flask serve its main app AND the /metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app(REGISTRY)
})
# ----------------------------------

print("Flask app initialized.")
logging.info("Flask app initialized.")

# --- Prometheus & Model Configuration ---
PROMETHEUS_URL = "http://localhost:9090"
print(f"Connecting to Prometheus at: {PROMETHEUS_URL}")

# ... (all your other configuration and helper functions like) ...
# CLEANED_METRIC_KEYS = [...]
# CSV_HEADERS = [...]
# pipeline = PredictPipeline()
# def get_prometheus_data():
# ... (this all stays the same) ...
CLEANED_METRIC_KEYS = [
    'Message_Size_Bytes', 'Latency_ms', 'Jitter_ms', 'Packet_Loss_Percent',
    'RTT_ms', 'Throughput_BytesPerSec', 'TCP_Retransmissions', 'Interface_Errors',
    'Link_Speed_Mbps', 'MQTT_Connection_State', 'MQTT_Message_Queue_Size',
    'QoS_Level', 'QoS_Success_Rate', 'Messages_Per_Minute', 'Failed_Delivery_Count',
    'CPU_Utilization_Percent', 'Memory_Usage_Percent', 'System_Load',
    'Network_Buffer_Status', 'Moving_Avg_Latency_ms', 'Rate_of_Change_Latency',
    'Sender_CPU_Freq', 'Sender_Memory_Percent', 'Sender_Reset_Cause',
    'Time_of_Day_Seconds', 'RP_Sensor_Value', 'RP_RSSI', 'RP_LQI',
    'RP_Sequence_Num', 'RP_Gateway_ID', 'RP_QoS_Level'
]
CSV_HEADERS = [
    "Timestamp", "Time_of_Day", "Sensor_ID", "Message_ID", "Published_Payload",
    "Received_Payload", "Message_Size_Bytes", "Latency_ms", "Jitter_ms",
    "Packet_Loss_Percent", "RTT_ms", "Throughput_BytesPerSec", "TCP_Retransmissions",
    "Interface_Errors", "Link_Speed_Mbps", "MQTT_Connection_State",
    "MQTT_Message_Queue_Size", "QoS_Level", "QoS_Success_Rate",
    "Messages_Per_Minute", "Failed_Delivery_Count", "CPU_Utilization_Percent",
    "Memory_Usage_Percent", "System_Load", "Network_Buffer_Status",
    "Network_Condition", "Moving_Avg_Latency_ms", "Rate_of_Change_Latency",
    "Sender_CPU_Freq", "Sender_Memory_Percent", "Sender_Reset_Cause",
    "Communication_Issue_Type", "Topic",
]
try:
    pipeline = PredictPipeline()
    print("Prediction pipeline initialized successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize prediction pipeline: {e}")
    pipeline = None

def get_prometheus_data():
    """
    Queries Prometheus for the latest set of metrics.
    """
    try:
        query = 'cpu_utilization_percent{job="iot_subscriber"}'
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        response.raise_for_status()
        results = response.json()['data']['result']
        if not results:
            raise Exception("No data found in Prometheus for key metric 'cpu_utilization_percent'")
        
        labels = results[0]['metric']
        if '__name__' in labels:
            del labels['__name__']
        
        label_query_string = ",".join([f'{k}="{v}"' for k, v in labels.items()])
        label_query = f"{{{label_query_string}}}"
        print(f"Found active sensor with labels: {label_query}")
    except Exception as e:
        print(f"ERROR: Failed to query Prometheus for active sensor. {e}")
        return None, None

    cleaned_data_map = {}
    cleaned_data_map.update(labels)
    full_query = label_query
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': full_query})
        response.raise_for_status()
        results = response.json()['data']['result']
        
        for metric in results:
            metric_name = metric['metric']['__name__']
            metric_value = float(metric['value'][1])
            for original_key in CLEANED_METRIC_KEYS:
                if original_key.lower().replace('%', 'percent').replace('-', '_') == metric_name:
                    cleaned_data_map[original_key] = metric_value
                    break
    except Exception as e:
        print(f"Error querying all metrics: {e}")
        return None, None
        
    for key in CLEANED_METRIC_KEYS:
        if key not in cleaned_data_map:
            cleaned_data_map[key] = 0.0
            
    raw_data_for_pipeline = {}
    for col in CSV_HEADERS:
        raw_data_for_pipeline[col] = cleaned_data_map.get(col, 0)
        
    try:
        raw_data_for_pipeline['Time_of_Day'] = time.strftime(
            '%H:%M:%S', time.gmtime(cleaned_data_map.get('Time_of_Day_Seconds', 0))
        )
        raw_data_for_pipeline['Received_Payload'] = (
            f"0,0,0,"
            f"{cleaned_data_map.get('RP_Sensor_Value', 0)},"
            f"{cleaned_data_map.get('RP_RSSI', 0)},"
            f"{cleaned_data_map.get('RP_LQI', 0)},"
            f"{cleaned_data_map.get('RP_Sequence_Num', 0)},"
            f"{cleaned_data_map.get('RP_Gateway_ID', 0)},"
            f"{cleaned_data_map.get('RP_QoS_Level', 0)}"
        )
        raw_data_for_pipeline['Sensor_ID'] = 'live_sensor'
        raw_data_for_pipeline['Network_Condition'] = 'unknown'
    except Exception as e:
        print(f"Error reverse-engineering raw data: {e}")
        return None, None

    return raw_data_for_pipeline, cleaned_data_map


# --- API Endpoints ---
@app.route('/api/get_live_data', methods=['GET'])
def get_live_data():
    logging.info("API endpoint /api/get_live_data was hit!")
    
    if pipeline is None:
        return jsonify({"error": "Prediction pipeline is not available."}), 500
        
    try:
        raw_data, kpi_data = get_prometheus_data()
        if raw_data is None:
            return jsonify({"error": "Failed to retrieve live data from Prometheus."}), 500

        current_row_df = pd.DataFrame([raw_data])
        current_row_df = current_row_df[CSV_HEADERS]
        
        prediction, score = pipeline.predict(current_row_df)
        
        # --- 4. SET THE GAUGE VALUES! ---
        # This makes the values available to Prometheus
        score_percentage = score * 100
        DETECTION_ANOMALY_GAUGE.set(prediction)
        DETECTION_SCORE_GAUGE.set(score_percentage)
        # ----------------------------------
        
        response_data = {
            "is_anomaly": bool(prediction),
            "anomaly_score": float(score),
            "kpi_data": {
                "Latency_ms": kpi_data.get('Latency_ms', 0),
                "CPU_Utilization_Percent": kpi_data.get('CPU_Utilization_Percent', 0),
                "TCP_Retransmissions": kpi_data.get('TCP_Retransmissions', 0),
                "Jitter_ms": kpi_data.get('Jitter_ms', 0)
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in API endpoint: {e}")
        print(f"ERROR in API endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# --- Main Execution ---
if __name__ == "__main__":
    print("Script execution has reached the main block.")
    logging.info("Starting Flask API server...")
    # Using port 5001 as we did to fix the port conflict
    app.run(host="0.0.0.0", port=5001, debug=True)