import requests
import time
import threading
from flask import Flask, jsonify
from prometheus_client import Gauge, make_wsgi_app, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)
PROMETHEUS_URL = "http://localhost:9090"

# --- RCA Metrics Exported to Prometheus ---
# 0=Healthy
# 1=Power Instability
# 2=Network/Signal Interference
# 3=TCP Congestion
# 4=Application Blocking
# 5=System Dead (No Heartbeat)
RCA_STATUS_CODE = Gauge('rca_status_code', 'Current Root Cause Code')
RCA_ACTIVE = Gauge('rca_is_active', '1 if diagnostics are running')

# --- Global State ---
current_diagnosis = {
    "status": "Healthy",
    "code": 0,
    "details": "System stable."
}

# Counters for Hysteresis (Level 1 Improvement)
# Prevents flickering by requiring 3 consecutive failures
error_counters = {
    "dead": 0,
    "power": 0,
    "network": 0,
    "tcp": 0,
    "app": 0
}
THRESHOLD_COUNT = 3  # 3 checks * 5 seconds = 15 seconds persistence

# Add /metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': make_wsgi_app(REGISTRY)})

def get_metric(name):
    """Helper to get the latest value of a metric from Prometheus"""
    try:
        # Query without labels to get the aggregate/dense value
        query = name.lower().replace('%', 'percent').replace('-', '_')
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        result = response.json()['data']['result']
        if result:
            return float(result[0]['value'][1])
    except:
        pass
    return 0.0

def rca_logic_engine():
    global current_diagnosis, error_counters
    print("RCA Engine Started (Prediction-Triggered Mode)...")
    
    while True:
        # 1. Gather Evidence
        # We look at the PREDICTION score first (Level 3 Improvement)
        pred_confidence = get_metric('prediction_confidence_score') # 0-100 scale
        
        # Also gather raw metrics for diagnosis
        time_since_hb = get_metric('time_since_last_heartbeat_seconds')
        reset_cause = get_metric('sender_reset_cause')
        mqtt_state = get_metric('mqtt_connection_state')
        packet_loss = get_metric('packet_loss_percent')
        rssi = get_metric('rp_rssi')
        retransmissions = get_metric('tcp_retransmissions')
        sender_cpu = get_metric('sender_cpu_freq') 

        # Debug print
        print(f"--- RCA Cycle ---")
        print(f"Prediction Confidence: {pred_confidence:.2f}%")
        print(f"Heartbeat Age: {time_since_hb}s")
        
        # --- LEVEL 3: PREDICTION TRIGGER ---
        # Only run full diagnostics if the model predicts an anomaly (>80% confidence)
        # OR if the system is actually dead (heartbeat missing)
        is_emergency = (time_since_hb > 15.0) or (pred_confidence > 80.0)
        
        if not is_emergency:
            # Reset everything if system is predicted healthy
            for key in error_counters: error_counters[key] = 0
            new_code = 0
            new_status = "Healthy"
            new_details = "Prediction model indicates normal operation."
            
        else:
            # --- LEVEL 2: CORRELATION LOGIC ---
            
            # 1. DEAD CHECK (Heartbeat)
            if time_since_hb > 15.0:
                error_counters["dead"] += 1
            else:
                error_counters["dead"] = 0

            # 2. PHYSICAL LAYER (Power)
            if reset_cause > 0 or mqtt_state != 0:
                error_counters["power"] += 1
            else:
                error_counters["power"] = 0

            # 3. NETWORK LAYER (Wi-Fi)
            if packet_loss > 5.0 and rssi > -75:
                error_counters["network"] += 1
            else:
                error_counters["network"] = 0

            # 4. TRANSPORT LAYER (TCP)
            if retransmissions > 5:
                error_counters["tcp"] += 1
            else:
                error_counters["tcp"] = 0

            # 5. APPLICATION LAYER (Code)
            if sender_cpu > 125000000: 
                error_counters["app"] += 1
            else:
                error_counters["app"] = 0
            
            # --- LEVEL 1: HYSTERESIS CHECK ---
            # Determine final status based on sustained counters
            new_code = 0
            new_status = "Investigating..."
            new_details = "Anomalous pattern detected. Gathering more samples..."

            if error_counters["dead"] >= THRESHOLD_COUNT:
                new_code = 5
                new_status = "CRITICAL: System Dead"
                new_details = f"No heartbeat for {int(time_since_hb)}s. Device powered off?"
            elif error_counters["power"] >= THRESHOLD_COUNT:
                new_code = 1
                new_status = "Physical: Power Instability"
                new_details = "Sustained connection flapping or reset detected."
            elif error_counters["network"] >= THRESHOLD_COUNT:
                new_code = 2
                new_status = "Network: Signal Interference"
                new_details = "High packet loss despite good RSSI signal."
            elif error_counters["tcp"] >= THRESHOLD_COUNT:
                new_code = 3
                new_status = "Transport: TCP Congestion"
                new_details = "Sustained high TCP retransmissions."
            elif error_counters["app"] >= THRESHOLD_COUNT:
                new_code = 4
                new_status = "App: Blocking Code"
                new_details = "High CPU usage suggests code hanging."
            else:
                # If prediction is high but no specific cause found yet
                if pred_confidence > 80.0:
                    new_code = 6 # New code for "Unknown/Predictive"
                    new_status = "Warning: High Failure Probability"
                    new_details = "Model predicts failure, but root cause not yet isolated."

        # Update Prometheus & API State
        RCA_STATUS_CODE.set(new_code)
        current_diagnosis = {"status": new_status, "code": new_code, "details": new_details}
        
        if new_code != 0:
            print(f"DIAGNOSIS: {new_status} | {new_details}")
            RCA_ACTIVE.set(1)
        else:
            RCA_ACTIVE.set(0)

        time.sleep(5) # Run diagnosis every 5 seconds

@app.route('/api/diagnosis', methods=['GET'])
def get_diagnosis():
    return jsonify(current_diagnosis)

if __name__ == "__main__":
    # Start the monitoring thread
    monitor = threading.Thread(target=rca_logic_engine)
    monitor.daemon = True
    monitor.start()
    
    print("RCA API running on port 5003...")
    app.run(host="0.0.0.0", port=5003, debug=True)