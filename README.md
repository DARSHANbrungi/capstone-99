# 🚀 Predictive Analysis of Communication Issues in IoT Systems Using Machine Learning

## 📌 Project Overview

This project addresses critical reliability challenges in Internet of Things (IoT) ecosystems. By bridging the gap between asynchronous IoT protocols (MQTT) and synchronous monitoring tools (Prometheus), it provides a robust, end-to-end framework for:

- **Real-Time Anomaly Detection**
  Using an Isolation Forest model to instantly flag communication failures.

- **Predictive Maintenance**
  Using a GRU (Gated Recurrent Unit) deep learning model to forecast system health 60 seconds in advance.

- **Root Cause Analysis (RCA)**
  Automated diagnosis of failure origin: Power, Network, Transport, or Application layer.

The system integrates hardware sensors, a custom Stateful Exporter, Prometheus, and Grafana, delivering a unified Single Pane of Glass dashboard.

## 📂 Project Structure

```
Capstone99-main/
├── Capstone-main/                  # Original repository
├── RCA_engine/                     # Root Cause Analysis module (rca_app.py)
├── frontend/                       # Web/React frontend (optional)
├── grafana-12.2.1/                 # Grafana installation
├── iot_anomaly_detection/          # Detection model + API (app.py)
│   ├── src/                        # Pipeline + utilities
│   ├── saved_models/               # Isolation Forest model (.pkl)
│   └── app.py                      # Flask API for real-time detection
├── iot_anomaly_prediction/         # GRU prediction model + API
│   ├── src/
│   ├── saved_models/               # GRU (.pth) + scaler (.pkl)
│   └── prediction_app.py           # Flask API (60s forecasting)
├── prometheus/
│   ├── prometheus.exe
│   └── prometheus.yml              # Scrape configuration
├── mqtt_exporter.py                # Stateful MQTT → Prometheus exporter
├── create_project_structure.py
├── deep_analysis.py
└── README.md
```

## ⚙️ Architecture & Workflow

### 1. Edge Layer (Raspberry Pi Pico W)
- Reads sensor data: DHT22, BMP280, MQ135
- Syncs time via NTP
- Publishes telemetry + heartbeat to MQTT broker

### 2. Transport Layer (Mosquitto)
- Central MQTT message bus

### 3. Bridge Layer (mqtt_exporter.py)
- Subscribes to telemetry topics
- Maintains global state of:
  - CPU
  - Latency
  - Jitter
  - Packet loss
  - Heartbeat timestamps
- Exposes a dense vector at: `http://localhost:8000/metrics`

### 4. Storage Layer (Prometheus)
- Scrapes exporter every 5 seconds
- Stores time-series metrics

### 5. Intelligence Layer
- **Anomaly Detection (app.py)**: Queries Prometheus → Detects anomaly → Exposes score
- **Prediction (prediction_app.py)**: Reads last 60s window → Forecasts failure → Exports confidence
- **RCA Engine (rca_app.py)**: Uses detection + prediction → Computes root cause → Exports status code

### 6. Visualization Layer (Grafana)
- Displays real-time metrics, anomaly scores, predictions, and RCA status

## 🛠️ Prerequisites

### Hardware
- Raspberry Pi Pico W
- Sensors: DHT22, BMP280, MQ135
- Host PC (Windows/Linux)

### Software
- Python 3.8+
- Mosquitto MQTT Broker
- Prometheus
- Grafana
- Anaconda (recommended)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Capstone99.git
cd Capstone99
```

### 2. Set Up Python Environment
```bash
conda create -n capstone python=3.9
conda activate capstone
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:
```text
flask
flask_cors
pandas
numpy
torch
joblib
paho-mqtt
psutil
prometheus-client
requests
```

### 3. Hardware Setup (Pico W)
- Flash Pico W with MicroPython
- Upload:
  - `main.py`
  - `config.py`
- Update `config.py`:
  - Wi-Fi SSID + password
  - Host PC IP
- Run `main.py`

### 4. Start the MQTT Broker
**Windows:**
- Open `services.msc`
- Start Mosquitto Broker

**Linux:**
```bash
sudo systemctl start mosquitto
```

### 5. Start All Services (Separate Terminals)

**Terminal 1: MQTT Exporter (Port 8000)**
```bash
python mqtt_exporter.py
```

**Terminal 2: Prometheus (Port 9090)**
```bash
cd prometheus
./prometheus.exe --config.file=prometheus.yml
```

**Terminal 3: Anomaly Detection API (Port 5001)**
```bash
cd iot_anomaly_detection
python app.py
```

**Terminal 4: Prediction API (Port 5002)**
```bash
cd iot_anomaly_prediction
python prediction_app.py
```

**Terminal 5: RCA Engine (Port 5003)**
```bash
cd RCA_engine
python rca_app.py
```

**Terminal 6: Grafana Dashboard (Port 3000)**
```bash
cd grafana-12.2.1/bin
./grafana-server.exe
```

## 📊 Grafana Dashboard Setup

1. **Open:** `http://localhost:3000`
2. **Login:** `admin` / `admin`
3. **Add Prometheus Data Source:**
   - Go to **Connections → Data Sources**
   - Select **Prometheus**
   - URL: `http://localhost:9090`
   - Save & Test
4. **Create Dashboard Panels:**
   - Anomaly Score → `detection_anomaly_score`
   - Prediction Confidence → `prediction_confidence_score`
   - RCA Status → `rca_status_code`
   - Map:
     - 0 = Healthy
     - 1 = Power
     - 2 = Network
     - 3 = Transport
     - 4 = Application
   - Live system metrics:
     - `cpu_utilization_percent`
     - `latency_ms`
     - `jitter_ms`

## 🔍 Troubleshooting

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Predictions stuck | Latency outliers | Add fix: `data_dict['Latency_ms'] = 0.0` |
| Sparse Data in Model | Exporter sending sparse metrics | Use updated Stateful Exporter |
| Connection refused | Incorrect IP or firewall | Check IP & allow ports 1883, 8000, 9090 |
| Grafana read-only | Browser cache issue | Ctrl+F5, logout/login |

## 👥 Contributors

- Darshan B M
- Channaveer Upase
- Dattaram
- Charan V
