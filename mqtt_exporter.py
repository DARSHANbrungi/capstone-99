import csv
import datetime
import json
import os
import platform
import socket
import statistics
import subprocess
import threading
import time

import paho.mqtt.client as mqtt
import psutil
from prometheus_client import start_http_server, Gauge

# --- 1. Global State Configuration ---
# This dictionary will hold the *current value* of ALL metrics.
# This is the "stateful" part.
CURRENT_METRIC_STATE = {}
LAST_HEARTBEAT_TIME = time.time()
# We need to list all our *cleaned* feature keys here.
# This must match the list in app.py
CLEANED_METRIC_KEYS = [
    'Message_Size_Bytes', 'Latency_ms', 'Jitter_ms', 'Packet_Loss_Percent',
    'RTT_ms', 'Throughput_BytesPerSec', 'TCP_Retransmissions', 'Interface_Errors',
    'Link_Speed_Mbps', 'MQTT_Connection_State', 'MQTT_Message_Queue_Size',
    'QoS_Level', 'QoS_Success_Rate', 'Messages_Per_Minute', 'Failed_Delivery_Count',
    'CPU_Utilization_Percent', 'Memory_Usage_Percent', 'System_Load',
    'Network_Buffer_Status', 'Moving_Avg_Latency_ms', 'Rate_of_Change_Latency',
    'Sender_CPU_Freq', 'Sender_Memory_Percent', 'Sender_Reset_Cause',
    'Time_of_Day_Seconds', 'RP_Sensor_Value', 'RP_RSSI', 'RP_LQI',
    'RP_Sequence_Num', 'RP_Gateway_ID', 'RP_QoS_Level','Time_Since_Last_Heartbeat_Seconds'
]

# Initialize all keys to 0
for key in CLEANED_METRIC_KEYS:
    CURRENT_METRIC_STATE[key] = 0.0

# We also need the original headers for the cleaning function
csv_headers = [
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

# --- 2. Prometheus Setup ---
# We create ONE gauge for EACH metric, with NO labels.
# This ensures we always expose a single, dense vector.
PROMETHEUS_GAUGES = {}
for key in CLEANED_METRIC_KEYS:
    prom_key = key.lower().replace('%', 'percent').replace('-', '_')
    PROMETHEUS_GAUGES[key] = Gauge(prom_key, f'Live metric for {key}')
print(f"Initialized {len(PROMETHEUS_GAUGES)} Prometheus gauges.")

# ------------------------------------

# MQTT Configuration
MQTT_BROKER = "10.185.144.223" # Make sure this is still correct
MQTT_PORT = 1883
TOPICS = [
    "sensor/dht22/temp",
    "sensor/dht22/humidity",
    "sensor/bmp280/temp",
    "sensor/bmp280/pressure",
    "sensor/mq135/air_quality",
    "system/heartbeat"
]

# Global variables
message_history = {topic: [] for topic in TOPICS}
latency_history = {topic: [] for topic in TOPICS}
message_counters = {topic: 0 for topic in TOPICS}
message_per_minute = {topic: 0 for topic in TOPICS}
failed_deliveries = {topic: 0 for topic in TOPICS}

#
# --- ALL YOUR ORIGINAL HELPER FUNCTIONS ---
#
def get_network_interface():
    """Determine the main network interface"""
    if platform.system() == "Windows":
        return "Ethernet"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((MQTT_BROKER, MQTT_PORT))
        ip = s.getsockname()[0]
        s.close()
        if os.path.exists("/proc/net/route"):
            with open("/proc/net/route", "r") as f:
                for line in f.readlines()[1:]:
                    fields = line.strip().split()
                    if fields[1] == "00000000":
                        return fields[0]
        return "eth0"
    except:
        return "eth0"

NETWORK_INTERFACE = get_network_interface()
print(f"Using network interface: {NETWORK_INTERFACE}")

def get_cpu_usage():
    return psutil.cpu_percent(interval=None) # Changed interval to non-blocking

def get_memory_usage():
    return psutil.virtual_memory().percent

def get_system_load():
    try:
        if platform.system() == "Windows":
            return psutil.cpu_percent()
        else:
            return os.getloadavg()[0]
    except:
        return psutil.cpu_percent()

def get_interface_stats():
    """Get network interface statistics"""
    try:
        if platform.system() == "Windows":
            net_io = psutil.net_io_counters(pernic=True).get(NETWORK_INTERFACE)
            if net_io:
                errors = net_io.errin + net_io.errout
                return errors, 1000
        else:
            try:
                cmd = subprocess.run(
                    ["ethtool", NETWORK_INTERFACE], capture_output=True, text=True
                )
                output = cmd.stdout
                speed = 1000
                for line in output.split("\n"):
                    if "Speed" in line:
                        speed_str = line.split(":")[1].strip()
                        if "Mb/s" in speed_str:
                            try:
                                speed = int(speed_str.replace("Mb/s", ""))
                            except:
                                pass
                if os.path.exists("/sbin/ifconfig"):
                    cmd = subprocess.run(
                        ["ifconfig", NETWORK_INTERFACE], capture_output=True, text=True
                    )
                    output = cmd.stdout
                    errors = 0
                    for line in output.split("\n"):
                        if "errors" in line.lower():
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part.lower() == "errors:" and i + 1 < len(parts):
                                    try:
                                        errors = int(parts[i + 1])
                                    except:
                                        pass
                    return errors, speed
            except:
                net_io = psutil.net_io_counters(
                    pernic=True).get(NETWORK_INTERFACE)
                if net_io:
                    errors = net_io.errin + net_io.errout
                    return errors, 1000
        return 0, 1000
    except Exception as e:
        print(f"Error getting interface stats: {e}")
        return 0, 1000

def get_tcp_retransmissions():
    """Get TCP retransmission count"""
    try:
        if platform.system() == "Linux":
            cmd = subprocess.run(
                ["netstat", "-s"], capture_output=True, text=True)
            output = cmd.stdout
            for line in output.split("\n"):
                if "retransmitted" in line.lower() and "segments" in line.lower():
                    parts = line.strip().split()
                    try:
                        return int(parts[0])
                    except:
                        pass
        return 0
    except:
        return 0

def measure_rtt():
    """Measure round trip time to the MQTT broker"""
    try:
        ping_count = "1"
        ping_cmd = []
        if platform.system() == "Windows":
            ping_cmd = ["ping", "-n", ping_count, MQTT_BROKER, "-w", "1000"] # 1 sec timeout
        else:
            ping_cmd = ["ping", "-c", ping_count, "-W", "1", MQTT_BROKER] # 1 sec timeout
        
        result = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=2)
        
        output = result.stdout
        rtt = 0
        if "time=" in output:
            for line in output.split("\n"):
                if "time=" in line:
                    try:
                        time_part = line.split("time=")[1].split()[0]
                        rtt = float(time_part.replace("ms", ""))
                        break
                    except:
                        pass
        return rtt
    except Exception as e:
        # print(f"Error measuring RTT: {e}") # Too noisy
        return 0 # Return 0 on timeout or error

def calculate_throughput(prev_bytes, curr_bytes, time_diff):
    if time_diff <= 0: return 0
    return (curr_bytes - prev_bytes) / time_diff

def calculate_jitter(latencies):
    if len(latencies) < 2: return 0
    differences = [abs(latencies[i] - latencies[i - 1]) for i in range(1, len(latencies))]
    if not differences: return 0
    return statistics.mean(differences)

def calculate_moving_average(values, window=5):
    if not values: return 0
    recent = values[-window:]
    return sum(recent) / len(recent)

def calculate_rate_of_change(values, window=5):
    if len(values) < 2: return 0
    recent = values[-window:]
    if len(recent) < 2: return 0
    return (recent[-1] - recent[0]) / len(recent)

def parse_enhanced_payload(payload):
    try:
        parts = payload.split(",")
        if len(parts) >= 10:
            return {
                "timestamp": float(parts[0]), "sensor_id": parts[1], "message_id": parts[2],
                "value": float(parts[3]), "wifi_rssi": int(parts[4]), "link_quality": int(parts[5]),
                "memory_percent": int(parts[6]), "cpu_freq": int(parts[7]), "reset_cause": int(parts[8]),
                "qos": int(parts[9]), "network_condition": parts[10] if len(parts) > 10 else "unknown",
            }
        else:
            return {
                "timestamp": float(parts[0]), "sensor_id": parts[1], "message_id": parts[2],
                "value": float(parts[3]), "wifi_rssi": int(parts[4]) if len(parts) > 4 else 0,
                "link_quality": 0, "memory_percent": 0, "cpu_freq": 0,
                "reset_cause": 0, "network_condition": "unknown",
            }
    except Exception as e:
        print(f"Error parsing payload: {e}")
        return {
            "timestamp": time.time(), "sensor_id": "unknown", "message_id": "unknown",
            "value": 0, "wifi_rssi": 0, "link_quality": 0, "memory_percent": 0,
            "cpu_freq": 0, "reset_cause": 0, "network_condition": "unknown",
        }

# --- 3. MODIFIED THREADS AND CLEANING ---

def get_part_safely(parts, index, convert_func, default=0):
    try:
        return convert_func(parts[index])
    except (IndexError, ValueError):
        return default

def process_and_update_state(log_data):
    """
    Cleans a new data row and updates the global state.
    This replaces the old `process_for_prometheus`.
    """
    global CURRENT_METRIC_STATE
    
    # 1. Convert row list to a dictionary for easier handling
    data_dict = dict(zip(csv_headers, log_data))

    # 2. Correct negative/offset latency (Cleaner Step 2 - "The Shortcut")
    # We will hardcode to 0 as a reliable fix.
    LATENCY_OFFSET_MS = 2000.0  # <--- THIS IS YOUR SHORTCUT
    
    # Get the absolute latency
    raw_latency = abs(data_dict.get('Latency_ms', 0))
    raw_avg_latency = abs(data_dict.get('Moving_Avg_Latency_ms', 0))

    # Subtract the offset, but use max(0, ...) to ensure it never goes negative
    # This perfectly mimics your training script's "shift" logic.
    data_dict['Latency_ms'] = max(0, abs(raw_latency - LATENCY_OFFSET_MS))
    data_dict['Moving_Avg_Latency_ms'] = max(0, abs(raw_avg_latency - LATENCY_OFFSET_MS))

    # 3. Transform Time_of_Day into seconds (Cleaner Step 3)
    try:
        time_str = data_dict.get('Time_of_Day', '00:00:00')
        time_parts = time_str.split(':')
        data_dict['Time_of_Day_Seconds'] = (
            int(time_parts[0]) * 3600 +
            int(time_parts[1]) * 60 +
            int(time_parts[2])
        )
    except Exception:
        data_dict['Time_of_Day_Seconds'] = 0

    # 4. Parse Received_Payload components (Cleaner Step 4)
    payload_str = data_dict.get('Received_Payload', '')
    payload_parts = payload_str.split(',')
    
    data_dict['RP_Sensor_Value'] = get_part_safely(payload_parts, 3, float)
    data_dict['RP_RSSI'] = get_part_safely(payload_parts, 4, int)
    data_dict['RP_LQI'] = get_part_safely(payload_parts, 5, int)
    data_dict['RP_Sequence_Num'] = get_part_safely(payload_parts, 6, int) 
    data_dict['RP_Gateway_ID'] = get_part_safely(payload_parts, 7, int)
    data_dict['RP_QoS_Level'] = get_part_safely(payload_parts, 8, int)

    # 5. Drop unnecessary columns (Cleaner Steps 1 & 5)
    # We don't need to drop, just iterate over the keys we care about
    
    # 6. --- UPDATE THE GLOBAL STATE ---
    for key in CLEANED_METRIC_KEYS:
        if key in data_dict:
            # Update the global state with the new value
            CURRENT_METRIC_STATE[key] = data_dict[key]
    
    # 7. --- UPDATE PROMETHEUS GAUGES ---
    # Set all gauges from the *complete* global state
    for key in CLEANED_METRIC_KEYS:
        try:
            prom_key = key.lower().replace('%', 'percent').replace('-', '_')
            if key in PROMETHEUS_GAUGES:
                PROMETHEUS_GAUGES[key].set(CURRENT_METRIC_STATE[key])
        except (TypeError, ValueError):
            pass # Skip non-numeric

def system_monitoring_thread():
    """
    Background thread to update system-wide metrics
    (CPU, Mem, RTT, Throughput, etc.)
    """
    global CURRENT_METRIC_STATE
    global CURRENT_METRIC_STATE
    last_bytes_total = 0
    last_check_time = time.time()
    
    rtt_history = []
    
    while True:
        try:
            # Get System Metrics
            CURRENT_METRIC_STATE['CPU_Utilization_Percent'] = get_cpu_usage()
            CURRENT_METRIC_STATE['Memory_Usage_Percent'] = get_memory_usage()
            CURRENT_METRIC_STATE['System_Load'] = get_system_load()

            # Get Network Metrics
            rtt = measure_rtt()
            rtt_history.append(rtt)
            if len(rtt_history) > 10:
                rtt_history.pop(0)
            CURRENT_METRIC_STATE['RTT_ms'] = calculate_moving_average(rtt_history)
            
            errors, link_speed = get_interface_stats()
            CURRENT_METRIC_STATE['Interface_Errors'] = errors
            CURRENT_METRIC_STATE['Link_Speed_Mbps'] = link_speed
            
            CURRENT_METRIC_STATE['TCP_Retransmissions'] = get_tcp_retransmissions()

            # Calculate throughput
            current_time = time.time()
            time_diff = current_time - last_check_time
            if time_diff >= 1.0:
                net_io = psutil.net_io_counters()
                current_bytes = net_io.bytes_sent + net_io.bytes_recv
                if last_bytes_total > 0:
                    CURRENT_METRIC_STATE['Throughput_BytesPerSec'] = calculate_throughput(
                        last_bytes_total, current_bytes, time_diff
                    )
                last_bytes_total = current_bytes
                last_check_time = current_time

            #HeartBeat
            time_since_heartbeat = time.time() - LAST_HEARTBEAT_TIME
            CURRENT_METRIC_STATE['Time_Since_Last_Heartbeat_Seconds'] = time_since_heartbeat
            
            # Update Prometheus Gauge
            if 'Time_Since_Last_Heartbeat_Seconds' in PROMETHEUS_GAUGES:
                PROMETHEUS_GAUGES['Time_Since_Last_Heartbeat_Seconds'].set(time_since_heartbeat)

            # Update all Prometheus gauges with the new system state
            for key in CLEANED_METRIC_KEYS:
                try:
                    prom_key = key.lower().replace('%', 'percent').replace('-', '_')
                    if key in PROMETHEUS_GAUGES:
                        PROMETHEUS_GAUGES[key].set(CURRENT_METRIC_STATE[key])
                except (TypeError, ValueError):
                    pass
            
            time.sleep(1) # Update system stats every second

        except Exception as e:
            print(f"Error in system monitoring thread: {e}")
            time.sleep(5)

# --- 4. MODIFIED MQTT CALLBACKS ---

def on_connect(client, userdata, flags, rc):
    connection_state = "Connected" if rc == 0 else f"Failed (code: {rc})"
    print(f"MQTT connection: {connection_state}")
    for topic in TOPICS:
        client.subscribe(topic)
        print(f"Subscribed to {topic}")

def on_disconnect(client, userdata, rc):
    print(f"MQTT disconnected with code: {rc}")

def on_message(client, userdata, msg):
    """
    Called when a message is received.
    This function is now MUCH simpler.
    It only calculates per-message metrics (Latency, Jitter, Sensor values)
    and updates the global state.
    """
    global CURRENT_METRIC_STATE
    global LAST_HEARTBEAT_TIME
    if msg.topic == "system/heartbeat":
        LAST_HEARTBEAT_TIME = time.time()
        # Optional: print a small log
        print(f"Heartbeat received at {time.strftime('%H:%M:%S')}")
        return 
    try:
        receive_time = time.time()
        payload = msg.payload.decode()
        message_data = parse_enhanced_payload(payload)

        # Calculate per-message metrics
        latency = (receive_time - message_data["timestamp"]) * 1000  # ms
        
        if msg.topic in latency_history:
            latency_history[msg.topic].append(latency)
            if len(latency_history[msg.topic]) > 20:
                latency_history[msg.topic].pop(0)
        
        jitter = calculate_jitter(latency_history[msg.topic])
        moving_avg_latency = calculate_moving_average(latency_history[msg.topic])
        rate_of_change = calculate_rate_of_change(latency_history[msg.topic])

        # --- Create a "log_data" row just for cleaning ---
        # Most values are now pulled from the *current state*
        log_data = [
            receive_time,
            datetime.datetime.fromtimestamp(receive_time).strftime("%H:%M:%S"),
            message_data["sensor_id"],
            message_data["message_id"],
            message_data["value"], # Published_Payload
            payload, # Received_Payload
            len(msg.payload), # Message_Size_Bytes
            latency, # Latency_ms
            jitter, # Jitter_ms
            CURRENT_METRIC_STATE['Packet_Loss_Percent'], # Placeholder
            CURRENT_METRIC_STATE['RTT_ms'],
            CURRENT_METRIC_STATE['Throughput_BytesPerSec'],
            CURRENT_METRIC_STATE['TCP_Retransmissions'],
            CURRENT_METRIC_STATE['Interface_Errors'],
            CURRENT_METRIC_STATE['Link_Speed_Mbps'],
            "Connected", # MQTT_Connection_State
            0, # MQTT_Message_Queue_Size (Placeholder)
            msg.qos, # QoS_Level
            100, # QoS_Success_Rate (Placeholder)
            0, # Messages_Per_Minute (Simplifying)
            0, # Failed_Delivery_Count (Simplifying)
            CURRENT_METRIC_STATE['CPU_Utilization_Percent'],
            CURRENT_METRIC_STATE['Memory_Usage_Percent'],
            CURRENT_METRIC_STATE['System_Load'],
            0, # Network_Buffer_Status (Simplifying)
            message_data["network_condition"],
            moving_avg_latency,
            rate_of_change,
            message_data["cpu_freq"],
            message_data["memory_percent"],
            message_data["reset_cause"],
            0, # Communication_Issue_Type (Simplifying)
            msg.topic,
        ]

        # Process this new data and update the global state/gauges
        process_and_update_state(log_data)

        # Print short status
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Message from {message_data['sensor_id']}: "
            f"value={message_data['value']:.2f}, "
            f"latency={latency:.2f}ms"
        )

    except Exception as e:
        print(f"Error processing message: {e}")

# --- 5. MODIFIED MAIN FUNCTION ---

def main():
    # Create MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    # Start system monitoring in background thread
    # This thread *replaces* the old network_monitoring_thread
    monitor_thread = threading.Thread(target=system_monitoring_thread)
    monitor_thread.daemon = True
    monitor_thread.start()

    try:
        # --- START PROMETHEUS SERVER ---
        start_http_server(8000)
        print(f"Prometheus metrics exporter running on http://{MQTT_BROKER}:8000")
        # -------------------------------

        # Connect to MQTT broker
        print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()

        print("MQTT Subscriber started. Press Ctrl+C to exit.")
        
        # Keep the main thread alive (the old menu is removed for simplicity)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except:
            pass
        print("Subscriber stopped.")


if __name__ == "__main__":
    main()