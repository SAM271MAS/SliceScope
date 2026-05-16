import time
import json
import urllib.parse
import sys
import os
from collections import deque
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from datetime import datetime

# --- CONFIGURATION ---
KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVER', 'demo-kafka-bootstrap.slicescope.svc.cluster.local:9082')
TOPIC = 'network-kpis'
LOG_FILE = '/var/log/squid/access.log'

FEATURES = [
    "UE1: web-rtc", "UE2: sipp", "UE3: web-server", # Throughputs
    "UE1-Jitter", "UE2-Jitter", "UE3-Jitter",       # Jitters
    "UE1-CQI", "UE2-CQI", "UE3-CQI"                 # CQIs
]

def log(msg: str):
    """Helper to print logs with a timestamp for K8s pod monitoring."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def create_producer():
    """Establishes a robust connection to the Kafka broker with automatic retries."""
    log(f"Connecting to Kafka at {KAFKA_SERVER}...")
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_SERVER],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                api_version=(2, 0, 2)
            )
            log("Successfully connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            log("Kafka not ready yet. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            log(f"Connection error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def stream_live_data():
    producer = create_producer()
    log("Starting LIVE Kafka Producer (Telemetry Injection + ML Windowing Mode)...")

    # The deque automatically drops the oldest item when it exceeds maxlen.
    sequence_window = deque(maxlen=30)
    
    # State dictionary to hold the parsed values from the URL
    current_state = {f: 0.0 for f in FEATURES}
    
    # Initialize CQI to 15 (perfect connection default)
    for f in FEATURES:
        if "CQI" in f: 
            current_state[f] = 15

    last_tick = time.time()

    # Open the Squid log and jump to the end
    try:
        file = open(LOG_FILE, 'r')
        file.seek(0, 2) 
    except FileNotFoundError:
        log(f"CRITICAL: Could not find {LOG_FILE}. Ensure the volume is mounted.")
        sys.exit(1)

    while True:
        line = file.readline()
        
        # 1. PARSE INJECTED DATA FROM SQUID LOGS
        if line and "ml_feature=" in line:
            try:
                parts = line.split()
                url = next((p for p in parts if "http" in p), "")
                
                parsed_url = urllib.parse.urlparse(url)
                query = urllib.parse.parse_qs(parsed_url.query)
                
                if 'ml_feature' in query:
                    feature = query['ml_feature'][0] # e.g., "UE1_web-rtc"
                    ue_id = feature.split('_')[0]    # "UE1"
                    app_id = feature.split('_')[1]   # "web-rtc"
                    
                    # Extract ground truth telemetry injected by the User Pods
                    cqi = int(query.get('cqi', [15])[0])
                    tput = float(query.get('tput', [0.0])[0])
                    jit = float(query.get('jit', [0.0])[0])
                    
                    # Update the running state for this exact second
                    current_state[f"{ue_id}-CQI"] = cqi
                    current_state[f"{ue_id}-Jitter"] = jit
                    current_state[f"{ue_id}: {app_id}"] = tput
            except Exception:
                pass # Ignore malformed logs

        # 2. ML CLOCK TICK: Evaluate the state every 1 second
        if time.time() - last_tick >= 1.0:
            
            # Snapshot the current state into real_now
            real_now = current_state.copy()
            
            sequence_window.append([real_now[f] for f in FEATURES])
            
            # Calculate total traffic volume to determine if the network is active
            total_traffic = sum(real_now[f] for f in FEATURES if "Jitter" not in f and "CQI" not in f)

            # Once the window is fully warmed up (30 seconds of memory), send to ML model
            if len(sequence_window) == 30:
                message = {
                    "timestamp": time.time(),
                    "sequence": list(sequence_window),
                    "real_now": real_now
                }
                producer.send(TOPIC, value=message)

                if total_traffic > 0:
                    # Print a dynamic summary of which UEs are actually transmitting
                    ue_app_map = {"UE1": "web-rtc", "UE2": "sipp", "UE3": "web-server"}
                    active_ues = [ue for ue, app in ue_app_map.items() if real_now.get(f"{ue}: {app}", 0) > 0]
                    log(f"--- SENT TO KAFKA | Active: {', '.join(active_ues)} | Total Traffic: {total_traffic:.2f} Mbps ---")
            else:
                if total_traffic > 0 or len(sequence_window) % 10 == 0:
                    log(f"Warming up LSTM sequence... {len(sequence_window)}/30")



            last_tick = time.time()
        else:
            # Prevent CPU thrashing if no new logs are written
            time.sleep(0.01)

if __name__ == "__main__":
    stream_live_data()

