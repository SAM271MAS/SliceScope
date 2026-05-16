import json
import time
import os
from datetime import datetime
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from influxdb import InfluxDBClient
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
from requests.exceptions import ConnectionError

# --- CONFIGURATION ---
KAFKA_SERVER = os.getenv('KAFKA_BOOTSTRAP_SERVER', 'demo-kafka-bootstrap.slicescope.svc.cluster.local:9082')
INFLUX_HOST = os.getenv('INFLUX_HOST', 'influxdb-service.slicescope.svc.cluster.local')
TOPIC = 'network-kpis'
INFLUX_PORT = int(os.getenv('INFLUX_PORT', '9084'))
DB_NAME = 'network_data'

def log(msg: str):
    """Helper to print logs with a timestamp for K8s pod monitoring."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def connect_to_influx():
    """Establishes connection to InfluxDB with retries."""
    log(f"Connecting to InfluxDB at {INFLUX_HOST}:{INFLUX_PORT}...")
    client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT)
    
    while True:
        try:
            client.ping()
            client.create_database(DB_NAME)
            client.switch_database(DB_NAME)
            log("Successfully connected to InfluxDB and selected database.")
            return client
        except (ConnectionError, InfluxDBClientError, InfluxDBServerError) as e:
            log(f"InfluxDB not ready yet ({e}). Retrying in 5 seconds...")
            time.sleep(5)

def connect_to_kafka():
    """Establishes connection to Kafka with retries."""
    log(f"Connecting to Kafka at {KAFKA_SERVER}...")
    while True:
        try:
            consumer = KafkaConsumer(
                TOPIC,
                bootstrap_servers=[KAFKA_SERVER],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=True,
                auto_offset_reset='latest',
                group_id='influx-consumer-group'
            )
            log(f"Successfully connected to Kafka! Listening to topic: '{TOPIC}'...")
            return consumer
        except NoBrokersAvailable:
            log("Kafka broker not ready yet. Retrying in 10 seconds...")
            time.sleep(10)
        except Exception as e:
            log(f"Kafka connection error ({e}). Retrying in 5 seconds...")
            time.sleep(5)

def run():
    log("Starting Data Consumer (Kafka -> InfluxDB)")
    
    influx_client = connect_to_influx()
    kafka_consumer = connect_to_kafka()

    try:
        for message in kafka_consumer:
            data = message.value
            real_now = data.get("real_now", {})
            
            # The 'sequence' array is also in data, but for InfluxDB 
            # we only want to store the real-time 'now' state.
            if not real_now:
                continue

            # Format for InfluxDB
            json_body = [{
                "measurement": "throughput_kpis",
                "tags": {"source": "squid_proxy"},
                "fields": real_now
            }]

            success = influx_client.write_points(json_body)
            
            if success:
                active_traffic = sum(v for k, v in real_now.items() if "Jitter" not in k and "CQI" not in k)
                if active_traffic > 0:
                    log(f"Saved to InfluxDB | Total Active Traffic: {active_traffic:.2f} Mbps")
            else:
                log("WARNING: Failed to write data point to InfluxDB.")

    except Exception as e:
        log(f"CRITICAL ERROR in consumer loop: {e}")

if __name__ == "__main__":
    run()

