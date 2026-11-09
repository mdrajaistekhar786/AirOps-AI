
import requests
import json
import time
from datetime import datetime, timezone
from kafka import KafkaProducer

# Configuration
OPENSKY_URL = "https://opensky-network.org/api/states/all"
KAFKA_TOPIC = "flight-events"
POLL_INTERVAL = 60  # Poll every 60 seconds

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9094",
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    key_serializer=lambda key: key.encode("utf-8")
)


def fetch_flight_data():

    response = requests.get(OPENSKY_URL, timeout=30)
    response.raise_for_status()

    data = response.json()

    states = data.get("states") or []
    snapshot_timestamp = data.get("time")

    flight_events = []

    for state in states:

        event = {
            "icao24": state[0],
            "callsign": state[1],
            "origin_country": state[2],
            "longitude": state[5],
            "latitude": state[6],
            "altitude": state[7],
            "on_ground": state[8],
            "velocity": state[9],
            "true_track": state[10],
            "vertical_rate": state[11],
            "source_timestamp": snapshot_timestamp,
            "ingestion_timestamp": datetime.now(
                timezone.utc
            ).isoformat()
        }

        flight_events.append(event)

    return flight_events


def publish_to_kafka(flight_events):

    for event in flight_events:

        producer.send(
            KAFKA_TOPIC,
            key=event["icao24"],
            value=event
        )

    producer.flush()


# Continuous ingestion
try:

    print("Starting continuous OpenSky ingestion...")

    while True:

        try:

            flight_events = fetch_flight_data()

            publish_to_kafka(flight_events)

            print(
                f"[{datetime.now()}] "
                f"Published {len(flight_events)} flight events"
            )

        except requests.exceptions.RequestException as e:

            print("OpenSky API error:", e)

        except Exception as e:

            print("Ingestion error:", e)

        time.sleep(POLL_INTERVAL)

except KeyboardInterrupt:

    print("Stopping flight ingestion...")

finally:

    producer.close()
    print("Kafka producer closed.")
