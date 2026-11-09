import json
import time
from datetime import datetime, timezone

import requests
from kafka import KafkaProducer
from kafka.errors import KafkaError


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = "https://aviationweather.gov/api/data/metar"

AIRPORTS = ["VIDP", "VABB", "VECC", "VOBL"]

KAFKA_TOPIC = "weather-events"
KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"

POLL_INTERVAL = 300

# Keep consistent with Spark's stale weather threshold.
MAX_OBSERVATION_AGE_HOURS = 6


# --------------------------------------------------
# KAFKA PRODUCER
# --------------------------------------------------

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    key_serializer=lambda key: key.encode("utf-8"),
    retries=5,
    acks="all",
    request_timeout_ms=30000
)


# Track latest successfully published observation.
last_seen = {}


# --------------------------------------------------
# FETCH WEATHER DATA
# --------------------------------------------------

def fetch_weather():

    params = {
        "ids": ",".join(AIRPORTS),
        "format": "json",
        "hours": MAX_OBSERVATION_AGE_HOURS
    }

    headers = {
        "User-Agent": "AirOpsAI/1.0"
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=headers,
        timeout=30
    )

    # API returns 204 when no observations are available.
    if response.status_code == 204:
        return []

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# VALIDATE WEATHER FRESHNESS
# --------------------------------------------------

def validate_observation(observation):

    airport = observation.get("icaoId")
    obs_timestamp = observation.get("obsTime")

    if not airport or obs_timestamp is None:
        return False, None

    obs_timestamp = int(obs_timestamp)

    current_timestamp = int(
        datetime.now(timezone.utc).timestamp()
    )

    age_minutes = (
        current_timestamp - obs_timestamp
    ) / 60

    # Reject future observations.
    if age_minutes < 0:
        print(
            f"[INVALID WEATHER] {airport}: "
            f"Future observation detected.",
            flush=True
        )
        return False, age_minutes

    # Reject observations older than configured threshold.
    if age_minutes > MAX_OBSERVATION_AGE_HOURS * 60:
        print(
            f"[STALE WEATHER] {airport}: "
            f"Observation age = {age_minutes:.2f} minutes",
            flush=True
        )
        return False, age_minutes

    return True, age_minutes


# --------------------------------------------------
# PUBLISH WEATHER TO KAFKA
# --------------------------------------------------

def publish_weather(weather_data):

    futures = []

    latest_seen = {}

    current_time = datetime.now(timezone.utc)

    for observation in weather_data:

        airport = observation.get("icaoId")

        if not airport:
            continue

        airport = airport.upper().strip()

        observation["icaoId"] = airport

        obs_timestamp = observation.get("obsTime")

        if obs_timestamp is None:
            continue

        obs_timestamp = int(obs_timestamp)

        # Validate freshness.
        is_valid, age_minutes = validate_observation(
            observation
        )

        if not is_valid:
            continue

        # Avoid publishing duplicate observations.
        if obs_timestamp <= last_seen.get(airport, -1):
            continue

        event = {
            **observation,

            "source_timestamp": obs_timestamp,

            "ingestion_timestamp": current_time.isoformat()
        }

        try:

            future = producer.send(
                KAFKA_TOPIC,
                key=airport,
                value=event
            )

            futures.append(
                (airport, obs_timestamp, age_minutes, future)
            )

        except KafkaError as error:

            print(
                f"[KAFKA ERROR] {airport}: {error}",
                flush=True
            )

    producer.flush()

    published = 0

    # Update last_seen separately for each successful delivery.
    for airport, obs_timestamp, age_minutes, future in futures:

        try:

            future.get(timeout=30)

            last_seen[airport] = max(
                last_seen.get(airport, -1),
                obs_timestamp
            )

            published += 1

            print(
                f"[PUBLISHED] {airport} | "
                f"Weather age: {age_minutes:.2f} minutes",
                flush=True
            )

        except KafkaError as error:

            print(
                f"[KAFKA DELIVERY ERROR] {airport}: {error}",
                flush=True
            )

    return published


# --------------------------------------------------
# MAIN INGESTION LOOP
# --------------------------------------------------

if __name__ == "__main__":

    print("Starting continuous weather ingestion...")
    print(f"Airports: {AIRPORTS}")
    print(f"Polling interval: {POLL_INTERVAL} seconds")
    print(
        f"Maximum observation age: "
        f"{MAX_OBSERVATION_AGE_HOURS} hours"
    )

    try:

        while True:

            try:

                weather_data = fetch_weather()

                published = publish_weather(weather_data)

                print(
                    f"\n[{datetime.now()}] "
                    f"Fetched: {len(weather_data)} observations | "
                    f"Published: {published} new observations",
                    flush=True
                )

                if not weather_data:

                    print(
                        "[INFO] No recent METAR observations "
                        "available for the selected airports.",
                        flush=True
                    )

            except requests.exceptions.RequestException as error:

                print(
                    f"[WEATHER API ERROR] {error}",
                    flush=True
                )

            except Exception as error:

                print(
                    f"[WEATHER INGESTION ERROR] {error}",
                    flush=True
                )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:

        print("Stopping weather ingestion...")

    finally:

        producer.flush()
        producer.close()

        print("Weather Kafka producer closed.")