from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, timestamp_seconds, radians, sin, cos,
    asin, sqrt, array, array_min, struct, lit, when,
    round, row_number, upper, trim, abs, unix_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, LongType, BooleanType
)
from pyspark.sql.window import Window
from pyspark.sql.streaming import StreamingQueryListener


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

AIRPORTS = [
    ("VIDP", 28.5562, 77.1000),
    ("VABB", 19.0896, 72.8656),
    ("VECC", 22.6547, 88.4467),
    ("VOBL", 13.1986, 77.7066)
]

AIRPORT_RADIUS_KM = 100
STALE_WEATHER_HOURS = 6

FLIGHT_TOPIC = "flight-events"
WEATHER_TOPIC = "weather-events"
KAFKA_SERVERS = "kafka:9092"

OUTPUT_PATH = "/opt/airops/storage/processed/flight_weather"
CHECKPOINT_PATH = "/opt/airops/storage/checkpoints/flight_weather_v6"


# --------------------------------------------------
# FLIGHT EVENT SCHEMA
# --------------------------------------------------

flight_schema = StructType([
    StructField("icao24", StringType(), True),
    StructField("callsign", StringType(), True),
    StructField("origin_country", StringType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("altitude", DoubleType(), True),
    StructField("on_ground", BooleanType(), True),
    StructField("velocity", DoubleType(), True),
    StructField("true_track", DoubleType(), True),
    StructField("vertical_rate", DoubleType(), True),
    StructField("source_timestamp", LongType(), True),
    StructField("ingestion_timestamp", StringType(), True)
])


# --------------------------------------------------
# WEATHER EVENT SCHEMA
# --------------------------------------------------

weather_schema = StructType([
    StructField("icaoId", StringType(), True),
    StructField("receiptTime", StringType(), True),
    StructField("obsTime", LongType(), True),
    StructField("temp", DoubleType(), True),
    StructField("dewp", DoubleType(), True),
    StructField("wdir", DoubleType(), True),
    StructField("wspd", DoubleType(), True),
    StructField("visib", StringType(), True),
    StructField("altim", DoubleType(), True),
    StructField("rawOb", StringType(), True),
    StructField("lat", DoubleType(), True),
    StructField("lon", DoubleType(), True),
    StructField("name", StringType(), True),
    StructField("cover", StringType(), True),
    StructField("fltCat", StringType(), True)
])


# --------------------------------------------------
# STREAMING PROGRESS LOGGER
# --------------------------------------------------

class ProgressLogger(StreamingQueryListener):

    def onQueryStarted(self, event):
        print(
            f"\n[STREAM STARTED] Query ID: {event.id}",
            flush=True
        )

    def onQueryProgress(self, event):
        progress = event.progress

        print("\n" + "=" * 55, flush=True)
        print(f"BATCH ID: {progress.batchId}", flush=True)
        print(
            f"TOTAL INPUT ROWS: {progress.numInputRows}",
            flush=True
        )

        for source in progress.sources:
            print(
                f"{source.description}: "
                f"{source.numInputRows} rows",
                flush=True
            )

        print("=" * 55, flush=True)

    def onQueryIdle(self, event):
        print(
            "[STREAM IDLE] Waiting for new flight events.",
            flush=True
        )

    def onQueryTerminated(self, event):
        print(
            f"\n[STREAM TERMINATED] Query ID: {event.id}",
            flush=True
        )

        if event.exception:
            print(f"Exception: {event.exception}", flush=True)


# --------------------------------------------------
# CALCULATE DISTANCE
# --------------------------------------------------

def airport_distance(lat, lon, airport_lat, airport_lon):

    lat_diff = radians(col(lat)) - radians(lit(airport_lat))
    lon_diff = radians(col(lon)) - radians(lit(airport_lon))

    a = (
        sin(lat_diff / 2) ** 2
        + cos(radians(col(lat)))
        * cos(radians(lit(airport_lat)))
        * sin(lon_diff / 2) ** 2
    )

    return 6371 * 2 * asin(sqrt(a))


# --------------------------------------------------
# ASSIGN NEAREST AIRPORT
# --------------------------------------------------

def assign_nearest_airport(flights):

    distances = []

    for code, lat, lon in AIRPORTS:

        distance = airport_distance(
            "latitude",
            "longitude",
            lat,
            lon
        )

        distances.append(
            struct(
                distance.alias("distance"),
                lit(code).alias("airport")
            )
        )

    nearest = array_min(array(*distances))

    return (
        flights
        .withColumn("nearest", nearest)
        .withColumn("distance_km", col("nearest.distance"))
        .withColumn("airport_icao", col("nearest.airport"))
        .drop("nearest")
        .filter(col("distance_km") <= AIRPORT_RADIUS_KM)
    )


# --------------------------------------------------
# READ LATEST WEATHER SNAPSHOT FROM KAFKA
# --------------------------------------------------

def read_weather_snapshot(spark):

    weather_kafka = (
        spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", WEATHER_TOPIC)
        .option("startingOffsets", "earliest")
        .option("endingOffsets", "latest")
        .load()
    )

    weather = (
        weather_kafka
        .select(
            from_json(
                col("value").cast("string"),
                weather_schema
            ).alias("weather")
        )
        .select("weather.*")
        .filter(
            col("icaoId").isNotNull()
            & col("obsTime").isNotNull()
        )
        .withColumn(
            "icaoId",
            upper(trim(col("icaoId")))
        )
        .withColumn(
            "weather_time",
            timestamp_seconds(col("obsTime"))
        )
        .withColumn(
            "visibility_km",
            when(
                col("visib").rlike(r"^\d+(\.\d+)?$"),
                round(
                    col("visib").cast("double") * 1.60934,
                    2
                )
            )
        )
        .withColumn(
            "wind_speed_kmh",
            round(
                col("wspd").cast("double") * 1.852,
                2
            )
        )
        .withColumn(
            "temperature_c",
            col("temp").cast("double")
        )
    )

    weather_window = (
        Window
        .partitionBy("icaoId")
        .orderBy(col("obsTime").desc())
    )

    latest_weather = (
        weather
        .withColumn(
            "_weather_rank",
            row_number().over(weather_window)
        )
        .filter(col("_weather_rank") == 1)
        .drop("_weather_rank")
        .select(
            col("icaoId"),
            col("temperature_c"),
            col("wind_speed_kmh"),
            col("visibility_km"),
            col("fltCat"),
            col("weather_time")
        )
    )

    return latest_weather


# --------------------------------------------------
# PROCESS EACH FLIGHT MICRO-BATCH
# --------------------------------------------------

def process_flight_batch(batch_df, batch_id, spark):

    print(
        f"\n[PROCESSING BATCH] Batch ID: {batch_id}",
        flush=True
    )

    # 1. Parse flight events.

    flights = (
        batch_df
        .select(
            from_json(
                col("value").cast("string"),
                flight_schema
            ).alias("flight")
        )
        .select("flight.*")
        .filter(
            col("icao24").isNotNull()
            & col("latitude").isNotNull()
            & col("longitude").isNotNull()
            & col("source_timestamp").isNotNull()
        )
        .withColumn(
            "flight_time",
            timestamp_seconds(col("source_timestamp"))
        )
        .dropDuplicates(["icao24", "source_timestamp"])
    )

    if flights.isEmpty():
        print("[BATCH] No valid flight records.", flush=True)
        return

    # 2. Assign nearest airport.

    flights = assign_nearest_airport(flights)

    flight_count = flights.count()

    if flight_count == 0:
        print(
            "[BATCH] No flights within airport radius.",
            flush=True
        )
        return

    print(
        f"[FLIGHTS] Records within airport radius: {flight_count}",
        flush=True
    )

    # 3. Read latest available weather per airport.

    weather = read_weather_snapshot(spark)

    weather_count = weather.count()

    print(
        f"[WEATHER] Latest airport observations: {weather_count}",
        flush=True
    )

    if weather_count > 0:
        print(
            "[WEATHER] Available airport codes:",
            [
                row["icaoId"]
                for row in weather.select("icaoId").collect()
            ],
            flush=True
        )

    # 4. Match by airport ICAO code.

    joined = (
        flights.alias("f")
        .join(
            weather.alias("w"),
            col("f.airport_icao") == col("w.icaoId"),
            "left"
        )
        .select(
            col("f.icao24"),
            col("f.callsign"),
            col("f.latitude"),
            col("f.longitude"),
            col("f.altitude"),
            col("f.velocity"),
            col("f.flight_time"),
            col("f.source_timestamp"),
            col("f.airport_icao"),
            col("f.distance_km"),
            col("w.temperature_c"),
            col("w.wind_speed_kmh"),
            col("w.visibility_km"),
            col("w.fltCat"),
            col("w.weather_time")
        )
    )

    # 5. Calculate weather age and freshness.

    enriched = (
        joined
        .withColumn(
            "weather_age_minutes",
            when(
                col("weather_time").isNotNull(),
                round(
                    abs(
                        unix_timestamp(col("flight_time"))
                        - unix_timestamp(col("weather_time"))
                    ) / 60,
                    2
                )
            )
        )
        .withColumn(
            "weather_is_stale",
            when(
                col("weather_time").isNull(),
                lit(True)
            ).otherwise(
                col("weather_age_minutes")
                > STALE_WEATHER_HOURS * 60
            )
        )
    )

    # 6. Calculate enrichment counts.

    record_count = enriched.count()

    weather_matched_count = (
        enriched
        .filter(col("weather_time").isNotNull())
        .count()
    )

    stale_count = (
        enriched
        .filter(col("weather_is_stale") == True)
        .count()
    )

    # 7. Weather age diagnostics.

    weather_age_stats = (
        enriched
        .filter(col("weather_age_minutes").isNotNull())
        .selectExpr(
            "MIN(weather_age_minutes) AS min_age",
            "AVG(weather_age_minutes) AS avg_age",
            "MAX(weather_age_minutes) AS max_age"
        )
        .first()
    )

    if weather_age_stats["min_age"] is not None:

        avg_age = float(weather_age_stats["avg_age"])

        print(
            f"[WEATHER AGE] Min: "
            f"{weather_age_stats['min_age']} min | "
            f"Average: {avg_age:.2f} min | "
            f"Max: {weather_age_stats['max_age']} min",
            flush=True
        )

    else:
        print(
            "[WEATHER AGE] No valid weather timestamps.",
            flush=True
        )

    # 8. Print enrichment summary.

    print(
        f"[ENRICHMENT] Total records: {record_count}",
        flush=True
    )

    print(
        f"[ENRICHMENT] Records with weather: "
        f"{weather_matched_count}",
        flush=True
    )

    print(
        f"[ENRICHMENT] Records with stale/missing weather: "
        f"{stale_count}",
        flush=True
    )

    if record_count == 0:
        print(
            "[BATCH] No records available for output.",
            flush=True
        )
        return

    # 9. Write enriched records.

    enriched.write.mode("append").parquet(OUTPUT_PATH)

    print(
        f"[BATCH COMPLETE] Batch: {batch_id} | "
        f"Enriched records: {record_count}",
        flush=True
    )


# --------------------------------------------------
# MAIN STREAMING PIPELINE
# --------------------------------------------------

def main():

    spark = (
        SparkSession.builder
        .appName("AirOps-Flight-Weather-Streaming")
        .master("spark://spark-master:7077")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    spark.streams.addListener(ProgressLogger())

    # Read only new flight events after query starts.

    flight_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", FLIGHT_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    query = (
        flight_stream.writeStream
        .foreachBatch(
            lambda batch_df, batch_id:
            process_flight_batch(batch_df, batch_id, spark)
        )
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="10 seconds")
        .start()
    )

    print(
        "\nFlight-Weather Streaming Pipeline started.",
        flush=True
    )

    print("Flight source: Kafka flight-events", flush=True)
    print("Weather source: Latest Kafka weather snapshot", flush=True)
    print("Join strategy: Latest weather per airport", flush=True)
    print(
        f"Stale weather threshold: {STALE_WEATHER_HOURS} hours",
        flush=True
    )
    print("Join type: Left join", flush=True)
    print("Checkpoint: flight_weather_v6", flush=True)
    print("Starting flight offsets: latest", flush=True)
    print(f"Output: {OUTPUT_PATH}", flush=True)

    query.awaitTermination()


if __name__ == "__main__":
    main()