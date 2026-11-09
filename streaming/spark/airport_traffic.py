
from pyspark.sql.functions import (
    col, from_json, radians, sin, cos,
    asin, sqrt, window, approx_count_distinct,
    from_unixtime
)

from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, BooleanType, LongType
)

from session import create_spark_session
from flight_processor import clean_flight_data


# Initialize Spark
spark = create_spark_session()
spark.sparkContext.setLogLevel("WARN")


# Incoming Kafka event schema
flight_schema = StructType([
    StructField("icao24", StringType(), True),
    StructField("callsign", StringType(), True),
    StructField("origin_country", StringType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("altitude", DoubleType(), True),
    StructField("on_ground", BooleanType(), True),
    StructField("velocity", DoubleType(), True),
    StructField("source_timestamp", LongType(), True)
])


# Airport coordinates
airports = [
    ("VIDP", 28.5562, 77.1000),
    ("VABB", 19.0896, 72.8656),
    ("VECC", 22.6547, 88.4467),
    ("VOBL", 13.1986, 77.7066)
]

airport_df = spark.createDataFrame(
    airports,
    ["airport", "airport_lat", "airport_lon"]
)


# Read flight events from Kafka
raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "flight-events")
    .option("startingOffsets", "latest")
    .load()
)


# Parse incoming JSON
flights = (
    raw_stream
    .select(
        from_json(
            col("value").cast("string"),
            flight_schema
        ).alias("flight")
    )
    .select("flight.*")
)


# Clean flight data
flights = clean_flight_data(flights)


# Convert Unix timestamp into Spark timestamp
flights = (
    flights
    .filter(col("source_timestamp").isNotNull())
    .withColumn(
        "event_time",
        from_unixtime(col("source_timestamp")).cast("timestamp")
    )
)


# Calculate distance from each airport using Haversine formula
nearby_flights = (
    flights
    .crossJoin(airport_df)
    .withColumn(
        "distance_km",
        6371 * 2 * asin(
            sqrt(
                sin(
                    radians(
                        col("latitude") - col("airport_lat")
                    ) / 2
                ) ** 2
                + cos(radians(col("latitude")))
                * cos(radians(col("airport_lat")))
                * sin(
                    radians(
                        col("longitude") - col("airport_lon")
                    ) / 2
                ) ** 2
            )
        )
    )
    .filter(col("distance_km") <= 100)
)


# Count unique aircraft per airport in 5-minute event-time windows
traffic = (
    nearby_flights
    .withWatermark("event_time", "10 minutes")
    .groupBy(
        window(
            col("event_time"),
            "5 minutes"
        ),
        col("airport")
    )
    .agg(
        approx_count_distinct("icao24").alias(
            "aircraft_count"
        )
    )
)


# Write finalized traffic windows to Parquet
query = (
    traffic.writeStream
    .format("parquet")
    .outputMode("append")
    .option(
        "path",
        "/opt/airops/storage/processed/airport_traffic"
    )
    .option(
        "checkpointLocation",
        "/opt/airops/storage/checkpoints/airport_traffic"
    )
    .start()
)

print("Airport traffic streaming started...")

query.awaitTermination()
