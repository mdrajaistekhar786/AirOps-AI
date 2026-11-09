
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    BooleanType,
    LongType
)

from session import create_spark_session

from flight_processor import (
    clean_flight_data,
    engineer_flight_features
)


# 1. Initialize Spark session using shared configuration
spark = create_spark_session()
spark.sparkContext.setLogLevel("WARN")


# 2. Define incoming flight-event schema
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


# 3. Read flight events from Kafka
flight_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "flight-events")
    .option("startingOffsets", "latest")
    .load()
)


# 4. Parse incoming JSON messages
flight_events = (
    flight_stream
    .select(
        from_json(
            col("value").cast("string"),
            flight_schema
        ).alias("flight")
    )
    .select("flight.*")
)


# 5. Clean flight data
cleaned_flights = clean_flight_data(flight_events)


# 6. Engineer flight features
flight_features = engineer_flight_features(cleaned_flights)


# 7. Write processed data to Parquet
query = (
    flight_features.writeStream
    .format("parquet")
    .outputMode("append")
    .option(
        "path",
        "/opt/airops/storage/processed/flight_features"
    )
    .option(
        "checkpointLocation",
        "/opt/airops/storage/checkpoints/flight_features"
    )
    .start()
)


# 8. Keep the streaming query active
query.awaitTermination()
