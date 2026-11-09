from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, LongType, ArrayType
)

from weather_processor import clean_weather_data

spark = (
    SparkSession.builder
    .appName("AirOps-Weather-Consumer")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

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

weather_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "weather-events")
    .option("startingOffsets", "earliest")
    .load()
)

weather_events = (
    weather_stream
    .select(
        from_json(
            col("value").cast("string"),
            weather_schema
        ).alias("weather")
    )
    .select("weather.*")
)

cleaned_weather = clean_weather_data(weather_events)

query = (
    cleaned_weather.writeStream
    .format("parquet")
    .outputMode("append")
    .option(
        "path",
        "/opt/airops/storage/processed/weather_features"
    )
    .option(
        "checkpointLocation",
        "/opt/airops/storage/checkpoints/weather_features"
    )
    .start()
)

query.awaitTermination()