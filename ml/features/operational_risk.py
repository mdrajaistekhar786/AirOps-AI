from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, radians, sin, cos, asin, sqrt
)


def calculate_operational_risk(spark):
    # Load flight-weather joined data
    df = spark.read.parquet(
        "/opt/airops/storage/processed/flight_weather"
    )

    # Classify weather severity
    df = df.withColumn(
        "weather_risk",
        when(col("fltCat") == "LIFR", "HIGH")
        .when(col("fltCat") == "IFR", "MEDIUM")
        .when(col("fltCat") == "MVFR", "LOW")
        .otherwise("NORMAL")
    )

    # Classify proximity to airport
    df = df.withColumn(
        "proximity_risk",
        when(col("distance_km") <= 25, "HIGH")
        .when(col("distance_km") <= 60, "MEDIUM")
        .otherwise("LOW")
    )

    # Combine weather and proximity risk
    df = df.withColumn(
        "operational_risk",
        when(
            (col("weather_risk") == "HIGH") &
            (col("proximity_risk") == "HIGH"),
            "CRITICAL"
        )
        .when(
            (col("weather_risk") == "HIGH") |
            (col("proximity_risk") == "HIGH"),
            "HIGH"
        )
        .when(
            (col("weather_risk") == "MEDIUM") |
            (col("proximity_risk") == "MEDIUM"),
            "MEDIUM"
        )
        .otherwise("LOW")
    )

    return df


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("AirOps-Operational-Risk")
        .getOrCreate()
    )

    result = calculate_operational_risk(spark)

    result.select(
        "icao24",
        "icaoId",
        "distance_km",
        "fltCat",
        "weather_risk",
        "proximity_risk",
        "operational_risk"
    ).show(20, truncate=False)

    result.write.mode("overwrite").parquet(
        "/opt/airops/storage/processed/operational_risk"
    )

    print("Operational risk records:", result.count())

    spark.stop()