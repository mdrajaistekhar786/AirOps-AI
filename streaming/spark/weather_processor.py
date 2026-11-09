
from pyspark.sql.functions import col, when, round


def clean_weather_data(df):
    return (
        df
        # Remove records with essential missing values
        .filter(
            col("icaoId").isNotNull() &
            col("temp").isNotNull() &
            col("wspd").isNotNull()
        )

        # Convert numeric visibility values to double
        .withColumn(
            "visibility_km",
            when(
                col("visib").rlike(r"^\d+(\.\d+)?$"),
                col("visib").cast("double")
            )
        )

        # Convert wind speed from knots to km/h
        .withColumn(
            "wind_speed_kmh",
            round(col("wspd").cast("double") * 1.852, 2)
        )

        # Retain temperature in Celsius
        .withColumn(
            "temperature_c",
            col("temp").cast("double")
        )
    )
