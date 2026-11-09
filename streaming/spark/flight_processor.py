
from pyspark.sql.functions import col, when, round


def clean_flight_data(df):
    return (
        df
        # Remove records with essential missing values
        .filter(
            col("icao24").isNotNull() &
            col("latitude").isNotNull() &
            col("longitude").isNotNull() &
            col("velocity").isNotNull()
        )

        # Validate geographical coordinates and velocity
        .filter(
            col("latitude").between(-90, 90) &
            col("longitude").between(-180, 180) &
            (col("velocity") >= 0)
        )
    )


def engineer_flight_features(df):
    return (
        df

        # Convert velocity from m/s to km/h
        .withColumn(
            "velocity_kmh",
            round(col("velocity") * 3.6, 2)
        )

        # Convert altitude from metres to feet
        .withColumn(
            "altitude_ft",
            round(col("altitude") * 3.28084, 2)
        )

        # Identify flight phase
        .withColumn(
            "flight_phase",
            when(col("on_ground").isNull(), "UNKNOWN")
            .when(col("on_ground") == True, "GROUND")
            .otherwise("AIRBORNE")
        )
    )
