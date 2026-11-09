
import time
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import datetime, timezone


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("storage/processed/flight_weather")

MODEL_PATH = Path("ml/models/risk_model.pkl")
ENCODER_PATH = Path("ml/models/label_encoder.pkl")
FEATURES_PATH = Path("ml/models/risk_features.pkl")

OUTPUT_PATH = Path("storage/processed/risk_predictions")
OUTPUT_FILE = OUTPUT_PATH / "predictions.parquet"
TEMP_FILE = OUTPUT_PATH / "predictions.tmp.parquet"

POLL_INTERVAL = 30

CATEGORY_MAP = {
    "VFR": 0,
    "MVFR": 1,
    "IFR": 2,
    "LIFR": 3
}


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

print("Loading trained risk prediction pipeline...")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = datetime.fromtimestamp(
    MODEL_PATH.stat().st_mtime,
    tz=timezone.utc
).isoformat()

print("Model loaded successfully.")
print("Model version:", MODEL_VERSION)
print("Features:", FEATURES)
print("Classes:", list(encoder.classes_))
print("Missing-value handling: Median imputation")


# --------------------------------------------------
# CREATE UNIQUE FLIGHT EVENT IDENTIFIER
# --------------------------------------------------

def create_prediction_key(df):

    timestamps = pd.to_datetime(
        df["flight_time"],
        utc=True,
        errors="coerce"
    )

    return (
        df["icao24"].astype("string").fillna("")
        + "|"
        + timestamps.astype("int64").astype("string")
        + "|"
        + df["airport_icao"].astype("string").fillna("")
    )


# --------------------------------------------------
# LOAD EXISTING PREDICTIONS
# --------------------------------------------------

def load_existing_predictions():

    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    try:
        existing = pd.read_parquet(OUTPUT_FILE)

        if existing.empty:
            return pd.DataFrame()

        if "_prediction_key" not in existing.columns:
            existing["_prediction_key"] = create_prediction_key(
                existing
            )

        return existing

    except Exception as error:
        raise RuntimeError(
            f"Could not read existing predictions: {error}"
        ) from error


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def prepare_features(df):

    df = df.copy()

    # Altitude: metres to feet
    df["altitude_ft"] = (
        pd.to_numeric(df["altitude"], errors="coerce")
        * 3.28084
    )

    # Velocity: metres/second to km/hour
    df["velocity_kmh"] = (
        pd.to_numeric(df["velocity"], errors="coerce")
        * 3.6
    )

    # Flight category encoding
    df["fltCat"] = (
        df["fltCat"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["flight_category_score"] = (
        df["fltCat"].map(CATEGORY_MAP)
    )

    # Verify required model features
    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = df[FEATURES].copy()

    # Convert all model features to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values with NaN
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df, X


# --------------------------------------------------
# PROCESS NEW FLIGHT-WEATHER RECORDS
# --------------------------------------------------

def process_new_records():

    print("\n" + "=" * 60)
    print("Checking for new flight-weather records...")
    print(
        "Time:",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )

    # 1. READ ENRICHED PARQUET DATA

    try:
        df = pd.read_parquet(DATA_PATH)

    except (FileNotFoundError, OSError):
        print("No processed Parquet data available yet.")
        return

    print("Total input records:", len(df))

    if df.empty:
        print("No input records available.")
        return

    # 2. VALIDATE REQUIRED SOURCE COLUMNS

    required_columns = [
        "icao24",
        "airport_icao",
        "flight_time",
        "altitude",
        "velocity",
        "fltCat"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing source columns: {missing_columns}"
        )

    # 3. FILTER VALID FLIGHT RECORDS

    df = df.dropna(
        subset=["icao24", "airport_icao", "flight_time"]
    ).copy()

    df["flight_time"] = pd.to_datetime(
        df["flight_time"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["flight_time"]
    ).copy()

    if df.empty:
        print("No valid flight records.")
        return

    # 4. REMOVE ALREADY PROCESSED RECORDS

    df["_prediction_key"] = create_prediction_key(df)

    # Remove duplicates within current input
    df = df.drop_duplicates(
        subset=["_prediction_key"],
        keep="last"
    ).copy()

    existing = load_existing_predictions()

    if not existing.empty:

        processed_keys = set(
            existing["_prediction_key"]
            .dropna()
            .astype(str)
        )

        df = df[
            ~df["_prediction_key"].astype(str).isin(
                processed_keys
            )
        ].copy()

    print("New records detected:", len(df))

    if df.empty:
        print("No new records to predict.")
        return

    # 5. FEATURE ENGINEERING

    df, X = prepare_features(df)

    print("\nMissing feature values before imputation:")
    print(X.isna().sum())

    missing_rows = X.isna().any(axis=1).sum()

    print("Records requiring imputation:", missing_rows)
    print("Records passed to model:", len(X))

    # 6. GENERATE RISK PREDICTIONS

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Convert encoded predictions to original risk labels
    predicted_labels = encoder.inverse_transform(
        np.asarray(predictions, dtype=int)
    )

    df["predicted_risk"] = predicted_labels

    # Overall prediction confidence
    df["risk_confidence"] = probabilities.max(axis=1)

    # Store probability for each risk category
    model_classes = encoder.inverse_transform(
        np.asarray(model.classes_, dtype=int)
    )

    for index, risk_class in enumerate(model_classes):

        probability_column = (
            f"risk_probability_{str(risk_class).lower()}"
        )

        df[probability_column] = probabilities[:, index]

    # Prediction metadata
    df["prediction_timestamp"] = datetime.now(
        timezone.utc
    ).isoformat()

    df["prediction_model_version"] = MODEL_VERSION

    # 7. UPDATE PREDICTION DATASET

    if not existing.empty:

        updated = pd.concat(
            [existing, df],
            ignore_index=True
        )

    else:
        updated = df.copy()

    updated = updated.drop_duplicates(
        subset=["_prediction_key"],
        keep="last"
    )

    # Write temporary file before replacing output
    updated.to_parquet(
        TEMP_FILE,
        index=False
    )

    TEMP_FILE.replace(OUTPUT_FILE)

    # 8. DISPLAY RESULTS

    print("\nNew predictions generated:", len(df))
    print("Total stored predictions:", len(updated))

    print("\nOverall risk distribution:")
    print(
        updated["predicted_risk"].value_counts()
    )

    print("\nAirport-wise risk distribution:")

    print(
        updated.groupby(
            ["airport_icao", "predicted_risk"]
        ).size().unstack(fill_value=0)
    )

    print("\nLatest predictions:")

    display_columns = [
        "icao24",
        "airport_icao",
        "predicted_risk",
        "risk_confidence"
    ]

    print(
        df[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\nPredictions updated successfully!")
    print("Output location:", OUTPUT_FILE)


# --------------------------------------------------
# CONTINUOUS PREDICTION LOOP
# --------------------------------------------------

def main():

    print("\nAirOps AI - Continuous Risk Prediction")
    print("Polling interval:", POLL_INTERVAL, "seconds")
    print("Output:", OUTPUT_FILE)

    while True:

        try:
            process_new_records()

        except Exception as error:

            print("\nPREDICTION ERROR:")
            print(type(error).__name__, str(error))

        print(
            f"\nNext prediction check in {POLL_INTERVAL} seconds.",
            flush=True
        )

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
