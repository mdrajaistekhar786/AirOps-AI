
import pandas as pd
from pathlib import Path


def flight_monitor_agent(state: dict) -> dict:
    """
    Analyze only the latest prediction for each aircraft.
    """

    predictions_path = (
        Path(__file__).resolve().parents[1]
        / "storage"
        / "processed"
        / "risk_predictions"
        / "predictions.parquet"
    )

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}"
        )

    df = pd.read_parquet(predictions_path)

    if df.empty:
        raise ValueError("Prediction dataset is empty.")

    required = {
        "icao24",
        "predicted_risk",
        "prediction_timestamp",
        "airport_icao"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {missing}"
        )

    # Sort newest predictions first
    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"],
        errors="coerce",
        utc=True
    )

    df = df.dropna(
        subset=["icao24", "prediction_timestamp"]
    )

    # Keep exactly one latest prediction per aircraft
    latest = (
        df.sort_values(
            "prediction_timestamp",
            ascending=False
        )
        .drop_duplicates(
            subset="icao24",
            keep="first"
        )
        .copy()
    )

    latest["predicted_risk"] = (
        latest["predicted_risk"]
        .astype(str)
        .str.upper()
    )

    high_risk = latest[
        latest["predicted_risk"] == "HIGH"
    ]

    airport_summary = (
        high_risk.groupby("airport_icao")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    findings = {
        "total_flights": int(len(latest)),
        "high_risk_flights": int(len(high_risk)),
        "airport_high_risk_count": {
            str(k): int(v)
            for k, v in airport_summary.items()
        },
        "high_risk_flight_ids": (
            high_risk["icao24"]
            .astype(str)
            .tolist()[:10]
        ),
        "latest_prediction_timestamp": (
            latest["prediction_timestamp"].max().isoformat()
        )
    }

    state["flight_findings"] = findings

    return state


if __name__ == "__main__":

    result = flight_monitor_agent({})

    print("\nLatest Flight Monitoring Results")
    print("--------------------------------")

    for key, value in result["flight_findings"].items():
        print(f"{key}: {value}")
