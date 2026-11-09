import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEATHER_DATA = (
    PROJECT_ROOT
    / "storage"
    / "processed"
    / "weather_features"
)


def get_airport_weather() -> list[dict]:
    """Retrieve the latest weather observation for each airport."""

    if not WEATHER_DATA.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {WEATHER_DATA}"
        )

    df = pd.read_parquet(WEATHER_DATA)

    if "icaoId" not in df.columns:
        raise ValueError("Missing icaoId column in weather dataset.")

    if "obsTime" not in df.columns:
        raise ValueError("Missing obsTime column in weather dataset.")

    df = df.dropna(
        subset=["icaoId", "obsTime"]
    )

    latest_weather = (
        df.sort_values("obsTime")
        .drop_duplicates(
            subset=["icaoId"],
            keep="last"
        )
    )

    latest_weather = latest_weather.astype(object).where(
        pd.notna(latest_weather), None
    )

    return latest_weather.to_dict(orient="records")