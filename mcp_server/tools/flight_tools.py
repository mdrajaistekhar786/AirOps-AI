import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLIGHT_DATA = (
    PROJECT_ROOT
    / "storage"
    / "processed"
    / "flight_weather"
)


def get_live_flights(limit: int = 20) -> list[dict]:
    """Retrieve recent enriched flight records."""

    if limit < 1 or limit > 500:
        raise ValueError("Limit must be between 1 and 500.")

    if not FLIGHT_DATA.exists():
        raise FileNotFoundError(
            f"Flight dataset not found: {FLIGHT_DATA}"
        )

    df = pd.read_parquet(FLIGHT_DATA)

    if "flight_time" in df.columns:
        df = df.sort_values(
            "flight_time",
            ascending=False
        )

    df = df.tail(limit)

    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient="records")