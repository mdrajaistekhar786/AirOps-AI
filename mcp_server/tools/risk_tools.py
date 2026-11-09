
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_DATA = (
    PROJECT_ROOT
    / "storage"
    / "processed"
    / "risk_predictions"
    / "predictions.parquet"
)


def get_risk_predictions(
    limit: int = 20,
    airport: str | None = None,
    risk_level: str | None = None
) -> list[dict]:
    """Retrieve latest risk prediction per aircraft."""

    if limit < 1 or limit > 1000:
        raise ValueError("Limit must be between 1 and 1000.")

    if not RISK_DATA.exists():
        raise FileNotFoundError(
            f"Risk prediction dataset not found: {RISK_DATA}"
        )

    df = pd.read_parquet(RISK_DATA)

    # Keep only the latest prediction for each aircraft
    df = df.sort_values(
        "prediction_timestamp",
        ascending=False
    )

    df = df.drop_duplicates(
        subset="icao24",
        keep="first"
    )

    # Apply optional filters after deduplication
    if airport:
        df = df[
            df["airport_icao"].str.upper() == airport.upper()
        ]

    if risk_level:
        risk_level = risk_level.upper()

        if risk_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError(
                "Risk level must be LOW, MEDIUM, or HIGH."
            )

        df = df[
            df["predicted_risk"].str.upper() == risk_level
        ]

    df = df.head(limit)

    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient="records")
