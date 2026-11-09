
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("storage/processed/flight_weather")
MODEL_DIR = Path("ml/models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "VFR": 0,
    "MVFR": 1,
    "IFR": 2,
    "LIFR": 3
}

FEATURES = [
    "distance_km",
    "temperature_c",
    "wind_speed_kmh",
    "visibility_km",
    "altitude_ft",
    "velocity_kmh",
    "flight_category_score"
]


# --------------------------------------------------
# 1. LOAD AND PREPARE DATA
# --------------------------------------------------

df = pd.read_parquet(DATA_PATH)

df = df.dropna(
    subset=["airport_icao", "flight_time"]
).copy()

df["flight_time"] = pd.to_datetime(
    df["flight_time"],
    errors="coerce",
    utc=True
)

df = df.dropna(subset=["flight_time"])

df = df.sort_values(
    "flight_time"
).reset_index(drop=True)

print("Total input records:", len(df))


# --------------------------------------------------
# 2. PREPARE RISK INPUTS
# --------------------------------------------------

for column in [
    "distance_km",
    "wind_speed_kmh",
    "visibility_km",
    "temperature_c"
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["fltCat"] = (
    df["fltCat"]
    .astype("string")
    .str.strip()
    .str.upper()
)

df["flight_category_score"] = (
    df["fltCat"].map(CATEGORY_MAP)
)

# Exclude records with no usable weather indicators.
weather_available = (
    df["fltCat"].isin(CATEGORY_MAP)
    | df["visibility_km"].notna()
    | df["wind_speed_kmh"].notna()
)

excluded = (~weather_available).sum()

df = df.loc[weather_available].copy()

print("Records excluded due to missing weather:", excluded)
print("Usable records:", len(df))


# --------------------------------------------------
# 3. CREATE PROXY RISK LABELS
# --------------------------------------------------

def assign_risk(row):

    score = 0

    # Visibility
    visibility = row["visibility_km"]

    if pd.notna(visibility):
        if visibility < 1.6:
            score += 3
        elif visibility <= 8:
            score += 1

    # Wind speed
    wind = row["wind_speed_kmh"]

    if pd.notna(wind):
        if wind >= 50:
            score += 2
        elif wind >= 30:
            score += 1

    # Distance from airport
    distance = row["distance_km"]

    if pd.notna(distance):
        if distance <= 25:
            score += 2
        elif distance <= 60:
            score += 1

    # Flight category
    category = row["fltCat"]

    if category == "LIFR":
        score += 2
    elif category == "IFR":
        score += 1

    if score >= 6:
        return "HIGH"

    elif score >= 3:
        return "MEDIUM"

    return "LOW"


df["risk_label"] = df.apply(
    assign_risk,
    axis=1
)


# --------------------------------------------------
# 4. INSPECT LABEL DISTRIBUTION
# --------------------------------------------------

print("\nOverall risk distribution:")

print(df["risk_label"].value_counts())

print("\nRisk percentages:")

print(
    (
        df["risk_label"].value_counts(normalize=True) * 100
    ).round(2)
)

print("\nHIGH-risk records by timestamp:")

print(
    df[df["risk_label"] == "HIGH"]
    .groupby("flight_time")
    .size()
)

print("\nHIGH-risk records by airport:")

print(
    df[df["risk_label"] == "HIGH"]
    .groupby("airport_icao")
    .size()
)


# --------------------------------------------------
# 5. FEATURE ENGINEERING
# --------------------------------------------------

# Altitude: metres to feet
df["altitude_ft"] = pd.to_numeric(
    df["altitude"],
    errors="coerce"
) * 3.28084

# Velocity: metres/second to kilometres/hour
df["velocity_kmh"] = pd.to_numeric(
    df["velocity"],
    errors="coerce"
) * 3.6

df[FEATURES] = df[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf],
    np.nan
)

print("\nMissing feature values:")

print(df[FEATURES].isna().sum())


# --------------------------------------------------
# 6. ENCODE TARGET
# --------------------------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(df["risk_label"])

X = df[FEATURES].copy()

print("\nRisk classes:", list(encoder.classes_))


# --------------------------------------------------
# 7. CHRONOLOGICAL TRAIN-TEST EVALUATION
# --------------------------------------------------

split_index = int(len(df) * 0.75)

if split_index == 0 or split_index >= len(df):
    raise ValueError("Insufficient records for train-test split.")

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("\nTraining period:")
print(
    df["flight_time"].iloc[0],
    "to",
    df["flight_time"].iloc[split_index - 1]
)

print("\nTesting period:")
print(
    df["flight_time"].iloc[split_index],
    "to",
    df["flight_time"].iloc[-1]
)

print("\nTraining distribution:")

print(
    pd.Series(
        encoder.inverse_transform(y_train)
    ).value_counts()
)

print("\nTesting distribution:")

print(
    pd.Series(
        encoder.inverse_transform(y_test)
    ).value_counts()
)


# --------------------------------------------------
# 8. BUILD MODEL PIPELINE
# --------------------------------------------------

def build_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])


# --------------------------------------------------
# 9. EVALUATE CHRONOLOGICAL HOLDOUT
# --------------------------------------------------

evaluation_model = build_model()

evaluation_model.fit(
    X_train,
    y_train
)

test_predictions = evaluation_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nChronological Test Accuracy:", round(accuracy, 4))

print("\nChronological Classification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

missing_test_classes = (
    set(encoder.classes_)
    - set(encoder.inverse_transform(y_test))
)

if missing_test_classes:

    print(
        "\nWARNING: These classes are absent from the "
        "chronological test period:",
        sorted(missing_test_classes)
    )

    print(
        "Their predictive performance cannot be evaluated "
        "using this holdout."
    )


# --------------------------------------------------
# 10. TRAIN FINAL PRODUCTION MODEL ON ALL DATA
# --------------------------------------------------

print("\nTraining final production model on all records...")

model = build_model()

model.fit(
    X,
    y
)

print("Final production model trained.")


# --------------------------------------------------
# 11. SAVE MODEL AND METADATA
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "risk_model.pkl"
)

joblib.dump(
    encoder,
    MODEL_DIR / "label_encoder.pkl"
)

joblib.dump(
    FEATURES,
    MODEL_DIR / "risk_features.pkl"
)

print("\nProduction model saved successfully!")

print("Total production training records:", len(X))
print("Risk classes:", list(encoder.classes_))
print("Features:", FEATURES)
print("Preprocessing: Median imputation")
print("Classifier: Random Forest")
