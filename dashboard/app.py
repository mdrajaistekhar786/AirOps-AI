
import streamlit as st
import pandas as pd
import requests

from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage" / "processed"

PREDICTIONS_PATH = (
    STORAGE / "risk_predictions" / "predictions.parquet"
)

TRAFFIC_PATH = STORAGE / "airport_traffic"

API_URL = "http://127.0.0.1:8000/operations/report"


# --------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AirOps AI Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AirOps AI")
st.caption("Real-Time Airline Operations Intelligence Platform")


# --------------------------------------------------
# LOAD ML PREDICTIONS
# --------------------------------------------------

def load_predictions():

    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    try:
        # Read the latest Parquet file contents on every refresh.
        df = pd.read_parquet(PREDICTIONS_PATH)

        if df.empty:
            return df

        if "icao24" in df.columns:
            df["icao24"] = (
                df["icao24"]
                .astype("string")
                .str.strip()
            )

        if "flight_time" in df.columns:
            df["flight_time"] = pd.to_datetime(
                df["flight_time"],
                errors="coerce",
                utc=True
            )

        if "prediction_timestamp" in df.columns:
            df["prediction_timestamp"] = pd.to_datetime(
                df["prediction_timestamp"],
                errors="coerce",
                utc=True
            )

        return df

    except Exception as error:
        st.error(f"Could not load ML predictions: {error}")
        return pd.DataFrame()


# --------------------------------------------------
# LATEST PREDICTION PER AIRCRAFT
# --------------------------------------------------

def get_latest_aircraft_predictions(df):

    if df.empty or "icao24" not in df.columns:
        return df.copy()

    latest = df.copy()

    # Remove records without a valid aircraft ID.
    latest = latest.dropna(subset=["icao24"])

    latest = latest[
        latest["icao24"].astype(str).str.strip() != ""
    ]

    # Use prediction timestamp as the primary ordering field.
    if "prediction_timestamp" in latest.columns:

        latest = latest.dropna(
            subset=["prediction_timestamp"]
        )

        sort_column = "prediction_timestamp"

    elif "flight_time" in latest.columns:

        latest = latest.dropna(
            subset=["flight_time"]
        )

        sort_column = "flight_time"

    else:
        return latest.drop_duplicates(
            subset="icao24",
            keep="last"
        )

    latest = latest.sort_values(
        sort_column,
        ascending=False
    )

    latest = latest.drop_duplicates(
        subset="icao24",
        keep="first"
    )

    return latest


# --------------------------------------------------
# FETCH LANGGRAPH AGENT REPORT
# --------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def load_agent_report():

    try:
        response = requests.get(
            API_URL,
            timeout=90
        )

        response.raise_for_status()

        return response.json(), None

    except Exception as error:
        return None, str(error)


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

@st.fragment(run_every="30s")
def render_dashboard():

    st.caption(
        f"Dashboard refreshed: "
        f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )

    # ==================================================
    # 1. ML RISK ANALYTICS
    # ==================================================

    st.divider()
    st.header("Live Flight Risk Intelligence")

    predictions = load_predictions()

    if predictions.empty:

        st.warning(
            "No ML predictions available yet. "
            "Waiting for Spark and the prediction pipeline."
        )

    else:

        # ----------------------------------------------
        # LATEST PREDICTION TIMESTAMP
        # ----------------------------------------------

        if "prediction_timestamp" in predictions.columns:

            latest_time = predictions[
                "prediction_timestamp"
            ].max()

            if pd.notna(latest_time):

                st.caption(
                    f"Latest prediction generated: {latest_time}"
                )

        # ----------------------------------------------
        # HISTORICAL RISK METRICS
        # ----------------------------------------------

        if "predicted_risk" not in predictions.columns:

            st.warning(
                "The prediction file does not contain "
                "the predicted_risk column."
            )

        else:

            risk = (
                predictions["predicted_risk"]
                .astype(str)
                .str.upper()
            )

            total_predictions = len(predictions)

            historical_high = int((risk == "HIGH").sum())
            historical_medium = int((risk == "MEDIUM").sum())
            historical_low = int((risk == "LOW").sum())

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Historical Predictions",
                total_predictions
            )

            c2.metric(
                "HIGH Risk Records",
                historical_high
            )

            c3.metric(
                "MEDIUM Risk Records",
                historical_medium
            )

            c4.metric(
                "LOW Risk Records",
                historical_low
            )

            st.caption(
                "Historical analytics include repeated predictions "
                "for aircraft across time."
            )

            # ------------------------------------------
            # HISTORICAL RISK DISTRIBUTION
            # ------------------------------------------

            st.subheader("Historical Risk Distribution")

            risk_distribution = (
                risk.value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH"],
                    fill_value=0
                )
            )

            left, right = st.columns(2)

            with left:
                st.bar_chart(risk_distribution)

            with right:

                st.subheader(
                    "Historical Airport-wise Risk Distribution"
                )

                if "airport_icao" in predictions.columns:

                    airport_risk = pd.crosstab(
                        predictions["airport_icao"],
                        risk
                    )

                    st.bar_chart(airport_risk)

            # ------------------------------------------
            # LATEST PREDICTION PER AIRCRAFT
            # ------------------------------------------

            st.subheader("Latest Flight Predictions")

            latest_aircraft = get_latest_aircraft_predictions(
                predictions
            )

            st.caption(
                f"Showing latest records for "
                f"{len(latest_aircraft)} unique aircraft."
            )

            # ------------------------------------------
            # CURRENT OPERATIONAL RISK METRICS
            # ------------------------------------------

            if "predicted_risk" in latest_aircraft.columns:

                current_risk = (
                    latest_aircraft["predicted_risk"]
                    .astype(str)
                    .str.upper()
                )

                current_high = int(
                    (current_risk == "HIGH").sum()
                )

                current_medium = int(
                    (current_risk == "MEDIUM").sum()
                )

                current_low = int(
                    (current_risk == "LOW").sum()
                )

                st.subheader(
                    "Current Risk Distribution "
                    "(Latest Prediction per Aircraft)"
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Unique Aircraft",
                    len(latest_aircraft)
                )

                c2.metric(
                    "HIGH Risk",
                    current_high
                )

                c3.metric(
                    "MEDIUM Risk",
                    current_medium
                )

                c4.metric(
                    "LOW Risk",
                    current_low
                )

                current_distribution = (
                    current_risk.value_counts()
                    .reindex(
                        ["LOW", "MEDIUM", "HIGH"],
                        fill_value=0
                    )
                )

                st.bar_chart(current_distribution)

            # ------------------------------------------
            # LATEST PREDICTION TABLE
            # ------------------------------------------

            display_columns = [
                "icao24",
                "airport_icao",
                "predicted_risk",
                "risk_confidence",
                "prediction_timestamp"
            ]

            available_columns = [
                column
                for column in display_columns
                if column in latest_aircraft.columns
            ]

            st.dataframe(
                latest_aircraft[available_columns].head(20),
                use_container_width=True,
                hide_index=True
            )

    # ==================================================
    # 2. AIRPORT TRAFFIC ANALYTICS
    # ==================================================

    st.divider()
    st.header("Airport Traffic Analytics")

    try:

        traffic = pd.read_parquet(TRAFFIC_PATH)

        traffic_summary = (
            traffic.groupby("airport")["aircraft_count"]
            .max()
            .sort_values(ascending=False)
        )

        st.bar_chart(traffic_summary)

    except Exception as error:

        st.info(
            f"Airport traffic data unavailable: {error}"
        )

    # ==================================================
    # 3. LANGGRAPH AGENT INTELLIGENCE
    # ==================================================

    st.divider()
    st.header("AI Agent Operations Intelligence")

    report, api_error = load_agent_report()

    if api_error:

        st.warning(
            f"Agent report unavailable: {api_error}"
        )

    else:

        flight = report.get("flight_monitoring", {})
        weather = report.get("weather_intelligence", [])

        recommendations = report.get(
            "operational_recommendations", []
        )

        assessment = report.get(
            "overall_assessment", ""
        )

        llm_model = report.get(
            "llm_model", "Not available"
        )

        llm_error = report.get("llm_error")

        # ----------------------------------------------
        # AGENT OVERVIEW
        # ----------------------------------------------

        st.subheader("Operational Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Unique Aircraft Monitored",
            flight.get("total_flights", 0)
        )

        c2.metric(
            "High-Risk Aircraft",
            flight.get("high_risk_flights", 0)
        )

        # Count only fresh weather disruptions.
        fresh_weather = [
            item for item in weather
            if not item.get("weather_is_stale", False)
        ]

        stale_weather = [
            item for item in weather
            if item.get("weather_is_stale", False)
        ]

        fresh_disruptions = sum(
            bool(item.get("disruption_detected"))
            for item in fresh_weather
        )

        c3.metric(
            "Airports with Disruptions",
            fresh_disruptions
        )

        c4.metric(
            "Operational Recommendations",
            len(recommendations)
        )

        st.caption(
            f"Fresh weather observations: {len(fresh_weather)} | "
            f"Stale observations excluded: {len(stale_weather)}"
        )

        # ----------------------------------------------
        # AIRPORT-WISE OPERATIONAL RISK
        # ----------------------------------------------

        st.subheader("Airport-wise High-Risk Aircraft")

        airport_counts = flight.get(
            "airport_high_risk_count", {}
        )

        if airport_counts:

            airport_df = pd.DataFrame(
                list(airport_counts.items()),
                columns=[
                    "Airport",
                    "High-Risk Aircraft"
                ]
            )

            airport_df = airport_df.sort_values(
                "High-Risk Aircraft",
                ascending=False
            )

            left, right = st.columns(2)

            with left:

                st.bar_chart(
                    airport_df.set_index("Airport")
                )

            with right:

                st.dataframe(
                    airport_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No airport-level risk findings available."
            )

        # ----------------------------------------------
        # GEMINI ASSESSMENT
        # ----------------------------------------------

        st.subheader("Gemini AI Operations Assessment")

        if llm_error:

            st.warning(
                f"AI assessment unavailable: {llm_error}"
            )

        elif assessment:

            with st.container(border=True):

                st.write(assessment)

                st.caption(
                    f"Powered by: {llm_model}"
                )

        else:

            st.info("No AI assessment available.")

        # ----------------------------------------------
        # WEATHER INTELLIGENCE
        # ----------------------------------------------

        st.subheader("Weather Intelligence")

        weather_df = pd.DataFrame(weather)

        if not weather_df.empty:

            weather_columns = [
                "airport",
                "flight_category",
                "weather_is_stale",
                "weather_age_minutes",
                "data_quality",
                "disruption_detected",
                "weather_risks"
            ]

            available_weather_columns = [
                column
                for column in weather_columns
                if column in weather_df.columns
            ]

            st.dataframe(
                weather_df[available_weather_columns],
                use_container_width=True,
                hide_index=True
            )

            if stale_weather:

                st.warning(
                    "Stale weather observations are displayed "
                    "for reference but excluded from current "
                    "operational decisions."
                )

        else:

            st.info("No weather findings available.")

        # ----------------------------------------------
        # WEATHER LOOKUP FOR RECOMMENDATIONS
        # ----------------------------------------------

        weather_lookup = {}

        for weather_item in weather:

            airport_key = str(
                weather_item.get("airport", "")
            ).strip().upper()

            if airport_key:
                weather_lookup[airport_key] = weather_item

        # ----------------------------------------------
        # OPERATIONAL RECOMMENDATIONS
        # ----------------------------------------------

        st.subheader(
            "AI-Generated Operational Recommendations"
        )

        if recommendations:

            for item in recommendations:

                priority = item.get(
                    "priority", "UNKNOWN"
                )

                airport = item.get(
                    "airport", "Unknown"
                )

                airport_key = str(airport).strip().upper()

                weather_item = weather_lookup.get(
                    airport_key, {}
                )

                with st.container(border=True):

                    st.markdown(
                        f"### {airport} — {priority} PRIORITY"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "High-Risk Aircraft",
                        item.get("high_risk_flights", 0)
                    )

                    # Prefer last-known weather from the
                    # Weather Intelligence Agent.
                    weather_category = weather_item.get(
                        "flight_category"
                    )

                    if not weather_category:
                        weather_category = item.get(
                            "weather_category",
                            "UNAVAILABLE"
                        )

                    weather_is_stale = weather_item.get(
                        "weather_is_stale", False
                    )

                    if weather_is_stale:
                        weather_category = (
                            f"{weather_category} (STALE)"
                        )

                    col2.metric(
                        "Weather Category",
                        weather_category
                    )

                    if item.get("reason"):

                        st.markdown("**AI Analysis**")

                        st.write(item["reason"])

                    if item.get("action"):

                        st.markdown("**Recommended Action**")

                        st.info(item["action"])

        else:

            st.success(
                "No operational recommendations generated."
            )

        # ----------------------------------------------
        # FLIGHT MONITORING DETAILS
        # ----------------------------------------------

        with st.expander("View High-Risk Aircraft IDs"):

            st.write(
                flight.get("high_risk_flight_ids", [])
            )

    # ==================================================
    # 4. FOOTER
    # ==================================================

    st.divider()

    st.caption(
        "AirOps AI | Kafka + Spark Structured Streaming "
        "+ Random Forest + LangGraph + FastAPI"
    )

    if st.button("Refresh Dashboard"):

        load_agent_report.clear()

        st.rerun(scope="app")


render_dashboard()
