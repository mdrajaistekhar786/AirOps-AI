
import os
import json
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


def extract_json(response):
    """Extract and parse JSON from Gemini's response."""

    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Gemini returned empty content.")

    content = re.sub(
        r"```(?:json)?",
        "",
        content,
        flags=re.IGNORECASE
    ).strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found.")

    result = json.loads(content[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")

    return result


def normal_status_agent(state: dict) -> dict:
    """
    Generate a normal-operations status when no disruption
    indicators are detected.
    """

    flight_data = state.get("flight_findings", {})
    weather_data = state.get("weather_findings", [])

    total_flights = flight_data.get("total_flights", 0)

    # Count only fresh weather observations
    fresh_weather = [
        item for item in weather_data
        if not item.get("weather_is_stale", False)
    ]

    stale_weather = [
        item for item in weather_data
        if item.get("weather_is_stale", False)
    ]

    airport_count = len(fresh_weather)

    state["operational_recommendations"] = []

    if stale_weather:
        state["overall_assessment"] = (
            f"Weather observations from {len(stale_weather)} "
            f"airports are stale and unavailable for current "
            f"weather-based decisions. {total_flights} flight "
            f"observations were analyzed. Fresh weather updates "
            f"are required before confirming normal operations."
        )
    else:
        state["overall_assessment"] = (
            f"Normal operations status: No significant flight or "
            f"weather disruptions detected across {airport_count} "
            f"monitored airports. {total_flights} flight observations "
            f"were analyzed. Continue routine monitoring."
        )

    state["llm_model"] = "Rule-based Normal Status Agent"
    state.pop("llm_error", None)

    print("Normal Status Agent: No disruption detected.")

    return state


def operations_agent(state: dict) -> dict:
    """
    Combine flight and fresh weather findings to generate
    evidence-based operational recommendations.

    Stale weather observations are retained for audit purposes
    but excluded from current operational decisions.
    """

    flight_data = state.get("flight_findings", {})
    all_weather_data = state.get("weather_findings", [])

    # Separate fresh and stale weather observations
    fresh_weather = [
        item for item in all_weather_data
        if not item.get("weather_is_stale", False)
    ]

    stale_weather = [
        item for item in all_weather_data
        if item.get("weather_is_stale", False)
    ]

    high_risk_airports = flight_data.get(
        "airport_high_risk_count", {}
    )

    # Include airports identified by flight predictions
    # or fresh weather observations
    airport_codes = set(high_risk_airports.keys())

    airport_codes.update(
        item["airport"]
        for item in fresh_weather
        if item.get("airport")
    )

    # Prepare stale weather warning
    stale_warning = ""

    if stale_weather:
        stale_airports = [
            item.get("airport")
            for item in stale_weather
            if item.get("airport")
        ]

        stale_warning = (
            f" WARNING: Weather observations from "
            f"{len(stale_weather)} airports "
            f"({', '.join(stale_airports)}) are stale and "
            f"excluded from current weather-based decisions. "
            f"Fresh weather updates are required."
        )

    # Handle missing airport-level findings
    if not airport_codes:

        state["operational_recommendations"] = []

        state["overall_assessment"] = (
            "No current airport-level findings available."
            + stale_warning
        )

        state["llm_model"] = "Rule-based Data Quality Check"

        state.pop("llm_error", None)

        return state

    # Build weather lookup using only fresh observations
    weather_lookup = {
        item["airport"]: item
        for item in fresh_weather
        if item.get("airport")
    }

    # Build consolidated airport evidence
    airport_evidence = []

    for airport in sorted(airport_codes):

        weather = weather_lookup.get(airport, {})

        airport_evidence.append({

            "airport": airport,

            "high_risk_flights": int(
                high_risk_airports.get(airport, 0)
            ),

            "weather_available": bool(weather),

            "flight_category": weather.get(
                "flight_category", "UNAVAILABLE"
            ),

            "weather_risks": weather.get(
                "weather_risks", []
            ),

            "disruption_detected": weather.get(
                "disruption_detected", False
            ),

            "visibility_km": weather.get("visibility_km"),

            "wind_speed_kmh": weather.get("wind_speed_kmh")

        })

    # Initialize Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if not api_key or not model_name:
        raise ValueError(
            "GEMINI_API_KEY or LLM_MODEL is missing from .env"
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        max_retries=2,
        response_mime_type="application/json"
    )

    # System instructions
    system_prompt = """
    You are an AI Airline Operations Decision Agent.

    Analyze the supplied airport-level flight and weather evidence.

    Rules:

    1. Assign priority: HIGH, MEDIUM, or LOW.
    2. Consider high-risk flight counts and available fresh weather.
    3. Recommend practical operational actions.
    4. Use only the supplied evidence.
    5. Never invent flight counts or weather observations.
    6. Do not claim delays or cancellations occurred.
    7. Consider airports with zero high-risk flights when fresh
       weather disruption indicators are present.
    8. Explain the evidence behind every recommendation.
    9. Weather marked unavailable must not be treated as
       evidence of normal weather or a current disruption.
    10. Never infer current weather conditions from stale data.
    11. Flight-risk predictions can still be considered even
        when weather observations are unavailable.

    Return only a valid JSON object:

    {
        "recommendations": [
            {
                "airport": "AIRPORT_CODE",
                "priority": "HIGH",
                "reason": "Evidence-based explanation.",
                "action": "Recommended action."
            }
        ],
        "overall_assessment": "Overall assessment."
    }
    """

    # User evidence
    human_prompt = f"""
    FLIGHT MONITORING FINDINGS:
    {json.dumps(flight_data)}

    FRESH WEATHER FINDINGS:
    {json.dumps(fresh_weather)}

    STALE WEATHER FINDINGS (AUDIT ONLY):
    {json.dumps(stale_weather)}

    CONSOLIDATED AIRPORT EVIDENCE:
    {json.dumps(airport_evidence)}

    Generate operational recommendations using
    flight predictions and fresh weather evidence only.

    Do not use stale weather observations as current evidence.
    """

    try:

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        result = extract_json(response)

        recommendations = result.get("recommendations", [])

        if not isinstance(recommendations, list):
            raise ValueError(
                "Recommendations must be a JSON list."
            )

        validated_recommendations = []

        for recommendation in recommendations:

            if not isinstance(recommendation, dict):
                continue

            airport = recommendation.get("airport")

            # Reject airports not present in current evidence
            if airport not in airport_codes:
                continue

            priority = recommendation.get(
                "priority", "MEDIUM"
            )

            if priority not in ["HIGH", "MEDIUM", "LOW"]:
                priority = "MEDIUM"

            weather = weather_lookup.get(airport, {})

            validated_recommendations.append({

                "airport": airport,

                "high_risk_flights": int(
                    high_risk_airports.get(airport, 0)
                ),

                "weather_available": bool(weather),

                "weather_category": weather.get(
                    "flight_category", "UNAVAILABLE"
                ),

                "weather_risks": weather.get(
                    "weather_risks", []
                ),

                "priority": priority,

                "reason": str(
                    recommendation.get("reason", "")
                ),

                "action": str(
                    recommendation.get("action", "")
                )

            })

        state["operational_recommendations"] = (
            validated_recommendations
        )

        state["overall_assessment"] = (
            str(result.get("overall_assessment", ""))
            + stale_warning
        )

        state["llm_model"] = model_name

        state.pop("llm_error", None)

    except Exception as e:

        print(f"Gemini Operations Agent Error: {e}")

        state["operational_recommendations"] = []

        state["overall_assessment"] = (
            "LLM-generated operational assessment unavailable."
            + stale_warning
        )

        state["llm_error"] = str(e)

    return state


if __name__ == "__main__":

    print("Gemini-powered Operations Decision Agent ready.")
    print("Normal Status Agent ready.")
    print("Waiting for LangGraph state input.")
