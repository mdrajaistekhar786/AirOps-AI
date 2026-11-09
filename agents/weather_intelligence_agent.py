
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def fetch_airport_weather():
    """Fetch latest airport weather observations through MCP."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        cwd=str(PROJECT_ROOT)
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "airport_weather",
                {}
            )

            if result.isError:
                raise RuntimeError(
                    f"MCP airport_weather failed: {result.content}"
                )

            if result.structuredContent is not None:
                data = result.structuredContent

                if isinstance(data, dict) and "result" in data:
                    data = data["result"]

                return data

            for item in result.content:
                if hasattr(item, "text"):
                    return json.loads(item.text)

            raise ValueError(
                "MCP returned no weather data."
            )


def weather_intelligence_agent(state: dict) -> dict:
    """
    Analyze airport weather observations,
    detect disruptions and identify stale data.
    """

    latest_weather = asyncio.run(fetch_airport_weather())

    findings = []

    for row in latest_weather:

        airport = row.get("icaoId")

        if not airport:
            continue

        category = str(
            row.get("fltCat", "UNKNOWN")
        ).upper()

        wind = row.get("wind_speed_kmh")
        visibility = row.get("visibility_km")
        temperature = row.get("temperature_c")

        # Calculate weather observation age
        obs_timestamp = row.get("obsTime")

        if obs_timestamp is not None:
            observation_datetime = datetime.fromtimestamp(
                float(obs_timestamp),
                tz=timezone.utc
            )

            current_datetime = datetime.now(timezone.utc)

            weather_age_minutes = (
                current_datetime - observation_datetime
            ).total_seconds() / 60

            weather_age_minutes = max(
                0, weather_age_minutes
            )

            weather_is_stale = weather_age_minutes > 30

        else:
            weather_age_minutes = None
            weather_is_stale = True

        # Identify weather risks
        risks = []

        if category in ["IFR", "LIFR"]:
            risks.append(
                f"Low flight category: {category}"
            )

        if wind is not None and wind >= 30:
            risks.append(
                f"High wind: {wind} km/h"
            )

        if visibility is not None and visibility < 3:
            risks.append(
                f"Low visibility: {visibility} km"
            )

        # Build airport-level findings
        findings.append({

            "airport": airport,

            "observation_time": str(
                obs_timestamp if obs_timestamp is not None else ""
            ),

            "flight_category": category,

            "temperature_c": temperature,

            "wind_speed_kmh": wind,

            "visibility_km": visibility,

            "weather_age_minutes": (
                round(weather_age_minutes, 2)
                if weather_age_minutes is not None
                else None
            ),

            "weather_is_stale": weather_is_stale,

            "data_quality": (
                "STALE" if weather_is_stale else "FRESH"
            ),

            "weather_risks": risks,

            "disruption_detected": bool(risks)

        })

    state["weather_findings"] = findings

    return state


if __name__ == "__main__":

    result = weather_intelligence_agent({})

    print("\nWeather Intelligence Agent Results:")
    print("----------------------------------")

    for finding in result["weather_findings"]:
        print(finding)
