
from mcp.server.fastmcp import FastMCP

from mcp_server.tools.flight_tools import get_live_flights
from mcp_server.tools.weather_tools import get_airport_weather
from mcp_server.tools.risk_tools import get_risk_predictions

mcp = FastMCP("AirOps AI MCP Server")


@mcp.tool()
def live_flights(limit: int = 20) -> list[dict]:
    """Retrieve recent enriched flight records."""
    return get_live_flights(limit)


@mcp.tool()
def airport_weather() -> list[dict]:
    """Retrieve the latest weather observation for each airport."""
    return get_airport_weather()


@mcp.tool()
def risk_predictions(
    limit: int = 20,
    airport: str | None = None,
    risk_level: str | None = None
) -> list[dict]:
    """Retrieve flight risk predictions with optional filters."""
    return get_risk_predictions(limit, airport, risk_level)


if __name__ == "__main__":
    mcp.run(transport="stdio") 
