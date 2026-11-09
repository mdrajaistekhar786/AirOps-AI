
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():

    # Configure MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_server.server"],
    )

    # Connect to MCP server
    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # Initialize connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()

            print("\nAvailable MCP tools:")
            for tool in tools.tools:
                print("-", tool.name)

            # 1. Test live flights tool
            flights = await session.call_tool(
                "live_flights",
                {"limit": 5},
            )

            print("\n========== LIVE FLIGHTS ==========")
            print(flights.content)

            # 2. Test airport weather tool
            weather = await session.call_tool(
                "airport_weather",
                {},
            )

            print("\n========== AIRPORT WEATHER ==========")
            print(weather.content)

            # 3. Test risk predictions tool
            predictions = await session.call_tool(
                "risk_predictions",
                {
                    "limit": 5,
                    "airport": "VOBL",
                },
            )

            print("\n========== RISK PREDICTIONS ==========")
            print(predictions.content)


if __name__ == "__main__":
    asyncio.run(main())
