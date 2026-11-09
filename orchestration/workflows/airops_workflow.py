
import sys
from pathlib import Path
from typing import TypedDict, Any
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, START, END

from agents.flight_monitor_agent import flight_monitor_agent
from agents.weather_intelligence_agent import weather_intelligence_agent
from agents.operations_agent import (
    operations_agent,
    normal_status_agent
)


# Shared state between agents
class AirOpsState(TypedDict, total=False):
    flight_findings: dict[str, Any]
    weather_findings: list[dict[str, Any]]
    operational_recommendations: list[dict[str, Any]]
    overall_assessment: str
    llm_model: str
    llm_error: str


# Measure agent execution time
def timed_agent(agent_name, agent_function):

    def wrapper(state):

        start_time = datetime.now()

        print(
            f"[{start_time:%H:%M:%S.%f}] "
            f"{agent_name} STARTED",
            flush=True
        )

        result = agent_function(state)

        end_time = datetime.now()

        print(
            f"[{end_time:%H:%M:%S.%f}] "
            f"{agent_name} FINISHED",
            flush=True
        )

        print(
            f"{agent_name} execution time: "
            f"{(end_time - start_time).total_seconds():.2f} seconds\n",
            flush=True
        )

        return result

    return wrapper


# Conditional routing decision
def check_disruption(state: AirOpsState) -> str:
    """
    Route to Operations Decision Agent when:
    1. High-risk flights are detected.
    2. Fresh weather observations indicate disruption.

    Stale weather observations cannot independently
    trigger operational disruption routing.
    """

    flight_data = state.get("flight_findings", {})
    weather_data = state.get("weather_findings", [])

    high_risk_flights = flight_data.get(
        "high_risk_flights", 0
    )

    # Consider only fresh weather observations
    fresh_weather = [
        item for item in weather_data
        if not item.get("weather_is_stale", False)
    ]

    stale_weather = [
        item for item in weather_data
        if item.get("weather_is_stale", False)
    ]

    # Check disruption using fresh weather only
    weather_disruption = any(
        item.get("disruption_detected", False)
        for item in fresh_weather
    )

    if stale_weather:
        print(
            f"\nWarning: {len(stale_weather)} stale weather "
            "observations excluded from routing decisions.",
            flush=True
        )

    # Route based on flight risks or fresh weather disruption
    if high_risk_flights > 0 or weather_disruption:

        print(
            "\nDisruption indicators detected: "
            "Routing to Operations Decision Agent.",
            flush=True
        )

        return "operations_decision"

    print(
        "\nNo current disruption indicators detected: "
        "Routing to Normal Status Agent.",
        flush=True
    )

    return "normal_status"


# Checkpoint node for conditional routing
def disruption_check_node(state: AirOpsState) -> dict:
    return {}


# Build LangGraph workflow
workflow = StateGraph(AirOpsState)


# Register agents
workflow.add_node(
    "flight_monitor",
    timed_agent(
        "Flight Monitoring Agent",
        flight_monitor_agent
    )
)

workflow.add_node(
    "weather_intelligence",
    timed_agent(
        "Weather Intelligence Agent",
        weather_intelligence_agent
    )
)

workflow.add_node(
    "operations_decision",
    timed_agent(
        "Operations Decision Agent",
        operations_agent
    )
)

workflow.add_node(
    "normal_status",
    timed_agent(
        "Normal Status Agent",
        normal_status_agent
    )
)

workflow.add_node(
    "check_disruption",
    disruption_check_node
)


# Parallel execution of monitoring agents
workflow.add_edge(START, "flight_monitor")
workflow.add_edge(START, "weather_intelligence")


# Wait for both agents before checking disruptions
workflow.add_edge(
    ["flight_monitor", "weather_intelligence"],
    "check_disruption"
)


# Conditional routing
workflow.add_conditional_edges(
    "check_disruption",
    check_disruption,
    {
        "operations_decision": "operations_decision",
        "normal_status": "normal_status"
    }
)


# Both paths terminate the workflow
workflow.add_edge("operations_decision", END)
workflow.add_edge("normal_status", END)


# Compile workflow
air_ops_graph = workflow.compile()


# Execute workflow
if __name__ == "__main__":

    print(
        "\nStarting AirOps AI workflow...\n",
        flush=True
    )

    workflow_start = datetime.now()

    result = air_ops_graph.invoke({})

    workflow_end = datetime.now()

    print(
        "\n========== AIR OPS INTELLIGENCE REPORT ==========\n"
    )

    print("FLIGHT FINDINGS:")
    print(result.get("flight_findings", {}))

    print("\nWEATHER FINDINGS:")

    for item in result.get("weather_findings", []):
        print(item)

    print("\nOPERATIONS ASSESSMENT:")
    print(
        result.get(
            "overall_assessment",
            "Not available"
        )
    )

    print("\nOPERATIONAL RECOMMENDATIONS:")

    for item in result.get(
        "operational_recommendations", []
    ):
        print(item)

    print("\nMODEL / AGENT:")
    print(
        result.get(
            "llm_model",
            "Not available"
        )
    )

    if result.get("llm_error"):

        print("\nLLM ERROR:")
        print(result["llm_error"])

    print("\nTOTAL WORKFLOW TIME:")

    print(
        f"{(workflow_end - workflow_start).total_seconds():.2f} seconds"
    )
