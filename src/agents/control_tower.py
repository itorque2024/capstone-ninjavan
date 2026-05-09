"""
Multi-Agent Control Tower — LangGraph orchestrator for NinjaVan daily operations.
Implements Problem 10: 5 coordinated agents that break operational silos.

Two modes:
  - Batch (event_type="daily"):
      demand → route → warehouse → pricing → proactive_customer_agent → coordinator → END
      Agents share TowerState: demand feeds route (riders needed), route feeds warehouse
      (zone assignments), warehouse feeds pricing (demand ratio), pricing feeds customer agent.

  - Reactive (event_type="customer"):
      customer query → multi-agent chatbot orchestrator → END

Note: Fraud Detection (P5) and the RAG Chatbot (P6) are standalone FastAPI endpoints.
They adapt to demand state via the global_demand_volume API parameter, not as nodes
in this LangGraph graph.
"""
import numpy as np
import time
import math
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Optional, List, Dict, Any
from .demand_agent import run_demand_agent
from .route_agent import run_route_agent
from .warehouse_agent import run_warehouse_agent
from .pricing_agent import run_pricing_agent

# ── Decision Log Entry ────────────────────────────────────────────────────────
class DecisionEntry(TypedDict):
    agent: str
    timestamp: float
    decision: str
    reasoning: str
    impact: str
    conflicts: List[str]

class TowerState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    event_type: str            # "daily" | "customer" | "scenario"
    scenario_id: Optional[str] # "spike", "lockdown", "fuel_surge"
    forecast_horizon: Optional[int]
    spike_threshold: Optional[float]
    delivery_zones: Optional[List[str]]
    customer_query: Optional[str]
    conversation_history: Optional[List[dict]]

    # ── Agent outputs ─────────────────────────────────────────────────────────
    demand_result: Optional[dict]
    route_result: Optional[dict]
    warehouse_result: Optional[dict]
    pricing_result: Optional[dict]
    customer_result: Optional[dict]

    # ── Coordination & Intelligence Layer ─────────────────────────────────────
    decision_logs: List[DecisionEntry]
    conflicts: List[str]
    metrics: Dict[str, Any]    # decision_latency, plan_stability, etc.
    alerts: List[str]


def _log(state: TowerState, agent: str, decision: str, reasoning: str, impact: str = "", conflicts: List[str] = None) -> List[DecisionEntry]:
    logs = list(state.get("decision_logs", []))
    logs.append({
        "agent": agent,
        "timestamp": time.time(),
        "decision": decision,
        "reasoning": reasoning,
        "impact": impact,
        "conflicts": conflicts or []
    })
    return logs

# ── Wrapped Agent Nodes ───────────────────────────────────────────────────────

def wrapped_demand_agent(state: TowerState) -> TowerState:
    start_time = time.time()
    res = run_demand_agent(state)
    d_res = res.get("demand_result", {})

    reasoning = "Normal trend analysis."
    if d_res.get("spike_detected"):
        reasoning = f"Detected demand spike of {d_res.get('peak_value'):,} parcels on {d_res.get('peak_date')}."

    impact = f"Required riders increased to {math.ceil(d_res.get('peak_value', 0)/80)}."

    new_logs = _log(state, "Demand Agent", "Forecast Finalized", reasoning, impact)
    return {**res, "decision_logs": new_logs}

def wrapped_route_agent(state: TowerState) -> TowerState:
    res = run_route_agent(state)
    r_res = res.get("route_result", {})

    reasoning = f"Optimized for {r_res.get('riders_needed', 0)} riders across {r_res.get('zones_covered', len(state.get('delivery_zones', [])))} zones."
    impact = f"Total distance: {r_res.get('total_distance_km', 0):.1f}km."

    new_logs = _log(state, "Route Agent", "Route Plan Generated", reasoning, impact)
    return {**res, "decision_logs": new_logs}

def wrapped_warehouse_agent(state: TowerState) -> TowerState:
    res = run_warehouse_agent(state)
    w_res = res.get("warehouse_result", {})

    reasoning = f"Zone assignments adjusted based on {w_res.get('total_daily_picks', 0)} predicted picks."
    impact = f"Walking distance reduced by {w_res.get('distance_saved_pct', 0):.1f}%."

    new_logs = _log(state, "Warehouse Agent", "Slotting Optimised", reasoning, impact)
    return {**res, "decision_logs": new_logs}

def wrapped_pricing_agent(state: TowerState) -> TowerState:
    res = run_pricing_agent(state)
    p_res = res.get("pricing_result", {})

    surge = "active" if p_res.get("surge_active") else "inactive"
    reasoning = f"Surge {surge} (Demand ratio: {p_res.get('demand_ratio')})."
    impact = f"Base multiplier: {p_res.get('demand_multiplier')}x."

    # Potential conflict: High surge might annoy customers
    conflicts = []
    if p_res.get("demand_multiplier", 1.0) > 1.3:
        conflicts.append("High pricing surge may lead to customer churn.")

    new_logs = _log(state, "Pricing Agent", "Dynamic Rates Set", reasoning, impact, conflicts)
    return {**state, "decision_logs": new_logs, "pricing_result": res.get("pricing_result"), "conflicts": conflicts}

def wrapped_proactive_customer_agent(state: TowerState) -> TowerState:
    """Agent 4: Proactive Customer Support (Daily Batch)"""
    p_res = state.get("pricing_result", {})
    r_res = state.get("route_result", {})
    d_res = state.get("demand_result", {})

    spike = d_res.get("spike_detected", False)
    delay_risk = r_res.get("estimated_duration_hrs", 0) > 8

    if spike and delay_risk:
        action = "Drafted proactive SMS/Email to customers."
        reasoning = "High demand and long route times detected. Pre-warning customers of 24h delays."
        impact = "Deflected est. 2,500 WISMO tickets."
        status = "⚠️ MASS ALERT DRAFTED"
    elif spike:
        action = "Updated Chatbot RAG context."
        reasoning = "Demand spike detected. Added context to Chatbot that deliveries may occur up to 10 PM."
        impact = "Deflected est. 500 tickets."
        status = "🟠 RAG UPDATED"
    else:
        action = "No proactive action needed."
        reasoning = "Operations are stable."
        impact = "Normal operations."
        status = "🟢 NORMAL"

    customer_proactive = {
        "status": status,
        "action": action,
        "impact": impact
    }

    new_logs = _log(state, "Customer Agent", action, reasoning, impact)
    return {**state, "decision_logs": new_logs, "customer_result": customer_proactive}

def coordinator_agent(state: TowerState) -> TowerState:
    """The Arbitrator: Reviews all decisions, flags conflicts, and calculates metrics."""
    logs = state.get("decision_logs", [])
    all_conflicts = list(state.get("conflicts", []))

    # 1. Calculate Decision Latency
    if logs:
        latency = logs[-1]["timestamp"] - logs[0]["timestamp"]
    else:
        latency = 0

    # 2. Strategic Arbitration
    # If route efficiency is low but pricing is high, that's a "System Tension"
    p_res = state.get("pricing_result", {})
    r_res = state.get("route_result", {})

    if p_res.get("surge_active") and r_res.get("estimated_duration_hrs", 0) > 8:
        all_conflicts.append("Dual Pressure: High prices AND long delivery times detected.")

    # 3. Final Recommendation
    recommendation = "Plan finalized. No critical blockers."
    if all_conflicts:
        recommendation = f"Plan active with {len(all_conflicts)} tactical conflicts. Monitor churn."

    demand_res = state.get("demand_result", {})
    route_res  = state.get("route_result", {})
    forecast   = demand_res.get("forecast", {})
    baseline   = forecast.get("baseline_avg", 1)
    peak       = demand_res.get("peak_value", baseline)
    spike_mult = round(peak / baseline, 2) if baseline else 1.0
    riders     = route_res.get("riders_needed", 0)
    agents_ran = len([l for l in logs if l["agent"] != "Control Tower"])

    metrics = {
        "decision_latency": round(latency, 2),
        "agents_coordinated": agents_ran,
        "demand_spike": spike_mult,
        "riders_dispatched": riders,
    }

    new_logs = _log(state, "Control Tower", "Coordination Complete", recommendation, f"Spike: {spike_mult}x | Riders: {riders}")

    # Final Alerts
    alerts = list(state.get("alerts", []))
    for c in all_conflicts:
        alerts.append(f"CONFLICT: {c}")

    return {**state, "decision_logs": new_logs, "conflicts": all_conflicts, "metrics": metrics, "alerts": alerts}

def _run_customer_agent(state: dict) -> dict:
    from src.agents.chatbot.orchestrator import chat
    query = state.get("customer_query", "")
    history = state.get("conversation_history", [])
    if not query:
        return {**state, "customer_result": {"error": "No query provided."}}
    result = chat(query, history)
    return {**state, "customer_result": result}


def router(state: TowerState) -> str:
    return "customer" if state.get("event_type") == "customer" else "daily"


def build_control_tower() -> StateGraph:
    graph = StateGraph(TowerState)

    graph.add_node("demand_agent",    wrapped_demand_agent)
    graph.add_node("route_agent",     wrapped_route_agent)
    graph.add_node("warehouse_agent", wrapped_warehouse_agent)
    graph.add_node("pricing_agent",   wrapped_pricing_agent)
    graph.add_node("proactive_customer_agent", wrapped_proactive_customer_agent)
    graph.add_node("customer_agent",  _run_customer_agent)
    graph.add_node("coordinator",     coordinator_agent)

    graph.add_conditional_edges(START, router, {
        "daily":    "demand_agent",
        "customer": "customer_agent",
    })

    graph.add_edge("demand_agent",    "route_agent")
    graph.add_edge("route_agent",     "warehouse_agent")
    graph.add_edge("warehouse_agent", "pricing_agent")
    graph.add_edge("pricing_agent",   "proactive_customer_agent")
    graph.add_edge("proactive_customer_agent", "coordinator")
    graph.add_edge("coordinator",     END)

    graph.add_edge("customer_agent", END)

    return graph.compile()


tower = build_control_tower()
