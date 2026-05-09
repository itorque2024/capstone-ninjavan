import sys
import os
import json
import math
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import AI Agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agents.demand_agent import run_demand_agent
from src.agents.route_agent import run_route_agent
from src.agents.fraud_agent import run_fraud_agent
from src.agents.chatbot.orchestrator import chat
from src.agents.control_tower import tower

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build ChromaDB from RAG documents on first startup if empty
    try:
        from src.utils.chroma_setup import _DEFAULT_CHROMA_PATH, build_chroma_from_files
        import chromadb
        _client = chromadb.PersistentClient(path=_DEFAULT_CHROMA_PATH)
        _col = _client.get_or_create_collection("ninjavan_kb")
        if _col.count() == 0:
            build_chroma_from_files()
    except Exception as e:
        print(f"ChromaDB init warning: {e}")
    yield

app = FastAPI(title="NinjaVan Control Tower API", lifespan=lifespan)

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

class DemandRequest(BaseModel):
    start_date: str
    horizon: int
    warehouse: str
    marketing: float
    is_sale: bool
    spike_pct: float
    sale_event: str = "none"

class RouteRequest(BaseModel):
    sim_mode: str
    manual_rain: float
    manual_riders: int
    dynamic_demand_volume: int

class ChatRequest(BaseModel):
    message: str
    global_demand_volume: int

class FraudRequest(BaseModel):
    threshold: float
    handled_claims: List[str]
    global_demand_volume: int

class ScenarioRequest(BaseModel):
    scenario_id: str
    custom_multiplier: Optional[float] = None

_SALE_EVENT_MAP = {
    "spike":       ("11.11 Mega Sale",       150),
    "sale_1212":   ("12.12 Year-End Sale",    120),
    "blackfriday": ("Black Friday",           100),
    "cny":         ("Chinese New Year",        80),
    "payday":      ("Payday Sale",             60),
    "flash":       ("Flash Sale",             200),
}

@app.post("/api/demand")
async def api_demand(req: DemandRequest):
    event_label = "sale event"
    is_sale = req.is_sale
    input_spike_pct = req.spike_pct
    if req.sale_event in _SALE_EVENT_MAP:
        event_label, input_spike_pct = _SALE_EVENT_MAP[req.sale_event]
        is_sale = True

    res = run_demand_agent({
        "event_type": "demand",
        "forecast_horizon": req.horizon,
        "forecast_start_date": req.start_date,
        "spike_threshold": 1.0 + input_spike_pct / 100,
        "warehouse": req.warehouse,
        "marketing_spend": req.marketing,
        "is_sale": is_sale,
        "scenario_id": req.sale_event if req.sale_event != "none" else None,
    })
    dres = res.get("demand_result", {})
    fc = dres.get("forecast", {})

    if not fc.get("values"):
        return {"error": "Model not trained."}

    dates = fc["dates"]
    values = fc["values"]
    baseline = fc.get("baseline_avg", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name="Forecast", line=dict(color="#10b981", width=3)))
    fig.add_hline(y=baseline, line_dash="dot", line_color="#475569", annotation_text="Baseline")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b")
    )

    peak_date = dres.get('peak_date', 'N/A')
    peak_val = dres.get('peak_value', 0)
    total = dres.get('total_parcels', sum(values))
    # spike_pct = peak day above daily average (NOT total volume change)
    spike_pct = round((peak_val / baseline - 1) * 100) if baseline > 0 else 0
    # total_increase_pct = how much MORE total parcels over the horizon vs a flat-baseline period
    baseline_total = baseline * req.horizon
    total_increase_pct = round((total - baseline_total) / max(baseline_total, 1) * 100, 1)
    spike_detected = dres.get('spike_detected', False) or is_sale

    riders_needed = math.ceil(peak_val / 80)
    warehouse_shifts = 3 if spike_pct > 100 else (2 if spike_pct > 30 else 1)
    risk_level = "high" if spike_pct > 100 else ("medium" if spike_pct > 30 else "low")

    surge_days = {"spike": 4, "sale_1212": 4, "blackfriday": 4, "cny": 7, "payday": 4, "flash": 3}
    n_surge_days = surge_days.get(req.sale_event, 1)

    if spike_detected and spike_pct > 100:
        model_decision = f"Peak day spike: {spike_pct}% above daily average on {peak_date}"
        why = (f"The {event_label} drives a {n_surge_days}-day demand surge. "
               f"Peak day ({peak_date}): {int(peak_val):,} parcels — {spike_pct}% above the daily avg of {int(baseline):,}. "
               f"Demand builds 1 day before (pre-sale orders), peaks on the sale day, "
               f"then tails off 2-3 days after (post-sale deliveries). "
               f"Total volume over the {req.horizon}-day horizon: {int(total):,} parcels "
               f"(+{total_increase_pct:.0f}% vs a normal {req.horizon}-day period of {int(baseline_total):,}).")
        recommendations = [
            f"Peak day {peak_date}: deploy {riders_needed} riders (normal ops = ~{int(baseline/80)+1})",
            f"Pre-sale day: increase staffing by ~60% — demand builds before the sale goes live",
            f"Post-sale days 1-2: maintain {int(riders_needed * 0.6)} riders for delivery processing",
            f"Run {warehouse_shifts} warehouse sorting shifts on {peak_date} and 2 shifts the days around it",
            "Pre-position stock in high-demand zones 1-2 days before peak",
        ]
    elif spike_detected:
        model_decision = f"Moderate surge: peak day +{spike_pct}% above daily average on {peak_date}"
        why = (f"The {event_label} creates a {n_surge_days}-day moderate surge. "
               f"Peak day reaches {int(peak_val):,} parcels (+{spike_pct}% vs daily avg). "
               f"Total volume over {req.horizon} days: +{total_increase_pct:.0f}%.")
        recommendations = [
            f"Peak day {peak_date}: deploy {riders_needed} riders (+{riders_needed - int(baseline/80) - 1} above normal)",
            f"Elevate staffing for {n_surge_days} days around {peak_date}",
            "Monitor throughput — escalate to 3 shifts if actual volume exceeds forecast",
        ]
    else:
        model_decision = f"Normal operations — demand stable at {int(baseline):,} parcels/day"
        why = (f"No sale event or anomaly detected over the {req.horizon}-day horizon. "
               f"All {req.horizon} days forecast at the daily baseline of {int(baseline):,} parcels. "
               f"Total expected volume: {int(total):,} parcels.")
        recommendations = [
            f"Standard deployment: {riders_needed} riders per day",
            "1 warehouse sorting shift — no extra staffing needed",
            "No proactive customer alerts required",
        ]

    return {
        "peak_date": peak_date,
        "peak_val": int(peak_val),
        "total": int(total),
        "baseline": int(baseline),
        "spike_pct": spike_pct,
        "total_increase_pct": total_increase_pct,
        "plot_json": json.loads(fig.to_json()),
        "global_demand_volume": int(peak_val) if is_sale else int(baseline),
        "insights": {
            "model_decision": model_decision,
            "why": why,
            "recommendations": recommendations,
            "risk_level": risk_level,
            "riders_needed": riders_needed,
            "warehouse_shifts": warehouse_shifts,
        }
    }

@app.post("/api/route")
async def api_route(req: RouteRequest):
    from src.utils.sg_districts import DISTRICTS, REGIONAL_HUBS

    state = {
        "event_type": "route",
        "demand_result": {"forecast": {"values": [req.dynamic_demand_volume]}},
        "simulation_mode": req.sim_mode,
        "manual_rain": req.manual_rain,
        "manual_riders": req.manual_riders
    }
    res = run_route_agent(state)
    rres = res.get("route_result", {})
    routes = rres.get("routes", [])

    fig = go.Figure()

    hub_lats = [DISTRICTS[h]["lat"] for h in REGIONAL_HUBS]
    hub_lons = [DISTRICTS[h]["lon"] for h in REGIONAL_HUBS]
    hub_names = [DISTRICTS[h]["name"] for h in REGIONAL_HUBS]

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=hub_lons,
        lat=hub_lats,
        marker=dict(size=15, color="#f43f5e", opacity=0.9),
        hoverinfo="text",
        hovertext=hub_names,
        name="Hubs"
    ))

    colors = px.colors.qualitative.Set3
    for idx, route_data in enumerate(routes):
        seq = route_data["sequence"]
        lats = [DISTRICTS[s]["lat"] for s in seq]
        lons = [DISTRICTS[s]["lon"] for s in seq]

        c = colors[idx % len(colors)]
        fig.add_trace(go.Scattermapbox(
            mode="lines+markers",
            lon=lons,
            lat=lats,
            marker=dict(size=8, color=c),
            line=dict(width=3, color=c),
            name=f"Rider {idx+1}"
        ))

    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_zoom=10.5,
        mapbox_center={"lat": 1.3521, "lon": 103.8198},
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.8)",
            font=dict(color="#f8fafc", size=10)
        )
    )

    metrics = {
        "total_distance": round(rres.get("total_distance_km", 0), 1),
        "total_time": round(rres.get("max_duration_hrs", 0) * 60, 0),
        "riders": rres.get("riders_needed", 0),
        "weather": rres.get("weather_delay_factor", 1.0)
    }

    return {
        "metrics": metrics,
        "plot_json": json.loads(fig.to_json())
    }

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    ops_context = ""
    if req.global_demand_volume > 10000:
        ops_context = (
            f"LIVE OPS ALERT: Current daily parcel volume is {req.global_demand_volume:,} — "
            "this is a major demand spike. Delivery SLAs are extended by 1–2 business days. "
            "Customers should expect delays and plan accordingly."
        )

    try:
        result = chat(req.message, [], ops_context=ops_context)
    except Exception as e:
        return {
            "answer": f"Sorry, the chatbot is unavailable: {str(e)}",
            "intent": "error", "agent_name": "Error", "agent_emoji": "⚠️",
            "agents_involved": [], "debug_log": str(e), "escalated": False, "sources": 0,
        }

    return {
        "answer":          result["answer"],
        "intent":          result["intent"],
        "agent_name":      result["agent_name"],
        "agent_emoji":     result["agent_emoji"],
        "agents_involved": result.get("agents_involved", []),
        "debug_log":       result["debug_log"],
        "escalated":       result["escalated"],
        "sources":         result["sources"],
        "llm_source":      result.get("llm_source", "gemini"),
    }

def _generate_fraud_sample(n: int = 500) -> pd.DataFrame:
    """Generate a reproducible synthetic fraud sample — no CSV dependency."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "parcel_id":        range(1, n + 1),
        "parcel_value":     rng.exponential(scale=150, size=n).clip(10, 5000).astype(int),
        "prior_claims":     rng.choice([0, 0, 0, 0, 1, 1, 2, 3], size=n),
        "account_age_days": rng.randint(30, 3650, size=n),
        "claim_lag_days":   rng.randint(0, 30, size=n),
    })
    # inject ~5% high-risk records
    mask = rng.rand(n) < 0.05
    df.loc[mask, "prior_claims"]     = rng.randint(3, 8,    size=mask.sum())
    df.loc[mask, "claim_lag_days"]   = rng.randint(25, 60,  size=mask.sum())
    df.loc[mask, "parcel_value"]     = rng.randint(500, 5000, size=mask.sum())
    return df


@app.post("/api/fraud")
async def api_fraud(req: FraudRequest):
    dynamic_threshold = req.threshold
    if req.global_demand_volume > 15000:
        dynamic_threshold += 0.15
    elif req.global_demand_volume > 8000:
        dynamic_threshold += 0.05

    # Try loading real CSV; fall back to synthetic data if not present (e.g. HF Spaces)
    try:
        from src.utils.data_loader import load_fraud_data
        df_raw = load_fraud_data()
        sample_fraud = df_raw.sample(n=min(500, len(df_raw)), random_state=42).copy()
    except Exception:
        sample_fraud = _generate_fraud_sample(500)

    _claim_cols = ["parcel_id", "parcel_value", "prior_claims", "account_age_days", "claim_lag_days"]
    _claims_input = sample_fraud[[c for c in _claim_cols if c in sample_fraud.columns]].rename(columns={"parcel_id": "claim_id"}).to_dict("records")

    try:
        res = run_fraud_agent({"event_type": "fraud", "claims": _claims_input})
    except Exception as e:
        return {"error": f"Fraud model error: {str(e)}", "auto_approved": 0, "flagged_count": 0, "total_claims": 0, "queue": [], "plot_json": None}
    fres = res.get("fraud_result") or {}
    if fres.get("error"):
        return {"error": fres["error"], "auto_approved": 0, "flagged_count": 0, "total_claims": 0, "queue": [], "plot_json": None}
    all_claims = fres.get("all_claims", [])

    flagged = [c for c in all_claims if c["risk_score"] >= dynamic_threshold and str(c["claim_id"]) not in req.handled_claims]

    total_claims = len(all_claims)
    auto_approved = total_claims - len(flagged)

    queue_data = [{"id": str(c["claim_id"]), "value": c["parcel_value"], "risk": c["risk_pct"], "action": c["action"]} for c in flagged]

    df_plot = pd.DataFrame(all_claims)
    fig = go.Figure()
    if not df_plot.empty and all(col in df_plot.columns for col in ["account_age_days", "prior_claims", "risk_score"]):
        fig.add_trace(go.Scatter(
            x=df_plot["account_age_days"].tolist(),
            y=df_plot["prior_claims"].tolist(),
            mode="markers",
            marker=dict(
                size=8,
                color=df_plot["risk_score"].tolist(),
                colorscale=[[0, "#10b981"], [0.4, "#f59e0b"], [0.75, "#ef4444"], [1, "#dc2626"]],
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(title="Risk", tickfont=dict(color="#94a3b8")),
                opacity=0.85,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>Account Age:</b> %{x} days<br>"
                "<b>Prior Claims:</b> %{y}<br>"
                "<b>Risk:</b> %{marker.color:.2f}<extra></extra>"
            ),
        ))
    else:
        fig.add_annotation(text="No claims data", showarrow=False, font=dict(color="#94a3b8", size=14))
    fig.update_layout(
        xaxis_title="Account Age (days)",
        yaxis_title="Prior Claims",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
    )

    claims_data = [
        {"id": str(c["claim_id"]), "age": int(c["account_age_days"]),
         "claims": int(c["prior_claims"]), "risk": round(c["risk_score"], 3)}
        for c in all_claims
    ]

    return {
        "auto_approved": auto_approved,
        "flagged_count": len(flagged),
        "total_claims": total_claims,
        "queue": queue_data,
        "claims_data": claims_data,
        "plot_json": json.loads(fig.to_json())
    }

@app.post("/api/simulate")
async def api_simulate(req: ScenarioRequest):
    from src.utils.sg_districts import ACTIVE_ZONES
    state = {
        "event_type": "daily",
        "scenario_id": req.scenario_id,
        "custom_multiplier": req.custom_multiplier,
        "forecast_horizon": 7,
        "delivery_zones": ACTIVE_ZONES
    }
    result = tower.invoke(state)

    d = result.get("demand_result", {})
    r = result.get("route_result", {})
    w = result.get("warehouse_result", {})
    p = result.get("pricing_result", {})
    c = result.get("customer_result", {})
    forecast = d.get("forecast", {})
    baseline = forecast.get("baseline_avg", 1)
    peak = d.get("peak_value", baseline)
    spike_mult = round(peak / baseline, 2) if baseline else 1.0

    agents = [
        {
            "name": "Demand Agent",
            "emoji": "📈",
            "role": "Forecasts how many parcels will arrive",
            "received": f"Scenario: {req.scenario_id or 'daily'} (×{spike_mult} multiplier)",
            "reasoning": f"Baseline avg: {int(baseline):,} parcels/day. Applied {spike_mult}× multiplier → peak of {int(peak):,} parcels on {d.get('peak_date', 'N/A')}.",
            "output": f"Peak demand: {int(peak):,} parcels",
            "feeds": f"Route Agent needs {int(peak/80)+1} riders to cover {int(peak):,} parcels",
            "status": "spike" if d.get("spike_detected") else "normal"
        },
        {
            "name": "Route Agent",
            "emoji": "🗺️",
            "role": "Plans delivery routes across Singapore",
            "received": f"Demand Agent → {int(peak):,} parcels to deliver",
            "reasoning": f"ceil({int(peak):,} parcels ÷ 80 per rider) = {r.get('riders_needed', 0)} riders. TSP nearest-neighbour over {r.get('zones_covered', 0)} active zones.",
            "output": f"{r.get('riders_needed', 0)} riders dispatched, {r.get('total_distance_km', 0):.1f} km total route",
            "feeds": f"Warehouse Agent: {r.get('zones_covered', 0)} zones need slotting for {r.get('riders_needed', 0)} riders",
            "status": "normal"
        },
        {
            "name": "Warehouse Agent",
            "emoji": "📦",
            "role": "Optimises parcel sorting and zone assignments",
            "received": f"Route Agent → {r.get('zones_covered', 0)} zones, {r.get('riders_needed', 0)} riders",
            "reasoning": f"Redistributed {w.get('total_daily_picks', 0):,} picks across zones based on rider assignments. Minimised walking distance between sort bays.",
            "output": f"Walking distance reduced by {w.get('distance_saved_pct', 0):.1f}%",
            "feeds": f"Pricing Agent: demand ratio {spike_mult}× for surge calculation",
            "status": "normal"
        },
        {
            "name": "Pricing Agent",
            "emoji": "💰",
            "role": "Sets dynamic delivery rates based on demand",
            "received": f"Demand Agent → demand ratio {p.get('demand_ratio', spike_mult)}×",
            "reasoning": f"Demand ratio {p.get('demand_ratio', spike_mult)}× exceeds surge threshold. Surge multiplier = {p.get('demand_multiplier', 1.0)}×.",
            "output": f"Surge {'ACTIVE' if p.get('surge_active') else 'INACTIVE'} — base rate ×{p.get('demand_multiplier', 1.0)}",
            "feeds": "Customer Agent: spike + pricing context for proactive comms",
            "status": "warning" if p.get("demand_multiplier", 1.0) > 1.3 else "normal",
            "conflict": "High surge pricing may cause customer churn" if p.get("demand_multiplier", 1.0) > 1.3 else None
        },
        {
            "name": "Customer Agent",
            "emoji": "💬",
            "role": "Proactively updates customers and chatbot",
            "received": f"Demand Agent → spike detected. Route Agent → delivery window extended.",
            "reasoning": c.get("action", "No action needed"),
            "output": c.get("impact", "Normal operations"),
            "feeds": "Control Tower: coordination complete",
            "status": "normal"
        }
    ]

    conflicts = result.get("conflicts", [])
    metrics = result.get("metrics", {})
    summary = {
        "recommendation": "Plan active with conflicts — monitor customer churn." if conflicts else "All agents aligned. Plan is active.",
        "spike": spike_mult,
        "riders": r.get("riders_needed", 0),
        "latency": metrics.get("decision_latency", 0),
        "conflicts": conflicts
    }

    return {
        "agents": agents,
        "summary": summary,
        "metrics": metrics,
        "global_demand_volume": int(peak)
    }

@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
