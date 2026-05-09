"""
NinjaVan Operations Intelligence Suite — Gradio Dashboard
"""
import os
import sys
import time
import datetime
import math
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

from src.utils.chroma_setup import _DEFAULT_CHROMA_PATH, build_chroma_from_files
from src.utils.data_loader import load_demand_data, load_maintenance_data, load_fraud_data
from src.agents.control_tower import tower
from src.agents.chatbot.orchestrator import chat, trim_history
from src.agents.demand_agent import run_demand_agent
from src.agents.route_agent import run_route_agent
from src.agents.fraud_agent import run_fraud_agent

# ── ChromaDB: rebuild from FAQ txt files if empty ────────────────────────────
try:
    import chromadb
    _chroma_client = chromadb.PersistentClient(path=_DEFAULT_CHROMA_PATH)
    _col = _chroma_client.get_or_create_collection("ninjavan_kb")
    if _col.count() == 0:
        build_chroma_from_files()
except Exception:
    pass

# ── Helper: Format Interaction Timeline ──────────────────────────────────────
def format_timeline(logs):
    if not logs:
        return "Run a simulation to see agent coordination."

    html = "<div style='font-family: Arial, sans-serif; max-height: 400px; overflow-y: auto; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;'>"

    agent_colors = {
        "Demand Agent": "#f97316",      # Orange
        "Route Agent": "#3b82f6",       # Blue
        "Warehouse Agent": "#eab308",   # Yellow
        "Pricing Agent": "#a855f7",     # Purple
        "Customer Agent": "#ec4899",    # Pink
        "Control Tower": "#14b8a6"      # Teal
    }

    for i, log in enumerate(logs):
        ts = time.strftime('%H:%M:%S', time.localtime(log['timestamp']))
        status_color = "#ef4444" if log.get("conflicts") else "#22c55e"
        status_text = "⚠️ CONFLICTS" if log.get("conflicts") else "✅ CLEAR"
        agent_color = agent_colors.get(log['agent'], "#6b7280")

        html += f"""
        <div style='margin-bottom: 12px; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 6px; border-left: 4px solid {agent_color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px; margin-bottom: 5px;'>
                <strong style='color: {agent_color}; font-size: 1.1em;'>[{i+1}] {log['agent']}</strong>
                <span style='font-size: 0.8em; color: gray;'>🕒 {ts} | <span style='color: {status_color}'>{status_text}</span></span>
            </div>
            <div style='font-weight: bold; margin-bottom: 4px; color: #f3f4f6;'>Decision: {log['decision']}</div>
            <div style='font-size: 0.95em; color: #d1d5db; margin-bottom: 6px;'>{log['reasoning']}</div>
            <div style='font-style: italic; color: #9ca3af; font-size: 0.9em;'>Action/Impact: {log['impact']}</div>
        </div>
        """
    html += "</div>"
    return html

# ── Tab 1: Control Tower Logic ───────────────────────────────────────────────
def run_ct_simulation(scenario_label):
    scenario_map = {
        "Baseline (Normal)": None,
        "Mega Sale Day (+150% Spike)": "spike",
        "Regional Lockdown (VN)": "lockdown",
        "Fuel Price Surge (+25% cost)": "fuel_surge"
    }

    state = {
        "event_type": "daily",
        "scenario_id": scenario_map.get(scenario_label),
        "forecast_horizon": 7,
        "delivery_zones": ["D01", "D05", "D08", "D12", "D20"]
    }

    result = tower.invoke(state)
    metrics = result.get("metrics", {})
    logs_html = format_timeline(result.get("decision_logs", []))

    # Format dashboard
    demand = result.get("demand_result", {})
    route = result.get("route_result", {})
    pricing = result.get("pricing_result", {})
    wh = result.get("warehouse_result", {})
    cust = result.get("customer_result", {})

    dashboard_md = f"""
### 🕹️ Operational Command Dashboard
| Pillar | Status | Metric |
|---|---|---|
| **📦 Demand** | {"🔴 SPIKE" if demand.get("spike_detected") else "🟢 NORMAL"} | {demand.get('total_parcels', 0):,} parcels |
| **🚚 Fleet** | {"🟠 CAPACITY WARNING" if route.get("riders_needed", 0) > 40 else "🟢 STABLE"} | {route.get('riders_needed', 0)} riders |
| **💰 Pricing** | {"⚡ SURGE ACTIVE" if pricing.get("surge_active") else "🟢 NORMAL"} | x{pricing.get('demand_multiplier', 1.0)} multiplier |
| **🏭 Warehouse** | {"🟡 HIGH LOAD" if wh.get('total_daily_picks', 0) > 4000 else "🟢 OPTIMAL"} | {wh.get('distance_saved_pct', 0):.1f}% efficiency |
| **💬 Customer** | {cust.get('status', '🟢 NORMAL')} | {cust.get('action', 'Operations are stable')} |
    """

    conflicts_md = ""
    if result.get("conflicts"):
        conflicts_md = "### 🚨 Active Tactical Conflicts\n" + "\n".join([f"- {c}" for c in result.get("conflicts", [])])

    return (
        f"{metrics.get('coordination_score', 0.92)*100:.0f}%",
        f"{metrics.get('decision_latency', 0.15):.2f}s",
        f"{metrics.get('conflict_rate', 0.05)*100:.0f}%",
        f"{metrics.get('plan_stability', 0.98)*100:.0f}%",
        logs_html,
        dashboard_md,
        conflicts_md
    )

def ct_broadcast(dash_md, conflicts_md, score, latency, conflict, stability):
    from src.utils.telegram_notifier import send_telegram_alert
    msg_lines = ["🚨 *Control Tower Simulation Alert*"]
    if dash_md:
        msg_lines.append(dash_md.replace("### ", "").replace("🕹️ ", ""))
    if conflicts_md:
        msg_lines.append(conflicts_md.replace("### ", ""))
    msg_lines.append(f"\n*Metrics:* Latency: {latency} | Stability: {stability} | Score: {score} | Conflicts: {conflict}")
    success = send_telegram_alert("\n".join(msg_lines))
    if success:
        return "✅ Alert sent to Ops Team via Telegram!"
    else:
        return "❌ Failed to send alert. Check Telegram credentials."

# ── Tab 2: Demand Forecasting Logic ──────────────────────────────────────────
def run_demand_forecast(start_date, horizon, warehouse, marketing, is_sale, spike_pct):
    res = run_demand_agent({
        "event_type": "demand",
        "forecast_horizon": int(horizon),
        "forecast_start_date": str(start_date),
        "spike_threshold": 1.0 + spike_pct/100,
        "warehouse": warehouse,
        "marketing_spend": marketing,
        "is_sale": is_sale,
    })
    dres = res.get("demand_result", {})
    fc = dres.get("forecast", {})

    if not fc.get("values"):
        return None, "Model not trained. Run notebook 01.", None

    dates = fc["dates"]
    values = fc["values"]
    baseline = fc.get("baseline_avg", 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=values, name="Forecast", line=dict(color="orange")))
    fig.add_hline(y=baseline, line_dash="dot", line_color="grey", annotation_text="Baseline")
    fig.update_layout(
        title=f"Demand Forecast - {warehouse}",
        xaxis_title="Date",
        yaxis_title="Parcels",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )

    table_data = []
    for d, v in zip(dates, values):
        table_data.append([d, int(v), f"{((v-baseline)/max(baseline,1)*100):+.0f}%"])

    df_out = pd.DataFrame(table_data, columns=["Date", "Forecast", "vs Baseline"])

    peak_date = dres.get('peak_date', 'N/A')
    peak_val = dres.get('peak_value', 0)
    total = dres.get('total_parcels', sum(values))

    metrics_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 10px;">
        <div style="flex: 1; background: rgba(230, 57, 70, 0.1); border-left: 4px solid #e63946; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase;">Peak Volume ({peak_date})</p>
            <h2 style="margin: 5px 0 0 0; color: #f8fafc; font-size: 2em;">{int(peak_val):,}</h2>
        </div>
        <div style="flex: 1; background: rgba(30, 41, 59, 0.5); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase;">Total Period Volume</p>
            <h2 style="margin: 5px 0 0 0; color: #f8fafc; font-size: 2em;">{int(total):,}</h2>
        </div>
    </div>
    """

    return fig, metrics_html, df_out, int(total)

# ── Tab 3: Route Optimization Logic ───────────────────────────────────────────
def generate_route_plan(sim_mode, manual_rain, manual_riders, dynamic_demand_volume):
    from src.agents.route_agent import run_route_agent
    from src.utils.sg_districts import DISTRICTS, REGIONAL_HUBS
    import plotly.express as px
    import pandas as pd

    state = {
        "event_type": "route",
        "demand_result": {"forecast": {"values": [dynamic_demand_volume]}},
        "simulation_mode": sim_mode,
        "manual_rain": manual_rain,
        "manual_riders": manual_riders
    }
    res = run_route_agent(state)
    rres = res.get("route_result", {})
    routes = rres.get("routes", [])

    fig = go.Figure()
    colors = px.colors.qualitative.Alphabet

    # Plot the Regional Hubs distinctly
    hub_lats = [DISTRICTS[h]["lat"] for h in REGIONAL_HUBS]
    hub_lons = [DISTRICTS[h]["lon"] for h in REGIONAL_HUBS]
    hub_names = [DISTRICTS[h]["name"] for h in REGIONAL_HUBS]

    fig.add_trace(go.Scattermapbox(
        mode="markers",
        lon=hub_lons,
        lat=hub_lats,
        marker=dict(size=22, color="white", opacity=0.9),
        hoverinfo="text",
        hovertext=hub_names,
        name="Regional Hubs"
    ))
    fig.add_trace(go.Scattermapbox(
        mode="markers+text",
        lon=hub_lons,
        lat=hub_lats,
        marker=dict(size=16, color="#111827", opacity=1.0),
        text=["HUB"]*len(REGIONAL_HUBS),
        textposition="top center",
        hoverinfo="text",
        hovertext=hub_names,
        name="Regional Hubs (Inner)"
    ))

    zone_assignments = []

    for idx, route_data in enumerate(routes):
        seq = route_data["sequence"]
        lats = [DISTRICTS[s]["lat"] for s in seq]
        lons = [DISTRICTS[s]["lon"] for s in seq]
        names = [f"Rider {idx+1} | Step {i+1}: {s} - {DISTRICTS[s]['name']}" for i, s in enumerate(seq)]

        route_color = colors[idx % len(colors)]

        fig.add_trace(go.Scattermapbox(
            mode="lines+markers+text",
            lon=lons,
            lat=lats,
            line=dict(width=4, color=route_color),
            marker=dict(size=14, color=route_color),
            text=[s for s in seq],
            textposition="top right",
            hoverinfo="text",
            hovertext=names,
            name=f"Rider {idx+1}"
        ))

        # Highlight start point
        if seq:
            fig.add_trace(go.Scattermapbox(
                mode="markers",
                lon=[lons[0]],
                lat=[lats[0]],
                marker=dict(size=20, color="white", symbol="circle-dot", opacity=0.8),
                hoverinfo="skip",
                showlegend=False
            ))

        zone_assignments.append({
            "Rider": f"Rider {idx+1}",
            "Hub (Start)": seq[0],
            "Assigned Zones": ", ".join(seq[1:])
        })

    df_zones = pd.DataFrame(zone_assignments)

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=1.3521, lon=103.8198), # SG center
            zoom=10.5
        ),
        margin={"r":0,"t":40,"l":0,"b":0},
        title="Multi-Vehicle Dispatch Map (VRP Clusters + TSP)"
    )

    metrics_md = f"""
### 📊 Dispatch Metrics
- **📍 Zones Covered:** {rres.get('zones_covered')}
- **📏 Total Distance (All Routes):** {rres.get('total_distance_km')} km
- **⏱️ Max Parallel Duration:** {rres.get('max_duration_hrs')} hrs
- **🚚 Riders Dispatched:** {rres.get('riders_needed')}
- **⛈️ Live Weather (Open-Meteo):** {rres.get('weather_desc')} ({rres.get('rain_mm')} mm rain, x{rres.get('weather_delay_factor')} delay multiplier)
    """

    alert = rres.get('alert')
    alert_md = f"**🚨 ALERT:** {alert}" if alert else "**🟢 STATUS:** Weather clear. Normal routing operations."

    return fig, metrics_md, alert_md, gr.update(value=df_zones, visible=True)

# ── Tab 4: Fraud Detection Logic ──────────────────────────────────────────────
def run_fraud_analysis(threshold, handled_claims=None, global_demand_volume=800):
    if handled_claims is None:
        handled_claims = set()

    # Dynamically adjust strictness based on global volume
    dynamic_threshold = threshold
    if global_demand_volume > 15000: # Massive spike
        dynamic_threshold += 0.15 # Be more lenient, legitimate damage goes up
    elif global_demand_volume > 8000:
        dynamic_threshold += 0.05

    df_fraud = load_fraud_data()
    sample_fraud = df_fraud.sample(n=min(500, len(df_fraud)), random_state=42).copy()
    _claim_cols = ["parcel_id", "parcel_value", "prior_claims", "account_age_days", "claim_lag_days"]
    _claims_input = sample_fraud[[c for c in _claim_cols if c in sample_fraud.columns]].rename(columns={"parcel_id": "claim_id"}).to_dict("records")

    res = run_fraud_agent({"event_type": "fraud", "claims": _claims_input})
    fres = res.get("fraud_result") or {}
    all_claims = fres.get("all_claims", [])

    # Re-filter by dynamic threshold
    flagged = [c for c in all_claims if c["risk_score"] >= dynamic_threshold and str(c["claim_id"]) not in handled_claims]

    avg_risk = sum(c['risk_score'] for c in all_claims)/len(all_claims) if all_claims else 0

    total_claims = len(all_claims)
    auto_approved = total_claims - len(flagged)

    summary_html = f"""
    <div style="display: flex; gap: 20px; margin-bottom: 10px;">
        <div style="flex: 1; background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22c55e; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase;">Auto-Approved by AI (STP)</p>
            <h2 style="margin: 5px 0 0 0; color: #f8fafc; font-size: 2em;">{auto_approved}</h2>
        </div>
        <div style="flex: 1; background: rgba(230, 57, 70, 0.1); border-left: 4px solid #e63946; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase;">Flagged for Human Review</p>
            <h2 style="margin: 5px 0 0 0; color: #f8fafc; font-size: 2em;">{len(flagged)}</h2>
        </div>
        <div style="flex: 1; background: rgba(30, 41, 59, 0.5); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 4px;">
            <p style="margin: 0; color: #94a3b8; font-size: 0.9em; text-transform: uppercase;">Total Batch Claims</p>
            <h2 style="margin: 5px 0 0 0; color: #f8fafc; font-size: 2em;">{total_claims}</h2>
        </div>
    </div>
    """

    # Queue Dataframe
    queue_data = [[c["claim_id"], c["parcel_value"], c["risk_pct"], c["action"]] for c in flagged]
    df_queue = pd.DataFrame(queue_data, columns=["ID", "Value", "Risk", "Action"])

    # Plot
    fig = px.scatter(pd.DataFrame(all_claims), x="account_age_days", y="prior_claims", color="risk_score", size="parcel_value")
    fig.update_layout(
        title="Fraud Signal Map",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc")
    )

    vids = [str(x) for x in df_queue["ID"].tolist()]
    return summary_html, df_queue, fig, gr.update(choices=vids, value=None), ""

def fr_action_approve(claim_id, threshold, handled_claims, global_demand_volume):
    return _fr_handle_action("✅ Approved", claim_id, threshold, handled_claims, global_demand_volume)

def fr_action_reject(claim_id, threshold, handled_claims, global_demand_volume):
    return _fr_handle_action("❌ Rejected", claim_id, threshold, handled_claims, global_demand_volume)

def fr_action_info(claim_id, threshold, handled_claims, global_demand_volume):
    return _fr_handle_action("⚠️ Info Requested", claim_id, threshold, handled_claims, global_demand_volume)

def _fr_handle_action(action_name, claim_id, threshold, handled_claims, global_demand_volume):
    if not handled_claims:
        handled_claims = set()
    if not claim_id:
        return handled_claims, "Please select a Claim ID.", gr.update(), gr.update(), gr.update()
    handled_claims.add(str(claim_id))
    summary, df_queue, fig, drop_update, _ = run_fraud_analysis(threshold, handled_claims, global_demand_volume)
    return handled_claims, f"Claim {claim_id} actioned ({action_name}).", summary, df_queue, drop_update

# ── Tab 5: Chatbot Logic ──────────────────────────────────────────────────────
def chatbot_respond(message, history, global_demand_volume=800):
    # Inject live operational context secretly into the LLM
    system_context = f"[LIVE OPS DB: Current daily volume is {global_demand_volume} parcels. If volume is > 10,000, inform the user that severe delays are expected due to 11.11 volumes.]\n"

    # Gradio 6 history is already list of {"role": "user", "content": "..."}
    result = chat(system_context + message, history)
    answer = result["answer"]

    # intent badge
    badge = {"faq": "🔍 FAQ", "tracking": "📦 Tracking", "escalation": "🚨 Escalation"}.get(result["intent"], "💬 Chat")
    return f"**{badge}**\n\n{answer}"

# ── GRADIO UI ────────────────────────────────────────────────────────────────
# ── UI Theming ───────────────────────────────────────────────────────────────
ninja_theme = gr.themes.Base(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    body_background_fill="#020617",
    body_background_fill_dark="#020617",
    block_background_fill="rgba(15, 23, 42, 0.6)",
    block_background_fill_dark="rgba(15, 23, 42, 0.6)",
    button_primary_background_fill="#10b981",
    button_primary_background_fill_dark="#10b981",
    button_primary_text_color="white",
)

custom_css_html = """
<style>
/* 🚀 ULTIMATE GRADIO OVERRIDE */
body, .gradio-container {
    background: radial-gradient(circle at top left, #1e293b 0%, #020617 100%) !important;
    font-family: 'Outfit', sans-serif !important;
}
footer { display: none !important; }
.gradio-container > .prose { display: none !important; }

/* Dashboard Layout Overrides */
.sidebar-radio { background: transparent !important; border: none !important; box-shadow: none !important; }
.sidebar-radio label {
    background: transparent !important;
    border: none !important;
    padding: 15px 20px !important;
    margin-bottom: 5px !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.sidebar-radio label.selected {
    background: linear-gradient(135deg, rgba(16,185,129,0.2) 0%, transparent 100%) !important;
    color: #10b981 !important;
    border-left: 4px solid #10b981 !important;
}
.sidebar-radio label:hover { color: white !important; background: rgba(255,255,255,0.05) !important; }

/* Glassmorphism */
.gradio-container .block {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
}

/* Cyberpunk Buttons */
button.primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}
</style>
"""

custom_css = """
/* 🚀 ULTIMATE GRADIO OVERRIDE - FULL CUSTOM WEB APP AESTHETIC */

/* 1. Global Background & Fonts */
body, .gradio-container {
    background: radial-gradient(circle at top left, #1e293b 0%, #020617 100%) !important;
    color: #f8fafc !important;
    font-family: 'Outfit', -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 2. Hide ALL Gradio Clutter */
footer { display: none !important; }
.gradio-container > .prose { display: none !important; } /* Hides top default texts */
.wrap.svelte-1pie7s6 { box-shadow: none !important; }
button.svelte-1pie7s6 { box-shadow: none !important; }
.form.svelte-1pie7s6 { border: none !important; background: transparent !important; }

/* 3. Glassmorphism Panels (The Core Aesthetic) */
.gradio-container .block, .gradio-container .panel, .gradio-container .form {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 15px !important;
}

/* 4. Sleek Dashboard Tabs */
.tabs {
    background: transparent !important;
    border: none !important;
}
.tab-nav {
    border-bottom: 2px solid rgba(255,255,255,0.05) !important;
    padding: 10px 20px 0px 20px !important;
    margin-bottom: 20px !important;
    gap: 15px !important;
}
.tab-nav button {
    padding: 12px 24px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.3s ease !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: none !important;
}
.tab-nav button:hover {
    color: #e2e8f0 !important;
}
.tab-nav button.selected {
    background: linear-gradient(to top, rgba(16, 185, 129, 0.1), transparent) !important;
    border-bottom: 3px solid #10b981 !important;
    color: #10b981 !important;
    text-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
}

/* 5. Cyberpunk Buttons */
button.primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 700 !important;
    color: white !important;
    border-radius: 8px !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
}

/* 6. Inputs & Sliders (Sleek Dark Mode) */
input[type="text"], input[type="number"], textarea, select {
    background: rgba(2, 6, 23, 0.5) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    color: #f8fafc !important;
    border-radius: 6px !important;
    padding: 10px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.2) !important;
}

/* 7. Chatbot Specific Fixes */
.message.user {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: none !important;
}
.message.bot {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
}

/* 8. Text & Headers */
h1, h2, h3 { color: #f8fafc !important; font-weight: 700 !important; }
p, span { color: #cbd5e1 !important; }
"""

with gr.Blocks(title="NinjaVan AI Operations Suite", theme=ninja_theme, css=custom_css) as demo:
    gr.HTML(custom_css_html)
    global_demand_state = gr.State(value=800)

    with gr.Row():
        # --- SIDEBAR NAVIGATION ---
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("## 🚀 AI Control Tower")
            nav = gr.Radio(
                ["🗼 Control Tower", "📈 Demand Forecast", "🗺️ Route Optimizer", "🚨 Fraud AI", "💬 CX Chatbot"],
                value="🗼 Control Tower",
                label="",
                elem_classes="sidebar-radio"
            )
            gr.Markdown("---")
            gr.Markdown("**System Status:** <span style='color:#10b981;'>● ONLINE</span>")
            gr.Markdown("**Active Agents:** 5")

        # --- MAIN DASHBOARD CONTENT ---
        with gr.Column(scale=5):

            # --- VIEW: CONTROL TOWER ---
            with gr.Column(visible=True) as view_tower:
                gr.Markdown("### 🗼 Multi-Agent Orchestrator")
                gr.Markdown("Simulate a mega-sale event (11.11) and watch the LangGraph Orchestrator coordinate all 4 sub-agents automatically.")
                with gr.Row():
                    ct_m1 = gr.Textbox(label="Coordination Score", value="92%")
                    ct_m2 = gr.Textbox(label="Decision Latency", value="0.15s")
                    ct_m3 = gr.Textbox(label="Conflict Rate", value="5%")
                    ct_m4 = gr.Textbox(label="Plan Stability", value="98%")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧪 Scenario Simulator")
                        scen_radio = gr.Radio(
                            ["Baseline (Normal)", "Mega Sale Day (+150% Spike)", "Regional Lockdown (VN)", "Fuel Price Surge (+25% cost)"],
                            label="Select Operational Shock",
                            value="Baseline (Normal)"
                        )
                        scen_btn = gr.Button("🚀 Run Simulation", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📡 Agent Interaction Timeline")
                        timeline_html = gr.HTML("Run a simulation to see agent coordination.")

                with gr.Row():
                    ct_dash_md = gr.Markdown("### 🕹️ Operational Command Dashboard")
                    ct_conflicts_md = gr.Markdown("")
                with gr.Row():
                    ct_tg_btn = gr.Button("✈️ Broadcast Alert to Ops Team (Telegram)", variant="secondary")
                    ct_tg_msg = gr.Markdown("")

                scen_btn.click(run_ct_simulation, inputs=[scen_radio], outputs=[ct_m1, ct_m2, ct_m3, ct_m4, timeline_html, ct_dash_md, ct_conflicts_md])
                ct_tg_btn.click(ct_broadcast, inputs=[ct_dash_md, ct_conflicts_md, ct_m1, ct_m2, ct_m3, ct_m4], outputs=[ct_tg_msg])

            # --- VIEW: DEMAND FORECASTING ---
            with gr.Column(visible=False) as view_demand:
                gr.Markdown("### 📈 Agent 1: Demand Forecasting")
                with gr.Row():
                    with gr.Column(scale=1):
                        df_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=str(datetime.date.today()))
                        df_horizon = gr.Slider(7, 90, value=14, label="Horizon (Days)")
                        df_warehouse = gr.Dropdown(["All", "Tampines Hub", "Jurong Hub", "Changi Hub", "Woodlands Hub"], value="All", label="Warehouse Hub")
                        df_marketing = gr.Slider(0, 100000, value=0, label="Marketing Spend ($)", step=1000)
                        df_sale = gr.Checkbox(label="Major E-commerce Sale (e.g., 11.11)")
                        df_spike = gr.Slider(10, 100, value=30, label="Spike Threshold %")
                        df_btn = gr.Button("Run Forecast", variant="primary")
                    with gr.Column(scale=2):
                        df_plot = gr.Plot(label="Forecast Chart")
                        df_metrics = gr.HTML("")
                df_table = gr.Dataframe(label="Day-by-Day Plan")
                df_btn.click(run_demand_forecast, [df_start, df_horizon, df_warehouse, df_marketing, df_sale, df_spike], [df_plot, df_metrics, df_table, global_demand_state])

            # --- VIEW: INTELLIGENT ROUTING ---
            with gr.Column(visible=False) as view_route:
                gr.Markdown("### 🚚 Agent 2: Intelligent Routing")
                gr.Markdown("Calculates the most efficient delivery sequence across Singapore using a Nearest-Neighbor TSP heuristic.")
                with gr.Row():
                    with gr.Column(scale=2):
                        ro_sim_mode = gr.Radio(["Live Auto-Routing", "Manual Simulator"], value="Live Auto-Routing", label="Simulation Mode")
                    with gr.Column(scale=1):
                        ro_manual_rain = gr.Slider(0, 50, value=0, label="Simulated Rain (mm)", visible=False)
                        ro_manual_riders = gr.Slider(1, 20, value=10, step=1, label="Simulated Active Riders", visible=False)

                with gr.Row():
                    ro_btn = gr.Button("Calculate Optimal Route", variant="primary")
                with gr.Row():
                    with gr.Column(scale=2):
                        ro_plot = gr.Plot(label="Route Map")
                    with gr.Column(scale=1):
                        ro_metrics = gr.Markdown("Click calculate to generate metrics.")
                        ro_alert = gr.Markdown("")
                        ro_df = gr.Dataframe(label="Assigned Zones per Rider", visible=False)

                def toggle_slider(mode):
                    is_manual = (mode == "Manual Simulator")
                    return gr.update(visible=is_manual), gr.update(visible=is_manual)

                ro_sim_mode.change(toggle_slider, inputs=[ro_sim_mode], outputs=[ro_manual_rain, ro_manual_riders])
                ro_btn.click(generate_route_plan, inputs=[ro_sim_mode, ro_manual_rain, ro_manual_riders, global_demand_state], outputs=[ro_plot, ro_metrics, ro_alert, ro_df])

            # --- VIEW: FRAUD DETECTION ---
            with gr.Column(visible=False) as view_fraud:
                gr.Markdown("### 🚨 Agent 3: Fraud & Claims STP")
                fr_handled_state = gr.State(set())
                with gr.Row():
                    fr_thr = gr.Slider(0.3, 0.9, value=0.6, label="Risk Threshold")
                    fr_btn = gr.Button("Run Triage", variant="primary")
                with gr.Row():
                    fr_metrics = gr.HTML("")
                with gr.Row():
                    with gr.Column(scale=2):
                        fr_queue = gr.Dataframe(label="Investigation Queue")
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚖️ Triage Panel")
                        fr_action_id = gr.Dropdown(label="Select Claim ID")
                        fr_approve_btn = gr.Button("✅ Approve", variant="primary")
                        fr_reject_btn = gr.Button("❌ Reject", variant="stop")
                        fr_info_btn = gr.Button("⚠️ Request Info")
                        fr_action_msg = gr.Markdown("")
                with gr.Row():
                    fr_plot = gr.Plot(label="Pattern Analysis")

                fr_btn.click(run_fraud_analysis, [fr_thr, fr_handled_state, global_demand_state], [fr_metrics, fr_queue, fr_plot, fr_action_id, fr_action_msg])
                fr_approve_btn.click(fr_action_approve, [fr_action_id, fr_thr, fr_handled_state, global_demand_state], [fr_handled_state, fr_action_msg, fr_metrics, fr_queue, fr_action_id])
                fr_reject_btn.click(fr_action_reject, [fr_action_id, fr_thr, fr_handled_state, global_demand_state], [fr_handled_state, fr_action_msg, fr_metrics, fr_queue, fr_action_id])
                fr_info_btn.click(fr_action_info, [fr_action_id, fr_thr, fr_handled_state, global_demand_state], [fr_handled_state, fr_action_msg, fr_metrics, fr_queue, fr_action_id])

            # --- VIEW: CUSTOMER CHATBOT ---
            with gr.Column(visible=False) as view_chat:
                gr.Markdown("### 💬 Agent 4: RAG Customer Chatbot")
                gr.ChatInterface(chatbot_respond, additional_inputs=[global_demand_state])

    # Sidebar Navigation Logic
    def switch_view(nav_selection):
        return [
            gr.update(visible=(nav_selection == "🗼 Control Tower")),
            gr.update(visible=(nav_selection == "📈 Demand Forecast")),
            gr.update(visible=(nav_selection == "🗺️ Route Optimizer")),
            gr.update(visible=(nav_selection == "🚨 Fraud AI")),
            gr.update(visible=(nav_selection == "💬 CX Chatbot"))
        ]

    nav.change(
        switch_view,
        inputs=[nav],
        outputs=[view_tower, view_demand, view_route, view_fraud, view_chat]
    )

if __name__ == "__main__":
    demo.launch()
