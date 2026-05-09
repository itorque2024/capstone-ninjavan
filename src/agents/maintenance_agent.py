"""
Predictive Maintenance Agent — decision-driving outputs for fleet managers.

Per vehicle returns:
  risk_score, failure_window_days, rul_km, trips_remaining,
  recommended_action, cost_now, cost_if_delayed,
  driver_behavior_score, confidence, sensor_alerts
Fleet-level returns:
  pct_high_risk, critical_count, weekly_cost_exposure, avg_health
"""
import math
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "maintenance_model.pkl"

# ── Financial constants (SGD) ──────────────────────────────────────────────────
_BASE_REPAIR        = 300      # Planned preventive repair
_BREAKDOWN_MULT     = 4.0      # Breakdown costs 4× planned
_DOWNTIME_PER_DAY   = 250      # Lost revenue per vehicle per day
_PENALTY_PER_PARCEL = 8        # SLA penalty per missed parcel
_PARCELS_PER_TRIP   = 40
_KM_PER_TRIP        = 80


def _failure_window(risk: float) -> tuple[int, int]:
    """Return (min_days, max_days) until likely failure."""
    if risk >= 0.85:   return (1,  2)
    if risk >= 0.70:   return (2,  4)
    if risk >= 0.55:   return (4,  7)
    if risk >= 0.40:   return (7, 14)
    return (14, 30)


def _rul_km(health: float, km_since_service: float) -> int:
    """Remaining Useful Life in km — degrades faster at low health."""
    service_limit = 15_000
    remaining_raw = max(0, service_limit - km_since_service)
    health_factor = (health / 100) ** 1.5   # non-linear degradation
    return max(0, int(remaining_raw * health_factor))


def _driver_behavior_score(vibration: float, engine_temp: float) -> int:
    """
    0–100 score. High vibration (harsh driving) and sustained high engine temp
    (over-revving, poor cooling) degrade the score.
    """
    vib_penalty  = min(50, int(vibration * 25))          # 0–50
    temp_penalty = min(50, max(0, int((engine_temp - 100) * 0.5)))  # 0–50
    return max(0, 100 - vib_penalty - temp_penalty)


def _sensor_alerts(row: dict) -> list[str]:
    alerts = []
    if row.get("engine_temp_c", 0) > 165:
        alerts.append(f"🌡️ Engine overheating ({row['engine_temp_c']:.0f}°C > 165°C limit)")
    if row.get("tyre_pressure_kpa", 999) < 180:
        alerts.append(f"🛞 Low tyre pressure ({row['tyre_pressure_kpa']:.0f} kPa < 180 threshold)")
    if row.get("vibration_g", 0) > 1.2:
        alerts.append(f"📳 Excessive vibration ({row['vibration_g']:.2f}g > 1.2g limit)")
    if row.get("km_since_service", 0) > 12_000:
        alerts.append(f"🔧 Overdue service ({int(row['km_since_service']):,} km since last service)")
    return alerts


def _action(risk: float) -> tuple[str, str]:
    """Return (badge, instruction)."""
    if risk >= 0.80:
        return ("🔴 STOP NOW",      "Take vehicle off road immediately. Do not start next route.")
    if risk >= 0.60:
        return ("🟠 URGENT",        "Schedule maintenance tonight. Complete current route only.")
    if risk >= 0.40:
        return ("🟡 SCHEDULE",      "Book maintenance within 3 days. Monitor closely on each shift.")
    return      ("🟢 CONTINUE",     "Safe to operate. Next scheduled service as planned.")


def _cost_estimate(risk: float, rul_km: int) -> tuple[int, int]:
    """(cost_now_sgd, cost_if_delayed_sgd) — delayed = wait until breakdown."""
    repair_now    = _BASE_REPAIR
    trips_left    = max(1, rul_km // _KM_PER_TRIP)
    # Breakdown: full repair × multiplier + downtime + SLA penalties
    breakdown     = int(_BASE_REPAIR * _BREAKDOWN_MULT
                        + _DOWNTIME_PER_DAY * 2
                        + _PENALTY_PER_PARCEL * _PARCELS_PER_TRIP * min(3, trips_left))
    return repair_now, breakdown


def _whatif_risk(base_risk: float, delay_days: int) -> float:
    """Simulate risk escalation if maintenance is delayed N days."""
    escalation = 0.06 * delay_days * (1 + base_risk)   # worsens faster at higher base
    return round(min(0.99, base_risk + escalation), 3)


def _whatif_cost(base_risk: float, delay_days: int, cost_now: int, cost_breakdown: int) -> int:
    """Additional cost incurred by delaying maintenance."""
    new_risk    = _whatif_risk(base_risk, delay_days)
    extra_prob  = new_risk - base_risk
    expected_extra = int(extra_prob * (cost_breakdown - cost_now)
                         + delay_days * _DOWNTIME_PER_DAY * extra_prob)
    return max(0, extra_prob and expected_extra or 0)


def run_maintenance_agent(state: dict) -> dict:
    """
    LangGraph-compatible node.
    Input state keys: vehicle_sensors (list[dict])
    Output: maintenance_result with per-vehicle decision data + fleet summary.
    """
    vehicles = state.get("vehicle_sensors", [])
    if not vehicles:
        return {**state, "maintenance_result": {"error": "No vehicle sensor data provided."}}
    if not MODEL_PATH.exists():
        return {**state, "maintenance_result": {"error": "Model not trained yet. Run notebook 02."}}

    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame(vehicles)
    risk_scores = model.predict_proba(df)[:, 1]

    enriched = []
    for i, v in enumerate(vehicles):
        risk  = float(risk_scores[i])
        health = float(v.get("vehicle_health_score", 70))
        km    = float(v.get("km_since_service", 5000))
        vib   = float(v.get("vibration_g", 0.2))
        temp  = float(v.get("engine_temp_c", 100))

        rul          = _rul_km(health, km)
        trips        = max(0, rul // _KM_PER_TRIP)
        win_lo, win_hi = _failure_window(risk)
        action_badge, action_text = _action(risk)
        cost_now, cost_delay = _cost_estimate(risk, rul)
        driver_score = _driver_behavior_score(vib, temp)
        alerts       = _sensor_alerts(v)

        # Confidence: how far from decision boundary (0.5)
        confidence = int(min(99, abs(risk - 0.5) * 200))

        enriched.append({
            "vehicle_id":           v.get("vehicle_id", i),
            "health_score":         round(health, 1),
            "risk_score":           round(risk, 3),
            "risk_pct":             f"{risk*100:.0f}%",
            "failure_window":       f"{win_lo}–{win_hi} days",
            "failure_window_lo":    win_lo,
            "failure_window_hi":    win_hi,
            "rul_km":               rul,
            "trips_remaining":      trips,
            "action":               action_badge,
            "action_text":          action_text,
            "cost_now_sgd":         cost_now,
            "cost_if_delayed_sgd":  cost_delay,
            "cost_saved_sgd":       cost_delay - cost_now,
            "driver_score":         driver_score,
            "confidence_pct":       confidence,
            "sensor_alerts":        alerts,
            "engine_temp_c":        round(temp, 1),
            "tyre_pressure_kpa":    round(float(v.get("tyre_pressure_kpa", 220)), 1),
            "vibration_g":          round(vib, 3),
            "km_since_service":     int(km),
        })

    # Fleet-level summary
    high_risk   = [e for e in enriched if e["risk_score"] >= 0.60]
    critical    = [e for e in enriched if e["risk_score"] >= 0.80]
    avg_health  = round(sum(e["health_score"] for e in enriched) / max(len(enriched), 1), 1)
    weekly_exposure = sum(e["cost_if_delayed_sgd"] - e["cost_now_sgd"]
                          for e in high_risk)
    top5 = sorted(enriched, key=lambda x: x["risk_score"], reverse=True)[:5]

    # Trend by component (vibration → brakes, temp → engine, pressure → tyres)
    avg_vib  = sum(e["vibration_g"] for e in enriched) / max(len(enriched), 1)
    avg_temp = sum(e["engine_temp_c"] for e in enriched) / max(len(enriched), 1)

    component_trends = []
    if avg_vib > 0.8:
        component_trends.append(f"📳 Brake/suspension issues trending (avg vibration {avg_vib:.2f}g)")
    if avg_temp > 140:
        component_trends.append(f"🌡️ Engine overheating trend (avg temp {avg_temp:.0f}°C)")

    return {
        **state,
        "maintenance_result": {
            "vehicles":            enriched,
            "flagged_vehicles":    high_risk,
            "top5_urgent":         top5,
            "fleet_summary": {
                "total":              len(enriched),
                "high_risk_count":    len(high_risk),
                "critical_count":     len(critical),
                "high_risk_pct":      round(len(high_risk) / max(len(enriched), 1) * 100, 1),
                "avg_health":         avg_health,
                "weekly_exposure_sgd": weekly_exposure,
                "component_trends":   component_trends,
            },
            "whatif_fn":   _whatif_risk,    # callable — used by Streamlit simulator
            "whatif_cost": _whatif_cost,
            "alert": (
                f"⚠️ {len(critical)} vehicle(s) require IMMEDIATE action. "
                f"{len(high_risk)} total at high risk. Est. cost exposure: S${weekly_exposure:,}."
            ) if critical else (
                f"{len(high_risk)} vehicle(s) need maintenance this week." if high_risk else None
            ),
        },
    }
