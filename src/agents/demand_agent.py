"""
Demand Forecast Agent — loads the trained forecast model and returns
shipment volume predictions + spike alerts for the Control Tower.
"""
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "demand_model.pkl"


def run_demand_agent(state: dict) -> dict:
    """
    LangGraph-compatible node. Accepts from state:
      forecast_horizon    int   (default 14)
      forecast_start_date str   (default tomorrow, YYYY-MM-DD)
      spike_threshold     float (default 1.3 = 30% above baseline)
      warehouse           str   ("All" | "Tampines Hub" | "Jurong Hub" | "Changi Hub" | "Woodlands Hub")
      marketing_spend     float
      is_sale             bool
    Returns demand_result with full forecast detail + operational recommendations.
    """
    horizon         = state.get("forecast_horizon", 14)
    start_date      = state.get("forecast_start_date")
    threshold       = state.get("spike_threshold", 1.3)
    warehouse       = state.get("warehouse", "All")
    marketing_spend = state.get("marketing_spend", 0.0)
    is_sale         = state.get("is_sale", False)
    scenario        = state.get("scenario_id")
    if scenario == "normal":
        scenario = None

    if not MODEL_PATH.exists():
        return {**state, "demand_result": {"error": "Model not trained yet."}}

    model = joblib.load(MODEL_PATH)
    forecast = model.predict(
        horizon=horizon,
        start_date=start_date,
        warehouse=warehouse,
        marketing_spend=marketing_spend,
        is_sale=is_sale
    )

    # ── Scenario Injector ──────────────────────────────────────────────────────
    custom_multiplier = state.get("custom_multiplier")

    # (peak_multiplier, calendar_date_key, label)
    SCENARIOS = {
        "spike":       (2.5, "11-11", "🔥 11.11 MEGA SALE"),
        "sale_1212":   (2.2, "12-12", "🛍️ 12.12 YEAR-END SALE"),
        "blackfriday": (2.0, "11-29", "🖤 BLACK FRIDAY"),  # 2024: Nov 29
        "cny":         (1.8, None,    "🧧 CHINESE NEW YEAR"),
        "payday":      (1.6, None,    "💰 PAYDAY SALE"),
        "flash":       (3.0, None,    "⚡ FLASH SALE"),
    }

    # Realistic multi-day surge curves: (day_offset_from_peak, multiplier_vs_daily_baseline)
    # Offset -1 = day before sale, 0 = peak day, +1/+2 = post-sale processing tail
    SURGE_CURVES = {
        "spike":       [(-1, 1.6), (0, 2.5), (1, 1.5), (2, 1.2)],            # 4-day surge
        "sale_1212":   [(-1, 1.4), (0, 2.2), (1, 1.5), (2, 1.2)],            # 4-day surge
        "blackfriday": [(-1, 1.4), (0, 2.0), (1, 1.4), (2, 1.2)],            # 4-day surge
        "cny":         [(-2, 1.3), (-1, 1.5), (0, 1.8), (1, 1.7), (2, 1.5), (3, 1.3), (4, 1.15)],  # 7-day festival
        "payday":      [(-1, 1.2), (0, 1.6), (1, 1.3), (2, 1.1)],            # 4-day surge
        "flash":       [(0, 3.0), (1, 1.4), (2, 1.15)],                       # 3-day (no pre-sale buildup)
    }

    base = forecast.get("baseline_avg", 500)
    n = len(forecast["values"])

    # When no scenario and no sale flag, normalise to flat baseline so the chart
    # shows true "normal operations" without the model's own learned holiday effects.
    if not scenario and not is_sale:
        forecast["values"]        = [base] * n
        forecast["events"]        = [""] * n
        forecast["riders_needed"] = [int(base / 80) + 1] * n

    if scenario in SCENARIOS:
        _, date_key, label = SCENARIOS[scenario]
        curve = SURGE_CURVES[scenario]

        # Normalise all days to flat baseline, then overlay the surge curve.
        forecast["values"]        = [base] * n
        forecast["events"]        = [""] * n
        forecast["riders_needed"] = [int(base / 80) + 1] * n

        # Find the peak day: match calendar date in horizon, otherwise use day 2
        # (leaving room for any pre-sale days at offsets -1 or -2).
        target_idx = 2
        if date_key:
            for i, d in enumerate(forecast["dates"]):
                if date_key in d:
                    target_idx = i
                    break

        for offset, day_mult in curve:
            idx = target_idx + offset
            if 0 <= idx < n:
                v = round(base * day_mult)
                if offset < 0:
                    ev = f"Pre-sale: {label}"
                elif offset == 0:
                    ev = label
                else:
                    ev = f"Post-sale processing: {label}"
                forecast["values"][idx]        = v
                forecast["events"][idx]        = ev
                forecast["riders_needed"][idx] = int(v / 80) + 1

    elif scenario == "lockdown":
        forecast["values"] = [round(v * 0.15) for v in forecast["values"]]
        forecast["events"] = ["🚫 REGIONAL LOCKDOWN"] * n
        forecast["riders_needed"] = [int(v / 80) + 1 for v in forecast["values"]]

    elif scenario == "custom" and custom_multiplier:
        # 3-day surge with tapering tail
        taper1 = round(1 + (custom_multiplier - 1) * 0.5, 2)
        taper2 = round(1 + (custom_multiplier - 1) * 0.2, 2)
        custom_curve = [(0, custom_multiplier), (1, taper1), (2, taper2)]
        forecast["values"]        = [base] * n
        forecast["events"]        = [""] * n
        forecast["riders_needed"] = [int(base / 80) + 1] * n
        for offset, day_mult in custom_curve:
            if offset < n:
                v = round(base * day_mult)
                forecast["values"][offset]        = v
                forecast["events"][offset]        = f"⚙️ CUSTOM ({custom_multiplier:.1f}×)" if offset == 0 else "⚙️ Post-spike tail"
                forecast["riders_needed"][offset] = int(v / 80) + 1

    baseline   = forecast["baseline_avg"]
    values     = forecast["values"]
    dates      = forecast["dates"]
    events     = forecast["events"]
    riders     = forecast["riders_needed"]

    # Classify each day
    spike_days = []
    shutdown_days = []
    action_plan = []

    for i, (d, v, e, r) in enumerate(zip(dates, values, events, riders)):
        pct = (v - baseline) / max(baseline, 1) * 100
        if v < baseline * 0.4:
            shutdown_days.append({"date": d, "event": e, "volume": v})
            action_plan.append(
                f"⚠️ **{d}** — Delivery slowdown ({e}). "
                f"Expected volume: {v:,} ({pct:+.0f}%). Pre-route remaining riders to other zones."
            )
        elif v > baseline * threshold:
            spike_days.append({"date": d, "event": e, "volume": v, "riders": r})
            action_plan.append(
                f"🔴 **{d}** — Demand spike ({e or 'volume surge'}). "
                f"Forecast: {v:,} parcels ({pct:+.0f}%). Deploy **{r} riders** (normal: {int(baseline/80)})."
            )

    spike_detected = len(spike_days) > 0
    peak_val  = max(values) if values else 0
    peak_date = dates[values.index(peak_val)] if values else ""
    peak_event = events[values.index(peak_val)] if values else ""

    return {
        **state,
        "demand_result": {
            "forecast":        forecast,
            "spike_detected":  spike_detected,
            "spike_days":      spike_days,
            "shutdown_days":   shutdown_days,
            "action_plan":     action_plan,
            "peak_date":       peak_date,
            "peak_value":      peak_val,
            "peak_event":      peak_event,
            "total_parcels":   sum(values),
            "alert": (
                f"{len(spike_days)} spike day(s) in forecast — increase fleet allocation."
                if spike_detected else None
            ),
        },
    }
