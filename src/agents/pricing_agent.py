"""
Dynamic Pricing Agent (Problem 7) — calculates recommended shipping prices
based on demand level, distance tier, and service urgency.

Formula:
  price = BASE_PRICE × demand_multiplier × distance_factor × urgency_factor

No ML model needed — rule-based pricing is standard in logistics
(Grab, Lalamove, and NinjaVan all use demand-sensitive surge pricing).
"""

BASE_PRICE_SGD = 4.50

DISTANCE_FACTORS = {
    "local":       1.0,   # within same postal district
    "zone1":       1.2,   # 1–2 districts away (< 10 km)
    "zone2":       1.5,   # cross-region (10–25 km)
    "interstate":  2.0,   # SG → Malaysia / long distance
}

URGENCY_FACTORS = {
    "standard":  1.0,   # 2–3 business days
    "express":   1.4,   # next day
    "same_day":  2.0,   # same day delivery
}

SURGE_THRESHOLD = 1.15   # demand > 15% above baseline triggers surge


def run_pricing_agent(state: dict) -> dict:
    """LangGraph-compatible node. Reads demand_result + route_result; outputs pricing_result."""
    demand = (state.get("demand_result") or {})
    forecast = demand.get("forecast", {})
    baseline_avg = forecast.get("baseline_avg", 500.0) or 500.0
    forecast_values = forecast.get("values", [])
    today_parcels = float(forecast_values[0]) if forecast_values else baseline_avg

    spike_detected = demand.get("spike_detected", False)
    demand_ratio = today_parcels / baseline_avg

    # ── Demand multiplier ──────────────────────────────────────────────────────
    if demand_ratio > SURGE_THRESHOLD:
        demand_multiplier = round(1.0 + (demand_ratio - 1.0) * 0.4, 3)
        surge_active = True
    else:
        demand_multiplier = 1.0
        surge_active = False

    # ── Build price table for all distance × urgency combinations ─────────────
    prices = {}
    for dist_key, dist_factor in DISTANCE_FACTORS.items():
        prices[dist_key] = {}
        for urg_key, urg_factor in URGENCY_FACTORS.items():
            raw = BASE_PRICE_SGD * demand_multiplier * dist_factor * urg_factor
            prices[dist_key][urg_key] = round(raw, 2)

    # ── Summary prices (most common tier: zone1 standard) ─────────────────────
    summary_prices = {
        "standard (local)":     prices["local"]["standard"],
        "standard (zone1)":     prices["zone1"]["standard"],
        "express (zone1)":      prices["zone1"]["express"],
        "same-day (zone1)":     prices["zone1"]["same_day"],
        "interstate (standard)": prices["interstate"]["standard"],
    }

    alert = None
    if surge_active:
        alert = (
            f"Surge pricing active — demand is {demand_ratio:.1f}× baseline. "
            f"Standard zone1 price raised to SGD {prices['zone1']['standard']:.2f}."
        )

    return {
        **state,
        "pricing_result": {
            "base_price_sgd": BASE_PRICE_SGD,
            "demand_multiplier": demand_multiplier,
            "surge_active": surge_active,
            "demand_ratio": round(demand_ratio, 2),
            "summary_prices": summary_prices,
            "full_price_table": prices,
            "alert": alert,
        },
    }
