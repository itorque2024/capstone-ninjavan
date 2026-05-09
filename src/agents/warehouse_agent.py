"""
Warehouse Picking Optimisation Agent (Problem 3) — assigns high-demand SKUs to
front warehouse zones to minimise picker walking distance.

Zone layout (distance from dispatch bay):
  Zone A — 0–30 m    (fastest picks, high-demand items)
  Zone B — 31–80 m   (medium demand)
  Zone C — 81–150 m  (slow-moving / bulky items)

Baseline: random assignment → avg 120 m/pick
Optimised: demand-ranked assignment → measured savings
"""
import math

# Mock warehouse inventory: 12 SKU categories mapped to demand region tiers
SKU_CATEGORIES = [
    "Electronics",
    "Fashion & Apparel",
    "Health & Beauty",
    "Food & Beverage",
    "Home & Living",
    "Sports & Outdoors",
    "Books & Stationery",
    "Toys & Games",
    "Automotive Parts",
    "Pet Supplies",
    "Office Supplies",
    "Baby & Kids",
]

ZONE_DISTANCES_M = {"A": 15, "B": 55, "C": 115}  # representative mid-point distances
BASELINE_DISTANCE_M = 120.0                         # random assignment average


def _assign_zones(demand_level: float) -> dict:
    """
    Given a demand level (fraction of baseline: 1.0 = normal, >1.0 = spike),
    rank SKUs by expected pick frequency and assign to zones.
    """
    # Simulate pick frequency: electronics/fashion highest, automotive lowest
    frequencies = [
        demand_level * 1.4,   # Electronics
        demand_level * 1.3,   # Fashion
        demand_level * 1.1,   # Health & Beauty
        demand_level * 1.0,   # F&B
        demand_level * 1.0,   # Home & Living
        demand_level * 0.9,   # Sports
        demand_level * 0.7,   # Books
        demand_level * 0.7,   # Toys
        demand_level * 0.4,   # Automotive
        demand_level * 0.5,   # Pet Supplies
        demand_level * 0.6,   # Office
        demand_level * 0.8,   # Baby & Kids
    ]
    ranked = sorted(zip(SKU_CATEGORIES, frequencies), key=lambda x: -x[1])

    zone_a = [sku for sku, _ in ranked[:4]]
    zone_b = [sku for sku, _ in ranked[4:8]]
    zone_c = [sku for sku, _ in ranked[8:]]

    return {"Zone A": zone_a, "Zone B": zone_b, "Zone C": zone_c}


def run_warehouse_agent(state: dict) -> dict:
    """LangGraph-compatible node. Reads demand_result; outputs warehouse_result."""
    demand = (state.get("demand_result") or {})
    forecast = demand.get("forecast", {})
    baseline_avg = forecast.get("baseline_avg", 500.0) or 500.0
    forecast_values = forecast.get("values", [])
    today_parcels = float(forecast_values[0]) if forecast_values else baseline_avg

    demand_level = today_parcels / baseline_avg

    zone_assignments = _assign_zones(demand_level)

    # Weighted average pick distance with optimised vs baseline assignment
    # Zone A gets 4 SKUs (highest frequency), B gets 4, C gets 4
    total_weight = sum([1.4, 1.3, 1.1, 1.0, 1.0, 0.9, 0.7, 0.7, 0.4, 0.5, 0.6, 0.8])
    zone_a_weight = sum([1.4, 1.3, 1.1, 1.0]) / total_weight
    zone_b_weight = sum([1.0, 0.9, 0.7, 0.7]) / total_weight
    zone_c_weight = sum([0.4, 0.5, 0.6, 0.8]) / total_weight

    optimised_avg_m = (
        zone_a_weight * ZONE_DISTANCES_M["A"] +
        zone_b_weight * ZONE_DISTANCES_M["B"] +
        zone_c_weight * ZONE_DISTANCES_M["C"]
    )
    distance_saved_pct = round((BASELINE_DISTANCE_M - optimised_avg_m) / BASELINE_DISTANCE_M * 100, 1)

    # Estimate daily picks per zone
    daily_picks = math.ceil(today_parcels * 1.2)  # 1.2 picks per parcel (some multi-item orders)
    picks_per_zone = {
        "A": math.ceil(daily_picks * zone_a_weight),
        "B": math.ceil(daily_picks * zone_b_weight),
        "C": math.ceil(daily_picks * zone_c_weight),
    }

    alert = None
    if demand_level > 1.3:
        alert = f"Demand spike ({demand_level:.1f}× baseline) — consider temporary Zone A expansion and extra pickers."

    return {
        **state,
        "warehouse_result": {
            "zone_assignments": zone_assignments,
            "picks_per_zone": picks_per_zone,
            "total_daily_picks": daily_picks,
            "optimised_avg_distance_m": round(optimised_avg_m, 1),
            "baseline_avg_distance_m": BASELINE_DISTANCE_M,
            "distance_saved_pct": distance_saved_pct,
            "demand_level": round(demand_level, 2),
            "alert": alert,
        },
    }
