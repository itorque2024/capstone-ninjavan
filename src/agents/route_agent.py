"""
Route Optimization Agent (Problem 2) — plans optimal delivery dispatch sequence
across Singapore postal districts using nearest-neighbour TSP + real weather factor.
"""
import math
import numpy as np
from sklearn.cluster import KMeans
from src.utils.sg_districts import ACTIVE_ZONES, DISTRICTS, nearest_neighbour_tsp, total_route_distance_km, REGIONAL_HUBS, haversine_km
from src.utils.weather_loader import get_sg_weather_forecast

AVG_SPEED_KMH = 25.0       # average urban delivery speed in SG
PARCELS_PER_RIDER = 80     # max parcels one rider handles per day
RAIN_DELAY_PER_MM = 0.015  # 1.5% extra time per mm of rain


def run_route_agent(state: dict) -> dict:
    """
    LangGraph-compatible node.
    Reads demand_result from state; outputs route_result.
    """
    demand = (state.get("demand_result") or {})
    forecast = demand.get("forecast", {})
    forecast_values = forecast.get("values", [])
    # Use the peak_value from demand agent when available (scenario simulations route
    # for the worst day, not just day 0). Standalone route planner sets forecast_values
    # directly and doesn't set peak_value, so fall back to forecast_values[0].
    peak_val = demand.get("peak_value")
    today_parcels = int(peak_val) if peak_val else (int(forecast_values[0]) if forecast_values else int(forecast.get("baseline_avg", 500)))

    delivery_zones = state.get("delivery_zones") or ACTIVE_ZONES

    # ── Rider allocation ───────────────────────────────────────────────────────
    sim_mode = state.get("simulation_mode", "Live Auto-Routing")

    if sim_mode == "Manual Simulator":
        riders_needed = int(state.get("manual_riders", 10))
    else:
        riders_needed = max(1, math.ceil(today_parcels / PARCELS_PER_RIDER))

    # ── VRP Clustering (KMeans) ────────────────────────────────────────────────
    k = min(riders_needed, len(delivery_zones))

    if k > 1:
        coords = np.array([[DISTRICTS[z]["lat"], DISTRICTS[z]["lon"]] for z in delivery_zones])
        # n_init=10 to suppress warnings and ensure stability
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords)
        clusters = {i: [] for i in range(k)}
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(delivery_zones[i])
        zone_groups = [group for group in clusters.values() if group]
    else:
        zone_groups = [delivery_zones]

    # ── TSP per cluster (Multi-Depot) ──────────────────────────────────────────
    routes = []
    total_distance_km = 0.0

    for group in zone_groups:
        # Find geographical center of the cluster
        avg_lat = sum(DISTRICTS[z]["lat"] for z in group) / len(group)
        avg_lon = sum(DISTRICTS[z]["lon"] for z in group) / len(group)

        # Find closest Regional Hub
        closest_hub = min(REGIONAL_HUBS, key=lambda h: haversine_km(DISTRICTS[h]["lat"], DISTRICTS[h]["lon"], avg_lat, avg_lon))

        # Add the hub to the group and calculate TSP starting from the hub
        full_group = [closest_hub] + group
        seq = nearest_neighbour_tsp(full_group, start=closest_hub)

        dist = total_route_distance_km(seq)
        routes.append({"sequence": seq, "distance_km": dist})
        total_distance_km += dist

    total_distance_km = round(total_distance_km, 1)

    # ── Weather adjustment & Max Duration ──────────────────────────────────────
    if sim_mode == "Manual Simulator":
        rain_mm = float(state.get("manual_rain", 0.0))
        weather_desc = "Simulated User Input"
    else:
        try:
            w_df = get_sg_weather_forecast()
            rain_mm = float(w_df.iloc[0]["rain_mm"])
            weather_desc = str(w_df.iloc[0]["weather_desc"])
        except Exception:
            rain_mm = 0.0
            weather_desc = "Unknown (API Failed)"

    weather_delay_factor = round(1.0 + rain_mm * RAIN_DELAY_PER_MM, 3)

    # Max duration is driven by the longest single route (parallel dispatch)
    longest_route_dist = max([r["distance_km"] for r in routes]) if routes else 0
    base_duration_hrs = longest_route_dist / AVG_SPEED_KMH
    adjusted_duration_hrs = round(base_duration_hrs * weather_delay_factor, 2)

    alert = None
    if rain_mm >= 10:
        alert = f"Heavy rain forecast ({rain_mm:.1f} mm) — delivery time extended by {(weather_delay_factor - 1) * 100:.0f}%. Consider notifying recipients."
    elif riders_needed > 15:
        alert = f"High demand: {riders_needed} riders needed today — consider activating on-demand fleet."

    return {
        **state,
        "route_result": {
            "routes": routes,
            "zones_covered": sum(len(r["sequence"]) for r in routes),
            "total_distance_km": total_distance_km,
            "max_duration_hrs": adjusted_duration_hrs,
            "riders_needed": riders_needed,
            "today_parcels": today_parcels,
            "rain_mm": rain_mm,
            "weather_desc": weather_desc,
            "weather_delay_factor": weather_delay_factor,
            "alert": alert,
        },
    }
