"""
Singapore postal districts — 28 districts with representative lat/lon and zone tier.
Used by route_agent.py for distance matrix and TSP optimisation.
"""
import math
from typing import Optional

# 28 Singapore postal districts
DISTRICTS = {
    "D01": {"name": "Raffles Place / Cecil / Marina / People's Park", "lat": 1.2836, "lon": 103.8507, "tier": "central"},
    "D02": {"name": "Anson / Tanjong Pagar", "lat": 1.2761, "lon": 103.8452, "tier": "central"},
    "D03": {"name": "Queenstown / Tiong Bahru", "lat": 1.2899, "lon": 103.8185, "tier": "central"},
    "D04": {"name": "Telok Blangah / Harbourfront", "lat": 1.2700, "lon": 103.8198, "tier": "central"},
    "D05": {"name": "Pasir Panjang / Clementi New Town", "lat": 1.3062, "lon": 103.7638, "tier": "west"},
    "D06": {"name": "High Street / Beach Road", "lat": 1.2966, "lon": 103.8520, "tier": "central"},
    "D07": {"name": "Middle Road / Golden Mile", "lat": 1.3016, "lon": 103.8570, "tier": "central"},
    "D08": {"name": "Little India", "lat": 1.3067, "lon": 103.8517, "tier": "central"},
    "D09": {"name": "Orchard / River Valley", "lat": 1.3045, "lon": 103.8318, "tier": "central"},
    "D10": {"name": "Ardmore / Bukit Timah", "lat": 1.3213, "lon": 103.8198, "tier": "central"},
    "D11": {"name": "Novena / Thomson", "lat": 1.3262, "lon": 103.8378, "tier": "central"},
    "D12": {"name": "Balestier / Toa Payoh / Serangoon", "lat": 1.3328, "lon": 103.8472, "tier": "north-east"},
    "D13": {"name": "Macpherson / Braddell", "lat": 1.3375, "lon": 103.8761, "tier": "north-east"},
    "D14": {"name": "Geylang / Eunos", "lat": 1.3175, "lon": 103.8930, "tier": "east"},
    "D15": {"name": "Katong / Joo Chiat / Amber Road", "lat": 1.3048, "lon": 103.9056, "tier": "east"},
    "D16": {"name": "Bedok / Upper East Coast / Eastwood", "lat": 1.3236, "lon": 103.9273, "tier": "east"},
    "D17": {"name": "Loyang / Changi", "lat": 1.3600, "lon": 103.9836, "tier": "east"},
    "D18": {"name": "Tampines / Pasir Ris", "lat": 1.3530, "lon": 103.9454, "tier": "east"},
    "D19": {"name": "Serangoon Garden / Hougang / Ponggol", "lat": 1.3714, "lon": 103.8938, "tier": "north-east"},
    "D20": {"name": "Bishan / Ang Mo Kio", "lat": 1.3650, "lon": 103.8455, "tier": "north"},
    "D21": {"name": "Upper Bukit Timah / Clementi Park", "lat": 1.3376, "lon": 103.7761, "tier": "west"},
    "D22": {"name": "Jurong", "lat": 1.3330, "lon": 103.7436, "tier": "west"},
    "D23": {"name": "Hillview / Dairy Farm / Bukit Panjang / Choa Chu Kang", "lat": 1.3800, "lon": 103.7470, "tier": "west"},
    "D24": {"name": "Lim Chu Kang / Tengah", "lat": 1.4000, "lon": 103.7200, "tier": "west"},
    "D25": {"name": "Kranji / Woodgrove", "lat": 1.4380, "lon": 103.7594, "tier": "north"},
    "D26": {"name": "Upper Thomson / Springleaf", "lat": 1.4015, "lon": 103.8198, "tier": "north"},
    "D27": {"name": "Yishun / Sembawang", "lat": 1.4304, "lon": 103.8354, "tier": "north"},
    "D28": {"name": "Seletar", "lat": 1.4048, "lon": 103.8693, "tier": "north"},
    "HUB-EAST": {"name": "Regional Hub (East)", "lat": 1.3530, "lon": 103.9454, "tier": "hub"},
    "HUB-WEST": {"name": "Regional Hub (West)", "lat": 1.3330, "lon": 103.7436, "tier": "hub"},
    "HUB-NORTH": {"name": "Regional Hub (North)", "lat": 1.4304, "lon": 103.8354, "tier": "hub"},
}

REGIONAL_HUBS = ["HUB-EAST", "HUB-WEST", "HUB-NORTH"]

ACTIVE_ZONES = [
    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10",
    "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20",
    "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28"
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_matrix(zone_codes: list[str]) -> dict[tuple, float]:
    """Returns pairwise Haversine distances (km) for the given district list."""
    matrix = {}
    for i, a in enumerate(zone_codes):
        for j, b in enumerate(zone_codes):
            if i != j:
                da, db = DISTRICTS[a], DISTRICTS[b]
                matrix[(a, b)] = haversine_km(da["lat"], da["lon"], db["lat"], db["lon"])
    return matrix


def nearest_neighbour_tsp(zone_codes: list[str], start: Optional[str] = None) -> list[str]:
    """
    Nearest-neighbour greedy TSP heuristic.
    Returns an ordered list of district codes representing the optimised visit sequence.
    """
    if not zone_codes:
        return []
    if len(zone_codes) == 1:
        return zone_codes[:]

    dist = distance_matrix(zone_codes)
    unvisited = set(zone_codes)
    current = start if start and start in unvisited else zone_codes[0]
    route = [current]
    unvisited.remove(current)

    while unvisited:
        nearest = min(unvisited, key=lambda z: dist.get((current, z), float("inf")))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    return route


def total_route_distance_km(route: list[str]) -> float:
    """Sum of Haversine distances along a route (open path)."""
    total = 0.0
    for i in range(len(route) - 1):
        a, b = DISTRICTS[route[i]], DISTRICTS[route[i + 1]]
        total += haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    return round(total, 1)
