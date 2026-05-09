"""
Multi-Country Region Definitions
Contains representative districts, hubs, and routing heuristics for Singapore, Malaysia, and Indonesia.
"""
import math
from typing import Optional, List, Dict, Tuple

# --- SINGAPORE (SG) ---
SG_DISTRICTS = {
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

SG_ACTIVE_ZONES = [f"D{str(i).zfill(2)}" for i in range(1, 29)]
SG_HUBS = ["HUB-EAST", "HUB-WEST", "HUB-NORTH"]

# --- MALAYSIA (MY) ---
MY_DISTRICTS = {
    # Klang Valley / Selangor
    "KL-CEN": {"name": "Kuala Lumpur City Center", "lat": 3.1390, "lon": 101.6869, "tier": "central"},
    "PJ": {"name": "Petaling Jaya", "lat": 3.1073, "lon": 101.6067, "tier": "central"},
    "SA": {"name": "Shah Alam", "lat": 3.0738, "lon": 101.5183, "tier": "west"},
    "SBG": {"name": "Subang Jaya", "lat": 3.0567, "lon": 101.5855, "tier": "west"},
    "PUCH": {"name": "Puchong", "lat": 3.0335, "lon": 101.6159, "tier": "south"},
    "KJG": {"name": "Kajang", "lat": 2.9935, "lon": 101.7891, "tier": "south-east"},
    "KLA": {"name": "Klang", "lat": 3.0449, "lon": 101.4456, "tier": "west"},
    "AMP": {"name": "Ampang", "lat": 3.1496, "lon": 101.7621, "tier": "east"},
    "CH": {"name": "Cheras", "lat": 3.1051, "lon": 101.7335, "tier": "south-east"},
    "RWD": {"name": "Rawang", "lat": 3.3225, "lon": 101.5739, "tier": "north"},
    "CYB": {"name": "Cyberjaya", "lat": 2.9228, "lon": 101.6572, "tier": "south"},
    "PUT": {"name": "Putrajaya", "lat": 2.9264, "lon": 101.6964, "tier": "south"},
    # Johor
    "JB": {"name": "Johor Bahru", "lat": 1.4927, "lon": 103.7414, "tier": "south"},
    "BP": {"name": "Batu Pahat", "lat": 1.8494, "lon": 102.9288, "tier": "south"},
    "KLU": {"name": "Kluang", "lat": 2.0251, "lon": 103.3328, "tier": "south"},
    "MG": {"name": "Pasir Gudang", "lat": 1.4646, "lon": 103.8824, "tier": "south"},
    # Penang
    "PNG-ISL": {"name": "Georgetown", "lat": 5.4141, "lon": 100.3288, "tier": "north"},
    "BW": {"name": "Butterworth", "lat": 5.3995, "lon": 100.3683, "tier": "north"},
    "BM": {"name": "Bukit Mertajam", "lat": 5.3630, "lon": 100.4660, "tier": "north"},
    "MY-HUB-CEN": {"name": "National Hub (KL)", "lat": 3.1390, "lon": 101.6869, "tier": "hub"},
    "MY-HUB-NTH": {"name": "Regional Hub (Penang)", "lat": 5.4141, "lon": 100.3288, "tier": "hub"},
    "MY-HUB-STH": {"name": "Regional Hub (Johor)", "lat": 1.4927, "lon": 103.7414, "tier": "hub"},
}

MY_ACTIVE_ZONES = [k for k in MY_DISTRICTS.keys() if "HUB" not in k]
MY_HUBS = ["MY-HUB-CEN", "MY-HUB-NTH", "MY-HUB-STH"]

# --- INDONESIA (ID) ---
ID_DISTRICTS = {
    # Jabodetabek
    "JKT-CEN": {"name": "Jakarta Pusat", "lat": -6.1805, "lon": 106.8283, "tier": "central"},
    "JKT-STH": {"name": "Jakarta Selatan", "lat": -6.2615, "lon": 106.8106, "tier": "south"},
    "JKT-WST": {"name": "Jakarta Barat", "lat": -6.1683, "lon": 106.7588, "tier": "west"},
    "JKT-EST": {"name": "Jakarta Timur", "lat": -6.2250, "lon": 106.9004, "tier": "east"},
    "JKT-NTH": {"name": "Jakarta Utara", "lat": -6.1384, "lon": 106.8640, "tier": "north"},
    "BOG": {"name": "Bogor", "lat": -6.5971, "lon": 106.7932, "tier": "south"},
    "DEP": {"name": "Depok", "lat": -6.4025, "lon": 106.7942, "tier": "south"},
    "TGR": {"name": "Tangerang", "lat": -6.1702, "lon": 106.6403, "tier": "west"},
    "BKS": {"name": "Bekasi", "lat": -6.2383, "lon": 106.9756, "tier": "east"},
    # West Java
    "BDO": {"name": "Bandung", "lat": -6.9175, "lon": 107.6191, "tier": "south-east"},
    "CMH": {"name": "Cimahi", "lat": -6.8723, "lon": 107.5459, "tier": "south-east"},
    "CRB": {"name": "Cirebon", "lat": -6.7320, "lon": 108.5523, "tier": "east"},
    # Central/East Java
    "SMG": {"name": "Semarang", "lat": -6.9667, "lon": 110.4167, "tier": "central-east"},
    "SBY": {"name": "Surabaya", "lat": -7.2504, "lon": 112.7688, "tier": "east"},
    "MLG": {"name": "Malang", "lat": -7.9797, "lon": 112.6304, "tier": "east"},
    "SDA": {"name": "Sidoarjo", "lat": -7.4478, "lon": 112.7183, "tier": "east"},
    # Sumatra
    "MDN": {"name": "Medan", "lat": 3.5952, "lon": 98.6722, "tier": "north-west"},
    "PLB": {"name": "Palembang", "lat": -2.9909, "lon": 104.7566, "tier": "west"},
    "PKU": {"name": "Pekanbaru", "lat": 0.5333, "lon": 101.4500, "tier": "west"},
    "ID-HUB-JKT": {"name": "National Hub (Jakarta)", "lat": -6.2000, "lon": 106.8166, "tier": "hub"},
    "ID-HUB-SBY": {"name": "Regional Hub (Surabaya)", "lat": -7.2504, "lon": 112.7688, "tier": "hub"},
    "ID-HUB-MDN": {"name": "Regional Hub (Medan)", "lat": 3.5952, "lon": 98.6722, "tier": "hub"},
}

ID_ACTIVE_ZONES = [k for k in ID_DISTRICTS.keys() if "HUB" not in k]
ID_HUBS = ["ID-HUB-JKT", "ID-HUB-SBY", "ID-HUB-MDN"]


REGION_DATA = {
    "SG": {
        "DISTRICTS": SG_DISTRICTS,
        "ACTIVE_ZONES": SG_ACTIVE_ZONES,
        "HUBS": SG_HUBS,
        "BASE_VOLUME": 800
    },
    "MY": {
        "DISTRICTS": MY_DISTRICTS,
        "ACTIVE_ZONES": MY_ACTIVE_ZONES,
        "HUBS": MY_HUBS,
        "BASE_VOLUME": 3000
    },
    "ID": {
        "DISTRICTS": ID_DISTRICTS,
        "ACTIVE_ZONES": ID_ACTIVE_ZONES,
        "HUBS": ID_HUBS,
        "BASE_VOLUME": 8000
    }
}

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_matrix(zone_codes: List[str], country: str = "SG") -> Dict[Tuple, float]:
    """Returns pairwise Haversine distances (km) for the given district list within a country."""
    matrix = {}
    districts = REGION_DATA[country]["DISTRICTS"]
    for i, a in enumerate(zone_codes):
        for j, b in enumerate(zone_codes):
            if i != j:
                da, db = districts[a], districts[b]
                matrix[(a, b)] = haversine_km(da["lat"], da["lon"], db["lat"], db["lon"])
    return matrix


def nearest_neighbour_tsp(zone_codes: List[str], country: str = "SG", start: Optional[str] = None) -> List[str]:
    """
    Nearest-neighbour greedy TSP heuristic.
    Returns an ordered list of district codes representing the optimised visit sequence.
    """
    if not zone_codes:
        return []
    if len(zone_codes) == 1:
        return zone_codes[:]

    dist = distance_matrix(zone_codes, country)
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


def total_route_distance_km(route: List[str], country: str = "SG") -> float:
    """Sum of Haversine distances along a route (open path)."""
    total = 0.0
    districts = REGION_DATA[country]["DISTRICTS"]
    for i in range(len(route) - 1):
        a, b = districts[route[i]], districts[route[i + 1]]
        total += haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    return round(total, 1)
