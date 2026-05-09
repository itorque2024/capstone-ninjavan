"""
Singapore weather loader — fetches daily weather from Open-Meteo (free, no API key).
Results are cached with joblib so repeated calls are instant.
"""
import datetime
import requests
import pandas as pd
from src.agents.chatbot._cache import memory

_SG_LAT = 1.3521
_SG_LON = 103.8198
_BASE_URL = "https://api.open-meteo.com/v1"

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


@memory.cache
def _fetch_forecast_raw(today_str: str) -> dict:
    """Cache key includes today's date so it auto-refreshes daily."""
    url = (
        f"{_BASE_URL}/forecast"
        f"?latitude={_SG_LAT}&longitude={_SG_LON}"
        f"&daily=precipitation_sum,weathercode,windspeed_10m_max"
        f"&timezone=Asia%2FSingapore&forecast_days=7"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


@memory.cache
def _fetch_historical_raw(start_date: str, end_date: str) -> dict:
    url = (
        f"{_BASE_URL}/archive"
        f"?latitude={_SG_LAT}&longitude={_SG_LON}"
        f"&daily=precipitation_sum,weathercode"
        f"&timezone=Asia%2FSingapore"
        f"&start_date={start_date}&end_date={end_date}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_sg_weather_forecast() -> pd.DataFrame:
    """
    Returns a 7-day SG weather forecast DataFrame:
      date, rain_mm, weather_code, weather_desc, wind_kmh
    """
    today = datetime.date.today().isoformat()
    try:
        data = _fetch_forecast_raw(today)
        daily = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "rain_mm": daily["precipitation_sum"],
            "weather_code": daily["weathercode"],
            "wind_kmh": daily["windspeed_10m_max"],
        })
        df["weather_desc"] = df["weather_code"].map(WMO_DESCRIPTIONS).fillna("Unknown")
        df["rain_mm"] = df["rain_mm"].fillna(0.0)
        return df
    except Exception:
        # Fallback: empty forecast
        dates = pd.date_range(start=today, periods=7)
        return pd.DataFrame({"date": dates, "rain_mm": [0.0] * 7, "weather_code": [0] * 7,
                             "weather_desc": ["Unknown"] * 7, "wind_kmh": [0.0] * 7})


def get_sg_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns historical daily SG weather DataFrame:
      date, rain_mm, weather_code, weather_desc
    start_date / end_date: 'YYYY-MM-DD'
    """
    try:
        data = _fetch_historical_raw(start_date, end_date)
        daily = data["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "rain_mm": daily["precipitation_sum"],
            "weather_code": daily["weathercode"],
        })
        df["weather_desc"] = df["weather_code"].map(WMO_DESCRIPTIONS).fillna("Unknown")
        df["rain_mm"] = df["rain_mm"].fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "rain_mm", "weather_code", "weather_desc"])


def today_rain_mm() -> float:
    """Returns today's forecast rainfall in mm (0.0 if unavailable)."""
    try:
        df = get_sg_weather_forecast()
        return float(df.iloc[0]["rain_mm"])
    except Exception:
        return 0.0
