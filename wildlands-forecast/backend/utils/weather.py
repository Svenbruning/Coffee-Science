# utils/weather.py
import requests
from datetime import date, timedelta
from typing import Dict, List, Tuple

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
DAILY_FIELDS = "temperature_2m_max,precipitation_sum"
TZ = "Europe/Amsterdam"
FORECAST_MAX_DAYS = 16  # Open-Meteo ~16 dagen vooruit

def _fetch(url: str, start: date, end: date, lat=52.78, lon=6.89) -> Dict[str, dict]:
    params = {
        "latitude": lat, "longitude": lon,
        "daily": DAILY_FIELDS, "timezone": TZ,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if "daily" not in data or not all(k in data["daily"] for k in ["time","temperature_2m_max","precipitation_sum"]):
        raise ValueError(f"Open-Meteo daily ontbreekt voor {start}–{end}")
    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_max"]
    rains = data["daily"]["precipitation_sum"]
    return {d: {"temperature": round(float(t), 1), "rain_mm": round(float(rmm), 1)}
            for d, t, rmm in zip(dates, temps, rains)}

def _fetch_archive(start: date, end: date, lat=52.78, lon=6.89) -> Dict[str, dict]:
    if start > end:
        return {}
    return _fetch(ARCHIVE_URL, start, end, lat, lon)

def _fetch_forecast_chunked(start: date, end: date, lat=52.78, lon=6.89) -> Dict[str, dict]:
    if start > end:
        return {}
    out: Dict[str, dict] = {}
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=FORECAST_MAX_DAYS - 1), end)
        try:
            out.update(_fetch(FORECAST_URL, cur, chunk_end, lat, lon))
        except Exception as e:
            print(f"[weather] WARN forecast {cur}–{chunk_end}: {e}")
        cur = chunk_end + timedelta(days=1)
    return out

def _fetch_daily_range(start: date, end: date, lat=52.78, lon=6.89) -> Dict[str, dict]:
    if start > end:
        return {}
    today = date.today()
    yesterday = today - timedelta(days=1)

    if end <= yesterday:
        try:
            return _fetch_archive(start, end, lat, lon)
        except Exception as e:
            print(f"[weather] WARN archive {start}–{end}: {e}")
            return {}

    if start >= today:
        return _fetch_forecast_chunked(start, end, lat, lon)

    left = {}
    right = {}
    try:
        left = _fetch_archive(start, yesterday, lat, lon)
    except Exception as e:
        print(f"[weather] WARN archive {start}–{yesterday}: {e}")
    try:
        right = _fetch_forecast_chunked(today, end, lat, lon)
    except Exception as e:
        print(f"[weather] WARN forecast {today}–{end}: {e}")
    left.update(right)
    return left

# --- Klimaat-fallback uit ruwe historiek (gemiddelde per maanddag) ---
# Let op: deze functie wordt door app.py opgebouwd; hier simpele versie:
_CLIM_CACHE: Dict[tuple, Tuple[float, float]] = {}

def set_climatology_cache(clim: Dict[tuple, Tuple[float, float]]) -> None:
    """Optioneel: app.py kan een (m,d)->(temp,rain) cache zetten."""
    _CLIM_CACHE.clear()
    _CLIM_CACHE.update(clim)

def _climate_for_date(d: date) -> Tuple[float, float]:
    v = _CLIM_CACHE.get((d.month, d.day))
    if v:
        return v
    # fallback als er geen cache is
    return 15.0, 0.0

def get_weather_for_date(d: date, lat=52.78, lon=6.89) -> Tuple[float, float, str]:
    """Retourneert (temperature, rain_mm, source) met source ∈ {'forecast','archive','climate'}."""
    today = date.today()
    # Buiten horizon → klimaat
    if d - today > timedelta(days=FORECAST_MAX_DAYS - 1):
        t, r = _climate_for_date(d)
        return t, r, "climate"

    try:
        # verleden
        if d <= today - timedelta(days=1):
            dd = _fetch_daily_range(d, d, lat, lon)
            v = dd.get(d.isoformat())
            if v:
                return v["temperature"], v["rain_mm"], "archive"
            t, r = _climate_for_date(d)
            return t, r, "climate"
        # vandaag/toekomst binnen horizon
        dd = _fetch_daily_range(d, d, lat, lon)
        v = dd.get(d.isoformat())
        if v:
            return v["temperature"], v["rain_mm"], "forecast"
        t, r = _climate_for_date(d)
        return t, r, "climate"
    except Exception as e:
        print(f"[weather] WARN day {d}: {e}")
        t, r = _climate_for_date(d)
        return t, r, "climate"
