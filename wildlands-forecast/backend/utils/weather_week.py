import requests
from datetime import timedelta

def get_week_weather(monday, lat=52.78, lon=6.89):
    sunday = monday + timedelta(days=6)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=Europe%2FAmsterdam"
        f"&start_date={monday.isoformat()}"
        f"&end_date={sunday.isoformat()}"
    )

    response = requests.get(url)
    data = response.json()

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_max"]
    rains = data["daily"]["precipitation_sum"]

    week = []
    for d, t, r in zip(dates, temps, rains):
        week.append({
            "date": d,
            "temperature": round(float(t), 1),
            "rain_mm": round(float(r), 1)
        })

    # Verwacht 7 dagen (ma–zo)
    return week
