import requests
from datetime import date, timedelta

def get_month_weather(year, month, lat=52.78, lon=6.89):
    # Eerste dag van de maand
    start_date = date(year, month, 1)

    # Laatste dag van de maand bepalen
    if month == 12:
        next_month_first = date(year + 1, 1, 1)
    else:
        next_month_first = date(year, month + 1, 1)
    end_date = next_month_first - timedelta(days=1)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        "&daily=temperature_2m_max,precipitation_sum"
        "&timezone=Europe%2FAmsterdam"
        f"&start_date={start_date.isoformat()}"
        f"&end_date={end_date.isoformat()}"
    )

    response = requests.get(url)
    data = response.json()

    dates = data["daily"]["time"]
    temps = data["daily"]["temperature_2m_max"]
    rains = data["daily"]["precipitation_sum"]

    month_days = []
    for d, t, r in zip(dates, temps, rains):
        month_days.append({
            "date": d,
            "temperature": round(float(t), 1),
            "rain_mm": round(float(r), 1)
        })

    return month_days
