import requests

def get_today_weather(lat=52.78, lon=6.89):
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&timezone=Europe%2FAmsterdam"
    )
    response = requests.get(url)
    data = response.json()
    today_temp = round(data["daily"]["temperature_2m_max"][0], 1)
    today_rain = round(data["daily"]["precipitation_sum"][0], 1)
    return today_temp, today_rain
