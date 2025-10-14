from flask import Flask, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from datetime import datetime
from utils.weather import get_today_weather
from utils.vacations import is_vacation_today
from utils.events import is_event_today


app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('..', 'ml_model', 'model.pkl')
DATA_PATH = os.path.join('..', 'data', 'raw_visitors.csv')
model = joblib.load(MODEL_PATH)
historical = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date')

@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    today = pd.Timestamp(datetime.today().date())

    # Lag/rolling features
    last_day = historical.iloc[-1]
    lag_1 = last_day['visitors']
    lag_7 = historical.iloc[-7]['visitors'] if len(historical) >=7 else lag_1
    roll_3 = historical['visitors'].iloc[-3:].mean() if len(historical) >=3 else lag_1
    roll_7 = historical['visitors'].iloc[-7:].mean() if len(historical) >=7 else lag_1

    # Weer en kalender
    temperature, rain_mm = get_today_weather()
    vacation = is_vacation_today()
    event = is_event_today()

    df = pd.DataFrame([{
        'temperature': temperature,
        'rain_mm': rain_mm,
        'vacation': vacation,
        'event': event,
        'lag_1': lag_1,
        'lag_7': lag_7,
        'roll_3': roll_3,
        'roll_7': roll_7
    }])

    prediction = model.predict(df)[0]

    return jsonify({
        'date': str(today.date()),
        'temperature': temperature,
        'rain_mm': rain_mm,
        'vacation': vacation,
        'event': event,
        'predicted_visitors': int(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
