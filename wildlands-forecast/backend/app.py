from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import pandas as pd
import os
from datetime import datetime
from utils.weather import get_today_weather
from utils.vacations import is_vacation_today
from utils.events import is_event_today

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml_model', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw_visitors.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'daily_predictions.csv')

# Model en historische data
model = joblib.load(MODEL_PATH)
historical = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date')

def generate_daily_prediction():
    today = datetime.today().strftime('%Y-%m-%d')
    print(f"[{datetime.now()}] Generating prediction for {today}...")

    last_day = historical.iloc[-1]
    lag_1 = last_day['visitors']
    lag_7 = historical['visitors'].iloc[-7] if len(historical) >= 7 else lag_1
    roll_3 = historical['visitors'].iloc[-3:].mean() if len(historical) >= 3 else lag_1
    roll_7 = historical['visitors'].iloc[-7:].mean() if len(historical) >= 7 else lag_1

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

    prediction = int(model.predict(df)[0])

    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        preds = pd.DataFrame(columns=['date','temperature','rain_mm','vacation','event','predicted_visitors'])

    preds = preds[preds['date'] != today]  # oude verwijderen
    new_row = pd.DataFrame([{
        'date': today,
        'temperature': temperature,
        'rain_mm': rain_mm,
        'vacation': vacation,
        'event': event,
        'predicted_visitors': prediction
    }])
    preds = pd.concat([preds, new_row], ignore_index=True)
    preds.to_csv(PREDICTIONS_PATH, index=False)

    print(f"✅ Prediction for {today}: {prediction} visitors saved.")

@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    today = datetime.today().strftime('%Y-%m-%d')
    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
        row = preds[preds['date'] == today]
        if not row.empty:
            result = row.iloc[0]
            return jsonify({
                'date': result['date'],
                'temperature': result['temperature'],
                'rain_mm': result['rain_mm'],
                'vacation': bool(result['vacation']),
                'event': bool(result['event']),
                'predicted_visitors': int(result['predicted_visitors'])
            })
    except FileNotFoundError:
        pass

    # Geen voorspelling → genereer nu
    generate_daily_prediction()
    return predict_today()

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(generate_daily_prediction, 'cron', hour=8, minute=0)
scheduler.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
