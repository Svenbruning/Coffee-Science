from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
import joblib
import pandas as pd
import os
from datetime import datetime, date, timedelta

from utils.weather import get_today_weather
from utils.weather_week import get_week_weather
from utils.weather_month import get_month_weather
from utils.vacations import is_vacation_today
from utils.events import is_event_today

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml_model', 'model_lgb.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'daily_visitors.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'daily_predictions.csv')
WEEKLY_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'weekly_predictions.csv')
MONTHLY_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly_predictions.csv')

# -------- Model & historical data --------
model = joblib.load(MODEL_PATH)
historical = pd.read_csv(DATA_PATH)

historical.rename(columns={
    'Date': 'date',
    'Nr Used Entrances': 'visitors',
    'Temperature (mean)': 'temperature',
    'Precipitation (sum)': 'rain_mm',
    'Vacation': 'vacation',
    'Event': 'event',
    'Campaign': 'campaign'
}, inplace=True)

historical['date'] = pd.to_datetime(historical['date'])
historical = historical.sort_values('date')

historical['day_of_week'] = historical['date'].dt.weekday
historical['is_weekend'] = historical['day_of_week'].isin([5, 6]).astype(int)
historical['month'] = historical['date'].dt.month


# -------- Helpers --------
def create_features(temperature, rain_mm, vacation, event, campaign,
                    day_of_week=None, month=None):

    last_day = historical.iloc[-1]
    lag_1 = last_day['visitors']
    lag_7 = historical['visitors'].iloc[-7] if len(historical) >= 7 else lag_1
    roll_3 = historical['visitors'].iloc[-3:].mean() if len(historical) >= 3 else lag_1
    roll_7 = historical['visitors'].iloc[-7:].mean() if len(historical) >= 7 else lag_1

    if day_of_week is None:
        day_of_week = datetime.today().weekday()
    if month is None:
        month = datetime.today().month

    is_weekend = 1 if day_of_week in [5, 6] else 0

    return pd.DataFrame([{
        'temperature': float(temperature),
        'rain_mm': float(rain_mm),
        'vacation': int(vacation),
        'event': int(event),
        'campaign': int(campaign),
        'lag_1': float(lag_1),
        'lag_7': float(lag_7),
        'roll_3': float(roll_3),
        'roll_7': float(roll_7),
        'day_of_week': int(day_of_week),
        'is_weekend': int(is_weekend),
        'month': int(month)
    }])


def get_monday_of_current_week(today=None):
    if today is None:
        today = date.today()
    weekday = today.isoweekday()
    monday = today - timedelta(days=weekday - 1)
    return monday


# -------- Daily prediction --------
def generate_daily_prediction():
    today_str = datetime.today().strftime('%Y-%m-%d')
    print(f"[{datetime.now()}] DAILY → generating prediction for {today_str}")

    temperature, rain_mm = get_today_weather()
    vacation = is_vacation_today()
    event = is_event_today()
    campaign = 0

    df = create_features(temperature, rain_mm, vacation, event, campaign)
    prediction = int(model.predict(df)[0])

    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        preds = pd.DataFrame(columns=[
            'Date', 'temperature', 'rain_mm', 'vacation', 'event', 'campaign',
            'day_of_week', 'is_weekend', 'month', 'predicted_visitors'
        ])

    preds = preds[preds['Date'] != today_str]
    new_row = df.copy()
    new_row['Date'] = today_str
    new_row['predicted_visitors'] = prediction

    preds = pd.concat([preds, new_row], ignore_index=True)
    preds.to_csv(PREDICTIONS_PATH, index=False)

    return prediction


# -------- Weekly prediction --------
def generate_weekly_prediction(force=False):
    today = date.today()
    monday = get_monday_of_current_week(today)
    week_key = monday.strftime('%Y-%m-%d')

    print(f"[{datetime.now()}] WEEK → generating week {week_key}")

    try:
        weekly = pd.read_csv(WEEKLY_PREDICTIONS_PATH)
    except FileNotFoundError:
        weekly = pd.DataFrame(columns=[
            'week_start', 'date', 'temperature', 'rain_mm',
            'vacation', 'event', 'campaign', 'predicted_visitors'
        ])

    if not force and not weekly.empty and (weekly['week_start'] == week_key).any():
        print("WEEK → already exists, skip.")
        return

    week_weather = get_week_weather(monday)

    rows = []
    for i in range(7):
        d = monday + timedelta(days=i)
        ww = week_weather[i]

        df_features = create_features(
            ww["temperature"],
            ww["rain_mm"],
            is_vacation_today(d),
            is_event_today(d),
            0,
            day_of_week=d.weekday(),
            month=d.month
        )

        pred = int(model.predict(df_features)[0])

        rows.append({
            'week_start': week_key,
            'date': d.strftime('%Y-%m-%d'),
            'temperature': ww["temperature"],
            'rain_mm': ww["rain_mm"],
            'vacation': is_vacation_today(d),
            'event': is_event_today(d),
            'campaign': 0,
            'predicted_visitors': pred
        })

    weekly = weekly[weekly['week_start'] != week_key]
    weekly = pd.concat([weekly, pd.DataFrame(rows)], ignore_index=True)
    weekly.to_csv(WEEKLY_PREDICTIONS_PATH, index=False)


# -------- Monthly prediction --------
def generate_monthly_prediction(force=False):
    today = date.today()
    month_start = date(today.year, today.month, 1)
    month_key = month_start.strftime('%Y-%m-01')

    print(f"[{datetime.now()}] MONTH → generating month {month_key}")

    try:
        monthly = pd.read_csv(MONTHLY_PREDICTIONS_PATH)
    except FileNotFoundError:
        monthly = pd.DataFrame(columns=[
            'month_start', 'date', 'temperature', 'rain_mm',
            'vacation', 'event', 'campaign', 'predicted_visitors'
        ])

    if not force and not monthly.empty and (monthly['month_start'] == month_key).any():
        print("MONTH → already exists, skip.")
        return

    weather = get_month_weather(month_start.year, month_start.month)

    rows = []
    for d in weather:
        dt = datetime.strptime(d["date"], "%Y-%m-%d").date()

        df_features = create_features(
            d["temperature"],
            d["rain_mm"],
            is_vacation_today(dt),
            is_event_today(dt),
            0,
            day_of_week=dt.weekday(),
            month=dt.month
        )

        pred = int(model.predict(df_features)[0])

        rows.append({
            'month_start': month_key,
            'date': d["date"],
            'temperature': d["temperature"],
            'rain_mm': d["rain_mm"],
            'vacation': is_vacation_today(dt),
            'event': is_event_today(dt),
            'campaign': 0,
            'predicted_visitors': pred
        })

    monthly = monthly[monthly['month_start'] != month_key]
    monthly = pd.concat([monthly, pd.DataFrame(rows)], ignore_index=True)
    monthly.to_csv(MONTHLY_PREDICTIONS_PATH, index=False)


# -------- API --------
@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    today_str = datetime.today().strftime('%Y-%m-%d')
    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
        row = preds[preds['Date'] == today_str]
        if not row.empty:
            r = row.iloc[0]
            return jsonify({
                'date': r['Date'],
                'temperature': float(r['temperature']),
                'rain_mm': float(r['rain_mm']),
                'vacation': bool(r['vacation']),
                'event': bool(r['event']),
                'campaign': bool(r['campaign']),
                'predicted_visitors': int(r['predicted_visitors'])
            })
    except:
        pass

    prediction = generate_daily_prediction()
    return jsonify({
        'date': today_str,
        'temperature': None,
        'rain_mm': None,
        'vacation': False,
        'event': False,
        'campaign': False,
        'predicted_visitors': prediction
    })


@app.route('/api/predict_custom', methods=['POST', 'OPTIONS'])
def predict_custom():

    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True)

        df = create_features(
            data.get('temperature', 20),
            data.get('rain_mm', 0),
            data.get('vacation', 0),
            data.get('event', 0),
            data.get('campaign', 0),
            day_of_week=data.get('day_of_week'),
            month=data.get('month')
        )

        pred = int(model.predict(df)[0])
        return jsonify({'predicted_visitors': pred})

    except Exception as e:
        print("Error in predict_custom:", e)
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict_week', methods=['GET'])
def predict_week():
    today = date.today()
    monday = get_monday_of_current_week(today)
    week_key = monday.strftime('%Y-%m-%d')

    try:
        weekly = pd.read_csv(WEEKLY_PREDICTIONS_PATH)
    except:
        weekly = pd.DataFrame()

    if weekly.empty or not (weekly['week_start'] == week_key).any():
        generate_weekly_prediction(force=True)
        weekly = pd.read_csv(WEEKLY_PREDICTIONS_PATH)

    this_week = weekly[weekly['week_start'] == week_key].copy()
    this_week = this_week.sort_values('date')

    iso_year, iso_week, _ = monday.isocalendar()

    weekday_names = [
        'Maandag', 'Dinsdag', 'Woensdag',
        'Donderdag', 'Vrijdag', 'Zaterdag', 'Zondag'
    ]

    days = []
    for _, row in this_week.iterrows():
        dt = datetime.strptime(row['date'], '%Y-%m-%d').date()

        days.append({
            'date': row['date'],
            'weekday': weekday_names[dt.weekday()],
            'temperature': float(row['temperature']),
            'rain_mm': float(row['rain_mm']),
            'vacation': bool(row['vacation']),
            'event': bool(row['event']),
            'campaign': bool(row['campaign']),
            'predicted_visitors': int(row['predicted_visitors'])
        })

    return jsonify({
        'week_start': monday.strftime('%Y-%m-%d'),
        'week_end': (monday + timedelta(days=6)).strftime('%Y-%m-%d'),
        'iso_week': iso_week,
        'year': iso_year,
        'days': days
    })


@app.route('/api/predict_month', methods=['GET'])
def predict_month():
    today = date.today()
    month_start = date(today.year, today.month, 1)
    month_key = month_start.strftime('%Y-%m-%d')

    try:
        monthly = pd.read_csv(MONTHLY_PREDICTIONS_PATH)
    except:
        monthly = pd.DataFrame()

    if monthly.empty or not (monthly['month_start'] == month_key).any():
        generate_monthly_prediction(force=True)
        monthly = pd.read_csv(MONTHLY_PREDICTIONS_PATH)

    this_month = monthly[monthly['month_start'] == month_key]

    total = int(this_month['predicted_visitors'].sum()) if not this_month.empty else 0

    month_names = [
        "", "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november", "december"
    ]

    # laatste dag van de maand
    if today.month == 12:
        month_end = date(today.year + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(today.year, today.month + 1, 1) - timedelta(days=1)

    return jsonify({
        'month_start': month_start.strftime('%Y-%m-%d'),
        'month_end': month_end.strftime('%Y-%m-%d'),
        'year': month_start.year,
        'month': month_start.month,
        'month_name': month_names[month_start.month],
        'total_predicted_visitors': total
    })


# -------- Scheduler --------
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(generate_daily_prediction, 'cron', hour=8, minute=0)
scheduler.add_job(generate_weekly_prediction, 'cron', day_of_week='mon', hour=0, minute=1)
scheduler.add_job(generate_monthly_prediction, 'cron', day=1, hour=0, minute=5)
scheduler.start()


if __name__ == "__main__":
    print("→ Running Wildlands Prediction Backend")
    print("→ Auto-generating daily, weekly & monthly predictions…")
    generate_daily_prediction()
    generate_weekly_prediction()
    generate_monthly_prediction()
    app.run(host="0.0.0.0", port=5000, debug=True)
