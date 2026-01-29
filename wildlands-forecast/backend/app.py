# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

import os, joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple

from utils.weather import get_weather_for_date, set_climatology_cache
from utils.vacations import vacation_flag_on
from utils.events import event_flag_on

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ------------------------------- Paths --------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'ml_model', 'model_lgb.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'daily_visitors.csv')
PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'daily_predictions.csv')
WEEKLY_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'weekly_predictions.csv')
MONTHLY_PREDICTIONS_PATH = os.path.join(BASE_DIR, '..', 'data', 'monthly_predictions.csv')

# ---------------------------- IO helpers ------------------------------
def _init_like(sample: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=sample.columns.tolist())

def _safe_read_or_init(path: str, sample: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        cols = [c.lower() for c in sample.columns]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype=sample[c].dtype if c in sample.columns else "float64")
        df = df[[c for c in cols]]
        for c in df.columns:
            if c in ["date", "week_start", "month_start"]:
                df[c] = df[c].astype(str)
        return df
    except Exception:
        return _init_like(sample)

def _safe_write(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def _csv_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

# ---------------------- Load model + historical -----------------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]
cal = bundle.get("calibration", {"a": 1.0, "b": 0.0})

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
past_series_base = historical['visitors'].astype(float).tolist()
LAST_HIST_DATE: date = historical['date'].max().date()

# --- klimatologie-cache (voor nette fallback) ---
hist_md = historical.copy()
hist_md['m'] = hist_md['date'].dt.month
hist_md['d'] = hist_md['date'].dt.day
CLIM = (
    hist_md.groupby(['m', 'd'])[['temperature', 'rain_mm']]
    .mean()
    .rename(columns={'temperature': 'clim_temp', 'rain_mm': 'clim_rain'})
)
clim_cache = {(int(m), int(d)): (float(round(row['clim_temp'], 1)),
                                 float(round(max(row['clim_rain'], 0.0), 1)))
              for (m, d), row in CLIM.iterrows()}
set_climatology_cache(clim_cache)

# --------------------------- Time helpers -----------------------------
def _weekday(d: date) -> int: return d.weekday()
def _weekofyear(d: date) -> int: return d.isocalendar().week
def _dayofyear(d: date) -> int: return d.timetuple().tm_yday
def _is_weekend(dow: int) -> int: return int(dow in (5, 6))
def _apply_calibration(y_hat: float) -> float:
    a, b = cal.get("a", 1.0), cal.get("b", 0.0)
    return a * y_hat + b
def _temperature_correction(temp: float, baseline: float = 18.0) -> float:
    factor = 1.0 + (temp - baseline) * 0.01
    return float(np.clip(factor, 0.85, 1.15))
def _rain_temperature_interaction(temp: float, rain_mm: float) -> float:
    if rain_mm <= 0:
        return 1.0

    rain_factor = min(rain_mm / 10.0, 1.0)  # 0..1 bij 0–10mm

    if temp < 10:
        penalty = 0.15 * rain_factor   # koud + regen = zwaar
    elif temp < 18:
        penalty = 0.10 * rain_factor
    elif temp < 25:
        penalty = 0.06 * rain_factor
    else:
        penalty = 0.03 * rain_factor   # warm + regen = licht

    return 1.0 - penalty


# -------------------------- Feature builder ---------------------------
def _features_for_date(d: date, temp: float, rain: float,
                       vac: int, evt: int, camp: int,
                       series: List[float]) -> pd.DataFrame:
    n = len(series)
    lag1  = series[-1] if n >= 1 else 0.0
    lag7  = series[-7] if n >= 7 else lag1
    lag14 = series[-14] if n >= 14 else lag7
    roll3  = float(np.mean(series[-3:]))  if n >= 1 else lag1
    roll7  = float(np.mean(series[-7:]))  if n >= 1 else lag1
    roll14 = float(np.mean(series[-14:])) if n >= 1 else lag1

    dow = _weekday(d); month = d.month; woy = _weekofyear(d); doy = _dayofyear(d)
    temp2 = temp ** 2
    rain_log1p = float(np.log1p(max(rain, 0.0)))
    weekend_temp = (1 if dow in (5, 6) else 0) * temp

    feat = {
        'temperature': float(temp), 'rain_mm': float(rain),
        'temp2': float(temp2), 'rain_log1p': rain_log1p, 'weekend_temp': float(weekend_temp),
        'vacation': int(vac), 'event': int(evt), 'campaign': int(camp),
        'lag_1': float(lag1), 'lag_7': float(lag7), 'lag_14': float(lag14),
        'roll_3': float(roll3), 'roll_7': float(roll7), 'roll_14': float(roll14),
        'day_of_week': int(dow), 'is_weekend': _is_weekend(dow),
        'month': int(month), 'weekofyear': int(woy),
        'sin_doy': float(np.sin(2 * np.pi * doy / 365.25)),
        'cos_doy': float(np.cos(2 * np.pi * doy / 365.25)),
    }
    return pd.DataFrame([feat])[FEATURES]

# --------------------- Exogenous (weer/vakantie/event) -----------------
def _exo_for_date(d: date) -> Dict[str, float | int]:
    t, r, _src = get_weather_for_date(d)
    return {
        "temperature": t,
        "rain_mm": r,
        "vacation": vacation_flag_on(d),
        "event": event_flag_on(d),
        "campaign": 0
    }

# ----------------- Rolling helpers to align all endpoints --------------
def _build_series_until(target: date) -> List[float]:
    """
    Vul 'gat-dagen' tussen LAST_HIST_DATE+1 en target-1 met modelvoorspellingen,
    zodat lags overal identiek zijn.
    """
    series = past_series_base.copy()
    cur = LAST_HIST_DATE + timedelta(days=1)
    while cur <= target - timedelta(days=1):
        exo = _exo_for_date(cur)
        X = _features_for_date(cur, exo["temperature"], exo["rain_mm"],
                               exo["vacation"], exo["event"], exo["campaign"],
                               series)
        y_raw = float(model.predict(X)[0])
        y_hat = max(0.0, _apply_calibration(y_raw))
        series.append(y_hat)
        cur += timedelta(days=1)
    return series

def predict_range_aligned(dates: List[date]) -> pd.DataFrame:
    dates = sorted(dates)
    if not dates:
        return pd.DataFrame()

    series = _build_series_until(dates[0])
    rows = []
    prev = dates[0] - timedelta(days=1)
    for d in dates:
        gap_start = prev + timedelta(days=1)
        if gap_start < d:
            cur = gap_start
            while cur <= d - timedelta(days=1):
                exo_gap = _exo_for_date(cur)
                X_gap = _features_for_date(cur, exo_gap["temperature"], exo_gap["rain_mm"],
                                           exo_gap["vacation"], exo_gap["event"], exo_gap["campaign"],
                                           series)
                y_raw_gap = float(model.predict(X_gap)[0])
                y_hat_gap = max(0.0, _apply_calibration(y_raw_gap))
                series.append(y_hat_gap)
                cur += timedelta(days=1)

        exo = _exo_for_date(d)
        X = _features_for_date(d, exo["temperature"], exo["rain_mm"],
                               exo["vacation"], exo["event"], exo["campaign"],
                               series)
        y_raw = float(model.predict(X)[0])
        y_hat = max(0.0, _apply_calibration(y_raw))
        series.append(y_hat)

        row = {"date": d.strftime('%Y-%m-%d'), **X.iloc[0].to_dict(),
               "predicted_visitors": int(round(y_hat))}
        rows.append(row)
        prev = d

    return pd.DataFrame(rows)

# --------------------------- Ranges utils ------------------------------
def get_monday_of_current_week(today: date | None = None) -> date:
    today = today or date.today()
    return today - timedelta(days=today.isoweekday() - 1)

def _month_range(today: date) -> Tuple[date, date]:
    start = date(today.year, today.month, 1)
    end = (date(today.year + (today.month // 12), ((today.month % 12) + 1), 1) - timedelta(days=1))
    return start, end

# --------------------------- Generators --------------------------------
def generate_daily_prediction() -> Dict:
    d = date.today()
    df_new = predict_range_aligned([d])

    preds = _safe_read_or_init(PREDICTIONS_PATH, df_new)
    preds = preds[preds['date'] != df_new.iloc[0]['date']]
    out = pd.concat([preds, df_new], ignore_index=True)
    _safe_write(PREDICTIONS_PATH, out)
    return df_new.iloc[0].to_dict()

def _weekly_exists_for(week_key: str) -> bool:
    if not _csv_exists(WEEKLY_PREDICTIONS_PATH):
        return False
    try:
        df = pd.read_csv(WEEKLY_PREDICTIONS_PATH)
        if "week_start" not in df.columns:
            return False
        return (df["week_start"].astype(str) == week_key).any()
    except Exception:
        return False

def _monthly_exists_for(month_key: str) -> bool:
    if not _csv_exists(MONTHLY_PREDICTIONS_PATH):
        return False
    try:
        df = pd.read_csv(MONTHLY_PREDICTIONS_PATH)
        if "month_start" not in df.columns:
            return False
        return (df["month_start"].astype(str) == month_key).any()
    except Exception:
        return False

def generate_weekly_prediction_if_missing() -> None:
    monday = get_monday_of_current_week(date.today())
    week_key = monday.strftime('%Y-%m-%d')
    if _weekly_exists_for(week_key):
        return  # al aanwezig

    dates = [monday + timedelta(days=i) for i in range(7)]
    df_week = predict_range_aligned(dates)
    df_week.insert(0, 'week_start', week_key)

    sample = df_week.copy()
    weekly = _safe_read_or_init(WEEKLY_PREDICTIONS_PATH, sample)
    weekly = weekly[weekly['week_start'] != week_key]
    weekly = pd.concat([weekly, df_week], ignore_index=True)
    _safe_write(WEEKLY_PREDICTIONS_PATH, weekly)

def generate_monthly_prediction_if_missing() -> None:
    today = date.today()
    start, end = _month_range(today)
    month_key = start.strftime('%Y-%m-%d')
    if _monthly_exists_for(month_key):
        return  # al aanwezig

    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur += timedelta(days=1)

    df_month = predict_range_aligned(dates)
    df_month.insert(0, 'month_start', month_key)

    sample = df_month.copy()
    monthly = _safe_read_or_init(MONTHLY_PREDICTIONS_PATH, sample)
    monthly = monthly[monthly['month_start'] != month_key]
    monthly = pd.concat([monthly, df_month], ignore_index=True)
    _safe_write(MONTHLY_PREDICTIONS_PATH, monthly)

# --------------------------- Boot helper -------------------------------
def boot_pipeline() -> Dict[str, int]:
    """
    Geen harde rebuild. We:
    - verversen de dagvoorspelling (altijd)
    - maken week/maand alleen aan als ze ontbreken
    """
    daily_obj = generate_daily_prediction()
    generate_weekly_prediction_if_missing()
    generate_monthly_prediction_if_missing()

    daily_df   = pd.read_csv(PREDICTIONS_PATH)   if _csv_exists(PREDICTIONS_PATH) else pd.DataFrame()
    weekly_df  = pd.read_csv(WEEKLY_PREDICTIONS_PATH)  if _csv_exists(WEEKLY_PREDICTIONS_PATH) else pd.DataFrame()
    monthly_df = pd.read_csv(MONTHLY_PREDICTIONS_PATH) if _csv_exists(MONTHLY_PREDICTIONS_PATH) else pd.DataFrame()
    return {
        "daily_rows":   int(len(daily_df)),
        "weekly_rows":  int(len(weekly_df)),
        "monthly_rows": int(len(monthly_df)),
        "today_pred":   int(daily_obj.get("predicted_visitors", 0))
    }

# -------------------------------- API ----------------------------------
@app.route('/api/predict_today', methods=['GET'])
def predict_today():
    today_str = date.today().strftime('%Y-%m-%d')
    df_sample = predict_range_aligned([date.today()])
    preds = _safe_read_or_init(PREDICTIONS_PATH, df_sample)
    row = preds.loc[preds['date'] == today_str]
    if not row.empty:
        return jsonify(row.iloc[0].to_dict())
    out = generate_daily_prediction()
    return jsonify(out)

@app.route('/api/predict_custom', methods=['POST', 'OPTIONS'])
def predict_custom():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    try:
        data = request.get_json(force=True)

        # ---- Datum (VERPLICHT) ----
        d = datetime.strptime(data["date"], "%Y-%m-%d").date()

        # ---- Exogene inputs ----
        temp = float(data.get('temperature', 20))
        rain = float(data.get('rain_mm', 0))
        vac  = int(data.get('vacation', 0))
        evt  = int(data.get('event', 0))
        camp = int(data.get('campaign', 0))

        # ---- Serie uitlijnen t/m dag vóór gekozen datum ----
        series = _build_series_until(d)

        # ---- Features bouwen (ALLE tijdsinfo uit datum!) ----
        X = _features_for_date(
            d,
            temp=temp,
            rain=rain,
            vac=vac,
            evt=evt,
            camp=camp,
            series=series
        )

        y_raw = float(model.predict(X)[0])
        y_hat = max(0.0, _apply_calibration(y_raw))

        y_hat *= _temperature_correction(temp)
        y_hat *= _rain_temperature_interaction(temp, rain)

        return jsonify({
            "date": d.strftime("%Y-%m-%d"),
            "predicted_visitors": int(round(y_hat))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict_week', methods=['GET'])
def api_predict_week():
    # NIET forceren; alleen aanmaken als ontbreekt
    generate_weekly_prediction_if_missing()
    monday = get_monday_of_current_week(date.today())
    week_key = monday.strftime('%Y-%m-%d')

    dates = [monday + timedelta(days=i) for i in range(7)]
    df_sample = predict_range_aligned(dates)
    sample_with_key = df_sample.assign(week_start=week_key)[['week_start'] + df_sample.columns.tolist()]
    weekly = _safe_read_or_init(WEEKLY_PREDICTIONS_PATH, sample_with_key)

    this_week = weekly[weekly['week_start'] == week_key].copy().sort_values('date')
    iso_year, iso_week, _ = monday.isocalendar()
    weekday_names = ['Maandag','Dinsdag','Woensdag','Donderdag','Vrijdag','Zaterdag','Zondag']
    days = []; total = 0
    for _, r in this_week.iterrows():
        dt = pd.to_datetime(r['date']).date()
        pv = int(r['predicted_visitors']); total += pv
        days.append({
            'date': r['date'],
            'weekday': weekday_names[dt.weekday()],
            'temperature': float(r['temperature']),
            'rain_mm': float(r['rain_mm']),
            'vacation': bool(int(r['vacation'])),
            'event': bool(int(r['event'])),
            'campaign': bool(int(r['campaign'])),
            'predicted_visitors': pv
        })
    return jsonify({
        'week_start': monday.strftime('%Y-%m-%d'),
        'week_end': (monday + timedelta(days=6)).strftime('%Y-%m-%d'),
        'iso_week': iso_week, 'year': iso_year,
        'days': days, 'total': total
    })

@app.route('/api/predict_month', methods=['GET'])
def api_predict_month():
    # NIET forceren; alleen aanmaken als ontbreekt
    generate_monthly_prediction_if_missing()
    today = date.today()
    start, end = _month_range(today)
    month_key = start.strftime('%Y-%m-%d')

    dates = []; cur = start
    while cur <= end:
        dates.append(cur); cur += timedelta(days=1)
    df_sample = predict_range_aligned(dates)
    sample_with_key = df_sample.assign(month_start=month_key)[['month_start'] + df_sample.columns.tolist()]
    monthly = _safe_read_or_init(MONTHLY_PREDICTIONS_PATH, sample_with_key)

    this_month = monthly[monthly['month_start'] == month_key]
    total = int(this_month['predicted_visitors'].sum()) if not this_month.empty else 0

    month_names = ["","januari","februari","maart","april","mei","juni",
                   "juli","augustus","september","oktober","november","december"]
    return jsonify({
        'month_start': start.strftime('%Y-%m-%d'),
        'month_end': end.strftime('%Y-%m-%d'),
        'year': start.year, 'month': start.month,
        'month_name': month_names[start.month],
        'total_predicted_visitors': total
    })

@app.route('/api/rebuild', methods=['POST'])
def api_rebuild():
    """
    Handmatige rebuild: gooit niets meer weg.
    Verversen we de dag en zorgen we dat week/maand aanwezig zijn.
    """
    try:
        stats = boot_pipeline()
        return jsonify({"status": "ok", **stats})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
        
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True}), 200




# --------------------------- Scheduler ---------------------------------
scheduler = BackgroundScheduler(daemon=True)
# Dagelijks: altijd bijwerken
scheduler.add_job(lambda: generate_daily_prediction(), 'cron', hour=8, minute=0)
# Wekelijks: alleen aanmaken als ontbreekt (maandag vlak na middernacht)
scheduler.add_job(lambda: generate_weekly_prediction_if_missing(), 'cron', day_of_week='mon', hour=0, minute=5)
# Maandelijks: alleen aanmaken als ontbreekt (dag 1)
scheduler.add_job(lambda: generate_monthly_prediction_if_missing(), 'cron', day=1, hour=0, minute=10)
scheduler.start()

if __name__ == "__main__":
    print("→ Running Wildlands Prediction Backend")
    print("→ Daily refresh; weekly/monthly only once per period…")
    # Start de zware init asynchroon zodat /health meteen beschikbaar is
    import threading
    threading.Thread(target=boot_pipeline, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)

