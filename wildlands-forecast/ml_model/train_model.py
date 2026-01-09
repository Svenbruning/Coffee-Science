# ml/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'daily_visitors.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'model_lgb.pkl')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---- Load ----
df = pd.read_csv(DATA_PATH)
df.rename(columns={
    'Date':'date','Nr Used Entrances':'visitors',
    'Temperature (mean)':'temperature','Precipitation (sum)':'rain_mm',
    'Vacation':'vacation','Event':'event','Campaign':'campaign'
}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Binariseer tekst/Yes → 1
for col in ['vacation','event','campaign']:
    df[col] = df[col].apply(
        lambda x: 0 if (pd.isna(x) or str(x).strip() in {'0','False','false','No','no',''}) else 1
    ).astype(int)

# ---- Features ----
def add_time_features(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x['day_of_week'] = x['date'].dt.weekday
    x['is_weekend']  = x['day_of_week'].isin([5,6]).astype(int)
    x['month']       = x['date'].dt.month
    x['weekofyear']  = x['date'].dt.isocalendar().week.astype(int)
    x['dayofyear']   = x['date'].dt.dayofyear
    x['sin_doy'] = np.sin(2*np.pi*x['dayofyear']/365.25)
    x['cos_doy'] = np.cos(2*np.pi*x['dayofyear']/365.25)
    return x

def add_lag_roll(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x['lag_1']  = x['visitors'].shift(1)
    x['lag_7']  = x['visitors'].shift(7)
    x['lag_14'] = x['visitors'].shift(14)
    x['roll_3']  = x['visitors'].shift(1).rolling(3, min_periods=1).mean()
    x['roll_7']  = x['visitors'].shift(1).rolling(7, min_periods=1).mean()
    x['roll_14'] = x['visitors'].shift(1).rolling(14, min_periods=1).mean()
    return x

def add_weather_interactions(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x['temp2'] = x['temperature']**2
    x['rain_log1p'] = np.log1p(x['rain_mm'])
    x['weekend_temp'] = x['is_weekend'] * x['temperature']
    return x

df = add_time_features(df)
df = add_lag_roll(df)
df = add_weather_interactions(df)
df = df.dropna().reset_index(drop=True)

FEATURES = [
    'temperature','rain_mm','temp2','rain_log1p','weekend_temp',
    'vacation','event','campaign',
    'lag_1','lag_7','lag_14','roll_3','roll_7','roll_14',
    'day_of_week','is_weekend','month','weekofyear','sin_doy','cos_doy'
]
X_full = df[FEATURES]
y_full = df['visitors'].astype(float)

# ---- TimeSeries CV + param keus ----
tscv = TimeSeriesSplit(n_splits=5)
param_grid = [
    dict(n_estimators=5000, learning_rate=0.02, num_leaves=63,
         subsample=0.8, colsample_bytree=0.8, min_data_in_leaf=50,
         reg_lambda=1.0, reg_alpha=0.0, objective='tweedie',
         tweedie_variance_power=1.2, random_state=42),
    dict(n_estimators=4000, learning_rate=0.03, num_leaves=127,
         subsample=0.9, colsample_bytree=0.9, min_data_in_leaf=80,
         reg_lambda=2.0, reg_alpha=0.0, objective='tweedie',
         tweedie_variance_power=1.3, random_state=42),
]

def fold_metric(y_true, y_pred) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    smape = np.mean(2*np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
    return mae, smape

best = (np.inf, np.inf); best_params = None; last_fold = None; best_model = None
for params in param_grid:
    maes, smapes, models = [], [], []
    for tr, va in tscv.split(X_full):
        Xt, Xv = X_full.iloc[tr], X_full.iloc[va]
        yt, yv = y_full.iloc[tr], y_full.iloc[va]
        m = lgb.LGBMRegressor(**params)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric='l2',
              callbacks=[lgb.early_stopping(200, verbose=False)])
        yp = m.predict(Xv)
        mae, smape = fold_metric(yv.values, yp)
        maes.append(mae); smapes.append(smape); models.append(m)
    avg = (float(np.mean(maes)), float(np.mean(smapes)))
    if (avg[1] < best[1]) or (avg[1] == best[1] and avg[0] < best[0]):
        best, best_params, best_model = avg, params, models[-1]
        last_idx = list(tscv.split(X_full))[-1][1]
        last_fold = (y_full.iloc[last_idx].values, models[-1].predict(X_full.iloc[last_idx]))

# ---- Calibratie (bias correctie) ----
if last_fold is not None:
    y_true, y_hat = last_fold
    A = np.vstack([y_hat, np.ones_like(y_hat)]).T
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
else:
    a, b = 1.0, 0.0

# ---- Retrain full ----
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_full, y_full)

bundle = {
    "model": final_model,
    "features": FEATURES,
    "calibration": {"a": float(a), "b": float(b)},
    "cv_score": {"mae": best[0], "sMAPE": best[1], "params": best_params},
}
joblib.dump(bundle, MODEL_PATH)
print("✅ saved:", MODEL_PATH)
print("CV:", best, "calibration a,b:", a, b)
