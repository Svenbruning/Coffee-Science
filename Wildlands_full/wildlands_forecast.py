#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wildlands visitors forecasting

Input files (put them in the same folder as this script):
  - visitordaily.csv                (semicolon-separated: AccessGroupId;Description;Date;NumberOfUsedEntrances)
  - weather.csv                     (Date,Temperature,Precipitation,ID_Time,Time) — hourly rows
  - events_2023_2024_2025.txt       (Date,Event)
  - holiday*/holidays* .txt files   for Netherlands & Germany, 2023–2025
      (headers may vary: Date/Name or date/holiday_name)

What the script does:
  1) Load and merge data (calendar, holidays NL/DE, events, weather)
  2) Build features incl. lags/rolling stats per AccessGroupId
  3) Train a LightGBM model and validate on the last 60 days
  4) Produce a 365-day recursive forecast per AccessGroupId

Outputs:
  - forecast_365d.csv           (per AccessGroupId daily forecast)
  - forecast_365d_total.csv     (sum across all AccessGroupId per day)
  - validation_metrics.txt      (MAE, MAPE on the validation window)
"""

import re
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib
from lightgbm import LGBMRegressor
try:
    from lightgbm import early_stopping, log_evaluation
except Exception:
    from lightgbm.callback import early_stopping, log_evaluation

# ----------------------- Paths -----------------------
# Change DATA_DIR if your files are located elsewhere
DATA_DIR = Path(".")
VISITORS_FILE = DATA_DIR / "visitordaily.csv"
WEATHER_FILE = DATA_DIR / "weather.csv"
EVENTS_FILE = DATA_DIR / "events_2023_2024_2025.txt"
HOLIDAY_FILES = sorted(list(DATA_DIR.glob("holiday*.txt")) + list(DATA_DIR.glob("holidays*.txt")))

# ----------------------- Helpers -----------------------
def parse_date_series(s: pd.Series) -> pd.Series:
    """Parse date strings robustly (supports ISO 'YYYY-MM-DD' and 'M/D/YYYY')."""
    s = s.astype(str).str.strip()
    try:
        return pd.to_datetime(s, format="%Y-%m-%d")
    except Exception:
        return pd.to_datetime(s, dayfirst=False)

def load_visitors(path: Path) -> pd.DataFrame:
    """Load visitors CSV (semicolon-separated) and aggregate to day × AccessGroupId."""
    df = pd.read_csv(path, sep=";", engine="python")
    expected = {"AccessGroupId", "Description", "Date", "NumberOfUsedEntrances"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"visitordaily.csv is missing columns: {missing}")
    df["Date"] = parse_date_series(df["Date"]).dt.normalize()
    df["AccessGroupId"] = df["AccessGroupId"].astype(str)
    # Sum to daily level for safety
    agg = (df.groupby(["AccessGroupId", "Date"], as_index=False)
             .agg(NumberOfUsedEntrances=("NumberOfUsedEntrances", "sum")))
    # Keep Description as a lookup table
    desc = df[["AccessGroupId", "Description"]].drop_duplicates()
    return agg.merge(desc, on="AccessGroupId", how="left")

def load_weather(path: Path) -> pd.DataFrame:
    """Load hourly weather and aggregate to daily mean temperature & daily precipitation sum."""
    w = pd.read_csv(path)
    for c in ["Date", "Temperature", "Precipitation"]:
        if c not in w.columns:
            raise ValueError(f"weather.csv is missing '{c}'")
    w["Date"] = parse_date_series(w["Date"]).dt.normalize()
    return (w.groupby("Date", as_index=False)
              .agg(daily_temp_mean=("Temperature", "mean"),
                   daily_precip_sum=("Precipitation", "sum")))

def load_events(path: Path) -> pd.DataFrame:
    """Load events (Date, Event) and reduce to a daily binary flag."""
    if not path.exists():
        return pd.DataFrame(columns=["Date", "has_event"])
    e = pd.read_csv(path)
    if "Date" not in e.columns:
        # Fallback: assume first column is date
        e = e.rename(columns={e.columns[0]: "Date"})
    e["Date"] = parse_date_series(e["Date"]).dt.normalize()
    out = e.groupby("Date", as_index=False).size().rename(columns={"size": "has_event"})
    out["has_event"] = 1
    return out


def load_holidays(paths) -> pd.DataFrame:
    """Load multiple holiday files (mixed headers) and build wide flags per country (DE/NL)."""
    rows = []
    for p in paths:
        df = pd.read_csv(p)
        # normalize headers to lower and find date/name columns
        df.columns = [c.lower() for c in df.columns]
        datecol = [c for c in df.columns if "date" in c][0]
        namecol = [c for c in df.columns if "name" in c][0]
        df = df.rename(columns={datecol: "Date", namecol: "Name"})
        # infer country from filename
        fname = p.name.lower()
        if "germany" in fname:
            country = "DE"
        elif "netherland" in fname:
            country = "NL"
        else:
            country = "UNK"
        df["Date"] = parse_date_series(df["Date"]).dt.normalize()
        df["Country"] = country
        rows.append(df[["Date", "Country"]])
    if not rows:
        return pd.DataFrame(columns=["Date", "is_holiday_DE", "is_holiday_NL"])
    hol = pd.concat(rows, ignore_index=True).dropna(subset=["Date"])
    hol["val"] = 1
    hol = (hol.pivot_table(index="Date", columns="Country", values="val",
                           aggfunc="max", fill_value=0).reset_index())
    if "DE" not in hol.columns: hol["DE"] = 0
    if "NL" not in hol.columns: hol["NL"] = 0
    hol = hol.rename(columns={"DE": "is_holiday_DE", "NL": "is_holiday_NL"})
    return hol

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features."""
    df["dow"] = df["Date"].dt.weekday           # 0=Mon
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["Date"].dt.month
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofyear"] = df["Date"].dt.dayofyear
    return df

def add_group_lags(gdf: pd.DataFrame) -> pd.DataFrame:
    """Create target lags and rolling stats within a single AccessGroupId."""
    gdf = gdf.sort_values("Date").copy()
    for L in (1, 7, 14, 28, 364):
        gdf[f"lag_{L}"] = gdf["NumberOfUsedEntrances"].shift(L)
    for w in (7, 28):
        gdf[f"roll_mean_{w}"] = gdf["NumberOfUsedEntrances"].shift(1).rolling(w).mean()
        gdf[f"roll_std_{w}"] = gdf["NumberOfUsedEntrances"].shift(1).rolling(w).std()
    return gdf

# ----------------------- Load & merge -----------------------
print("Loading data...")
vis = load_visitors(VISITORS_FILE)
weather = load_weather(WEATHER_FILE)
events = load_events(EVENTS_FILE)
holidays = load_holidays(HOLIDAY_FILES)

print("Merging features...")
df = (vis.merge(holidays, on="Date", how="left")
         .merge(weather, on="Date", how="left")
         .merge(events, on="Date", how="left"))

# Fill missing feature values
for c in ["is_holiday_NL", "is_holiday_DE", "has_event"]:
    df[c] = df.get(c, 0)
    df[c] = df[c].fillna(0).astype(int)
df["daily_temp_mean"] = df["daily_temp_mean"].interpolate(limit_direction="both")
df["daily_precip_sum"] = df["daily_precip_sum"].fillna(0)

# Calendar features and target lags
df = add_time_features(df)
df = (df.sort_values("Date")
        .groupby("AccessGroupId", group_keys=False, sort=False)
        .apply(lambda g: add_group_lags(g)))
df = df.dropna().reset_index(drop=True)

# ----------------------- Train/validation split -----------------------
val_days = 60
max_date = df["Date"].max()
val_start = max_date - pd.Timedelta(days=val_days - 1)
train_df = df[df["Date"] < val_start].copy()
val_df   = df[df["Date"] >= val_start].copy()

feature_cols = [c for c in df.columns if c not in ["NumberOfUsedEntrances", "Date"]]
X_tr, y_tr = train_df[feature_cols], train_df["NumberOfUsedEntrances"]
X_va, y_va = val_df[feature_cols], val_df["NumberOfUsedEntrances"]

from pandas.api.types import CategoricalDtype
cat_cols = [c for c in ["AccessGroupId", "Description"] if c in X_tr.columns]
cat_dtypes = {}

for c in cat_cols:
    #build unique set of categories from both train and validation
    all_vals = pd.concat([X_tr[c], X_va[c]], ignore_index=True).astype(str)
    cats = pd.Index(all_vals.unique())
    dtype = CategoricalDtype(categories=cats, ordered=False)
    #cast both splits to the same category dtype
    X_tr[c] = X_tr[c].astype(str).astype(dtype)
    X_va[c] = X_va[c].astype(str).astype(dtype)
    #remember dtype for inference
    cat_dtypes[c] = dtype

# ----------------------- Train LightGBM -----------------------
print("Training LightGBM...")
model = LGBMRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=127,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="l1",
    callbacks=[
        early_stopping(stopping_rounds=300),
        log_evaluation(period=200)
    ]
)


val_pred = model.predict(X_va)
mae = mean_absolute_error(y_va, val_pred)
mape = (np.abs((y_va - val_pred) / np.maximum(1, y_va))).mean()
Path("validation_metrics.txt").write_text(f"MAE={mae:.3f}\nMAPE={mape*100:.2f}%\n", encoding="utf-8")
print(f"Validation MAE={mae:.3f}, MAPE={mape*100:.2f}%")

#---------------------- Save model -----------------------------------
cat_info = {c: list(cat_dtypes[c].categories) for c in cat_cols}
artifact = {
    "model": model,
    "feature_cols": feature_cols,
    "cat_cols": cat_cols,
    "cat_info": cat_info,
}
joblib.dump(artifact, "lgbm_model.joblib")
print("Saved lgbm_model.joblib")

# ----------------------- Build 365-day future -----------------------
print("Preparing 365-day future...")
FORECAST_DAYS = 365
future_dates = pd.date_range(df["Date"].max() + pd.Timedelta(days=1),
                             periods=FORECAST_DAYS, freq="D")
future = pd.DataFrame({"Date": future_dates})
future = add_time_features(future)

# Holidays & events for future dates (merge if provided; otherwise zeros)
future = future.merge(holidays, on="Date", how="left").merge(events, on="Date", how="left")
for c in ["is_holiday_NL", "is_holiday_DE", "has_event"]:
    future[c] = future.get(c, 0)
    future[c] = future[c].fillna(0).astype(int)

# Weather for future: use a climatology (mean by day-of-year)
clim = (weather.assign(dayofyear=lambda d: d["Date"].dt.dayofyear)
                .groupby("dayofyear", as_index=False)
                .agg(temp=("daily_temp_mean","mean"),
                     prec=("daily_precip_sum","mean")))
future = future.merge(clim, on="dayofyear", how="left")
future["daily_temp_mean"] = future["temp"]
future["daily_precip_sum"] = future["prec"]
future = future.drop(columns=["temp", "prec"])

# ----------------------- Recursive forecast per group -----------------------
print("Forecasting per AccessGroupId...")

lag_cols = [c for c in df.columns if c.startswith("lag_")]
roll_windows = [7, 28]  # used above

desc_map = {}
if "Description" in df.columns:
    desc_map = (df[["AccessGroupId", "Description"]]
                .drop_duplicates()
                .set_index("AccessGroupId")["Description"]
                .to_dict())

preds = []
for gid, g_hist in df.groupby("AccessGroupId"):
    # Keep last ~400 days to have enough context for lag_364
    history = g_hist.sort_values("Date")[["Date", "NumberOfUsedEntrances"]].copy()
    recent = history.tail(400).copy()
    # Precompute weekday averages for cold start fallback
    dow_avg = history.groupby(history["Date"].dt.weekday)["NumberOfUsedEntrances"].mean()

    for _, row in future.iterrows():
        # Build feature row for this date & group
        feat = row.to_dict()
        feat["AccessGroupId"] = gid
        if "Description" in feature_cols:
            feat["Description"] = desc_map.get(gid, "unknown")
        # Lags from recent actuals/predictions
        for col in lag_cols:
            L = int(col.split("_")[1])
            feat[col] = recent["NumberOfUsedEntrances"].iloc[-L] if len(recent) >= L else np.nan
        # Rolling stats (based on previous values)
        for w in roll_windows:
            tail = recent["NumberOfUsedEntrances"].iloc[-w:]
            feat[f"roll_mean_{w}"] = tail.mean() if len(tail) else np.nan
            feat[f"roll_std_{w}"] = tail.std(ddof=0) if len(tail) else np.nan

        # Fallback if NaNs (not enough history at the very beginning)
        if any(pd.isna(feat.get(c)) for c in lag_cols):
            guess = dow_avg.loc[feat["dow"]] if feat["dow"] in dow_avg.index else history["NumberOfUsedEntrances"].mean()
            for col in lag_cols:
                if pd.isna(feat[col]):
                    feat[col] = guess
            for w in roll_windows:
                if pd.isna(feat.get(f"roll_mean_{w}")):
                    feat[f"roll_mean_{w}"] = guess
                    feat[f"roll_std_{w}"] = history["NumberOfUsedEntrances"].std()

        feat_df = pd.DataFrame([feat])[feature_cols]
        for c in cat_cols:
            if c in feat_df.columns:
                feat_df[c] = feat_df[c].astype(str).astype(cat_dtypes[c])
        yhat = float(model.predict(feat_df)[0])
        preds.append({"AccessGroupId": gid, "Date": row["Date"], "yhat": yhat})
        # Append prediction to recent to advance lags
        recent = pd.concat([recent, pd.DataFrame({"Date": [row["Date"]],
                                                  "NumberOfUsedEntrances": [yhat]})],
                           ignore_index=True)

# Save per-group forecast
pred = pd.DataFrame(preds).sort_values(["AccessGroupId", "Date"])
pred.to_csv("forecast_365d.csv", index=False, encoding="utf-8")
print("Saved forecast_365d.csv")


# Save total (sum over all groups) per-day forecast
total = (pred.groupby("Date", as_index=False)["yhat"].sum()
            .rename(columns={"yhat": "yhat_total"}))
total.to_csv("forecast_365d_total.csv", index=False, encoding="utf-8")
print("Saved forecast_365d_total.csv")

print("Done. Check validation_metrics.txt, forecast_365d.csv, forecast_365d_total.csv")
