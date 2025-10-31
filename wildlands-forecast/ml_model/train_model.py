import pandas as pd
import lightgbm as lgb
import joblib
import os

# Data
DATA_PATH = os.path.join('..', 'data', 'daily_visitors.csv')
df = pd.read_csv(DATA_PATH)

# Rename
df.rename(columns={
    'Date':'date',
    'Nr Used Entrances':'visitors',
    'Temperature (mean)':'temperature',
    'Precipitation (sum)':'rain_mm',
    'Vacation':'vacation',
    'Event':'event',
    'Campaign':'campaign'
}, inplace=True)

df['date'] = pd.to_datetime(df['date'])

# Binary columns
for col in ['vacation','event','campaign']:
    df[col] = df[col].apply(lambda x: 0 if pd.isna(x) or x in [0,'0'] else 1)

# Lag & rolling features
df['lag_1'] = df['visitors'].shift(1)
df['lag_7'] = df['visitors'].shift(7)
df['roll_3'] = df['visitors'].shift(1).rolling(3, min_periods=1).mean()
df['roll_7'] = df['visitors'].shift(1).rolling(7, min_periods=1).mean()

# Extra features
df['day_of_week'] = df['date'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['month'] = df['date'].dt.month

df = df.dropna()

# Features & target
features = ['temperature','rain_mm','vacation','event','campaign',
            'lag_1','lag_7','roll_3','roll_7','day_of_week','is_weekend','month']
X = df[features]
y = df['visitors']

# Train LightGBM
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)
model.fit(X, y)

# Save model
MODEL_PATH = os.path.join('..','ml_model','model_lgb.pkl')
joblib.dump(model, MODEL_PATH)
print("✅ LightGBM model trained and saved as model_lgb.pkl")
