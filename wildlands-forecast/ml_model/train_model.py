import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Pad naar dataset
DATA_PATH = os.path.join('..', 'data', 'raw_visitors.csv')
df = pd.read_csv(DATA_PATH, parse_dates=['date']).sort_values('date')

# Lag features
df['lag_1'] = df['visitors'].shift(1)
df['lag_7'] = df['visitors'].shift(7)

# Rolling features
df['roll_3'] = df['visitors'].shift(1).rolling(3, min_periods=1).mean()
df['roll_7'] = df['visitors'].shift(1).rolling(7, min_periods=1).mean()

# Drop NaN
df = df.dropna()

# Features en target
features = ['temperature','rain_mm','vacation','event','lag_1','lag_7','roll_3','roll_7']
X = df[features]
y = df['visitors']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
MODEL_PATH = os.path.join('..','ml_model','model.pkl')
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved as model.pkl")
