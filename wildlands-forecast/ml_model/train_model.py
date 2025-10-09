import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Path to dataset
DATA_PATH = os.path.join('..', 'data', 'visitors.csv')

# Read data
data = pd.read_csv(DATA_PATH)

# Features and target
X = data[['temperature', 'rain', 'vacation', 'event']]
y = data['visitors']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
MODEL_PATH = 'model.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
