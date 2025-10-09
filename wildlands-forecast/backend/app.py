from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Model laden
MODEL_PATH = os.path.join('..', 'ml_model', 'model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Dataframe van 1 rij met exact dezelfde feature namen
    df = pd.DataFrame([{
        'temperature': data.get('temperature'),
        'rain': data.get('rain'),
        'vacation': data.get('vacation'),
        'event': data.get('event')
    }])
    prediction = model.predict(df)[0]
    return jsonify({'predicted_visitors': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
