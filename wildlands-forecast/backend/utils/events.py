import pandas as pd
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # twee niveaus omhoog
EVENTS_PATH = os.path.join(BASE_DIR, 'data', 'events.csv')

def is_event_today(date=None):
    today = datetime.today().date() if date is None else pd.to_datetime(date).date()
    df = pd.read_csv(EVENTS_PATH, parse_dates=["start_date", "end_date"])
    for _, row in df.iterrows():
        if row["start_date"].date() <= today <= row["end_date"].date():
            return 1
    return 0
