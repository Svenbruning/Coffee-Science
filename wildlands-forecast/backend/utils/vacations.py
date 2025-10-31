import pandas as pd
from datetime import datetime
import os

VACATIONS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'vacations.csv')

def is_vacation_today(date=None):
    today = datetime.today().date() if date is None else pd.to_datetime(date).date()
    df = pd.read_csv(VACATIONS_PATH, parse_dates=["start_date", "end_date"])
    for _, row in df.iterrows():
        if row["start_date"].date() <= today <= row["end_date"].date():
            return 1
    return 0
