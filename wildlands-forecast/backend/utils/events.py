# utils/events.py
import os
import pandas as pd
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
EVENTS_PATH = os.path.join(BASE_DIR, 'data', 'events.csv')

if os.path.exists(EVENTS_PATH):
    _events_df = pd.read_csv(EVENTS_PATH, parse_dates=["start_date", "end_date"])
else:
    _events_df = pd.DataFrame(columns=["start_date", "end_date"])

def _to_date(d) -> date:
    return d if isinstance(d, date) else pd.to_datetime(d).date()

def event_flag_on(d: date) -> int:
    if _events_df.empty:
        return 0
    dd = _to_date(d)
    mask = (_events_df["start_date"].dt.date <= dd) & (_events_df["end_date"].dt.date >= dd)
    return int(mask.any())

def is_event_today(d=None) -> int:
    return event_flag_on(_to_date(d or datetime.today().date()))
