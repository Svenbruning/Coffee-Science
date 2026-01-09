# utils/vacations.py
import os
import pandas as pd
from datetime import datetime, date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VACATIONS_PATH = os.path.join(BASE_DIR, 'data', 'vacations.csv')

if os.path.exists(VACATIONS_PATH):
    _vac_df = pd.read_csv(VACATIONS_PATH, parse_dates=["start_date", "end_date"])
else:
    _vac_df = pd.DataFrame(columns=["start_date", "end_date"])

def _to_date(d) -> date:
    return d if isinstance(d, date) else pd.to_datetime(d).date()

def vacation_flag_on(d: date) -> int:
    if _vac_df.empty:
        return 0
    dd = _to_date(d)
    mask = (_vac_df["start_date"].dt.date <= dd) & (_vac_df["end_date"].dt.date >= dd)
    return int(mask.any())

def is_vacation_today(d=None) -> int:
    return vacation_flag_on(_to_date(d or datetime.today().date()))
