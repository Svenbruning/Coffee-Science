import pandas as pd
from datetime import datetime

VACATIONS_PATH = "../data/vacations.csv"

def is_vacation_today():
    today = datetime.today().date()
    df = pd.read_csv(VACATIONS_PATH, parse_dates=["start_date", "end_date"])
    for _, row in df.iterrows():
        if row["start_date"].date() <= today <= row["end_date"].date():
            return 1
    return 0
