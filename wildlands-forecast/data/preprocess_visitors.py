import pandas as pd

# Laad volledige CSV
df = pd.read_csv('visitors.csv', parse_dates=['Date'])

# Groepeer per dag
df_daily = df.groupby('Date').agg({
    'Nr Used Entrances': 'sum',               # totaal bezoekers
    'Temperature (mean)': 'mean',             # gemiddelde temperatuur per dag
    'Precipitation (sum)': 'mean',            # gebruik gemiddelde in plaats van sum
    'Holiday_Germany': 'max',
    'Holiday_Netherlands': 'max',
    'Event': 'max',
    'Campaign_Netherlands': 'max',
    'Campaign_Germany': 'max'
}).reset_index()

# Combineer beide vakantie-kolommen tot 1
df_daily['Vacation'] = df_daily[['Holiday_Germany','Holiday_Netherlands']].max(axis=1)
df_daily['Campaign'] = df_daily[['Campaign_Netherlands','Campaign_Germany']].max(axis=1)

# Selecteer relevante kolommen en rond af op 1 decimaal
df_daily = df_daily[['Date', 'Nr Used Entrances', 'Temperature (mean)', 'Precipitation (sum)', 'Vacation', 'Event', 'Campaign']]
df_daily['Temperature (mean)'] = df_daily['Temperature (mean)'].round(1)
df_daily['Precipitation (sum)'] = df_daily['Precipitation (sum)'].round(1)

# Sla op
df_daily.to_csv('daily_visitors.csv', index=False)
print("✅ daily_visitors.csv aangemaakt in data/ met afgeronde waarden")
