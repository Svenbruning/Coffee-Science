df["day_of_week_sin"] =np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_of_week_cos"] =np.cos(2 * np.pi * df["day_of_week"] / 7)

df["month_sin"] =np.sin(2 * np.pi * df["month"] / 7)
df["month_cos"] =np.cos(2 * np.pi * df["month"] / 7)
