import pandas as pd

RAW_PATH = "data/raw/hakhathon_data.csv"
CLEAN_PATH = "data/processed/clean.csv"
GAP_SECONDS = 60

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

data = pd.read_csv(RAW_PATH)
rows_before = len(data)

data["Time"] = pd.to_datetime(data["Time"], format="%d.%m.%Y %H:%M:%S")
data = data.sort_values(["Time", "ID"], kind="stable")
data = data.drop_duplicates(subset="ID")
data = data.reset_index(drop=True)

data["step_sec"] = data["Time"].diff().dt.total_seconds()
data["series_id"] = (data["step_sec"] > GAP_SECONDS).cumsum()

for series in data["series_id"].unique():
    mask = data["series_id"] == series
    data.loc[mask, signals] = data.loc[mask, signals].interpolate().bfill().ffill()

data["step_sec"] = data["step_sec"].fillna(0)

data.to_csv(CLEAN_PATH, index=False)

print("строк было:", rows_before)
print("строк стало:", len(data))
print("пропусков в сигналах:", int(data[signals].isna().sum().sum()))
print("серий:", data["series_id"].nunique())
print(data.groupby("series_id").agg(строк=("ID", "count"),
                                    первый_ID=("ID", "min"),
                                    последний_ID=("ID", "max"),
                                    начало=("Time", "min"),
                                    конец=("Time", "max")).to_string())
print("порядок ID сохранён:", bool(data["ID"].is_monotonic_increasing))
