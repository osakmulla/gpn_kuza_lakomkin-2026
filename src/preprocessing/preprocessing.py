import pandas as pd

RAW_PATH = "data/raw/hakhathon_data.csv"
CLEAN_PATH = "data/processed/clean.csv"
GAP_SECONDS = 60

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

SPIKE_SIGNALS = ["HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
                 "InletFlow", "Load", "Turn_rotor", "WedgePosition"]

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

fixed_report = {}
for name in SPIKE_SIGNALS:
    low = data[name].quantile(0.001)
    high = data[name].quantile(0.999)
    noise_floor = 0.02 * (high - low)
    fixed_report[name] = 0
    for series in data["series_id"].unique():
        mask = data["series_id"] == series
        x = data.loc[mask, name].reset_index(drop=True)
        center = x.rolling(11, center=True, min_periods=1).median()
        spread = (x - center).abs().rolling(11, center=True, min_periods=1).median() * 1.4826
        spread = spread.where(spread > noise_floor, noise_floor)
        limit = 6 * spread

        jump_from_left = (x - x.shift(1)).abs() > limit
        jump_to_right = (x - x.shift(-1)).abs() > limit
        same_level_around = (x.shift(1) - x.shift(-1)).abs() < 2 * spread
        one_point = jump_from_left & jump_to_right & same_level_around

        next_jump = (x.shift(-1) - x.shift(-2)).abs() > limit
        pair_is_close = (x - x.shift(-1)).abs() < 2 * spread
        same_level_outside = (x.shift(1) - x.shift(-2)).abs() < 2 * spread
        pair_first = jump_from_left & next_jump & pair_is_close & same_level_outside
        pair_second = pair_first.shift(1, fill_value=False)

        bad = one_point | pair_first | pair_second
        x = x.mask(bad).interpolate().bfill().ffill()
        data.loc[mask, name] = x.values
        fixed_report[name] = fixed_report[name] + int(bad.sum())

data.to_csv(CLEAN_PATH, index=False)

print("исправлено одиночных выбросов по сигналам:", fixed_report)
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
