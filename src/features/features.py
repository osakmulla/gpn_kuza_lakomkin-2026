import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLEAN_PATH = "data/processed/clean.csv"
OUT_FOLDER = "data/processed/"

PUMP_LIMIT = 1
ROTOR_LIMIT = 5
BOTTOM_LIMIT = 3
WEDGE_LIMIT = 0.5

SPEED_WIDTH = 10
CHANGE_WIDTH = 60
MEAN_WIDTHS = [30, 150, 600]
STD_WIDTHS = [30, 150]

HEAVY = ["HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
         "InletFlow", "Load", "Turn_rotor", "BitPosition"]


def centered_stats(values, times, width):
    half = width / 2
    low = np.searchsorted(times, times - half, side="left")
    high = np.searchsorted(times, times + half, side="right")
    count = high - low
    shift = values.mean()
    moved = values - shift
    sums = np.concatenate([[0.0], np.cumsum(moved)])
    squares = np.concatenate([[0.0], np.cumsum(moved * moved)])
    mean_moved = (sums[high] - sums[low]) / count
    variance = (squares[high] - squares[low]) / count - mean_moved * mean_moved
    variance = np.maximum(variance, 0)
    return mean_moved + shift, np.sqrt(variance)


def side_mean(values, times, width, side):
    if side == "left":
        low = np.searchsorted(times, times - width, side="left")
        high = np.searchsorted(times, times, side="right")
    else:
        low = np.searchsorted(times, times, side="left")
        high = np.searchsorted(times, times + width, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    return (sums[high] - sums[low]) / (high - low)


data = pd.read_csv(CLEAN_PATH)
data["Time"] = pd.to_datetime(data["Time"])

parts = []
for series in sorted(data["series_id"].unique()):
    part = data[data["series_id"] == series].reset_index(drop=True)
    times = (part["Time"] - part["Time"].iloc[0]).dt.total_seconds().values

    cols = {}
    cols["ID"] = part["ID"].values
    cols["series_id"] = part["series_id"].values

    continuous = []
    for name in HEAVY:
        cols["z_" + name] = np.arcsinh(part[name].values)
        continuous.append("z_" + name)
    cols["WedgePosition"] = part["WedgePosition"].values
    continuous.append("WedgePosition")

    speed_sources = {"bit_speed": "BitPosition",
                     "block_speed": "TravelingBlockPosition",
                     "depth_speed": "Depth"}
    for new_name in speed_sources:
        values = part[speed_sources[new_name]].values
        right = side_mean(values, times, SPEED_WIDTH, "right")
        left = side_mean(values, times, SPEED_WIDTH, "left")
        cols[new_name] = (right - left) / SPEED_WIDTH
        continuous.append(new_name)

    cols["pump_on"] = (part["InletFlow"].values > PUMP_LIMIT).astype(float)
    cols["rotating"] = (part["Turn_rotor"].abs().values > ROTOR_LIMIT).astype(float)
    cols["on_bottom"] = (part["BitPosition"].values < BOTTOM_LIMIT).astype(float)
    cols["wedge_high"] = (part["WedgePosition"].values > WEDGE_LIMIT).astype(float)
    flags = ["pump_on", "rotating", "on_bottom", "wedge_high"]

    state_names = continuous + flags
    window_names = []
    change_names = []

    for name in state_names:
        values = cols[name]
        for width in MEAN_WIDTHS:
            mean, std = centered_stats(values, times, width)
            cols[name + "_m" + str(width)] = mean
            window_names.append(name + "_m" + str(width))
            if name in continuous and width in STD_WIDTHS:
                cols[name + "_s" + str(width)] = std
                window_names.append(name + "_s" + str(width))

    for name in continuous:
        values = cols[name]
        right = side_mean(values, times, CHANGE_WIDTH, "right")
        left = side_mean(values, times, CHANGE_WIDTH, "left")
        cols[name + "_d" + str(CHANGE_WIDTH)] = right - left
        change_names.append(name + "_d" + str(CHANGE_WIDTH))

    parts.append(pd.DataFrame(cols))

table = pd.concat(parts, ignore_index=True)

sets = {}
sets["state"] = state_names
sets["window"] = state_names + window_names
sets["full"] = state_names + window_names + change_names

for set_name in sets:
    names = sets[set_name]
    scaled = StandardScaler().fit_transform(table[names])
    result = pd.DataFrame(scaled, columns=names)
    result.insert(0, "series_id", table["series_id"].values)
    result.insert(0, "ID", table["ID"].values)
    file_name = OUT_FOLDER + "features_" + set_name + ".csv.gz"
    result.to_csv(file_name, index=False, float_format="%.4f", compression="gzip")
    print(file_name, "| строк:", len(result), "| признаков:", len(names),
          "| пропусков:", int(result.isna().sum().sum()),
          "| бесконечных:", int(np.isinf(result[names].values).sum()))
