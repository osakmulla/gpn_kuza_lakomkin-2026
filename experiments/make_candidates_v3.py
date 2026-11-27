import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CLEAN_PATH = "data/processed/clean.csv"
STATE_PATH = "data/processed/features_state.csv.gz"
OUT_FOLDER = "submission/"
PUMP_LIMIT = 1
ROTOR_LIMIT = 5
BOTTOM_LIMIT = 3
WEDGE_LIMIT = 0.5
MOVE_LIMIT = 0.05
SPEED_WIDTH = 10
SMOOTH_WIDTH = 30
CLASSES = 9


def side_mean(values, times, width, side):
    if side == "left":
        low = np.searchsorted(times, times - width, side="left")
        high = np.searchsorted(times, times, side="right")
    else:
        low = np.searchsorted(times, times, side="left")
        high = np.searchsorted(times, times + width, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    return (sums[high] - sums[low]) / (high - low)


def smooth_labels(labels, times, width):
    half = width / 2
    low = np.searchsorted(times, times - half, side="left")
    high = np.searchsorted(times, times + half, side="right")
    best_share = np.full(len(labels), -1.0)
    best_label = np.zeros(len(labels), dtype=int)
    for k in range(CLASSES):
        sums = np.concatenate([[0.0], np.cumsum((labels == k).astype(float))])
        share = (sums[high] - sums[low]) / (high - low)
        better = share > best_share
        best_label[better] = k
        best_share[better] = share[better]
    return best_label


def save(name, labels):
    result = pd.DataFrame({"ID": data["ID"].values, "label": labels})
    result.to_csv(OUT_FOLDER + "candidate_" + name + ".csv", index=False)
    changes = 0
    for series in series_list:
        rows = np.where(data["series_id"].values == series)[0]
        series_labels = labels[rows]
        changes = changes + int((series_labels[1:] != series_labels[:-1]).sum())
    shares = np.bincount(labels, minlength=CLASSES) / len(labels) * 100
    print(name, "| интервалов:", changes + len(series_list), "| классов:", len(np.unique(labels)),
          "| доли %:", np.round(shares, 1).tolist())


data = pd.read_csv(CLEAN_PATH)
data["Time"] = pd.to_datetime(data["Time"])
series_list = sorted(data["series_id"].unique())

all_times = np.zeros(len(data))
block_speed = np.zeros(len(data))
for series in series_list:
    rows = np.where(data["series_id"].values == series)[0]
    first = data["Time"].iloc[rows[0]]
    times = (data["Time"].iloc[rows] - first).dt.total_seconds().values
    all_times[rows] = times
    block = data["TravelingBlockPosition"].values[rows]
    right = side_mean(block, times, SPEED_WIDTH, "right")
    left = side_mean(block, times, SPEED_WIDTH, "left")
    block_speed[rows] = (right - left) / SPEED_WIDTH

pump = (data["InletFlow"].values > PUMP_LIMIT).astype(int)
rotor = (data["Turn_rotor"].abs().values > ROTOR_LIMIT).astype(int)
bottom = (data["BitPosition"].values < BOTTOM_LIMIT).astype(int)
wedge = (data["WedgePosition"].values > WEDGE_LIMIT).astype(int)
moving = (np.abs(block_speed) > MOVE_LIMIT).astype(int)
code = pump * 16 + rotor * 8 + bottom * 4 + wedge * 2 + moving

counts = np.bincount(code, minlength=32)
top_codes = np.argsort(-counts)[:CLASSES]

code_to_label = np.zeros(32, dtype=int)
for one_code in range(32):
    best_distance = 99
    best_label = 0
    for label in range(CLASSES):
        differences = bin(one_code ^ top_codes[label]).count("1")
        if differences < best_distance:
            best_distance = differences
            best_label = label
    code_to_label[one_code] = best_label

rule_labels = code_to_label[code]
smoothed = rule_labels.copy()
for series in series_list:
    rows = np.where(data["series_id"].values == series)[0]
    smoothed[rows] = smooth_labels(rule_labels[rows], all_times[rows], SMOOTH_WIDTH)
save("R_rules", rule_labels)
save("R30_rules_smooth", smoothed)

table = pd.read_csv(STATE_PATH)
names = [c for c in table.columns if c != "ID" and c != "series_id"]
state_labels = KMeans(n_clusters=CLASSES, n_init=10, random_state=0).fit_predict(table[names].values)
save("E_state_points", state_labels)
