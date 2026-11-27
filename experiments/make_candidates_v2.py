import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLEAN_PATH = "data/processed/clean.csv"
OUT_FOLDER = "submission/"
VARIANTS = {"D300s300": [300, 300], "D300s900": [300, 900]}
MIN_RUN = 30
PUMP_LIMIT = 1
ROTOR_LIMIT = 5
BOTTOM_LIMIT = 3
WEDGE_LIMIT = 0.5
CLASSES = 9


def window_stats(values, times, width):
    half = width / 2
    low = np.searchsorted(times, times - half, side="left")
    high = np.searchsorted(times, times + half, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    squares = np.concatenate([[0.0], np.cumsum(values * values)])
    mean = (sums[high] - sums[low]) / (high - low)
    variance = (squares[high] - squares[low]) / (high - low) - mean * mean
    variance = np.maximum(variance, 0)
    return mean, np.sqrt(variance)


def side_mean(values, times, width, side):
    if side == "left":
        low = np.searchsorted(times, times - width, side="left")
        high = np.searchsorted(times, times, side="right")
    else:
        low = np.searchsorted(times, times, side="left")
        high = np.searchsorted(times, times + width, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    return (sums[high] - sums[low]) / (high - low)


def make_features(part, times, width):
    result = {}
    pump = (part["InletFlow"].values > PUMP_LIMIT).astype(float)
    rotor = (part["Turn_rotor"].abs().values > ROTOR_LIMIT).astype(float)
    bottom = (part["BitPosition"].values < BOTTOM_LIMIT).astype(float)
    wedge = (part["WedgePosition"].values > WEDGE_LIMIT).astype(float)
    wedge_change = np.abs(np.diff(wedge, prepend=wedge[0]))
    bit = part["BitPosition"].values
    depth = part["Depth"].values

    result["pump"] = window_stats(pump, times, width)[0]
    result["rotor"] = window_stats(rotor, times, width)[0]
    result["bottom"] = window_stats(bottom, times, width)[0]
    result["bit_level"] = window_stats(np.arcsinh(bit), times, width)[0]
    bit_move = (side_mean(bit, times, width, "right") - side_mean(bit, times, width, "left")) / width * 60
    result["bit_trend"] = np.arcsinh(bit_move)
    result["block_std"] = window_stats(part["TravelingBlockPosition"].values, times, width)[1]
    result["wedge_rate"] = window_stats(wedge_change, times, width)[0]
    result["load_mean"], result["load_std"] = window_stats(np.arcsinh(part["Load"].values), times, width)
    result["hl"] = window_stats(part["HL"].values, times, width)[0]
    result["pressure"] = window_stats(np.arcsinh(part["ManifoldPressure"].values), times, width)[0]
    if part["ROP"].std() > 0.001:
        result["rop"] = window_stats(np.arcsinh(part["ROP"].values), times, width)[0]
    else:
        result["rop"] = np.zeros(len(part))
    depth_move = (side_mean(depth, times, width, "right") - side_mean(depth, times, width, "left")) / width * 3600
    result["depth_speed"] = depth_move
    return pd.DataFrame(result)


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


def remove_short_runs(labels):
    result = labels.copy()
    start = 0
    while start < len(result):
        end = start
        while end < len(result) and result[end] == result[start]:
            end = end + 1
        if end - start < MIN_RUN and start > 0:
            result[start:end] = result[start - 1]
        start = end
    return result


data = pd.read_csv(CLEAN_PATH)
data["Time"] = pd.to_datetime(data["Time"])
series_list = sorted(data["series_id"].unique())

all_times = np.zeros(len(data))
for series in series_list:
    rows = np.where(data["series_id"].values == series)[0]
    first = data["Time"].iloc[rows[0]]
    all_times[rows] = (data["Time"].iloc[rows] - first).dt.total_seconds().values

for variant in VARIANTS:
    width = VARIANTS[variant][0]
    smooth_width = VARIANTS[variant][1]

    parts = []
    for series in series_list:
        rows = np.where(data["series_id"].values == series)[0]
        part = data.iloc[rows].reset_index(drop=True)
        parts.append(make_features(part, all_times[rows], width))
    table = pd.concat(parts, ignore_index=True)

    scaled = StandardScaler().fit_transform(table)
    labels = KMeans(n_clusters=CLASSES, n_init=10, random_state=0).fit_predict(scaled)

    final = labels.copy()
    for series in series_list:
        rows = np.where(data["series_id"].values == series)[0]
        smoothed = smooth_labels(labels[rows], all_times[rows], smooth_width)
        final[rows] = remove_short_runs(smoothed)

    result = pd.DataFrame({"ID": data["ID"].values, "label": final})
    result.to_csv(OUT_FOLDER + "candidate_" + variant + ".csv", index=False)

    changes = 0
    for series in series_list:
        rows = np.where(data["series_id"].values == series)[0]
        series_labels = final[rows]
        changes = changes + int((series_labels[1:] != series_labels[:-1]).sum())
    shares = np.bincount(final, minlength=CLASSES) / len(final) * 100
    print(variant, "| интервалов:", changes + len(series_list), "| классов:", len(np.unique(final)),
          "| доли %:", np.round(shares, 1).tolist())
