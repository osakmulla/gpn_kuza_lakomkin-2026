import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

OUT_FOLDER = "submission/"
CHANGE_WIDTH = 60
MIN_GAP = 60
QUANTILE = 0.85
SEED = 0

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]


def side_mean(values, times, width, side):
    if side == "left":
        low = np.searchsorted(times, times - width, side="left")
        high = np.searchsorted(times, times, side="right")
    else:
        low = np.searchsorted(times, times, side="left")
        high = np.searchsorted(times, times + width, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    return (sums[high] - sums[low]) / (high - low)


def save(file_name, labels):
    result = pd.DataFrame({"ID": clean["ID"].values, "label": labels})
    result.to_csv(OUT_FOLDER + file_name, index=False)
    sizes = sorted(np.round(np.bincount(labels, minlength=9) / len(labels) * 100, 1).tolist(), reverse=True)
    print(file_name, "| строк:", len(result), "| классов:", result["label"].nunique(),
          "| пропусков:", int(result.isna().sum().sum()), "| размеры классов, %:", sizes)


clean = pd.read_csv("data/processed/clean.csv")
clean["Time"] = pd.to_datetime(clean["Time"])
series_ids = clean["series_id"].values

parts = []
for series in sorted(clean["series_id"].unique()):
    part = clean[clean["series_id"] == series]
    parts.append(part[signals].rolling(30, center=True, min_periods=1).mean())
old_features = StandardScaler().fit_transform(pd.concat(parts))
labels_old = KMeans(n_clusters=9, n_init=5, random_state=SEED).fit_predict(old_features)
save("candidate_A_old_features.csv", labels_old)

window_table = pd.read_csv("data/processed/features_window.csv.gz")
window_names = [c for c in window_table.columns if c != "ID" and c != "series_id"]
labels_window = KMeans(n_clusters=9, n_init=5, random_state=SEED).fit_predict(window_table[window_names].values)
save("candidate_B_window_points.csv", labels_window)

state_table = pd.read_csv("data/processed/features_state.csv.gz")
state_names = [c for c in state_table.columns if c != "ID" and c != "series_id"]
Z = state_table[state_names].values

scores = np.zeros(len(clean))
times_by_series = {}
for series in sorted(clean["series_id"].unique()):
    rows = np.where(series_ids == series)[0]
    times = (clean["Time"].iloc[rows] - clean["Time"].iloc[rows[0]]).dt.total_seconds().values
    times_by_series[series] = times
    total = np.zeros(len(rows))
    for column in range(Z.shape[1]):
        values = Z[rows, column]
        right = side_mean(values, times, CHANGE_WIDTH, "right")
        left = side_mean(values, times, CHANGE_WIDTH, "left")
        total = total + (right - left) ** 2
    scores[rows] = np.sqrt(total)

limit = np.quantile(scores, QUANTILE)
segment_id = np.zeros(len(clean), dtype=int)
next_id = 0
for series in sorted(clean["series_id"].unique()):
    rows = np.where(series_ids == series)[0]
    times = times_by_series[series]
    series_scores = pd.Series(scores[rows], index=clean["Time"].iloc[rows].values)
    local_max = series_scores.rolling(pd.Timedelta(seconds=2 * MIN_GAP + 1), center=True, min_periods=1).max().values
    candidates = np.where((scores[rows] >= local_max) & (scores[rows] > limit))[0]
    marks = np.zeros(len(rows), dtype=int)
    last_time = -1e9
    for index in candidates:
        if times[index] - last_time >= MIN_GAP:
            marks[index] = 1
            last_time = times[index]
    local_ids = np.cumsum(marks)
    segment_id[rows] = local_ids + next_id
    next_id = next_id + local_ids.max() + 1

frame = pd.DataFrame(Z, columns=state_names)
frame["segment"] = segment_id
segment_means = frame.groupby("segment")[state_names].mean().values
segment_sizes = np.bincount(segment_id).astype(float)
segment_labels = KMeans(n_clusters=9, n_init=10, random_state=SEED).fit_predict(segment_means, sample_weight=segment_sizes)
labels_segments = segment_labels[segment_id]
save("candidate_C_segments.csv", labels_segments)
print("сегментов:", len(segment_means))

print("согласие кандидатов между собой (ARI):")
print("A-B:", round(adjusted_rand_score(labels_old, labels_window), 3))
print("A-C:", round(adjusted_rand_score(labels_old, labels_segments), 3))
print("B-C:", round(adjusted_rand_score(labels_window, labels_segments), 3))
