import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SEEDS = [0, 1, 2, 3, 4]
CHANGE_WIDTH = 60
MIN_GAP = 60
QUANTILES = [0.97, 0.94, 0.85]


def side_mean(values, times, width, side):
    if side == "left":
        low = np.searchsorted(times, times - width, side="left")
        high = np.searchsorted(times, times, side="right")
    else:
        low = np.searchsorted(times, times, side="left")
        high = np.searchsorted(times, times + width, side="right")
    sums = np.concatenate([[0.0], np.cumsum(values)])
    return (sums[high] - sums[low]) / (high - low)


clean = pd.read_csv("data/processed/clean.csv")
clean["Time"] = pd.to_datetime(clean["Time"])
series_ids = clean["series_id"].values

table = pd.read_csv("data/processed/features_state.csv.gz")
names = [c for c in table.columns if c != "ID" and c != "series_id"]
Z = table[names].values

state_code = ((clean["InletFlow"] > 1).astype(int) * 1
              + (clean["Turn_rotor"].abs() > 5).astype(int) * 2
              + (clean["BitPosition"] < 3).astype(int) * 4
              + (clean["WedgePosition"] > 0.5).astype(int) * 8).values

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


def find_segments(quantile):
    limit = np.quantile(scores, quantile)
    segment_id = np.zeros(len(clean), dtype=int)
    next_id = 0
    for series in sorted(clean["series_id"].unique()):
        rows = np.where(series_ids == series)[0]
        times = times_by_series[series]
        series_scores = pd.Series(scores[rows], index=clean["Time"].iloc[rows].values)
        local_max = series_scores.rolling(pd.Timedelta(seconds=2 * MIN_GAP + 1), center=True, min_periods=1).max().values
        candidates = np.where((scores[rows] >= local_max) & (scores[rows] > limit))[0]
        cuts = []
        last_time = -1e9
        for index in candidates:
            if times[index] - last_time >= MIN_GAP:
                cuts.append(index)
                last_time = times[index]
        marks = np.zeros(len(rows), dtype=int)
        for index in cuts:
            marks[index] = 1
        local_ids = np.cumsum(marks)
        segment_id[rows] = local_ids + next_id
        next_id = next_id + local_ids.max() + 1
    return segment_id


print("квантиль порога | сегментов | сид (разброс) | между сериями (разброс) | переключений на 1000 | мин. кластер % | согласие с состояниями")

for quantile in QUANTILES:
    segment_id = find_segments(quantile)
    frame = pd.DataFrame(Z, columns=names)
    frame["segment"] = segment_id
    frame["length"] = 1.0
    means = frame.groupby("segment")[names].mean()
    sizes = frame.groupby("segment")["length"].sum()
    series_of_segment = pd.Series(series_ids).groupby(segment_id).first()
    segment_x = means.values
    segment_weight = sizes.values
    segment_series = series_of_segment.values
    point_segment = segment_id

    all_labels = []
    across_list = []
    shares_list = []
    switches_list = []
    state_list = []

    for seed in SEEDS:
        model = KMeans(n_clusters=9, n_init=5, random_state=seed)
        segment_labels = model.fit_predict(segment_x, sample_weight=segment_weight)
        labels = segment_labels[point_segment]
        all_labels.append(labels)

        big = segment_series <= 1
        small = segment_series >= 2
        model_big = KMeans(n_clusters=9, n_init=5, random_state=seed)
        model_big.fit(segment_x[big], sample_weight=segment_weight[big])
        predicted = model_big.predict(segment_x[small])
        model_small = KMeans(n_clusters=9, n_init=5, random_state=seed)
        independent = model_small.fit_predict(segment_x[small], sample_weight=segment_weight[small])
        small_points = series_ids >= 2
        across_list.append(adjusted_rand_score(predicted[point_segment[small_points] - point_segment[small_points].min()],
                                               independent[point_segment[small_points] - point_segment[small_points].min()]))

        shares_list.append((np.bincount(labels, minlength=9) / len(labels) * 100).min())

        changes = 0
        for series in sorted(clean["series_id"].unique()):
            series_labels = labels[series_ids == series]
            changes = changes + int((series_labels[1:] != series_labels[:-1]).sum())
        switches_list.append(changes / len(labels) * 1000)
        state_list.append(adjusted_rand_score(state_code, labels))
    if quantile in [0.97, 0.85]:
    result = clean[["ID", "Time", "series_id"]].copy()
    result["segment_id"] = segment_id

    for i, seed in enumerate(SEEDS):
        result[f"cluster_seed_{seed}"] = all_labels[i]

    result.to_csv(
        f"data/processed/segments_{quantile}.csv",
        index=False
    )
    seed_list = []
    for i in range(1, len(SEEDS)):
        seed_list.append(adjusted_rand_score(all_labels[0], all_labels[i]))

    print(quantile, "|", len(segment_x), "|",
          round(np.mean(seed_list), 3), "(", round(np.std(seed_list), 3), ") |",
          round(np.mean(across_list), 3), "(", round(np.std(across_list), 3), ") |",
          round(np.mean(switches_list), 1), "|",
          round(np.mean(shares_list), 2), "|",
          round(np.mean(state_list), 3), "(", round(np.std(state_list), 3), ")")
