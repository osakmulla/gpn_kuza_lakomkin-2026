import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 4]

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

clean = pd.read_csv("data/processed/clean.csv")
series_ids = clean["series_id"].values

parts = []
for series in sorted(clean["series_id"].unique()):
    part = clean[clean["series_id"] == series]
    parts.append(part[signals].rolling(30, center=True, min_periods=1).mean())
old_features = pd.concat(parts)

sets = {}
sets["old_mean30"] = StandardScaler().fit_transform(old_features)
for set_name in ["state", "window", "full"]:
    table = pd.read_csv("data/processed/features_" + set_name + ".csv.gz")
    names = [c for c in table.columns if c != "ID" and c != "series_id"]
    sets[set_name] = table[names].values

state_code = ((clean["InletFlow"] > 1).astype(int) * 1
              + (clean["Turn_rotor"].abs() > 5).astype(int) * 2
              + (clean["BitPosition"] < 3).astype(int) * 4
              + (clean["WedgePosition"] > 0.5).astype(int) * 8).values

big = series_ids <= 1
small = series_ids >= 2

print("среднее по", len(SEEDS), "сидам (в скобках разброс)")
print("набор | мин. кластер % | устойчивость сида | между сериями | переключений на 1000 | согласие с состояниями")

for set_name in sets:
    X = sets[set_name]

    all_labels = []
    across_list = []
    shares_list = []
    switches_list = []
    state_list = []

    for seed in SEEDS:
        labels = KMeans(n_clusters=9, n_init=2, random_state=seed).fit_predict(X)
        all_labels.append(labels)

        model_big = KMeans(n_clusters=9, n_init=2, random_state=seed).fit(X[big])
        predicted = model_big.predict(X[small])
        independent = KMeans(n_clusters=9, n_init=2, random_state=seed).fit_predict(X[small])
        across_list.append(adjusted_rand_score(predicted, independent))

        shares_list.append((np.bincount(labels, minlength=9) / len(labels) * 100).min())

        changes = 0
        for series in sorted(clean["series_id"].unique()):
            series_labels = labels[series_ids == series]
            changes = changes + int((series_labels[1:] != series_labels[:-1]).sum())
        switches_list.append(changes / len(labels) * 1000)

        state_list.append(adjusted_rand_score(state_code, labels))

    seed_list = []
    for i in range(1, len(SEEDS)):
        seed_list.append(adjusted_rand_score(all_labels[0], all_labels[i]))

    print(set_name, "|",
          round(np.mean(shares_list), 2), "|",
          round(np.mean(seed_list), 3), "(", round(np.std(seed_list), 3), ") |",
          round(np.mean(across_list), 3), "(", round(np.std(across_list), 3), ") |",
          round(np.mean(switches_list), 1), "(", round(np.std(switches_list), 1), ") |",
          round(np.mean(state_list), 3), "(", round(np.std(state_list), 3), ")")
