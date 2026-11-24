import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data/raw/hakhathon_data.csv")
data["Time"] = pd.to_datetime(data["Time"], format="%d.%m.%Y %H:%M:%S")
data = data.sort_values("Time")
data = data.reset_index(drop=True)

signals = ["Depth", "HL", "TravelingBlockPosition", "ManifoldPressure", "ROP",
           "InletFlow", "Load", "BitPosition", "WedgePosition", "Turn_rotor"]

data[signals] = data[signals].interpolate()
data[signals] = data[signals].bfill()
data = data.iloc[:60000]

print("окно | вариант | silhouette | устойчивость | переключений")

for w in [12, 30, 60, 120, 240]:
    means = data[signals].rolling(w, center=True, min_periods=1).mean()
    means.columns = [s + "_mean" for s in signals]

    variants = {}
    variants["сигналы+mean"] = pd.concat([data[signals], means], axis=1)
    variants["только mean"] = means

    for name in variants:
        X = variants[name]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model_a = KMeans(n_clusters=9, n_init=3, random_state=0)
        labels_a = model_a.fit_predict(X)
        model_b = KMeans(n_clusters=9, n_init=3, random_state=1)
        labels_b = model_b.fit_predict(X)

        sil = silhouette_score(X, labels_a, sample_size=5000, random_state=0)
        stable = adjusted_rand_score(labels_a, labels_b)

        switches = 0
        for i in range(1, len(labels_a)):
            if labels_a[i] != labels_a[i - 1]:
                switches = switches + 1

        print(w, "|", name, "|", round(sil, 3), "|", round(stable, 3), "|", switches)
