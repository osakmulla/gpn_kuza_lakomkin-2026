import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

RANDOM_STATE = 42
N_CLUSTERS = 9
EXCLUDE_COLS = {'ID', 'series_id'}


def main():
    df = pd.read_csv('data/processed/features_window.csv.gz')
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = np.nan_to_num(df[cols].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10, max_iter=500).fit_predict(Xs)

    submission = pd.DataFrame({'ID': df['ID'].astype(int), 'class': labels.astype(int)})
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()