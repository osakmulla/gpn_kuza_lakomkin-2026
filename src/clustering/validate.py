import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

RANDOM_STATE = 42
N_CLUSTERS = 9
EXCLUDE_COLS = {'ID', 'series_id'}


def get_X(df):
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return np.nan_to_num(df[cols].to_numpy(dtype=np.float64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='window', choices=['state', 'window', 'full'])
    parser.add_argument('--model', default='gmm', choices=['kmeans', 'gmm'])
    args = parser.parse_args()

    df = pd.read_csv('data/processed/features_' + args.features + '.csv.gz')
    X = get_X(df)

    train_mask = df['series_id'].isin([0, 1])
    test_mask = df['series_id'].isin([2, 3])

    X_train, X_test = X[train_mask], X[test_mask]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    if args.model == 'kmeans':
        m_train = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        m_test = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    else:
        m_train = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE, n_init=3)
        m_test = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE, n_init=3)

    m_train.fit(X_train_s)
    pred_test_from_train = m_train.predict(X_test_s)

    pred_test_own = m_test.fit_predict(X_test_s)

    ari = adjusted_rand_score(pred_test_from_train, pred_test_own)

    with open('validation_report.txt', 'a') as f:
        f.write(f'{args.features} {args.model} ARI: {ari:.4f}\n')


if __name__ == '__main__':
    main()
    