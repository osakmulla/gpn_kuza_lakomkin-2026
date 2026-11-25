import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42
N_CLUSTERS = 9
EXCLUDE_COLS = {'ID', 'series_id'}


def load_features(name):
    return pd.read_csv('data/processed/features_' + name + '.csv.gz')


def get_feature_matrix(df):
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit_kmeans(X):
    Xs = StandardScaler().fit_transform(X)
    return KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10, max_iter=500).fit_predict(Xs)


def fit_gmm(X):
    Xs = StandardScaler().fit_transform(X)
    return GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE,
                           covariance_type='full', n_init=3).fit_predict(Xs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='state', choices=['state', 'window', 'full'])
    parser.add_argument('--model', default='kmeans', choices=['kmeans', 'gmm'])
    parser.add_argument('--out', default='submission.csv')
    args = parser.parse_args()

    df = load_features(args.features)
    X = get_feature_matrix(df)
    labels = fit_kmeans(X) if args.model == 'kmeans' else fit_gmm(X)

    pd.DataFrame({'ID': df['ID'].astype(int), 'class': labels.astype(int)}).to_csv(args.out, index=False)
    pd.Series(labels).value_counts().sort_index().to_csv('clusters_report.txt', header=False)


if __name__ == '__main__':
    main()