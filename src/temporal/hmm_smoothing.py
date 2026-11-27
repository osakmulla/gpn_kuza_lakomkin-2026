import argparse
import pandas as pd
import numpy as np
from hmmlearn import hmm

def apply_hmm(series_labels, n_classes=9):
    if len(np.unique(series_labels)) < 2:
        return series_labels

    model = hmm.CategoricalHMM(n_components=n_classes, n_iter=100, random_state=42)
    
    try:
        model.fit(series_labels.reshape(-1, 1))
        hidden_states = model.predict(series_labels.reshape(-1, 1))
        
        mapped = np.zeros_like(hidden_states)
        for state in range(n_classes):
            mask = hidden_states == state
            if np.any(mask):
                mode_val = pd.Series(series_labels[mask]).mode()[0]
                mapped[mask] = mode_val
        return mapped
    except Exception as e:
        return series_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='candidate_D300s300.csv')
    parser.add_argument('--out', default='final_submission_hmm.csv')
    args = parser.parse_args()

    features = pd.read_csv('data/processed/features_state.csv.gz')
    candidate_file = 'submission/' + args.file
    
    df = pd.read_csv(candidate_file)

    label_col = None
    if 'operation' in df.columns:
        label_col = 'operation'
    elif 'class' in df.columns:
        label_col = 'class'
    elif 'cluster' in df.columns:
        label_col = 'cluster'
    else:
        label_col = df.columns[1]

    df = df.rename(columns={label_col: 'operation'})
    df = df.merge(features[['ID', 'series_id']], on='ID', how='left')
    
    smoothed_operations = []
    
    for series_id in df['series_id'].unique():
        mask = df['series_id'] == series_id
        series_ops = df.loc[mask, 'operation'].values
        smoothed = apply_hmm(series_ops)
        smoothed_operations.extend(smoothed)
        
    df['operation_smoothed'] = smoothed_operations
    final_sub = df[['ID', 'operation_smoothed']].rename(columns={'operation_smoothed': 'operation'})
    
    out_path = 'submission/' + args.out
    final_sub.to_csv(out_path, index=False)
    print(f"Cохранено в {out_path}")

if __name__ == '__main__':
    main()