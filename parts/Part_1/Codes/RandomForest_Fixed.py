
import pandas as pd
import numpy as np
np.random.seed(42)
import os
import glob
import re
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from portfolio_lite import evaluate_portfolio, format_metrics

def get_sorted_parquet_files(data_path):
    files = glob.glob(os.path.join(data_path, "part_*.parquet"))
    def get_year(f):
        match = re.search(r'part_(\d+).parquet', f)
        return int(match.group(1)) if match else 0
    return sorted(files, key=get_year)

def run_rf():
    print("Running Random Forest (Fixed Split - Single Pass Data Collection)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    
    files = get_sorted_parquet_files(data_path)
    targets = ['ret', 'ret12']
    
    train_data = {t: {'X': [], 'y': []} for t in targets}
    test_data = {t: {'X': [], 'df': []} for t in targets}
    feature_cols, exclude = [], ['permno', 'DATE', 'ret', 'ret12', 'sic2', 'shrcd', 'exchcd', 'retvol', 'maxret']
    
    print("  Collecting data (Single Pass)...")
    for f in files:
        match = re.search(r'part_(\d+).parquet', f); year = int(match.group(1)) if match else 0
        try: 
            df_chunk = pd.read_parquet(f)
            if year > 0: print(f"    Processing {year}...")
        except: continue
        if not feature_cols: feature_cols = [c for c in df_chunk.columns if c not in exclude]
        
        for t in targets:
            if t not in df_chunk.columns: continue
            valid_chunk = df_chunk.dropna(subset=[t])
            if valid_chunk.empty: continue
            
            if year < 2005:
                df_sample = valid_chunk.sample(frac=0.1, random_state=42)
                train_data[t]['X'].append(np.nan_to_num(df_sample[feature_cols].values.astype(np.float32)))
                train_data[t]['y'].append(df_sample[t].values.astype(np.float32))
            else:
                test_data[t]['X'].append(np.nan_to_num(valid_chunk[feature_cols].values.astype(np.float32)))
                test_data[t]['df'].append(valid_chunk[['DATE', 'permno', t]])

    for t in targets:
        print(f"  Training & Evaluating {t}...")
        if not train_data[t]['X']: continue
        X_train = np.vstack(train_data[t]['X']); y_train = np.concatenate(train_data[t]['y'])
        X_test = np.vstack(test_data[t]['X']); test_df = pd.concat(test_data[t]['df'])
        
        model = RandomForestRegressor(n_estimators=100, max_depth=3, max_features='sqrt', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        metrics = evaluate_portfolio(test_df, model.predict(X_test), t, f"RF Fixed_{t}", results_base)
        
        metrics = evaluate_portfolio(test_df, model.predict(X_test), t, f"RF Fixed_{t}", results_base)
        
        with open(os.path.join(results_base, 'datas', f'RF_{t}_report.txt'), 'w') as f_out:
            f_out.write(f"RF Results ({t})\n" + "="*20 + "\n" + format_metrics(metrics))
            
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            f_out.write("\n\nTop 20 Features (by importance):\n")
            for i, idx in enumerate(top_idx, 1):
                f_out.write(f"{i:2d}. {feature_cols[idx]:<30}: {importances[idx]:.6f}\n")
    print("\nDone.")

if __name__ == "__main__":
    run_rf()
