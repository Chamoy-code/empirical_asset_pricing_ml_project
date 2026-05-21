
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

def get_best_param(csv_path, model_name, target, param_name, default_value):
    if not os.path.exists(csv_path): return default_value
    try:
        df = pd.read_csv(csv_path)
        val = df[(df['Model'] == model_name) & (df['Target'] == target) & (df['Best_Parameter'] == param_name)]['Best_Value'].iloc[-1]
        return float(val)
    except: return default_value

def run_rf_optimized():
    print("Running Random Forest (Optimized Fixed - Single Pass)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    csv_path = os.path.join(results_base, 'datas', 'best_parameters.csv')
    
    files = get_sorted_parquet_files(data_path)
    targets = ['ret', 'ret12']
    best_depths = {t: int(get_best_param(csv_path, 'RandomForest', t, 'max_depth', 3)) for t in targets}
    best_nests = {t: int(get_best_param(csv_path, 'RandomForest', t, 'n_estimators', 100)) for t in targets}
    print(f"  Best params: depths={best_depths}, nests={best_nests}")
    
    data = {t: {'train_X': [], 'train_y': [], 'test_X': [], 'test_df': []} for t in targets}
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
            X = np.nan_to_num(valid_chunk[feature_cols].values.astype(np.float32))
            if year < 2005:
                idx = valid_chunk.sample(frac=0.1, random_state=42).index
                data[t]['train_X'].append(X[valid_chunk.index.get_indexer(idx)])
                data[t]['train_y'].append(valid_chunk.loc[idx, t].values.astype(np.float32))
            else:
                data[t]['test_X'].append(X); data[t]['test_df'].append(valid_chunk[['DATE', 'permno', t]])

    for t in targets:
        print(f"  Training & Evaluating {t}...")
        X_train, y_train = np.vstack(data[t]['train_X']), np.concatenate(data[t]['train_y'])
        X_test, test_df = np.vstack(data[t]['test_X']), pd.concat(data[t]['test_df'])
        
        model = RandomForestRegressor(n_estimators=best_nests[t], max_depth=best_depths[t], max_features='sqrt', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        metrics = evaluate_portfolio(test_df, model.predict(X_test), t, f"RF_Opt_{t}", results_base)
        with open(os.path.join(results_base, 'datas', f'RF_{t}_optimized_fixed_report.txt'), 'w') as f:
            f.write(f"RF Optimized Results ({t})\nParams: depth={best_depths[t]}, nest={best_nests[t]}\n" + "="*40 + "\n" + format_metrics(metrics))
            
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            f.write("\n\nTop 20 Features (by importance):\n")
            for i, idx in enumerate(top_idx, 1):
                f.write(f"{i:2d}. {feature_cols[idx]:<30}: {importances[idx]:.6f}\n")
    print("\nDone.")

if __name__ == "__main__":
    run_rf_optimized()
