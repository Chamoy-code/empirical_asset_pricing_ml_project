
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

def run_rf_tuning():
    print("Running Random Forest Tuning (3-Way Split - Single Pass)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    csv_path = os.path.join(results_base, 'datas', 'best_parameters.csv')
    
    files = get_sorted_parquet_files(data_path)
    TRAIN_END, VAL_END = 1990, 2005
    depths, estimators, targets = [1, 3, 6], [100, 300], ['ret', 'ret12']
    
    data = {t: {'train_X': [], 'train_y': [], 'val_X': [], 'val_y': [], 'test_X': [], 'test_df': []} for t in targets}
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
            
            if year < TRAIN_END:
                idx = valid_chunk.sample(frac=0.05, random_state=42).index
                data[t]['train_X'].append(X[valid_chunk.index.get_indexer(idx)])
                data[t]['train_y'].append(valid_chunk.loc[idx, t].values.astype(np.float32))
            elif year < VAL_END:
                data[t]['val_X'].append(X)
                data[t]['val_y'].append(valid_chunk[t].values.astype(np.float32))
            else:
                data[t]['test_X'].append(X)
                data[t]['test_df'].append(valid_chunk[['DATE', 'permno', t]])
        
    for t in targets:
        print(f"  Tuning {t}...")
        X_train, y_train = np.vstack(data[t]['train_X']), np.concatenate(data[t]['train_y'])
        X_val, y_val = np.vstack(data[t]['val_X']), np.concatenate(data[t]['val_y'])
        X_test, test_df = np.vstack(data[t]['test_X']), pd.concat(data[t]['test_df'])
        
        best_p, best_mse, winner = None, float('inf'), None
        for d in depths:
            for nest in estimators:
                print(f"    Testing: nest={nest}, depth={d}...")
                model = RandomForestRegressor(n_estimators=nest, max_depth=d, max_features='sqrt', n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
                mse = np.mean((y_val - model.predict(X_val))**2)
                if mse < best_mse: 
                    best_mse, best_p, winner = mse, {'n_estimators': nest, 'max_depth': d}, model
        
        print(f"    Target {t} -> Best Params: {best_p}")
        pd.DataFrame([
            {'Model': 'RandomForest', 'Target': t, 'Best_Parameter': 'max_depth', 'Best_Value': best_p['max_depth']},
            {'Model': 'RandomForest', 'Target': t, 'Best_Parameter': 'n_estimators', 'Best_Value': best_p['n_estimators']}
        ]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        metrics = evaluate_portfolio(test_df, winner.predict(X_test), t, f"RF 3-Way nest={best_p['n_estimators']} depth={best_p['max_depth']}_{t}", results_base)
        with open(os.path.join(results_base, 'datas', f'RF_{t}_tuning_report.txt'), 'w') as f_out:
            f_out.write(f"RF Results ({t})\nWinner Params: {best_p}\n" + "="*30 + "\n" + format_metrics(metrics))
            
            importances = winner.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            f_out.write("\n\nTop 20 Features (by importance):\n")
            for i, idx in enumerate(top_idx, 1):
                f_out.write(f"{i:2d}. {feature_cols[idx]:<30}: {importances[idx]:.6f}\n")
    print("\nDone.")

if __name__ == "__main__":
    run_rf_tuning()
