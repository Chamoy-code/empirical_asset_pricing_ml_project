
import pandas as pd
import numpy as np
np.random.seed(42)
import os
import glob
import re
from sklearn.linear_model import SGDRegressor
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from portfolio_lite import evaluate_portfolio, format_metrics

def get_sorted_parquet_files(data_path):
    files = glob.glob(os.path.join(data_path, "part_*.parquet"))
    def get_year(f):
        match = re.search(r'part_(\d+).parquet', f)
        return int(match.group(1)) if match else 0
    return sorted(files, key=get_year)

def run_linear_tuning():
    print("Running OLS & Lasso Tuning (3-Way Split - Single Pass Streaming)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    csv_path = os.path.join(results_base, 'datas', 'best_parameters.csv')
    if os.path.exists(csv_path): os.remove(csv_path) # Clear old results

    files = get_sorted_parquet_files(data_path)
    TRAIN_END = 1990
    VAL_END = 2005
    alphas = [1e-7, 1e-6, 1e-5, 1e-4]
    targets = ['ret', 'ret12']
    
    # Init Models
    ols = {t: SGDRegressor(penalty=None, random_state=42) for t in targets}
    lasso_grid = {t: {a: SGDRegressor(penalty='l1', alpha=a, random_state=42) for a in alphas} for t in targets}
    
    val_errors = {t: {a: [] for a in alphas} for t in targets}
    test_data = {t: {'y': [], 'dates': [], 'permnos': [], 'preds_ols': [], 'preds_lasso': {a: [] for a in alphas}} for t in targets}
    feature_cols, exclude = [], ['permno', 'DATE', 'ret', 'ret12', 'sic2', 'shrcd', 'exchcd', 'retvol', 'maxret']
    
    print("  Streaming data (Single Pass)...")
    for f in files:
        match = re.search(r'part_(\d+).parquet', f); year = int(match.group(1)) if match else 0
        try: df_chunk = pd.read_parquet(f)
        except: continue
        if not feature_cols: feature_cols = [c for c in df_chunk.columns if c not in exclude]
        
        for t in targets:
            if t not in df_chunk.columns: continue
            valid_chunk = df_chunk.dropna(subset=[t])
            if valid_chunk.empty: continue
            X = np.nan_to_num(valid_chunk[feature_cols].values.astype(np.float32))
            y = valid_chunk[t].values.astype(np.float32)
            
            if year < TRAIN_END:
                ols[t].partial_fit(X, y)
                for m in lasso_grid[t].values(): m.partial_fit(X, y)
            elif year < VAL_END:
                for a, m in lasso_grid[t].items():
                    val_errors[t][a].extend((y - m.predict(X))**2)
            else:
                test_data[t]['y'].extend(y); test_data[t]['dates'].extend(valid_chunk['DATE'].values); test_data[t]['permnos'].extend(valid_chunk['permno'].values)
                test_data[t]['preds_ols'].extend(ols[t].predict(X))
                for a, m in lasso_grid[t].items():
                    test_data[t]['preds_lasso'][a].extend(m.predict(X))
        
    for t in targets:
        print(f"  Tuning {t}...")
        best_alpha, best_mse = None, float('inf')
        for a in alphas:
            mse = np.mean(val_errors[t][a])
            if mse < best_mse: best_mse = mse; best_alpha = a
        print(f"    Target {t} -> Best Alpha: {best_alpha}")
        
        # Save Parameter
        pd.DataFrame([{'Model': 'Lasso', 'Target': t, 'Best_Parameter': 'alpha', 'Best_Value': best_alpha}]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        # Portfolios
        test_df = pd.DataFrame({'DATE': test_data[t]['dates'], 'permno': test_data[t]['permnos'], t: test_data[t]['y']})
        for name, preds, model_obj in [
            ('OLS 3-Way', test_data[t]['preds_ols'], ols[t]), 
            (f'Lasso 3-Way alpha={best_alpha}', test_data[t]['preds_lasso'][best_alpha], lasso_grid[t][best_alpha])
        ]:
            metrics = evaluate_portfolio(test_df, np.array(preds), t, f"{name}_{t}", results_base)
            report_path = os.path.join(results_base, 'datas', f'{name}_{t}_tuning_report.txt')
            with open(report_path, 'w') as f_out:
                f_out.write(f"{name} Results ({t})\n" + "="*30 + "\n")
                f_out.write(format_metrics(metrics))
                
                # Top 20 Features
                coefs = model_obj.coef_
                top_idx = np.argsort(np.abs(coefs))[::-1][:20]
                f_out.write("\n\nTop 20 Features (by absolute coefficient):\n")
                for i, idx in enumerate(top_idx, 1):
                    f_out.write(f"{i:2d}. {feature_cols[idx]:<30}: {coefs[idx]:.6f}\n")
    print("\nDone.")

if __name__ == "__main__":
    run_linear_tuning()
