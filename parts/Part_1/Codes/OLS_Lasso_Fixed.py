
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

def run_linear():
    print("Running OLS & Lasso (Fixed Split - Single Pass Streaming)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    
    files = get_sorted_parquet_files(data_path)
    if not files:
        print("Error: No data files found.")
        return

    targets = ['ret', 'ret12']
    # Initialize models for each target
    model_configs = [('OLS', None, 0), ('Lasso', 'l1', 1e-5)]
    models = {t: {name: SGDRegressor(penalty=p, alpha=a, random_state=42) 
                  for name, p, a in model_configs} for t in targets}
    
    test_data = {t: {'y': [], 'dates': [], 'permnos': [], 'preds': {name: [] for name, _, _ in model_configs}} for t in targets}
    feature_cols = []
    exclude = ['permno', 'DATE', 'ret', 'ret12', 'sic2', 'shrcd', 'exchcd', 'retvol', 'maxret']
    
    print("  Streaming data (Single Pass)...")
    for f in files:
        match = re.search(r'part_(\d+).parquet', f); year = int(match.group(1)) if match else 0
        try: df_chunk = pd.read_parquet(f)
        except: continue
        
        if not feature_cols:
            feature_cols = [c for c in df_chunk.columns if c not in exclude]
        
        # Process each target in the same chunk
        for t in targets:
            if t not in df_chunk.columns: continue
            valid_chunk = df_chunk.dropna(subset=[t])
            if valid_chunk.empty: continue
            
            X = np.nan_to_num(valid_chunk[feature_cols].values.astype(np.float32))
            y = valid_chunk[t].values.astype(np.float32)
            
            if year < 2005:
                for m in models[t].values(): m.partial_fit(X, y)
            else:
                test_data[t]['y'].extend(y)
                test_data[t]['dates'].extend(valid_chunk['DATE'].values)
                test_data[t]['permnos'].extend(valid_chunk['permno'].values)
                for name, m in models[t].items():
                    test_data[t]['preds'][name].extend(m.predict(X))
                    
    # Evaluation
    for t in targets:
        print(f"  Evaluating {t}...")
        if not test_data[t]['y']: continue
        test_df = pd.DataFrame({'DATE': test_data[t]['dates'], 'permno': test_data[t]['permnos'], t: test_data[t]['y']})
        for name in models[t].keys():
            metrics = evaluate_portfolio(test_df, np.array(test_data[t]['preds'][name]), t, f"{name} Fixed_{t}", results_base)
            report_path = os.path.join(results_base, 'datas', f'{name}_{t}_report.txt')
            with open(report_path, 'w') as f_out:
                f_out.write(f"{name} Results ({t})\n" + "="*20 + "\n")
                f_out.write(format_metrics(metrics))
                
                # Top 20 Features
                coefs = models[t][name].coef_
                top_idx = np.argsort(np.abs(coefs))[::-1][:20]
                f_out.write("\n\nTop 20 Features (by absolute coefficient):\n")
                for i, idx in enumerate(top_idx, 1):
                    f_out.write(f"{i:2d}. {feature_cols[idx]:<30}: {coefs[idx]:.6f}\n")
        
        metrics = evaluate_portfolio(test_df, preds, t, f"{name} Fixed_{t}", results_base)

    print("\nDone.")

if __name__ == "__main__":
    run_linear()
