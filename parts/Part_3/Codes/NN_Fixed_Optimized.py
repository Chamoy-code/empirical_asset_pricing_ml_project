
import pandas as pd
import numpy as np
import os
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from portfolio_lite import evaluate_portfolio, format_metrics

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FinanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation='relu'):
        super(NeuralNet, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU() if activation == 'relu' else nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

def get_sorted_parquet_files(data_path):
    files = glob.glob(os.path.join(data_path, "part_*.parquet"))
    def get_year(f):
        match = re.search(r'part_(\d+).parquet', f); return int(match.group(1)) if match else 0
    return sorted(files, key=get_year)

def get_best_param(csv_path, target, param_name, default_value):
    if not os.path.exists(csv_path): return default_value
    try:
        df = pd.read_csv(csv_path)
        val = df[(df['Model'] == 'NN') & (df['Target'] == target) & (df['Best_Parameter'] == param_name)]['Best_Value'].iloc[-1]
        return val
    except: return default_value

def run_nn_optimized():
    print(f"Running NN Optimized Fixed (Memory Optimized) on {device}...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    csv_path = os.path.join(results_base, 'datas', 'best_parameters.csv')
    
    files = get_sorted_parquet_files(data_path)
    arch_map = {'NN1': [32], 'NN2': [32, 16], 'NN3': [32, 16, 8]}
    targets = ['ret', 'ret12']
    
    raw_data = {t: {'train_X': [], 'train_y': [], 'test_X': [], 'test_df': []} for t in targets}
    feature_cols, exclude = [], ['permno', 'DATE', 'ret', 'ret12', 'sic2', 'shrcd', 'exchcd', 'retvol', 'maxret']
    
    print("  Collecting data (Streaming)...")
    for f in files:
        match = re.search(r'part_(\d+).parquet', f); year = int(match.group(1)) if match else 0
        print(f"    Processing {year}...", end='\r')
        try: df_chunk = pd.read_parquet(f)
        except: continue
        if not feature_cols: feature_cols = [c for c in df_chunk.columns if c not in exclude]
        
        for t in targets:
            if t not in df_chunk.columns: continue
            valid_chunk = df_chunk.dropna(subset=[t])
            if valid_chunk.empty: continue
            X = np.nan_to_num(valid_chunk[feature_cols].values.astype(np.float32))
            y = valid_chunk[t].values.astype(np.float32)
            
            if year < 2005:
                idx = valid_chunk.sample(frac=0.2, random_state=42).index
                raw_data[t]['train_X'].append(X[valid_chunk.index.get_indexer(idx)])
                raw_data[t]['train_y'].append(y[valid_chunk.index.get_indexer(idx)])
            else:
                raw_data[t]['test_X'].append(X); raw_data[t]['test_df'].append(valid_chunk[['DATE', 'permno', t]])

    for t in targets:
        best_arch = get_best_param(csv_path, t, 'architecture', 'NN2')
        best_act = get_best_param(csv_path, t, 'activation', 'relu')
        layers = arch_map.get(best_arch, [32, 16])
        print(f"\n  Training Optimized Model {best_arch} ({best_act}) for {t}...")
        
        X_train, y_train = np.vstack(raw_data[t]['train_X']), np.concatenate(raw_data[t]['train_y'])
        X_test, test_df = np.vstack(raw_data[t]['test_X']), pd.concat(raw_data[t]['test_df'])
        
        raw_data[t] = None; gc.collect()
        
        scaler = StandardScaler(copy=False)
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        
        loader = DataLoader(FinanceDataset(X_train, y_train), batch_size=8192, shuffle=True)
        model = NeuralNet(len(feature_cols), layers, best_act).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        for epoch in range(25):
            model.train()
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device); optimizer.zero_grad(); criterion(model(bx), by).backward(); optimizer.step()

        model.eval(); test_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 8192):
                bx = torch.from_numpy(X_test[i:i+8192]).to(device).float()
                test_preds.append(model(bx).cpu().numpy())
        test_preds = np.vstack(test_preds).flatten()
        
        metrics = evaluate_portfolio(test_df, test_preds, t, f"NN Optimized Fixed_{t}", results_base)

        curve_dir = os.path.abspath(os.path.join(results_base, '..', '..', 'Résultats', 'datas', 'curves'))
        os.makedirs(curve_dir, exist_ok=True)
        def assign_decile(x): return (x.rank(method='first', pct=True) * 10 - 1e-9).astype(int)
        tdfc = test_df.copy(); tdfc['pred'] = test_preds
        tdfc['decile'] = tdfc.groupby('DATE')['pred'].transform(assign_decile)
        port_rets = tdfc.groupby(['DATE', 'decile'])[t].mean().unstack()
        curve_df = pd.DataFrame({'NN3_Optimized': port_rets[9], 'Market': port_rets.mean(axis=1)})
        curve_df.to_csv(os.path.join(curve_dir, f"Q3_NN_Optimized_{t}_curves.csv"))

        with open(os.path.join(results_base, 'datas', f'NN_{t}_optimized_fixed_report.txt'), 'w') as f:
            f.write(f"NN Optimized Fixed Results ({t})\nArch: {best_arch} ({best_act})\n" + "="*40 + "\n" + format_metrics(metrics))
        
        del X_train, X_test, y_train, test_df; gc.collect()

    print("\nDone.")

if __name__ == "__main__":
    run_nn_optimized()

if __name__ == "__main__":
    run_nn_optimized()
