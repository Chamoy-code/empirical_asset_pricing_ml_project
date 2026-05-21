
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
    torch.backends.cudnn.benchmark = False
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

def run_nn_tuning():
    print(f"Running NN Tuning (3-Way Split - Memory Optimized) on {device}...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.abspath(os.path.join(base_dir, '..', '..', 'data', 'rebuilt_dataset_parts'))
    results_base = os.path.join(base_dir, 'Résultats')
    csv_path = os.path.join(results_base, 'datas', 'best_parameters.csv')
    
    files = get_sorted_parquet_files(data_path)
    TRAIN_END, VAL_END = 1990, 2005
    targets = ['ret', 'ret12']
    
    architectures = {'NN1': [32], 'NN2': [32, 16], 'NN3': [32, 16, 8]}
    activations = ['relu', 'tanh']
    
    raw_data = {t: {'train_X': [], 'train_y': [], 'val_X': [], 'val_y': [], 'test_X': [], 'test_df': []} for t in targets}
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
            
            if year < TRAIN_END:
                idx = valid_chunk.sample(frac=0.1, random_state=42).index
                raw_data[t]['train_X'].append(X[valid_chunk.index.get_indexer(idx)])
                raw_data[t]['train_y'].append(y[valid_chunk.index.get_indexer(idx)])
            elif year < VAL_END:
                idx = valid_chunk.sample(frac=0.2, random_state=42).index
                raw_data[t]['val_X'].append(X[valid_chunk.index.get_indexer(idx)])
                raw_data[t]['val_y'].append(y[valid_chunk.index.get_indexer(idx)])
            else:
                raw_data[t]['test_X'].append(X)
                raw_data[t]['test_df'].append(valid_chunk[['DATE', 'permno', t]])

    for t in targets:
        print(f"\n  Tuning {t}...")
        X_train = np.vstack(raw_data[t]['train_X']); y_train = np.concatenate(raw_data[t]['train_y'])
        X_val = np.vstack(raw_data[t]['val_X']); y_val = np.concatenate(raw_data[t]['val_y'])
        X_test = np.vstack(raw_data[t]['test_X']); test_df = pd.concat(raw_data[t]['test_df'])
        
        raw_data[t] = None; gc.collect()
        
        scaler = StandardScaler(copy=False)
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        
        train_loader = DataLoader(FinanceDataset(X_train, y_train), batch_size=8192, shuffle=True)
        val_loader = DataLoader(FinanceDataset(X_val, y_val), batch_size=8192, shuffle=False)
        
        best_p, best_mse, winner_state = None, float('inf'), None
        
        for arch_name, layers in architectures.items():
            for act in activations:
                print(f"    Testing {arch_name} ({act})...")
                model = NeuralNet(len(feature_cols), layers, act).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                criterion = nn.MSELoss()
                
                best_val_loss, patience, wait = float('inf'), 3, 0
                for epoch in range(50):
                    model.train()
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad(); criterion(model(bx), by).backward(); optimizer.step()
                    
                    model.eval(); val_loss = 0
                    with torch.no_grad():
                        for bx, by in val_loader:
                            bx, by = bx.to(device), by.to(device)
                            val_loss += criterion(model(bx), by).item() * len(bx)
                    val_loss /= len(X_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss; wait = 0
                        if val_loss < best_mse:
                            best_mse = val_loss
                            best_p = {'arch': arch_name, 'activation': act, 'layers': layers}
                            winner_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        wait += 1
                        if wait >= patience: break
        
        print(f"    Best: {best_p['arch']} ({best_p['activation']})")
        pd.DataFrame([
            {'Model': 'NN', 'Target': t, 'Best_Parameter': 'architecture', 'Best_Value': best_p['arch']},
            {'Model': 'NN', 'Target': t, 'Best_Parameter': 'activation', 'Best_Value': best_p['activation']}
        ]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        
        winner_model = NeuralNet(len(feature_cols), best_p['layers'], best_p['activation']).to(device)
        winner_model.load_state_dict(winner_state); winner_model.eval()
        
        test_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 8192):
                bx = torch.from_numpy(X_test[i:i+8192]).to(device).float()
                test_preds.append(winner_model(bx).cpu().numpy())
        test_preds = np.vstack(test_preds).flatten()
        
        metrics = evaluate_portfolio(test_df, test_preds, t, f"NN 3-Way {best_p['arch']}_{t}", results_base)
        with open(os.path.join(results_base, 'datas', f'NN_{t}_tuning_report.txt'), 'w') as f:
            f.write(f"NN Tuning Results ({t})\nBest Arch: {best_p['arch']} ({best_p['activation']})\n" + "="*40 + "\n" + format_metrics(metrics))
        
        del X_train, X_val, X_test, y_train, y_val, test_df
        gc.collect()

    print("\nDone.")

if __name__ == "__main__":
    run_nn_tuning()
