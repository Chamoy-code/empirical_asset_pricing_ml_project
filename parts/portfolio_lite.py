
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_portfolio(test_df, predictions, target, model_name, results_path):
    df = test_df[['DATE', 'permno', target]].copy()
    df['pred'] = predictions
    df = df.dropna(subset=[target, 'pred'])
    
    def assign_decile(x):
        pct = x.rank(method='first', pct=True)
        return (pct * 10 - 1e-9).astype(int)
        
    df['decile'] = df.groupby('DATE')['pred'].transform(assign_decile)
    port_rets = df.groupby(['DATE', 'decile'])[target].mean().unstack()
    
    if 9 not in port_rets.columns or 0 not in port_rets.columns:
        return None 
        
    long_ret = port_rets[9]
    short_ret = port_rets[0]
    ls_ret = long_ret - short_ret
    market_ret = port_rets.mean(axis=1) # Market is the average of all deciles
    
    ann_factor = 1.0 if target == 'ret12' else 12.0
    metrics = {}
    
    # R2 OOS Calculation
    y_test = df[target].values
    y_pred = df['pred'].values
    r2_oos = 1 - np.sum((y_test - y_pred)**2) / np.sum(y_test**2)
    metrics['R2_OOS'] = r2_oos

    for name, series in zip(["Long", "Short", "Long-Short"], [long_ret, short_ret, ls_ret]):
        mean = series.mean() * ann_factor
        vol = series.std() * np.sqrt(ann_factor)
        metrics[name] = {"Return": mean, "Vol": vol, "Sharpe": mean/vol if vol > 0 else 0}
        
    # Plotting (Generates cumulative curves for any target)
    if True:
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative returns starting from 1.0
        cum_long = (1 + long_ret).cumprod()
        cum_short = (1 + short_ret).cumprod()
        cum_ls = (1 + ls_ret).cumprod()
        cum_market = (1 + market_ret).cumprod()
        
        plt.plot(cum_long.index, cum_long, label='Long (Top 10%)', color='green')
        plt.plot(cum_short.index, cum_short, label='Short (Bottom 10%)', color='red')
        plt.plot(cum_ls.index, cum_ls, label='Long-Short', color='blue', linewidth=2)
        plt.plot(cum_market.index, cum_market, label='Market (Eq. Wgt)', color='black', linestyle='--')
        
        plt.title(f'Cumulative Equity Curve - {model_name} ({"1-Month Horizon" if target=="ret" else "12-Month Horizon"})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Wealth ($1 Initial)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_dir = os.path.join(results_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        safe_name = model_name.replace(" ", "_").replace("=", "").replace(",", "")
        plt.savefig(os.path.join(plot_dir, f"{safe_name}_{target}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
    return metrics

def format_metrics(metrics):
    r2 = metrics.get('R2_OOS', 0)
    s = f"R2 OOS: {r2:.4%}\n"
    s += "Portfolio Metrics (Annualized):\n"
    for name in ["Long", "Short", "Long-Short"]:
        m = metrics[name]
        s += f"  {name:10}: Return={m['Return']:.4f}, Vol={m['Vol']:.4f}, Sharpe={m['Sharpe']:.4f}\n"
    return s
