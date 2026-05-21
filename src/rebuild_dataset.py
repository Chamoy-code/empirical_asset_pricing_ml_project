
import pandas as pd
import numpy as np
import os
import gc

def load_and_preprocess_macro(file_path):
    print("Loading macro data...")
    try:
        df = pd.read_excel(file_path, sheet_name='Monthly')
    except Exception as e:
        print(f"Error loading excel: {e}. Trying csv if available or checking path.")
        return None

    # Parse dates (yyyymm format expected)
    df['DATE'] = pd.to_datetime(df['yyyymm'].astype(str), format='%Y%m') + pd.offsets.MonthEnd(0)
    
    # Select and rename columns
    # We need: dp, ep, bm, ntis, tbl, tms, dfy, infl
    # dp = D12 / Index
    # ep = E12 / Index
    # tms = lty - tbl
    # dfy = BAA - AAA (default yield spread)
    
    # Check column names
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    
    # Calculate derived variables
    if 'Index' in df.columns and 'D12' in df.columns:
        df['dp'] = np.log(df['D12'].astype(float) / df['Index'].astype(float))
    if 'Index' in df.columns and 'E12' in df.columns:
        df['ep'] = np.log(df['E12'].astype(float) / df['Index'].astype(float))
        
    if 'lty' in df.columns and 'tbl' in df.columns:
        df['tms'] = df['lty'].astype(float) - df['tbl'].astype(float)
        
    # 'dfy' and 'infl' and 'ntis' and 'bm' (b/m) usually exist directly
    # 'b/m' might be named 'b/m'
    if 'b/m' in df.columns:
        df['bm'] = df['b/m']
        
    needed_cols = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'infl']
    
    # Filter for required columns
    final_cols = ['DATE'] + [c for c in needed_cols if c in df.columns]
    macro_df = df[final_cols].copy()
    
    # Rename macro columns to avoid collision with stock characteristics
    # e.g. stock 'ep' vs macro 'ep'
    rename_map = {c: f'macro_{c}' for c in macro_df.columns if c != 'DATE'}
    macro_df.rename(columns=rename_map, inplace=True)
    
    # Fill missing? Macro data is usually complete but check.
    macro_df.ffill(inplace=True)
    
    # Filter Date Range for Macro (matching stock data filtering later)
    # But macro is joined, so we keep all for now.
    
    print(f"Macro data loaded. Columns: {macro_df.columns.tolist()}")
    return macro_df

def load_stock_data_optimized(file_path):
    print("Loading stock data (optimized)...")
    # Columns to exclude from load if possible to save memory, but we basically need all chars.
    # We do NOT need 'sic2' as int, can handle it.
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Parse DATE
    df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d')
    
    # Filter 1957-03 to 2021-12 (Extended based on user request/data availability)
    start_date = '1957-03-01'
    end_date = '2021-12-31'
    df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)].reset_index(drop=True)
    
    return df

def process_sic_dummies(df):
    print("Generating SIC dummies...")
    if 'sic2' not in df.columns:
        print("Warning: sic2 column missing.")
        return df, []
        
    # Create dummies
    # sic2 is likely already 2 digits.
    dummies = pd.get_dummies(df['sic2'], prefix='sic', dtype=np.float32)
    dummy_cols = dummies.columns.tolist()
    
    # Concatenate - inefficient for memory to concat big DF.
    # Better to join?
    # Actually, we can return dummies and join later or processing chunks.
    # For now, let's append to df if memory allows.
    df = pd.concat([df, dummies], axis=1)
    
    return df, dummy_cols

def rank_and_normalize(df, char_cols):
    print("Ranking and Normalizing features...")
    # Group by DATE
    # Apply rank(pct=True) -> map to [-1, 1]
    # Formula: (Rank - 0.5) / Count * 2 - 1 ? 
    # Or just 2 * rank_pct - 1. 
    # rank_pct gives values in (0, 1]. 
    # min rank_pct is 1/N. max is 1.
    # 2 * (1/N) - 1 approx -1. 2*1 - 1 = 1.
    
    def normalize_step(group):
        # Fill missing with median first?
        # User requirement: "Missing data handling (cross-sectional median)"
        # "Toute caractéristique manquante... remplacée par la médiane"
        
        # We assume 'group' contains only char_cols
        # Impute
        median = group.median()
        group_imputed = group.fillna(median)
        
        # Rank
        # method='dense'? 'average'? Usually average.
        ranks = group_imputed.rank(pct=True, method='average')
        
        # Scale to [-1, 1]
        scaled = 2 * ranks - 1
        return scaled

    # Process per date to save memory?
    # Applying groupby transform on 30k*600 rows (18M) with 94 cols is heavy.
    # Optimization: Iterating over groups might be slower but safer for memory if we overwrite.
    
    # Let's try transform.
    # We only apply to char_cols.
    
    df[char_cols] = df.groupby('DATE')[char_cols].transform(normalize_step)
    
    return df

def rebuilder(datashare_path, macro_path, output_path):
    
    # 1. Load Macro
    macro_df = load_and_preprocess_macro(macro_path)
    if macro_df is None: return

    # 2. Load Stock & Filter Universe
    stock_df = load_stock_data_optimized(datashare_path)
    print(f"Stock data after filter: {stock_df.shape}")
    
    # Identify Characteristic Columns
    # Usually all columns except permno, DATE, sic2, ret...
    # We assume known characteristics list or exclude non-chars.
    non_char_cols = ['permno', 'DATE', 'sic2', 'ret', 'shrcd', 'exchcd', 'retvol', 'maxret'] # Adjust based on inspection
    # Note: 'retvol', 'maxret' ARE characteristics in Gu et al.
    # We need to exclude ONLY identifiers.
    identifiers = ['permno', 'DATE', 'sic2']
    # And maybe 'ret' if we have it? datashare might not have 'ret'. check later.
    
    char_cols = [c for c in stock_df.columns if c not in identifiers]
    print(f"Identified {len(char_cols)} characteristics.")
    
    # 3. Create Targets
    # Target 1: 1-month forward return (r_{t+1})
    # Assumption: 'mom1m' in current row is return t-1 -> t.
    # So to get return t -> t+1, we shift mom1m from next row (t+1) to current row (t).
    stock_df.sort_values(['permno', 'DATE'], inplace=True)
    stock_df['ret'] = stock_df.groupby('permno')['mom1m'].shift(-1)
    
    # Target 2: 12-month forward return (r_{t+1, t+12})
    # Calculation: Cumulative return of the next 12 months.
    # (1+r_1)(1+r_2)...(1+r_12) - 1.
    # Using log returns for summation: sum(ln(1+r))
    
    # helper for log return (handle potential <= -1 returns gracefully?)
    # usually returns are > -1.
    stock_df['log_ret_fwd'] = np.log1p(stock_df['ret'])
    
    # Rolling sum forward.
    # Standard rolling is backward. We can use backward rolling on shifted data or just shift result feature.
    # rolling(12) at index t+11 gives sum(t...t+11).
    # We want sum(t+1 ... t+12) at index t.
    # This corresponds to rolling(12) at index t+12 shifted back by 12?
    # No.
    # Let's align: 
    # 'log_ret_fwd' at t is r_{t+1}.
    # We want sum(r_{t+1}, ..., r_{t+12}).
    # This is a forward rolling sum of length 12 starting at t.
    # Pandas: rolling(12) sums [t-11, ... t].
    # Reversing the series, rolling, reversing back is a trick.
    # Or: use shift.
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=12)
    stock_df['cum_log_ret12'] = stock_df.groupby('permno')['log_ret_fwd'].rolling(window=indexer).sum().reset_index(level=0, drop=True)
    
    stock_df['ret12'] = np.expm1(stock_df['cum_log_ret12'])
    
    # Drop temp cols
    stock_df.drop(columns=['log_ret_fwd', 'cum_log_ret12'], inplace=True)
    
    print("Targets 'ret' (1m) and 'ret12' (12m) created.")
    
    # 4. Process SIC Dummies
    stock_df, dummy_cols = process_sic_dummies(stock_df)
    print(f"Created {len(dummy_cols)} SIC dummies.")
    
    # 5. Rank & Normalize Characteristics (ONLY chars, not dummies, not target)
    stock_df = rank_and_normalize(stock_df, char_cols)
    
    # ... (Previous steps 1-5 same) ...
    # 6. Macro Interactions & Saving (Chunked)
    print("Merging macro data...")
    merged_df = pd.merge(stock_df, macro_df, on='DATE', how='left')
    
    # Garbage collect to free detailed memory
    del stock_df
    gc.collect()

    macro_vars = [c for c in macro_df.columns if c != 'DATE']
    print(f"Creating interactions with {len(macro_vars)} macro variables...")
    
    # Create output directory for partitioned dataset or single file?
    # User might prefer single file, but we can write pieces and then they can be read.
    # Actually, parquet dataset (directory) is standard.
    # Let's write to a folder `data/rebuilt_dataset_parts/` and then maybe concat if needed, 
    # or just instruct user to read the folder (pd.read_parquet handles folders).
    
    dataset_dir = output_path.replace('.parquet', '_parts')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    unique_years = merged_df['DATE'].dt.year.unique()
    print(f"Processing {len(unique_years)} years...")
    
    for year in sorted(unique_years):
        print(f"Processing year {year}...")
        
        # Subset
        mask = merged_df['DATE'].dt.year == year
        subset = merged_df[mask].copy()
        
        if subset.empty: continue
        
        # Generate Interactions for subset
        interaction_data = {}
        for char in char_cols:
             for macro in macro_vars:
                col_name = f"{char}_x_{macro}"
                interaction_data[col_name] = subset[char] * subset[macro]
        
        inter_subset = pd.DataFrame(interaction_data, index=subset.index).astype(np.float32)
        
        # Combine
        # subset has [identifiers, chars, ret, dummies, macro_cols]
        # We need check if 'ret' in subset columns. Yes.
        
        final_subset = pd.concat([subset, inter_subset], axis=1)
        
        # Select columns
        # We need to construct final_cols list based on available columns to be safe
        # Or just drop macro cols if not needed.
        # Required: permno, DATE, ret, chars, interactions, dummies.
        # We exclude only intermediate macro columns?
        # Select cols
        # Ensure ret12 is included if created
        final_cols = ['permno', 'DATE', 'ret']
        if 'ret12' in final_subset.columns: # Check in final_subset, not merged_df
            final_cols.append('ret12')
            
        # Add characteristic columns, interaction columns, and dummy columns
        final_cols.extend(char_cols)
        final_cols.extend(inter_subset.columns.tolist()) # Use inter_subset.columns
        final_cols.extend(dummy_cols)
        
        # Filter final_subset to only include the desired columns
        final_subset = final_subset[final_cols].copy()
        
        # Drop rows with NaN target? 
        final_subset.dropna(subset=['ret'], inplace=True)
        
        if final_subset.empty: continue
        
        # Save part
        part_path = os.path.join(dataset_dir, f"part_{year}.parquet")
        final_subset.to_parquet(part_path, index=False)
        
        del subset, inter_subset, final_subset, interaction_data
        gc.collect()

    print(f"Saved chunked dataset to {dataset_dir}")
    print("You can read it using: pd.read_parquet(path)")
    
    # Optional: If user really wants single file and has disk space but not RAM for processing...
    # We can try to concat on disk? No, just leave as dataset.


if __name__ == "__main__":
    base_dir = r'c:\Users\charl\Desktop\Projet Professionel'
    datashare = os.path.join(base_dir, 'datashare.csv')
    macro = os.path.join(base_dir, 'PredictorData2022 (1).xlsx')
    output = os.path.join(base_dir, 'data', 'rebuilt_dataset.parquet')
    
    if not os.path.exists(os.path.join(base_dir, 'data')):
        os.makedirs(os.path.join(base_dir, 'data'))
        
    rebuilder(datashare, macro, output)
