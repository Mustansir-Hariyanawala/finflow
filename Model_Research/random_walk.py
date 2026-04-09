import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def calculate_metrics(y_true, y_pred, y_lag):
    # Masking zeros to avoid division by zero errors in MAPE/MdAPE/MAAPE
    mask = y_true != 0
    yt = y_true[mask]
    yp = y_pred[mask]
    
    if len(yt) == 0: return {m: np.nan for m in ['MAE','MSE','RMSE','MAPE','MAAPE','MdAPE','DA']}

    # 1-3: Absolute Errors
    mae = np.mean(np.abs(yt - yp))
    mse = np.mean((yt - yp)**2)
    rmse = np.sqrt(mse)
    
    # 4-6: Percentage Errors
    abs_pe = np.abs((yt - yp) / yt)
    mape = np.mean(abs_pe) * 100
    maape = np.mean(np.arctan(abs_pe)) # Mean Absolute Arch Percentage Error
    mdape = np.median(abs_pe) * 100
    
    # 7: Directional Accuracy (DA)
    # Correct if sign of actual change == sign of predicted change
    actual_dir = np.sign(y_true - y_lag)
    pred_dir = np.sign(y_pred - y_lag)
    da = np.mean(actual_dir == pred_dir)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MAAPE': maape, 'MdAPE': mdape, 'DA': da}

# Path Setup
raw_data_path = r'C:\mustansir\House Of Code\My_projects\finetuning_IPD\Data\train\RAW'
files = glob.glob(os.path.join(raw_data_path, "*.csv"))
target_col = 'Income_Total Revenue' # Example target

results = []

for f in tqdm(files, desc="Evaluating Random Walk"):
    try:
        df = pd.read_csv(f)
        # Assuming the first column is the Date/Index
        df = df.sort_values(df.columns[0]).reset_index(drop=True)
        
        if target_col not in df.columns: continue
            
        vals = df[target_col].values
        
        # Random Walk Logic
        y_true = vals[1:]      # Current values (t)
        y_pred = vals[:-1]     # Forecasts (value at t-1)
        y_lag = vals[:-1]      # Used for Directional Accuracy
        
        metrics = calculate_metrics(y_true, y_pred, y_lag)
        metrics['File'] = os.path.basename(f)
        results.append(metrics)
    except Exception as e:
        print(f"Skipping {f} due to error: {e}")

# Save and View Summary
results_df = pd.DataFrame(results)
results_df.to_csv("RW_Evaluation_Results.csv", index=False)
print("\nGlobal Mean Metrics Across All Files:")
print(results_df.drop(columns='File').mean())