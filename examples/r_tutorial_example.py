"""
R Tutorial Example: Downscaling land-use change projections in Argentina

This script implements the workflow from the original R tutorial at
https://tkrisztin.github.io/downscalr/articles/downscalr_tutorial.html
while using the direct data loading approach from full_argentina_example.py.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from downscalepy import downscale, mnlogit, luc_plot, save_luc_plot

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def load_argentina_data():
    """
    Load the Argentina data directly from CSV files or generate synthetic data.
    
    This function looks for data in multiple possible locations to handle
    different deployment environments. If real data is not found, it generates
    synthetic data for testing.
    """
    log("Loading Argentina data...")
    
    possible_paths = [
        "/storage/lopesas/downscalepy/downscalepy/data/converted",
        "/storage/lopesas/downscalepy/downscalepy/data",
        "/storage/lopesas/downscalepy/data/converted",
        "/storage/lopesas/downscalepy/data",
        os.path.join(os.path.dirname(__file__), "../downscalepy/data/converted"),
        os.path.join(os.path.dirname(__file__), "../downscalepy/data"),
        os.path.join(os.path.dirname(__file__), "../data/converted"),
        os.path.join(os.path.dirname(__file__), "../data")
    ]
    
    log("Checking the following paths:")
    for path in possible_paths:
        log(f"  - {path}")
    
    data_path = None
    raster_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            log(f"Path exists: {path}")
            try:
                files = os.listdir(path)
                log(f"Files in {path}: {', '.join(files)}")
            except Exception as e:
                log(f"Error listing files in {path}: {str(e)}")
                continue
                
            luc_file_exists = any(f in files for f in ["argentina_luc.csv", "argentina_df_luc.csv"])
            xmat_file_exists = any(f in files for f in ["argentina_xmat.csv", "argentina_df_xmat.csv"])
            lu_levels_file_exists = any(f in files for f in ["argentina_lu_levels.csv", "argentina_df_lu_levels.csv"])
            fable_file_exists = any(f in files for f in ["argentina_FABLE.csv", "argentina_df_FABLE.csv"])
            
            if luc_file_exists and xmat_file_exists and lu_levels_file_exists and fable_file_exists:
                data_path = path
                log(f"Found CSV data at: {data_path}")
                
                if "argentina_raster.tif" in files:
                    raster_path = os.path.join(path, "argentina_raster.tif")
                    log(f"Found raster data at: {raster_path}")
                
                break
    
    if data_path is None:
        log("Real data not found. Generating synthetic data for testing...")
        return generate_synthetic_data()
    
    try:
        luc_file = "argentina_df_luc.csv" if os.path.exists(os.path.join(data_path, "argentina_df_luc.csv")) else "argentina_luc.csv"
        log(f"Loading {luc_file} from {data_path}")
        argentina_luc = pd.read_csv(os.path.join(data_path, luc_file))
        
        xmat_file = "argentina_df_xmat.csv" if os.path.exists(os.path.join(data_path, "argentina_df_xmat.csv")) else "argentina_xmat.csv"
        log(f"Loading {xmat_file} from {data_path}")
        xmat = pd.read_csv(os.path.join(data_path, xmat_file))
        
        lu_levels_file = "argentina_df_lu_levels.csv" if os.path.exists(os.path.join(data_path, "argentina_df_lu_levels.csv")) else "argentina_lu_levels.csv"
        log(f"Loading {lu_levels_file} from {data_path}")
        lu_levels = pd.read_csv(os.path.join(data_path, lu_levels_file))
        
        fable_file = "argentina_df_FABLE.csv" if os.path.exists(os.path.join(data_path, "argentina_df_FABLE.csv")) else "argentina_FABLE.csv"
        log(f"Loading {fable_file} from {data_path}")
        fable = pd.read_csv(os.path.join(data_path, fable_file))
        
        data = {
            'argentina_luc': argentina_luc,
            'argentina_df': {
                'xmat': xmat,
                'lu_levels': lu_levels
            },
            'argentina_FABLE': fable,
            'argentina_raster': raster_path
        }
        
        log("Data loaded successfully!")
        return data
    
    except Exception as e:
        log(f"ERROR loading data: {str(e)}")
        log("Falling back to synthetic data...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic data for testing when real data is not available.
    """
    log("Generating synthetic data...")
    
    n_samples = 100
    lu_classes = ['Cropland', 'Forest', 'Pasture', 'Urban', 'OtherLand', 'Plantations']
    ks = [f'k{i}' for i in range(1, 26)]  # Match the real data dimensions
    ns = [f'ns{i}' for i in range(1, n_samples + 1)]
    times = [2000, 2010, 2020, 2030]
    
    luc_data = []
    for t in [2000]:  # Only for year 2000
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                for n in ns:
                    if lu_from != lu_to:
                        luc_data.append({
                            'Ts': t,
                            'lu.from': lu_from,
                            'lu.to': lu_to,
                            'ns': n,
                            'value': np.random.uniform(0, 1)
                        })
    
    argentina_luc = pd.DataFrame(luc_data)
    
    xmat_data = []
    for n in ns:
        for k in ks:
            xmat_data.append({
                'ns': n,
                'ks': k,
                'value': np.random.normal()
            })
    
    xmat = pd.DataFrame(xmat_data)
    
    lu_levels_data = []
    for n in ns:
        for lu in lu_classes:
            lu_levels_data.append({
                'ns': n,
                'lu.from': lu,
                'value': np.random.uniform(5, 10)
            })
    
    lu_levels = pd.DataFrame(lu_levels_data)
    
    fable_data = []
    for t in times:
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                if lu_from != lu_to:
                    fable_data.append({
                        'times': t,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': np.random.uniform(50, 100)
                    })
    
    fable = pd.DataFrame(fable_data)
    
    data = {
        'argentina_luc': argentina_luc,
        'argentina_df': {
            'xmat': xmat,
            'lu_levels': lu_levels
        },
        'argentina_FABLE': fable,
        'argentina_raster': None
    }
    
    log("Synthetic data generated successfully!")
    return data

def run_r_tutorial_example(debug=True):
    """
    Run the R tutorial example for downscaling land-use change projections in Argentina.
    
    This function follows the workflow from the original R tutorial at
    https://tkrisztin.github.io/downscalr/articles/downscalr_tutorial.html
    while using the direct data loading approach.
    
    Parameters
    ----------
    debug : bool, default=True
        Whether to print debug information.
        
    Returns
    -------
    dict
        The result of the downscaling process.
    """
    example_LU_from = "Cropland"
    example_time = 2010
    
    log(f"Starting R tutorial example (debug={debug})...")
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    log("=== Input data & Prior module - Preliminaries ===")
    data = load_argentina_data()
    if data is None:
        log("ERROR: Failed to load data. Exiting.")
        return None
    
    argentina_luc = data['argentina_luc']
    argentina_df = data['argentina_df']
    argentina_FABLE = data['argentina_FABLE']
    
    if debug:
        log("\nData summary:")
        log(f"argentina_luc shape: {argentina_luc.shape}")
        log(f"argentina_luc columns: {argentina_luc.columns.tolist()}")
        log(f"argentina_df['xmat'] shape: {argentina_df['xmat'].shape}")
        log(f"argentina_df['lu_levels'] shape: {argentina_df['lu_levels'].shape}")
        log(f"argentina_FABLE shape: {argentina_FABLE.shape}")
    
    log("\nPreparing data for econometric Multinomial logistic (MNL) regression...")
    
    mask = (argentina_luc['lu.from'] == example_LU_from) & ((argentina_luc['Ts'] == '2000') | (argentina_luc['Ts'] == 2000))
    if mask.sum() == 0:
        log(f"No data for lu.from={example_LU_from}, Ts=2000. Exiting.")
        return None
    
    Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
    
    X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
    X_data = X_data.reindex(Y_data.index)
    
    if X_data.isna().any().any() or Y_data.isna().any().any():
        log(f"Warning: Missing values in data for lu.from={example_LU_from}. Filling with zeros.")
        X_data = X_data.fillna(0)
        Y_data = Y_data.fillna(0)
    
    baseline = Y_data.columns.get_loc(example_LU_from) if example_LU_from in Y_data.columns else None
    
    if debug:
        log(f"Y_data shape: {Y_data.shape}")
        log(f"X_data shape: {X_data.shape}")
        log(f"baseline: {baseline}")
    
    log("\n=== Input data & Prior module - Estimate MNL model ===")
    log("Computing MNL model with settings from R tutorial (niter=100, nburn=50)...")
    
    try:
        results_MNL = mnlogit(
            X=X_data.values,
            Y=Y_data.values,
            baseline=baseline,
            niter=100,  # Matching R tutorial
            nburn=50,   # Matching R tutorial
            A0=1e4,     # Matching R tutorial
            jitter=1e-6  # Added for numerical stability
        )
        
        log("MNL model computed successfully!")
        
        log("\n=== Downscale module - Preliminaries ===")
        
        arg_targets_crop_2010 = argentina_FABLE[
            (argentina_FABLE['lu.from'] == example_LU_from) & 
            ((argentina_FABLE['times'] == str(example_time)) | (argentina_FABLE['times'] == example_time))
        ]
        
        if len(arg_targets_crop_2010) == 0:
            log(f"Warning: No targets for {example_time}. Using first available time period.")
            first_time = argentina_FABLE['times'].iloc[0]
            arg_targets_crop_2010 = argentina_FABLE[
                (argentina_FABLE['lu.from'] == example_LU_from) & 
                ((argentina_FABLE['times'] == first_time) | (argentina_FABLE['times'] == int(first_time) if first_time.isdigit() else False))
            ]
        
        X_long = X_data.reset_index()
        X_long = X_long.melt(id_vars='ns', var_name='ks', value_name='value')
        
        log("Computing mean of coefficient posterior draws...")
        
        postb_without_baseline = np.delete(results_MNL['postb'], baseline, axis=1)
        pred_coeff = np.mean(postb_without_baseline, axis=2)
        
        pred_coeff_long = []
        for k_idx, k in enumerate(X_data.columns):
            for lu_idx, lu_to in enumerate(Y_data.columns):
                if lu_to != example_LU_from:  # Skip baseline
                    adjusted_lu_idx = lu_idx if lu_idx < baseline else lu_idx - 1
                    pred_coeff_long.append({
                        'ks': k,
                        'lu.from': example_LU_from,
                        'lu.to': lu_to,
                        'value': pred_coeff[k_idx, adjusted_lu_idx]
                    })
        
        pred_coeff_long = pd.DataFrame(pred_coeff_long)
        
        arg_start_areas_crop = argentina_df['lu_levels'][
            argentina_df['lu_levels']['lu.from'] == example_LU_from
        ]
        
        log("\n=== Downscale module - Downscaling computation ===")
        log("Running downscaling with bias correction...")
        
        results_DS = downscale(
            targets=arg_targets_crop_2010,
            start_areas=arg_start_areas_crop,
            xmat=X_long,
            betas=pred_coeff_long
        )
        
        downscaled_LUC = results_DS['out_res']
        
        log("Downscaling complete!")
        log(f"Results shape: {downscaled_LUC.shape}")
        
        log("\n=== Visualize results ===")
        
        raster_path = data.get('argentina_raster')
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        if raster_path:
            log(f"Creating raster visualizations using '{raster_path}'...")
            
            try:
                with rasterio.open(raster_path) as raster:
                    luc_plot_result = luc_plot(
                        res=results_DS,
                        raster_file=raster,
                        figsize=(10, 8)
                    )
                    
                    luc_plot_path = os.path.join(output_dir, 'argentina_luc_plot.png')
                    save_luc_plot(luc_plot_result, luc_plot_path)
                    log(f"LUC plot saved to '{luc_plot_path}'")
            except Exception as e:
                log(f"Error creating raster visualizations: {str(e)}")
        else:
            log("No raster path provided. Creating alternative visualizations...")
            
            start_totals = arg_start_areas_crop.groupby('lu.from')['value'].sum().reset_index()
            start_totals = start_totals.rename(columns={'lu.from': 'lu', 'value': 'start_value'})
            
            end_totals = downscaled_LUC.groupby('lu.to')['value'].sum().reset_index()
            end_totals = end_totals.rename(columns={'lu.to': 'lu', 'value': 'end_value'})
            
            totals = pd.merge(start_totals, end_totals, on='lu', how='outer').fillna(0)
            
            plt.figure(figsize=(10, 6))
            
            x = np.arange(len(totals))
            width = 0.35
            
            plt.bar(x - width/2, totals['start_value'], width, label='Starting Areas')
            plt.bar(x + width/2, totals['end_value'], width, label='Downscaled Areas')
            
            plt.xlabel('Land-Use Class')
            plt.ylabel('Total Area')
            plt.title(f'Comparison of Starting and Downscaled Areas for {example_LU_from}')
            plt.xticks(x, totals['lu'])
            plt.legend()
            
            plt.tight_layout()
            bar_chart_path = os.path.join(output_dir, 'argentina_bar_chart.png')
            plt.savefig(bar_chart_path)
            plt.close()
            
            log(f"Bar chart saved to '{bar_chart_path}'")
        
        log("R tutorial example completed successfully!")
        return results_DS
    
    except Exception as e:
        log(f"ERROR: {str(e)}")
        return None

def main():
    """Main function to run the R tutorial example."""
    result = run_r_tutorial_example()
    
    if result is not None:
        log("R tutorial example completed successfully!")
        return 0
    else:
        log("R tutorial example failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
