"""
Full Argentina example script with direct data loading and complete downscaling.

This script combines the direct data loading approach from direct_argentina_example.py
with the full downscaling functionality from argentina_example.py. It loads the real
Argentina data directly from CSV files and runs the complete downscaling process.
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

def run_full_example(debug=True):
    """
    Run the full Argentina example with direct data loading and complete downscaling.
    
    This function loads the Argentina data directly from CSV files and runs the
    complete downscaling process.
    
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
    
    log(f"Starting full Argentina example (debug={debug})...")
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
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
    
    required_cols = {
        'argentina_luc': ['ns', 'lu.from', 'lu.to', 'Ts', 'value'],
        'argentina_df_xmat': ['ns', 'ks', 'value'],
        'argentina_df_lu_levels': ['ns', 'lu.from', 'value'],
        'argentina_FABLE': ['times', 'lu.from', 'lu.to', 'value']
    }
    
    for dataset, cols in required_cols.items():
        if dataset == 'argentina_luc':
            missing = [col for col in cols if col not in argentina_luc.columns]
            if missing:
                log(f"WARNING: Missing columns in {dataset}: {missing}")
                if 'Ts' in missing and 'ts' in argentina_luc.columns:
                    log("Renaming 'ts' to 'Ts'")
                    argentina_luc = argentina_luc.rename(columns={'ts': 'Ts'})
                    missing.remove('Ts')
                if missing:
                    raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_df_xmat':
            missing = [col for col in cols if col not in argentina_df['xmat'].columns]
            if missing:
                log(f"WARNING: Missing columns in {dataset}: {missing}")
                raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_df_lu_levels':
            missing = [col for col in cols if col not in argentina_df['lu_levels'].columns]
            if missing:
                log(f"WARNING: Missing columns in {dataset}: {missing}")
                raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_FABLE':
            missing = [col for col in cols if col not in argentina_FABLE.columns]
            if missing:
                log(f"WARNING: Missing columns in {dataset}: {missing}")
                if 'times' in missing and 'time' in argentina_FABLE.columns:
                    log("Renaming 'time' to 'times'")
                    argentina_FABLE = argentina_FABLE.rename(columns={'time': 'times'})
                    missing.remove('times')
                if missing:
                    raise ValueError(f"Missing columns in {dataset}: {missing}")
    
    log("\nEstimating coefficients using mnlogit...")
    betas = []
    for lu_from in argentina_luc['lu.from'].unique():
        if debug:
            log(f"Processing lu.from={lu_from}")
        
        mask = (argentina_luc['lu.from'] == lu_from) & ((argentina_luc['Ts'] == '2000') | (argentina_luc['Ts'] == 2000))
        if mask.sum() == 0:
            log(f"No data for lu.from={lu_from}, Ts=2000. Skipping.")
            continue
            
        Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
        
        X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
        X_data = X_data.reindex(Y_data.index)
        
        if X_data.isna().any().any() or Y_data.isna().any().any():
            log(f"Warning: Missing values in data for lu.from={lu_from}. Filling with zeros.")
            X_data = X_data.fillna(0)
            Y_data = Y_data.fillna(0)
        
        baseline = Y_data.columns.get_loc(lu_from) if lu_from in Y_data.columns else None
        
        if debug:
            log(f"  X_data shape: {X_data.shape}")
            log(f"  Y_data shape: {Y_data.shape}")
            log(f"  baseline: {baseline}")
        
        try:
            res = mnlogit(
                X=X_data.values,
                Y=Y_data.values,
                baseline=baseline,
                niter=10,  # Increased from 3 for better convergence
                nburn=5,   # Increased from 2 for better convergence
                jitter=1e-6  # Add small jitter for numerical stability
            )
            
            beta_mean = np.mean(res['postb'], axis=2)
            
            for k_idx, k in enumerate(X_data.columns):
                for lu_idx, lu_to in enumerate(Y_data.columns):
                    betas.append({
                        'ks': k,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': beta_mean[k_idx, lu_idx]
                    })
                    
            log(f"SUCCESS: mnlogit completed for {lu_from}")
            
        except Exception as e:
            log(f"ERROR in mnlogit for {lu_from}: {str(e)}")
    
    betas_df = pd.DataFrame(betas)
    
    if debug:
        log(f"\nbetas_df shape: {betas_df.shape}")
        log(f"betas_df columns: {betas_df.columns.tolist() if not betas_df.empty else []}")
    
    if len(betas_df) == 0:
        log("No betas were generated. Exiting.")
        return None
    
    ns_list = argentina_df['lu_levels']['ns'].unique()
    priors = pd.DataFrame({
        'ns': ns_list,
        'lu.from': example_LU_from,
        'lu.to': 'Forest',
        'value': np.random.uniform(0, 1, size=len(ns_list))
    })
    
    targets_2010 = argentina_FABLE[(argentina_FABLE['times'] == '2010') | (argentina_FABLE['times'] == 2010)]
    if len(targets_2010) == 0:
        log("Warning: No targets for 2010. Using first available time period.")
        first_time = argentina_FABLE['times'].iloc[0]
        targets_2010 = argentina_FABLE[argentina_FABLE['times'] == first_time]
    
    filtered_betas = betas_df[
        ~((betas_df['lu.from'] == example_LU_from) & (betas_df['lu.to'] == 'Forest'))
    ]
    
    log("\nRunning downscaling...")
    try:
        result = downscale(
            targets=targets_2010,
            start_areas=argentina_df['lu_levels'],
            xmat=argentina_df['xmat'],
            betas=filtered_betas,
            priors=priors
        )
        
        log("Downscaling complete!")
        if 'out_res' in result:
            log(f"Results shape: {result['out_res'].shape}")
            
            raster_path = data.get('argentina_raster')
            if raster_path:
                log(f"Using raster data from: {raster_path}")
            else:
                log("No raster data available for visualization.")
            
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
            os.makedirs(output_dir, exist_ok=True)
            
            plot_results(result, argentina_df['lu_levels'], raster_path, output_dir, example_LU_from)
        else:
            log("Warning: No results returned from downscaling.")
        
        return result
    
    except Exception as e:
        log(f"ERROR in downscaling: {str(e)}")
        return None

def plot_results(result, start_areas, raster_path=None, output_dir=None, example_LU_from="Cropland"):
    """
    Plot the results of the downscaling process.
    
    Parameters
    ----------
    result : dict
        The result of the downscaling process.
    start_areas : pd.DataFrame
        The starting areas.
    raster_path : str, optional
        Path to the raster file for visualization.
    output_dir : str, optional
        Directory to save the output plots.
    example_LU_from : str, default="Cropland"
        The land use change origin class used in the example, matching the R example.
    """
    log("Plotting results...")
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    out_res = result['out_res']
    
    start_totals = start_areas.groupby('lu.from')['value'].sum().reset_index()
    start_totals = start_totals.rename(columns={'lu.from': 'lu', 'value': 'start_value'})
    
    end_totals = out_res.groupby('lu.to')['value'].sum().reset_index()
    end_totals = end_totals.rename(columns={'lu.to': 'lu', 'value': 'end_value'})
    
    totals = pd.merge(start_totals, end_totals, on='lu', how='outer').fillna(0)
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(totals))
    width = 0.35
    
    plt.bar(x - width/2, totals['start_value'], width, label='Starting Areas')
    plt.bar(x + width/2, totals['end_value'], width, label='Downscaled Areas')
    
    plt.xlabel('Land-Use Class')
    plt.ylabel('Total Area')
    plt.title('Comparison of Starting and Downscaled Areas')
    plt.xticks(x, totals['lu'])
    plt.legend()
    
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'argentina_bar_chart.png')
    plt.savefig(bar_chart_path)
    plt.close()
    
    log(f"Bar chart saved to '{bar_chart_path}'")
    
    if raster_path:
        log(f"Creating raster visualizations using '{raster_path}'...")
        
        try:
            with rasterio.open(raster_path) as raster:
                all_plot = luc_plot(
                    res=result,
                    raster_file=raster,
                    figsize=(12, 10)
                )
                
                all_plot_path = os.path.join(output_dir, 'argentina_all_landuses_times.png')
                save_luc_plot(all_plot, all_plot_path)
                
                first_time = out_res['times'].unique()[0]
                cropland_to_forest = out_res[
                    (out_res['times'] == first_time) & 
                    (out_res['lu.from'] == example_LU_from) & 
                    (out_res['lu.to'] == 'Forest')
                ]
                
                if not cropland_to_forest.empty:
                    cropland_plot = luc_plot(
                        res=result,
                        raster_file=raster,
                        year=first_time,
                        lu='Forest',
                        cmap='Greens',
                        figsize=(8, 6)
                    )
                    
                    cropland_plot_path = os.path.join(output_dir, f'argentina_cropland_to_forest_{first_time}.png')
                    save_luc_plot(cropland_plot, cropland_plot_path)
                
                time_plot = luc_plot(
                    res=result,
                    raster_file=raster,
                    year=first_time,
                    cmap='viridis',
                    figsize=(12, 6)
                )
                
                time_plot_path = os.path.join(output_dir, f'argentina_all_landuses_{first_time}.png')
                save_luc_plot(time_plot, time_plot_path)
                
                log(f"Raster visualizations saved to '{output_dir}'")
        except Exception as e:
            log(f"Error creating raster visualizations: {str(e)}")
    else:
        log("No raster path provided. Skipping raster visualizations.")

def main():
    """Main function to run the full Argentina example."""
    result = run_full_example()
    
    if result is not None:
        log("Full Argentina example completed successfully!")
        return 0
    else:
        log("Full Argentina example failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
