"""
Direct Argentina example script that bypasses package installation issues.

This script directly loads the data files from the known path and runs the
Argentina example without relying on the installed package. This helps
diagnose issues with package installation or execution.

Run this script directly:
python examples/direct_argentina_example.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from downscalepy.core.downscale import downscale
from downscalepy.models.mnlogit import mnlogit
from downscalepy.visualization.luc_plot import luc_plot

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    sys.stderr.write(full_message + "\n")
    sys.stderr.flush()

def direct_load_argentina_data():
    """
    Directly load the Argentina data from the known path.
    
    This function bypasses the normal loading mechanism and directly loads
    the data files from the known path.
    
    Returns
    -------
    dict
        A dictionary containing the Argentina data.
    """
    log("DIRECT LOADING: Starting direct data loading...")
    
    data_path = "/storage/lopesas/downscalepy/downscalepy/data/converted"
    log(f"DIRECT LOADING: Using path: {data_path}")
    
    result = {}
    
    try:
        file_path = os.path.join(data_path, 'argentina_luc.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        argentina_luc = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_luc.csv, shape: {argentina_luc.shape}")
        result['argentina_luc'] = argentina_luc
    except Exception as e:
        log(f"ERROR: Failed to load argentina_luc.csv: {e}")
        raise
    
    try:
        file_path = os.path.join(data_path, 'argentina_FABLE.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        argentina_FABLE = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_FABLE.csv, shape: {argentina_FABLE.shape}")
        result['argentina_FABLE'] = argentina_FABLE
    except Exception as e:
        log(f"ERROR: Failed to load argentina_FABLE.csv: {e}")
        raise
    
    try:
        file_path = os.path.join(data_path, 'argentina_df_xmat.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        xmat = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_df_xmat.csv, shape: {xmat.shape}")
        
        file_path = os.path.join(data_path, 'argentina_df_lu_levels.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        lu_levels = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_df_lu_levels.csv, shape: {lu_levels.shape}")
        
        file_path = os.path.join(data_path, 'argentina_df_restrictions.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        restrictions = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_df_restrictions.csv, shape: {restrictions.shape}")
        
        file_path = os.path.join(data_path, 'argentina_df_pop_data.csv')
        log(f"DIRECT LOADING: Loading {file_path}")
        pop_data = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded argentina_df_pop_data.csv, shape: {pop_data.shape}")
        
        result['argentina_df'] = {
            'xmat': xmat,
            'lu_levels': lu_levels,
            'restrictions': restrictions,
            'pop_data': pop_data
        }
        log("SUCCESS: Created argentina_df dictionary with all components")
    except Exception as e:
        log(f"ERROR: Failed to load argentina_df components: {e}")
        raise
    
    try:
        raster_path = os.path.join(data_path, 'argentina_raster.tif')
        log(f"DIRECT LOADING: Checking for raster at {raster_path}")
        
        if os.path.exists(raster_path):
            log(f"SUCCESS: Found raster file at {raster_path}")
            result['argentina_raster'] = raster_path
        else:
            log(f"WARNING: Raster file not found at {raster_path}")
            import rasterio
            from rasterio.transform import from_origin
            
            temp_dir = os.path.join(os.getcwd(), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            synthetic_raster_path = os.path.join(temp_dir, 'synthetic_raster.tif')
            
            nrows, ncols = 100, 100
            data = np.random.rand(nrows, ncols).astype(np.float32)
            
            transform = from_origin(0, 0, 1, 1)
            
            with rasterio.open(
                synthetic_raster_path,
                'w',
                driver='GTiff',
                height=nrows,
                width=ncols,
                count=1,
                dtype=data.dtype,
                crs='+proj=latlong',
                transform=transform,
            ) as dst:
                dst.write(data, 1)
            
            log(f"SUCCESS: Created synthetic raster at {synthetic_raster_path}")
            result['argentina_raster'] = synthetic_raster_path
    except Exception as e:
        log(f"ERROR: Exception while handling raster: {e}")
        result['argentina_raster'] = None
    
    log("DIRECT LOADING SUMMARY:")
    for key in result:
        if key == 'argentina_df':
            log(f"- {key}: Dictionary with {len(result[key])} components")
            for subkey, value in result[key].items():
                log(f"  - {subkey}: Shape {value.shape}")
        elif key == 'argentina_raster':
            log(f"- {key}: {result[key]}")
        else:
            log(f"- {key}: Shape {result[key].shape}")
    
    return result

def run_example():
    """
    Run the Argentina example using directly loaded data.
    """
    log("Starting Argentina example with direct data loading...")
    
    data = direct_load_argentina_data()
    
    argentina_luc = data['argentina_luc']
    argentina_df = data['argentina_df']
    argentina_FABLE = data['argentina_FABLE']
    
    log("Preparing coefficients using mnlogit...")
    
    np.random.seed(42)
    
    betas = []
    
    lu_from = 'Cropland'
    
    mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == 2000)
    
    if mask.sum() == 0:
        log(f"WARNING: No data found for lu.from={lu_from} and Ts=2000")
        mask = (argentina_luc['lu.from'] == lu_from)
        if mask.sum() == 0:
            log(f"ERROR: No data found for lu.from={lu_from}")
            return
        else:
            time_step = argentina_luc.loc[mask, 'Ts'].unique()[0]
            log(f"Using time step {time_step} instead")
            mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == time_step)
    
    ns_values = argentina_luc.loc[mask, 'ns'].unique()
    
    lu_to_values = argentina_luc.loc[mask, 'lu.to'].unique()
    
    log(f"Found {len(ns_values)} spatial units and {len(lu_to_values)} destination land use classes")
    
    Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
    
    Y_data = Y_data.fillna(0)
    
    X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
    
    X_data = X_data.reindex(Y_data.index)
    
    X_data = X_data.fillna(0)
    
    log(f"X data shape: {X_data.shape}")
    log(f"Y data shape: {Y_data.shape}")
    
    baseline = None
    if lu_from in Y_data.columns:
        baseline = Y_data.columns.get_loc(lu_from)
        log(f"Using {lu_from} as baseline (index {baseline})")
    else:
        log(f"Baseline {lu_from} not found in Y data columns")
    
    log("Running mnlogit...")
    res = mnlogit(
        X=X_data.values,
        Y=Y_data.values,
        baseline=baseline,
        niter=10,  # Use a small number for demonstration
        nburn=5
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
    
    betas_df = pd.DataFrame(betas)
    log(f"Created betas DataFrame with shape: {betas_df.shape}")
    
    targets_2010 = argentina_FABLE[argentina_FABLE['times'] == '2010']
    log(f"Filtered targets for 2010, shape: {targets_2010.shape}")
    
    log("Running downscaling...")
    result = downscale(
        targets=targets_2010,
        start_areas=argentina_df['lu_levels'],
        xmat=argentina_df['xmat'],
        betas=betas_df
    )
    
    log(f"Downscaling complete! Results shape: {result['out_res'].shape}")
    
    if data['argentina_raster'] is not None:
        try:
            log("Creating visualization...")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            luc_plot(
                result['out_res'],
                raster_path=data['argentina_raster'],
                lu_from='Cropland',
                lu_to='Forest',
                ax=ax,
                title='Land-use change: Cropland to Forest (2010)'
            )
            
            output_path = os.path.join(os.getcwd(), 'argentina_luc_plot.png')
            plt.savefig(output_path)
            log(f"Saved visualization to {output_path}")
            
            plt.close(fig)
        except Exception as e:
            log(f"ERROR: Failed to create visualization: {e}")
    
    return result

if __name__ == "__main__":
    try:
        run_example()
        log("Script completed successfully!")
    except Exception as e:
        log(f"ERROR: Script failed with exception: {e}")
        import traceback
        log(traceback.format_exc())
