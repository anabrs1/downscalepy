"""
Direct Argentina example script for testing the numerical stability fixes in mnlogit.

This script directly loads the Argentina data and runs the mnlogit function
with the numerical stability improvements. It's designed to be a simple test
case for verifying that the fixes work correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from downscalepy.models.mnlogit import mnlogit

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
    for path in possible_paths:
        if os.path.exists(path):
            log(f"Path exists: {path}")
            try:
                files = os.listdir(path)
                log(f"Files in {path}: {', '.join(files)}")
            except Exception as e:
                log(f"Error listing files in {path}: {str(e)}")
                continue
                
            if os.path.exists(os.path.join(path, "argentina_luc.csv")):
                data_path = path
                log(f"Found CSV data at: {data_path}")
                break
            
            if os.path.exists(os.path.join(path, "argentina_raster.tif")):
                log(f"Found GeoTIFF data at: {path}")
                data_path = "synthetic"
                break
    
    if data_path == "synthetic" or data_path is None:
        log("Real data not found. Generating synthetic data for testing...")
        return generate_synthetic_data()
    
    try:
        log(f"Loading argentina_luc.csv from {data_path}")
        argentina_luc = pd.read_csv(os.path.join(data_path, "argentina_luc.csv"))
        
        log(f"Loading argentina_xmat.csv from {data_path}")
        xmat = pd.read_csv(os.path.join(data_path, "argentina_xmat.csv"))
        
        log(f"Loading argentina_lu_levels.csv from {data_path}")
        lu_levels = pd.read_csv(os.path.join(data_path, "argentina_lu_levels.csv"))
        
        log(f"Loading argentina_FABLE.csv from {data_path}")
        fable = pd.read_csv(os.path.join(data_path, "argentina_FABLE.csv"))
        
        data = {
            'argentina_luc': argentina_luc,
            'argentina_df': {
                'xmat': xmat,
                'lu_levels': lu_levels
            },
            'argentina_FABLE': fable
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
    lu_classes = ['Cropland', 'Forest', 'Pasture', 'Urban', 'OtherLand']
    ks = [f'k{i}' for i in range(1, 5)]
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
        'argentina_FABLE': fable
    }
    
    log("Synthetic data generated successfully!")
    return data

def run_mnlogit_test(data):
    """
    Run a test of the mnlogit function with the Argentina data.
    
    This function tests the numerical stability fixes in the mnlogit function.
    """
    log("Running mnlogit test...")
    
    argentina_luc = data['argentina_luc']
    argentina_df = data['argentina_df']
    
    results = {}
    for lu_from in argentina_luc['lu.from'].unique():
        log(f"Testing mnlogit for lu.from={lu_from}")
        
        mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == 2000)
        
        if mask.sum() == 0:
            log(f"WARNING: No data found for lu.from={lu_from} and Ts=2000")
            continue
        
        Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
        
        X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
        X_data = X_data.reindex(Y_data.index)
        
        baseline = Y_data.columns.get_loc(lu_from) if lu_from in Y_data.columns else None
        
        log(f"Running mnlogit for {lu_from} with data shape: X={X_data.shape}, Y={Y_data.shape}")
        
        try:
            res = mnlogit(
                X=X_data.values,
                Y=Y_data.values,
                baseline=baseline,
                niter=10,  # Small number for testing
                nburn=5,
                jitter=1e-6  # Add small jitter for numerical stability
            )
            
            beta_mean = np.mean(res['postb'], axis=2)
            
            results[lu_from] = {
                'beta_mean': beta_mean,
                'X_shape': X_data.shape,
                'Y_shape': Y_data.shape,
                'success': True
            }
            
            log(f"SUCCESS: mnlogit completed for {lu_from}")
            
        except Exception as e:
            log(f"ERROR in mnlogit for {lu_from}: {str(e)}")
            results[lu_from] = {
                'error': str(e),
                'X_shape': X_data.shape,
                'Y_shape': Y_data.shape,
                'success': False
            }
    
    return results

def main():
    """Main function to run the direct Argentina example."""
    log("Starting direct Argentina example...")
    
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    data = load_argentina_data()
    if data is None:
        log("ERROR: Failed to load data. Exiting.")
        return 1
    
    results = run_mnlogit_test(data)
    
    log("Test summary:")
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    log(f"Successful runs: {success_count}/{total_count}")
    
    for lu_from, result in results.items():
        if result.get('success', False):
            log(f"  {lu_from}: SUCCESS")
        else:
            log(f"  {lu_from}: FAILED - {result.get('error', 'Unknown error')}")
    
    log("Direct Argentina example completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
