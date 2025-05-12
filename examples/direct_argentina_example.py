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
    Load the Argentina data directly from CSV files.
    
    This function looks for data in multiple possible locations to handle
    different deployment environments.
    """
    log("Loading Argentina data...")
    
    possible_paths = [
        "/storage/lopesas/downscalepy/downscalepy/data",
        "/storage/lopesas/downscalepy/data",
        os.path.join(os.path.dirname(__file__), "../downscalepy/data"),
        os.path.join(os.path.dirname(__file__), "../data")
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            log(f"Checking path: {path}")
            if os.path.exists(os.path.join(path, "argentina_luc.csv")):
                data_path = path
                log(f"Found data at: {data_path}")
                break
    
    if data_path is None:
        log("ERROR: Could not find data directory. Please specify the correct path.")
        return None
    
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
        return None

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
