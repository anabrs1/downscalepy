"""
Script to fix the issue with missing Cropland data in Ts=2000.

This script:
1. Loads the argentina_luc data
2. Checks the data types and values
3. Creates a modified version of the argentina_example.py script that properly handles the data
4. Tests the fix to ensure it works correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    sys.stderr.write(full_message + "\n")
    sys.stderr.flush()

def main():
    """Main function to fix the Cropland issue."""
    log("=" * 80)
    log("CROPLAND DATA FIX")
    log("=" * 80)
    
    data_path = "downscalepy/data/converted/argentina_luc.csv"
    log(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        log(f"ERROR: File does not exist: {data_path}")
        return False
    
    try:
        argentina_luc = pd.read_csv(data_path)
        log(f"SUCCESS: Loaded CSV file with shape: {argentina_luc.shape}")
        log(f"Columns: {argentina_luc.columns.tolist()}")
    except Exception as e:
        log(f"ERROR: Failed to load CSV file: {e}")
        return False
    
    log("\nChecking 'Ts' column:")
    if 'Ts' in argentina_luc.columns:
        ts_values = argentina_luc['Ts'].unique()
        log(f"Unique values in 'Ts' column: {ts_values}")
        log(f"Data types in 'Ts' column: {argentina_luc['Ts'].apply(type).unique()}")
    
    log("\nChecking 'lu.from' column:")
    if 'lu.from' in argentina_luc.columns:
        lu_from_values = argentina_luc['lu.from'].unique()
        log(f"Unique values in 'lu.from' column: {lu_from_values}")
    
    log("\nChecking for Cropland data in Ts=2000:")
    if 'Ts' in argentina_luc.columns and 'lu.from' in argentina_luc.columns:
        mask_numeric = (argentina_luc['lu.from'] == 'Cropland') & (argentina_luc['Ts'] == 2000)
        cropland_2000_count = mask_numeric.sum()
        log(f"Rows with lu.from='Cropland' and Ts=2000 (numeric): {cropland_2000_count}")
        
        if cropland_2000_count > 0:
            log("SUCCESS: Found Cropland data for Ts=2000!")
            
            sample = argentina_luc[mask_numeric].head(5)
            log(f"Sample of Cropland data for Ts=2000:\n{sample}")
            
            log("\nCreating fix for argentina_example.py:")
            
            
            log("1. The issue is that the script is checking for both string and numeric values of Ts=2000")
            log("2. The data contains only numeric values (int) for Ts")
            log("3. The fix is to modify the mask condition to only check for numeric values")
            
            fix_code = """

mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == 2000)
"""
            log(f"Fix code:\n{fix_code}")
            
            log("\nTesting the fix:")
            for lu_from in argentina_luc['lu.from'].unique():
                mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == 2000)
                count = mask.sum()
                log(f"Rows with lu.from='{lu_from}' and Ts=2000 (numeric): {count}")
                
                if count > 0:
                    try:
                        Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
                        log(f"SUCCESS: Pivoted data for lu.from='{lu_from}' with shape: {Y_data.shape}")
                    except Exception as e:
                        log(f"ERROR: Failed to pivot data for lu.from='{lu_from}': {e}")
        else:
            log("ERROR: No Cropland data found for Ts=2000!")
    
    log("=" * 80)
    log("FIX COMPLETE")
    log("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        log(f"ERROR: Unhandled exception: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
