"""
Script to check the structure of the Argentina data and identify issues.

This script:
1. Loads the argentina_luc data
2. Checks what values exist for the 'Ts' column
3. Checks what values exist for the 'lu.from' column
4. Identifies what data is actually available
5. Suggests a fix for the argentina_example.py script
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

def check_csv_file(file_path):
    """Check the structure of a CSV file."""
    log(f"Checking CSV file: {file_path}")
    
    if not os.path.exists(file_path):
        log(f"ERROR: File does not exist: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        log(f"SUCCESS: Loaded CSV file with shape: {df.shape}")
        log(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        log(f"ERROR: Failed to load CSV file: {e}")
        return None

def main():
    """Main function to check the data structure."""
    log("=" * 80)
    log("ARGENTINA DATA STRUCTURE CHECKER")
    log("=" * 80)
    
    data_paths = [
        "/storage/lopesas/downscalepy/downscalepy/data/converted/argentina_luc.csv",
        "/storage/lopesas/downscalepy/data/converted/argentina_luc.csv",
        "downscalepy/data/converted/argentina_luc.csv"
    ]
    
    argentina_luc = None
    for path in data_paths:
        log(f"Trying to load from: {path}")
        df = check_csv_file(path)
        if df is not None:
            argentina_luc = df
            log(f"SUCCESS: Loaded argentina_luc from {path}")
            break
    
    if argentina_luc is None:
        log("ERROR: Could not load argentina_luc data from any path.")
        return False
    
    log("\nChecking 'Ts' column:")
    if 'Ts' in argentina_luc.columns:
        ts_values = argentina_luc['Ts'].unique()
        log(f"Unique values in 'Ts' column: {ts_values}")
        log(f"Data types in 'Ts' column: {argentina_luc['Ts'].apply(type).unique()}")
    else:
        log("ERROR: 'Ts' column not found in argentina_luc.")
        
        for col in argentina_luc.columns:
            if col.lower() in ['time', 'times', 'year', 'years', 't']:
                log(f"Found potential time column: {col}")
                ts_values = argentina_luc[col].unique()
                log(f"Unique values in '{col}' column: {ts_values}")
    
    log("\nChecking 'lu.from' column:")
    if 'lu.from' in argentina_luc.columns:
        lu_from_values = argentina_luc['lu.from'].unique()
        log(f"Unique values in 'lu.from' column: {lu_from_values}")
    else:
        log("ERROR: 'lu.from' column not found in argentina_luc.")
        
        for col in argentina_luc.columns:
            if col.lower() in ['lufrom', 'lu_from', 'from', 'source']:
                log(f"Found potential 'lu.from' column: {col}")
                lu_from_values = argentina_luc[col].unique()
                log(f"Unique values in '{col}' column: {lu_from_values}")
    
    log("\nChecking for Cropland data in Ts=2000:")
    if 'Ts' in argentina_luc.columns and 'lu.from' in argentina_luc.columns:
        mask_exact = (argentina_luc['lu.from'] == 'Cropland') & (argentina_luc['Ts'] == '2000')
        log(f"Rows with lu.from='Cropland' and Ts='2000': {mask_exact.sum()}")
        
        mask_numeric = (argentina_luc['lu.from'] == 'Cropland') & (argentina_luc['Ts'] == 2000)
        log(f"Rows with lu.from='Cropland' and Ts=2000 (numeric): {mask_numeric.sum()}")
        
        mask_cropland = argentina_luc['lu.from'] == 'Cropland'
        log(f"Total rows with lu.from='Cropland': {mask_cropland.sum()}")
        
        if mask_cropland.sum() > 0:
            cropland_ts = argentina_luc.loc[mask_cropland, 'Ts'].unique()
            log(f"Ts values for Cropland: {cropland_ts}")
    
    log("\nSuggested fix:")
    log("1. Check if 'Cropland' exists in the lu.from column")
    log("2. Check what time values are actually available in the Ts column")
    log("3. Update the argentina_example.py script to use the available time values")
    log("4. If 'Cropland' doesn't exist, use another land use class that does exist")
    
    log("=" * 80)
    log("DATA STRUCTURE CHECK COMPLETE")
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
