"""
Super direct data file finder script.

This script will search for the required data files in various locations
and print detailed information about what it finds. It will also try to
directly load the files to verify they can be read.

Run this script directly:
python examples/find_data_files.py
"""

import os
import sys
import pandas as pd
import glob
import traceback
from datetime import datetime

REQUIRED_FILES = [
    "argentina_luc.csv",
    "argentina_FABLE.csv",
    "argentina_df_xmat.csv",
    "argentina_df_lu_levels.csv",
    "argentina_df_restrictions.csv",
    "argentina_df_pop_data.csv"
]

PRIMARY_PATH = "/storage/lopesas/downscalepy/downscalepy/data/converted"

ALTERNATIVE_PATHS = [
    "/storage/lopesas/downscalepy/data/converted",
    "/storage/lopesas/downscalepy/converted",
    "/storage/lopesas/downscalepy/downscalepy/data",
    "/storage/lopesas/downscalepy",
    "/storage/lopesas/downscalepy/downscalepy",
    "/storage/lopesas/downscalepy/data",
    "/storage/lopesas"
]

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    sys.stderr.write(full_message + "\n")
    sys.stderr.flush()

def check_file(file_path):
    """Check if a file exists and try to load it."""
    log(f"Checking file: {file_path}")
    
    if os.path.exists(file_path):
        log(f"  ✓ File exists: {file_path}")
        log(f"  ✓ File size: {os.path.getsize(file_path)} bytes")
        
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                log(f"  ✓ Successfully loaded CSV file")
                log(f"  ✓ Shape: {data.shape}")
                log(f"  ✓ Columns: {data.columns.tolist()}")
                return True
            else:
                log(f"  ✓ File exists but is not a CSV file")
                return True
        except Exception as e:
            log(f"  ✗ Failed to load file: {str(e)}")
            return False
    else:
        log(f"  ✗ File does not exist")
        return False

def check_directory(directory):
    """Check if a directory exists and list its contents."""
    log(f"Checking directory: {directory}")
    
    if os.path.exists(directory):
        log(f"  ✓ Directory exists")
        
        try:
            contents = os.listdir(directory)
            log(f"  ✓ Directory contains {len(contents)} items")
            
            csv_files = [f for f in contents if f.endswith('.csv')]
            if csv_files:
                log(f"  ✓ Found {len(csv_files)} CSV files:")
                for csv_file in sorted(csv_files):
                    file_path = os.path.join(directory, csv_file)
                    log(f"    - {csv_file} ({os.path.getsize(file_path)} bytes)")
            else:
                log(f"  ✗ No CSV files found in this directory")
            
            return True
        except Exception as e:
            log(f"  ✗ Failed to list directory contents: {str(e)}")
            return False
    else:
        log(f"  ✗ Directory does not exist")
        return False

def find_csv_files_recursively(base_path, max_depth=3):
    """Find all CSV files recursively up to a certain depth."""
    log(f"Searching recursively for CSV files in: {base_path} (max depth: {max_depth})")
    
    if not os.path.exists(base_path):
        log(f"  ✗ Base path does not exist")
        return []
    
    csv_files = []
    
    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth > max_depth:
            continue
        
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                csv_files.append(full_path)
    
    if csv_files:
        log(f"  ✓ Found {len(csv_files)} CSV files recursively:")
        for file_path in sorted(csv_files):
            log(f"    - {file_path} ({os.path.getsize(file_path)} bytes)")
    else:
        log(f"  ✗ No CSV files found recursively")
    
    return csv_files

def copy_files_if_found(source_dir, target_dir):
    """Copy required files from source to target directory if they exist."""
    if not os.path.exists(source_dir):
        log(f"  ✗ Source directory does not exist: {source_dir}")
        return False
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        log(f"  ✓ Created target directory: {target_dir}")
    except Exception as e:
        log(f"  ✗ Failed to create target directory: {str(e)}")
        return False
    
    success = True
    
    for file_name in REQUIRED_FILES:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(source_path):
            try:
                import shutil
                shutil.copy2(source_path, target_path)
                log(f"  ✓ Copied {file_name} to {target_dir}")
            except Exception as e:
                log(f"  ✗ Failed to copy {file_name}: {str(e)}")
                success = False
        else:
            log(f"  ✗ Source file does not exist: {source_path}")
            success = False
    
    return success

def main():
    """Main function to find data files."""
    log("=" * 80)
    log("SUPER DIRECT DATA FILE FINDER")
    log("=" * 80)
    
    log(f"Current working directory: {os.getcwd()}")
    log(f"Python executable: {sys.executable}")
    log(f"Python version: {sys.version.replace(chr(10), ' ')}")
    
    log("\nCHECKING PRIMARY PATH")
    log("-" * 80)
    primary_dir_exists = check_directory(PRIMARY_PATH)
    
    if primary_dir_exists:
        log("\nCHECKING REQUIRED FILES IN PRIMARY PATH")
        log("-" * 80)
        
        all_found = True
        for file_name in REQUIRED_FILES:
            file_path = os.path.join(PRIMARY_PATH, file_name)
            if not check_file(file_path):
                all_found = False
        
        if all_found:
            log("\n✓ All required files found in primary path!")
            return
    
    log("\nCHECKING ALTERNATIVE PATHS")
    log("-" * 80)
    
    for alt_path in ALTERNATIVE_PATHS:
        log(f"\nChecking alternative path: {alt_path}")
        if check_directory(alt_path):
            all_found = True
            for file_name in REQUIRED_FILES:
                file_path = os.path.join(alt_path, file_name)
                if not check_file(file_path):
                    all_found = False
            
            if all_found:
                log(f"\n✓ All required files found in alternative path: {alt_path}")
                
                log("\nCOPYING FILES TO PRIMARY PATH")
                log("-" * 80)
                if copy_files_if_found(alt_path, PRIMARY_PATH):
                    log(f"\n✓ Successfully copied all files to primary path")
                else:
                    log(f"\n✗ Failed to copy all files to primary path")
                
                return
    
    log("\nRECURSIVE SEARCH")
    log("-" * 80)
    
    base_search_path = "/storage/lopesas/downscalepy"
    csv_files = find_csv_files_recursively(base_search_path)
    
    if csv_files:
        log("\nCHECKING IF REQUIRED FILES WERE FOUND RECURSIVELY")
        log("-" * 80)
        
        for required_file in REQUIRED_FILES:
            found = False
            for csv_file in csv_files:
                if os.path.basename(csv_file) == required_file:
                    log(f"  ✓ Found {required_file} at {csv_file}")
                    found = True
                    break
            
            if not found:
                log(f"  ✗ Did not find {required_file} in recursive search")
    
    log("\nSUMMARY")
    log("=" * 80)
    log("Could not find all required files in any of the checked paths.")
    log("Please ensure the data files are available in one of the following paths:")
    log(f"1. {PRIMARY_PATH} (primary path)")
    for i, alt_path in enumerate(ALTERNATIVE_PATHS):
        log(f"{i+2}. {alt_path}")
    
    log("\nRequired files:")
    for file_name in REQUIRED_FILES:
        log(f"- {file_name}")
    
    log("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {str(e)}")
        log(traceback.format_exc())
