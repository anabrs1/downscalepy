"""
Direct data loading test script that explicitly checks for files in the exact path.

This script attempts to directly load the data files from the exact path
where they are known to exist, bypassing the normal loading mechanism.
It will print detailed information about what it finds.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def print_message(message):
    """Print a message with timestamp and also write to stderr."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    
    print(formatted_message, flush=True)
    sys.stderr.write(formatted_message + "\n")
    sys.stderr.flush()

def print_separator(char="="):
    """Print a separator line."""
    separator = char * 80
    print_message(separator)

def check_file(file_path):
    """Check if a file exists and print its details."""
    print_message(f"Checking file: {file_path}")
    
    if os.path.exists(file_path):
        print_message(f"SUCCESS: File exists: {file_path}")
        print_message(f"File size: {os.path.getsize(file_path)} bytes")
        
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                print_message(f"SUCCESS: Loaded CSV file")
                print_message(f"Shape: {data.shape}")
                print_message(f"Columns: {data.columns.tolist()}")
                return data
            else:
                print_message(f"Not a CSV file, skipping content check")
                return None
        except Exception as e:
            print_message(f"ERROR: Failed to load file content: {str(e)}")
            return None
    else:
        print_message(f"ERROR: File does not exist: {file_path}")
        return None

def check_directory(directory_path):
    """Check if a directory exists and list its contents."""
    print_message(f"Checking directory: {directory_path}")
    
    if os.path.exists(directory_path):
        print_message(f"SUCCESS: Directory exists: {directory_path}")
        
        try:
            contents = os.listdir(directory_path)
            print_message(f"Directory contents ({len(contents)} items):")
            for item in sorted(contents):
                item_path = os.path.join(directory_path, item)
                if os.path.isdir(item_path):
                    print_message(f"  - {item}/ (directory)")
                else:
                    print_message(f"  - {item} ({os.path.getsize(item_path)} bytes)")
            return contents
        except Exception as e:
            print_message(f"ERROR: Failed to list directory contents: {str(e)}")
            return []
    else:
        print_message(f"ERROR: Directory does not exist: {directory_path}")
        
        parent_dir = os.path.dirname(directory_path)
        if os.path.exists(parent_dir):
            print_message(f"Parent directory exists: {parent_dir}")
            try:
                contents = os.listdir(parent_dir)
                print_message(f"Parent directory contents ({len(contents)} items):")
                for item in sorted(contents):
                    print_message(f"  - {item}")
            except Exception as e:
                print_message(f"ERROR: Failed to list parent directory contents: {str(e)}")
        return []

def main():
    """Main function to test direct data loading."""
    print_separator("#")
    print_message("DIRECT DATA LOADING TEST")
    print_message("This script checks for data files in the exact path")
    print_separator("#")
    
    cwd = os.getcwd()
    print_message(f"Current working directory: {cwd}")
    
    print_message("Python executable: " + sys.executable)
    print_message("Python version: " + sys.version.replace('\n', ' '))
    
    data_path = "/storage/lopesas/downscalepy/downscalepy/data/converted"
    
    print_separator()
    print_message("CHECKING PRIMARY DATA DIRECTORY")
    print_separator()
    contents = check_directory(data_path)
    
    required_files = [
        "argentina_luc.csv",
        "argentina_FABLE.csv",
        "argentina_df_xmat.csv",
        "argentina_df_lu_levels.csv",
        "argentina_df_restrictions.csv",
        "argentina_df_pop_data.csv"
    ]
    
    print_separator()
    print_message("CHECKING REQUIRED FILES")
    print_separator()
    
    loaded_data = {}
    missing_files = []
    
    for file_name in required_files:
        file_path = os.path.join(data_path, file_name)
        data = check_file(file_path)
        if data is not None:
            loaded_data[file_name] = data
        else:
            missing_files.append(file_name)
    
    if missing_files:
        print_separator()
        print_message(f"MISSING FILES: {missing_files}")
        print_message("CHECKING ALTERNATIVE PATHS")
        print_separator()
        
        alternative_paths = [
            "/storage/lopesas/downscalepy/data/converted",
            "/storage/lopesas/downscalepy/converted",
            "/storage/lopesas/downscalepy/downscalepy/data",
            "/storage/lopesas/downscalepy",
            "/storage/lopesas/downscalepy/downscalepy",
            "/storage/lopesas/downscalepy/data",
            "/storage/lopesas"
        ]
        
        for alt_path in alternative_paths:
            print_message(f"Checking alternative path: {alt_path}")
            check_directory(alt_path)
            
            if os.path.exists(alt_path):
                for item in os.listdir(alt_path):
                    if item.endswith('.csv'):
                        print_message(f"Found CSV file: {os.path.join(alt_path, item)}")
    
    print_separator()
    print_message("SUMMARY")
    print_separator()
    
    if len(loaded_data) == len(required_files):
        print_message("SUCCESS: All required files were found and loaded!")
    else:
        print_message(f"ERROR: Only {len(loaded_data)} out of {len(required_files)} files were loaded successfully.")
        print_message(f"Missing files: {missing_files}")
    
    print_separator("#")
    print_message("END OF DIRECT DATA LOADING TEST")
    print_separator("#")

if __name__ == "__main__":
    main()
