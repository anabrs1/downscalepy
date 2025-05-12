"""
Direct data loading test script.

This script attempts to directly load the data files from the exact path
where they are known to exist, bypassing the normal loading mechanism.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

RED = "\033[1;91m"
GREEN = "\033[1;92m"
YELLOW = "\033[1;93m"
BLUE = "\033[1;94m"
RESET = "\033[0m"

def print_colored(message, color=None):
    """Print a message with color and also write to stderr."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if color:
        formatted_message = f"{color}[{timestamp}] {message}{RESET}"
    else:
        formatted_message = f"[{timestamp}] {message}"
    
    print(formatted_message, flush=True)
    sys.stderr.write(formatted_message + "\n")
    sys.stderr.flush()

def print_separator(char="=", color=BLUE):
    """Print a separator line with color."""
    separator = char * 80
    print_colored(separator, color)

def load_file(file_path, file_type="csv"):
    """Attempt to load a file and return its contents."""
    print_colored(f"Attempting to load {file_path}", YELLOW)
    
    if not os.path.exists(file_path):
        print_colored(f"ERROR: File does not exist: {file_path}", RED)
        return None
    
    try:
        if file_type == "csv":
            data = pd.read_csv(file_path)
            print_colored(f"SUCCESS: Loaded CSV file: {file_path}", GREEN)
            print_colored(f"Shape: {data.shape}", GREEN)
            print_colored(f"Columns: {data.columns.tolist()}", GREEN)
            return data
        else:
            print_colored(f"ERROR: Unsupported file type: {file_type}", RED)
            return None
    except Exception as e:
        print_colored(f"ERROR: Failed to load file: {file_path}", RED)
        print_colored(f"Exception: {str(e)}", RED)
        return None

def main():
    """Main function to test direct data loading."""
    print_separator("#", RED)
    print_colored("DIRECT DATA LOADING TEST", RED)
    print_colored("This script attempts to directly load data files from the known path", RED)
    print_separator("#", RED)
    
    data_path = "/storage/lopesas/downscalepy/downscalepy/data/converted"
    
    print_colored(f"Data path: {data_path}", BLUE)
    print_colored(f"Current working directory: {os.getcwd()}", BLUE)
    
    files = [
        "argentina_luc.csv",
        "argentina_FABLE.csv",
        "argentina_df_xmat.csv",
        "argentina_df_lu_levels.csv",
        "argentina_df_restrictions.csv",
        "argentina_df_pop_data.csv"
    ]
    
    loaded_data = {}
    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        data = load_file(file_path)
        if data is not None:
            loaded_data[file_name] = data
    
    if len(loaded_data) == len(files):
        print_colored("SUCCESS: All files loaded successfully!", GREEN)
        
        try:
            argentina_data = {
                'argentina_luc': loaded_data['argentina_luc.csv'],
                'argentina_FABLE': loaded_data['argentina_FABLE.csv'],
                'argentina_df': {
                    'xmat': loaded_data['argentina_df_xmat.csv'],
                    'lu_levels': loaded_data['argentina_df_lu_levels.csv'],
                    'restrictions': loaded_data['argentina_df_restrictions.csv'],
                    'pop_data': loaded_data['argentina_df_pop_data.csv']
                }
            }
            print_colored("SUCCESS: Created argentina_data dictionary!", GREEN)
        except Exception as e:
            print_colored(f"ERROR: Failed to create argentina_data dictionary: {str(e)}", RED)
    else:
        print_colored(f"ERROR: Only {len(loaded_data)} out of {len(files)} files were loaded successfully.", RED)
    
    print_separator("#", RED)
    print_colored("END OF DIRECT DATA LOADING TEST", RED)
    print_separator("#", RED)

if __name__ == "__main__":
    main()
