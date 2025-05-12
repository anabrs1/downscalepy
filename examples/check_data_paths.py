"""
Standalone script to check data paths for downscalepy.

This script will:
1. Check for data files in various locations
2. Print colorized output that's impossible to miss
3. Show exactly what paths are available on the system
"""

import os
import sys
import glob
from datetime import datetime

RED = "\033[1;91m"
GREEN = "\033[1;92m"
YELLOW = "\033[1;93m"
BLUE = "\033[1;94m"
MAGENTA = "\033[1;95m"
CYAN = "\033[1;96m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

def print_colored(message, color=None):
    """Print a message with color."""
    if color:
        print(f"{color}{message}{RESET}", flush=True)
    else:
        print(message, flush=True)

def print_separator(char="=", color=BLUE):
    """Print a separator line with color."""
    separator = char * 80
    print_colored(separator, color)

def check_file_exists(path):
    """Check if a file exists and print the result."""
    if os.path.exists(path):
        print_colored(f"✓ File exists: {path}", GREEN)
        print_colored(f"  Size: {os.path.getsize(path)} bytes", GREEN)
        return True
    else:
        print_colored(f"✗ File does not exist: {path}", RED)
        return False

def check_directory(path):
    """Check if a directory exists and list its contents."""
    print_separator("-", CYAN)
    if os.path.exists(path):
        print_colored(f"✓ Directory exists: {path}", GREEN)
        try:
            contents = os.listdir(path)
            print_colored(f"Contents ({len(contents)} items):", YELLOW)
            for item in sorted(contents):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    print_colored(f"  📁 {item}/", BLUE)
                elif item.endswith('.csv'):
                    print_colored(f"  📄 {item} ({os.path.getsize(item_path)} bytes)", GREEN)
                else:
                    print_colored(f"  📄 {item}", RESET)
        except Exception as e:
            print_colored(f"Error listing directory contents: {e}", RED)
    else:
        print_colored(f"✗ Directory does not exist: {path}", RED)
        
        parent_dir = os.path.dirname(path)
        if os.path.exists(parent_dir):
            print_colored(f"Parent directory exists: {parent_dir}", YELLOW)
            try:
                print_colored(f"Parent directory contents:", YELLOW)
                for item in sorted(os.listdir(parent_dir)):
                    print_colored(f"  - {item}", RESET)
            except Exception as e:
                print_colored(f"Error listing parent directory contents: {e}", RED)

def main():
    """Main function to run all checks."""
    print_separator("#", RED)
    print_colored("DOWNSCALEPY DATA PATH CHECKER", RED + BOLD)
    print_colored("This script will help diagnose data path issues", RED + BOLD)
    print_separator("#", RED)
    
    print_colored(f"Current working directory: {os.getcwd()}", CYAN)
    print_colored(f"Python executable: {sys.executable}", CYAN)
    
    print_separator("=", BLUE)
    print_colored("CHECKING PRIMARY DATA PATH", BLUE + BOLD)
    print_separator("=", BLUE)
    
    primary_path = '/storage/lopesas/downscalepy/downscalepy/data/converted'
    check_directory(primary_path)
    
    print_separator("=", BLUE)
    print_colored("CHECKING FOR REQUIRED CSV FILES", BLUE + BOLD)
    print_separator("=", BLUE)
    
    required_files = [
        'argentina_luc.csv',
        'argentina_FABLE.csv',
        'argentina_df_xmat.csv',
        'argentina_df_lu_levels.csv',
        'argentina_df_restrictions.csv',
        'argentina_df_pop_data.csv'
    ]
    
    for req_file in required_files:
        file_path = os.path.join(primary_path, req_file)
        check_file_exists(file_path)
    
    print_separator("=", BLUE)
    print_colored("CHECKING ALTERNATIVE PATHS", BLUE + BOLD)
    print_separator("=", BLUE)
    
    alternative_paths = [
        '/storage/lopesas/downscalepy/data/converted',
        '/storage/lopesas/downscalepy/converted',
        '/storage/lopesas/downscalepy/downscalepy/data',
        '/storage/lopesas/downscalepy',
        '/storage/lopesas/downscalepy/downscalepy',
        '/storage/lopesas/downscalepy/data',
        '/storage/lopesas'
    ]
    
    for alt_path in alternative_paths:
        check_directory(alt_path)
    
    print_separator("=", BLUE)
    print_colored("SEARCHING FOR ANY CSV FILES", BLUE + BOLD)
    print_separator("=", BLUE)
    
    csv_files = glob.glob('/storage/lopesas/downscalepy/**/*.csv', recursive=True)
    if csv_files:
        print_colored(f"Found {len(csv_files)} CSV files:", GREEN)
        for csv_file in sorted(csv_files):
            print_colored(f"  - {csv_file} ({os.path.getsize(csv_file)} bytes)", GREEN)
    else:
        print_colored("No CSV files found in /storage/lopesas/downscalepy", RED)
    
    print_separator("#", RED)
    print_colored("END OF DATA PATH CHECK", RED + BOLD)
    print_separator("#", RED)

if __name__ == "__main__":
    main()
