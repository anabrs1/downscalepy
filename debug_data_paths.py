"""
Standalone debug script for diagnosing data path issues in downscalepy.

This script will:
1. Check for data files in various locations
2. Print colorized output that's impossible to miss
3. Write debug info to a log file in the user's home directory
4. Show exactly what paths are available on the system
"""

import os
import sys
import shutil
import glob
import traceback
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

LOG_FILE = os.path.expanduser("~/downscalepy_debug.log")

def log_message(message, color=None, file=None):
    """Log a message to stdout, stderr, and the log file with color."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if color:
        formatted_message = f"{color}{message}{RESET}"
    else:
        formatted_message = message
    
    full_message = f"[{timestamp}] {message}"
    
    print(formatted_message, flush=True)
    
    sys.stderr.write(formatted_message + "\n")
    sys.stderr.flush()
    
    with open(LOG_FILE, "a") as f:
        f.write(full_message + "\n")
    
    if file:
        with open(file, "a") as f:
            f.write(full_message + "\n")

def print_separator(char="=", color=BLUE):
    """Print a separator line with color."""
    separator = char * 80
    log_message(separator, color)

def check_directory(path, required_files=None):
    """Check if a directory exists and contains required files."""
    print_separator("-", CYAN)
    if os.path.exists(path):
        log_message(f"✓ Directory exists: {path}", GREEN)
        try:
            contents = os.listdir(path)
            log_message(f"Contents ({len(contents)} items):", YELLOW)
            for item in sorted(contents):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    log_message(f"  📁 {item}/", BLUE)
                elif item.endswith('.csv'):
                    log_message(f"  📄 {item} ({os.path.getsize(item_path)} bytes)", GREEN)
                else:
                    log_message(f"  📄 {item}", RESET)
            
            if required_files:
                log_message("\nChecking for required files:", YELLOW)
                all_found = True
                for req_file in required_files:
                    file_path = os.path.join(path, req_file)
                    if os.path.exists(file_path):
                        log_message(f"  ✓ Found: {req_file} ({os.path.getsize(file_path)} bytes)", GREEN)
                    else:
                        all_found = False
                        found_case_insensitive = False
                        for existing_file in contents:
                            if existing_file.lower() == req_file.lower():
                                log_message(f"  ⚠ Case mismatch: Required '{req_file}', found '{existing_file}'", YELLOW)
                                found_case_insensitive = True
                                break
                        
                        if not found_case_insensitive:
                            log_message(f"  ✗ Missing: {req_file}", RED)
                
                if all_found:
                    log_message("\n✓ All required files found in this directory!", GREEN + BOLD)
                    return True
                else:
                    log_message("\n✗ Some required files are missing from this directory.", RED)
        except Exception as e:
            log_message(f"Error listing directory contents: {e}", RED)
    else:
        log_message(f"✗ Directory does not exist: {path}", RED)
        
        parent_dir = os.path.dirname(path)
        if os.path.exists(parent_dir):
            log_message(f"Parent directory exists: {parent_dir}", YELLOW)
            try:
                log_message(f"Parent directory contents:", YELLOW)
                for item in sorted(os.listdir(parent_dir)):
                    log_message(f"  - {item}", RESET)
            except Exception as e:
                log_message(f"Error listing parent directory contents: {e}", RED)
    
    return False

def find_csv_files_recursively(base_path, max_depth=3):
    """Find all CSV files recursively up to a certain depth."""
    print_separator("-", MAGENTA)
    log_message(f"Searching recursively for CSV files in: {base_path} (max depth: {max_depth})", MAGENTA)
    
    if not os.path.exists(base_path):
        log_message(f"Base path does not exist: {base_path}", RED)
        return
    
    csv_files_found = []
    
    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth > max_depth:
            continue
        
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            rel_path = os.path.relpath(root, base_path)
            log_message(f"Found {len(csv_files)} CSV files in: {rel_path}/", GREEN)
            for csv_file in csv_files:
                full_path = os.path.join(root, csv_file)
                csv_files_found.append(full_path)
                log_message(f"  - {csv_file} ({os.path.getsize(full_path)} bytes)", CYAN)
    
    if not csv_files_found:
        log_message(f"No CSV files found in {base_path} (up to depth {max_depth})", YELLOW)
    else:
        log_message(f"Total CSV files found: {len(csv_files_found)}", GREEN)
    
    return csv_files_found

def check_environment():
    """Check environment variables and Python environment."""
    print_separator("=", BLUE)
    log_message("ENVIRONMENT INFORMATION", BLUE + BOLD)
    print_separator("=", BLUE)
    
    log_message(f"Current working directory: {os.getcwd()}", CYAN)
    log_message(f"Python executable: {sys.executable}", CYAN)
    log_message(f"Python version: {sys.version}", CYAN)
    log_message(f"Platform: {sys.platform}", CYAN)
    
    log_message("\nEnvironment Variables:", YELLOW)
    for key, value in sorted(os.environ.items()):
        if key.startswith('PYTHON') or key.startswith('PATH') or 'DOWNSCALE' in key:
            log_message(f"  {key}={value}", CYAN)
    
    log_message("\nPython Path:", YELLOW)
    for path in sys.path:
        log_message(f"  {path}", CYAN)

def main():
    """Main function to run all checks."""
    with open(LOG_FILE, "w") as f:
        f.write(f"Downscalepy Debug Log - {datetime.now()}\n\n")
    
    print_separator("#", RED)
    log_message("DOWNSCALEPY SUPER EXTREME DEBUGGING", RED + BOLD)
    log_message("This script will help diagnose data path issues", RED + BOLD)
    print_separator("#", RED)
    
    log_message(f"Debug log file: {LOG_FILE}", YELLOW)
    
    check_environment()
    
    required_files = [
        'argentina_luc.csv',
        'argentina_FABLE.csv',
        'argentina_df_xmat.csv',
        'argentina_df_lu_levels.csv',
        'argentina_df_restrictions.csv',
        'argentina_df_pop_data.csv'
    ]
    
    print_separator("=", BLUE)
    log_message("CHECKING PRIMARY DATA PATH", BLUE + BOLD)
    print_separator("=", BLUE)
    
    primary_path = '/storage/lopesas/downscalepy/downscalepy/data/converted'
    found_in_primary = check_directory(primary_path, required_files)
    
    print_separator("=", BLUE)
    log_message("CHECKING ALTERNATIVE PATHS", BLUE + BOLD)
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
    
    found_in_alt = False
    found_path = None
    
    for alt_path in alternative_paths:
        if check_directory(alt_path, required_files):
            found_in_alt = True
            found_path = alt_path
            log_message(f"\n✓ Found all required files in alternative path: {alt_path}", GREEN + BOLD)
            
            try:
                if not os.path.exists(primary_path):
                    os.makedirs(primary_path, exist_ok=True)
                    log_message(f"Created primary path directory: {primary_path}", GREEN)
                
                for req_file in required_files:
                    src = os.path.join(alt_path, req_file)
                    dst = os.path.join(primary_path, req_file)
                    if os.path.exists(src) and not os.path.exists(dst):
                        shutil.copy(src, dst)
                        log_message(f"Copied {req_file} to primary path", GREEN)
                
                log_message("All files copied to primary path successfully!", GREEN + BOLD)
            except Exception as e:
                log_message(f"Error copying files to primary path: {e}", RED)
                log_message(traceback.format_exc(), RED)
            
            break
    
    if not found_in_primary and not found_in_alt:
        print_separator("=", BLUE)
        log_message("RECURSIVE SEARCH FOR CSV FILES", BLUE + BOLD)
        print_separator("=", BLUE)
        
        find_csv_files_recursively('/storage/lopesas/downscalepy')
    
    print_separator("=", BLUE)
    log_message("SUMMARY", BLUE + BOLD)
    print_separator("=", BLUE)
    
    if found_in_primary:
        log_message("✓ All required files found in primary path!", GREEN + BOLD)
        log_message(f"Primary path: {primary_path}", GREEN)
    elif found_in_alt:
        log_message("✓ All required files found in alternative path!", YELLOW + BOLD)
        log_message(f"Alternative path: {found_path}", YELLOW)
        log_message("Files were copied to the primary path.", YELLOW)
    else:
        log_message("✗ Required files not found in any checked location!", RED + BOLD)
        log_message("Please ensure the data files are available in one of the checked paths.", RED)
    
    print_separator("#", RED)
    log_message(f"Debug log written to: {LOG_FILE}", YELLOW + BOLD)
    log_message("END OF SUPER EXTREME DEBUGGING", RED + BOLD)
    print_separator("#", RED)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_separator("!", RED)
        log_message(f"ERROR IN DEBUG SCRIPT: {e}", RED + BOLD)
        log_message(traceback.format_exc(), RED)
        print_separator("!", RED)
        sys.exit(1)
