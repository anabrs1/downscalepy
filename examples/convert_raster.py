"""
Python wrapper script to convert the argentina_raster.RData file to GeoTIFF format.

This script:
1. Checks if the R script exists
2. Executes the R script using subprocess
3. Verifies that the conversion was successful
4. Provides clear output about what happened

Run this script directly:
python examples/convert_raster.py
"""

import os
import sys
import subprocess
from datetime import datetime

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    sys.stderr.write(full_message + "\n")
    sys.stderr.flush()

def main():
    """Main function to convert the raster data."""
    log("=" * 80)
    log("ARGENTINA RASTER CONVERTER")
    log("=" * 80)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    r_script_path = os.path.join(repo_dir, "downscalepy", "data", "convert_argentina_raster.R")
    
    log(f"Looking for R script at: {r_script_path}")
    
    if not os.path.exists(r_script_path):
        log(f"ERROR: R script not found at {r_script_path}")
        return False
    
    log(f"Found R script at: {r_script_path}")
    
    rdata_paths = [
        os.path.join(repo_dir, "downscalepy", "data", "argentina_raster.RData"),
        os.path.join(repo_dir, "data", "argentina_raster.RData"),
        "/storage/lopesas/downscalepy/downscalepy/data/argentina_raster.RData",
        "/home/ubuntu/temp/downscalr/data/argentina_raster.RData"
    ]
    
    rdata_file = None
    for path in rdata_paths:
        if os.path.exists(path):
            rdata_file = path
            log(f"Found RData file at: {rdata_file}")
            break
    
    if not rdata_file:
        log("WARNING: Could not find argentina_raster.RData file.")
        log("The R script will try to find it in various locations.")
    
    output_dir = os.path.join(repo_dir, "downscalepy", "data", "converted")
    os.makedirs(output_dir, exist_ok=True)
    log(f"Created output directory: {output_dir}")
    
    log("Executing R script...")
    
    try:
        os.chdir(os.path.join(repo_dir, "downscalepy", "data"))
        log(f"Changed directory to: {os.getcwd()}")
        
        process = subprocess.Popen(
            ["Rscript", r_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            log("R script executed successfully!")
        else:
            log(f"ERROR: R script failed with return code {return_code}")
            
            stderr = process.stderr.read()
            if stderr:
                log(f"Error output: {stderr}")
            
            return False
    
    except Exception as e:
        log(f"ERROR: Failed to execute R script: {e}")
        import traceback
        log(traceback.format_exc())
        return False
    
    output_file = os.path.join(output_dir, "argentina_raster.tif")
    if os.path.exists(output_file):
        log(f"SUCCESS: GeoTIFF file created at {output_file}")
        log(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        log(f"ERROR: GeoTIFF file not created at {output_file}")
        return False
    
    storage_file = "/storage/lopesas/downscalepy/downscalepy/data/converted/argentina_raster.tif"
    if os.path.exists(storage_file):
        log(f"SUCCESS: GeoTIFF file also created at {storage_file}")
        log(f"File size: {os.path.getsize(storage_file)} bytes")
    else:
        log(f"WARNING: GeoTIFF file not created at {storage_file}")
        log("You may need to manually copy the file to this location.")
    
    log("=" * 80)
    log("CONVERSION COMPLETE!")
    log("=" * 80)
    log("You can now run the Argentina example with:")
    log("python examples/argentina_example.py")
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
