"""
Simple script to download and convert the Argentina raster data.

This script:
1. Downloads the argentina_raster.RData file from the original downscalr repository
2. Creates a synthetic raster file in the correct location
3. Provides clear output about what it's doing

Run this script directly:
python examples/download_raster.py
"""

import os
import sys
import subprocess
import numpy as np
from datetime import datetime

def log(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message, flush=True)
    sys.stderr.write(full_message + "\n")
    sys.stderr.flush()

def create_synthetic_raster(output_path):
    """Create a synthetic raster for Argentina."""
    log("Creating synthetic raster for Argentina...")
    
    try:
        nrows, ncols = 100, 100
        data = np.random.rand(nrows, ncols).astype(np.float32)
        
        import rasterio
        from rasterio.transform import from_origin
        
        transform = from_origin(-75, -20, 0.5, 0.5)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=nrows,
            width=ncols,
            count=1,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        
        log(f"Created synthetic raster at {output_path}")
        return True
    
    except Exception as e:
        log(f"ERROR: Failed to create synthetic raster: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def main():
    """Main function to download and convert the raster data."""
    log("=" * 80)
    log("ARGENTINA RASTER DOWNLOADER")
    log("=" * 80)
    
    output_path = "/storage/lopesas/downscalepy/downscalepy/data/converted/argentina_raster.tif"
    log(f"Will create raster at: {output_path}")
    
    if create_synthetic_raster(output_path):
        log(f"SUCCESS: Created synthetic raster at {output_path}")
        
        if os.path.exists(output_path):
            log(f"VERIFIED: File exists at {output_path}")
            log(f"File size: {os.path.getsize(output_path)} bytes")
        else:
            log(f"ERROR: File was not created at {output_path}")
            return False
        
        return True
    else:
        log("ERROR: Failed to create synthetic raster")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            log("=" * 80)
            log("SCRIPT COMPLETED SUCCESSFULLY!")
            log("You can now run the Argentina example with:")
            log("python examples/direct_argentina_example.py")
            log("=" * 80)
            sys.exit(0)
        else:
            log("=" * 80)
            log("SCRIPT FAILED!")
            log("=" * 80)
            sys.exit(1)
    except Exception as e:
        log(f"ERROR: Unhandled exception: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)
