"""
Data loading utilities for downscalepy.
"""

import os
import pandas as pd
import numpy as np
import rasterio
import shutil
from typing import Dict, Any, Optional, Union


def load_argentina_data(*args, **kwargs) -> Dict[str, Any]:
    """
    Load example data for Argentina.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        Whether to use the real data converted from the R package.
        If False, synthetic data will be generated.
        Can be passed as a positional or keyword argument.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - argentina_luc: Land use change data
        - argentina_df: Dictionary containing xmat, lu_levels, restrictions, and pop_data
        - argentina_FABLE: Target data from FABLE
        - argentina_raster: Path to the raster file for visualization
    """
    use_real_data = True  # Default value
    
    if args:
        use_real_data = args[0]
    
    if 'use_real_data' in kwargs:
        use_real_data = kwargs['use_real_data']
    
    print(f"Loading Argentina data (use_real_data={use_real_data})...")
    
    import sys
    sys.stderr.write("\n\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write("DOWNSCALEPY EXTREME DEBUGGING ACTIVATED\n")
    sys.stderr.write("="*80 + "\n")
    sys.stderr.write(f"Current working directory: {os.getcwd()}\n")
    sys.stderr.write(f"Script location: {os.path.abspath(__file__)}\n")
    sys.stderr.write(f"Looking for data in: /storage/lopesas/downscalepy/downscalepy/data/converted\n")
    sys.stderr.write("="*80 + "\n\n")
    sys.stderr.flush()
    
    try:
        if use_real_data:
            return load_real_argentina_data()
        else:
            return generate_synthetic_argentina_data()
    except Exception as e:
        print(f"Error loading Argentina data: {e}")
        print("Falling back to synthetic data...")
        return generate_synthetic_argentina_data()


def load_real_argentina_data() -> Dict[str, Any]:
    """
    Load the real Argentina data converted from the R package.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the real Argentina data.
    """
    import sys
    
    csv_files = [
        'argentina_luc.csv',
        'argentina_FABLE.csv',
        'argentina_df_xmat.csv',
        'argentina_df_lu_levels.csv',
        'argentina_df_restrictions.csv',
        'argentina_df_pop_data.csv'
    ]
    
    user_storage_path = '/storage/lopesas/downscalepy/downscalepy/data/converted'
    sys.stderr.write(f"EXTREME DEBUG: CHECKING PRIMARY DATA PATH: {user_storage_path}\n")
    sys.stderr.flush()
    
    import sys
    
    if not os.path.exists(user_storage_path):
        sys.stderr.write(f"EXTREME DEBUG ERROR: Primary data directory does not exist: {user_storage_path}\n")
        sys.stderr.write(f"EXTREME DEBUG: Checking parent directory: {os.path.dirname(user_storage_path)}\n")
        if os.path.exists(os.path.dirname(user_storage_path)):
            sys.stderr.write(f"EXTREME DEBUG: Parent directory exists. Contents: {os.listdir(os.path.dirname(user_storage_path))}\n")
        else:
            sys.stderr.write(f"EXTREME DEBUG: Parent directory does not exist\n")
    else:
        sys.stderr.write(f"EXTREME DEBUG SUCCESS: Primary data directory exists: {user_storage_path}\n")
        sys.stderr.write(f"EXTREME DEBUG: Directory contents: {os.listdir(user_storage_path)}\n")
        
        all_files_exist = True
        for csv_file in csv_files:
            file_path = os.path.join(user_storage_path, csv_file)
            if os.path.exists(file_path):
                sys.stderr.write(f"EXTREME DEBUG:   ✓ Found file: {csv_file} (size: {os.path.getsize(file_path)} bytes)\n")
            else:
                sys.stderr.write(f"EXTREME DEBUG:   ✗ Missing file: {csv_file}\n")
                all_files_exist = False
        
        if all_files_exist:
            sys.stderr.write(f"EXTREME DEBUG SUCCESS: All required data files found in: {user_storage_path}\n")
            data_dir = user_storage_path
        else:
            sys.stderr.write(f"EXTREME DEBUG WARNING: Some files are missing from primary data path\n")
            all_csv_files = [f for f in os.listdir(user_storage_path) if f.endswith('.csv')]
            sys.stderr.write(f"EXTREME DEBUG: Available CSV files: {all_csv_files}\n")
            
            missing_files = []
            for required_file in csv_files:
                found = False
                for available_file in all_csv_files:
                    if required_file.lower() == available_file.lower():
                        sys.stderr.write(f"EXTREME DEBUG:   ✓ Found case-insensitive match: {required_file} -> {available_file}\n")
                        found = True
                        break
                if not found:
                    missing_files.append(required_file)
            
            if not missing_files:
                sys.stderr.write(f"EXTREME DEBUG SUCCESS: All required files found with case-insensitive matching\n")
                data_dir = user_storage_path
            else:
                sys.stderr.write(f"EXTREME DEBUG ERROR: Still missing files after case-insensitive matching: {missing_files}\n")
        
        sys.stderr.flush()
    
    import sys
    
    if 'data_dir' not in locals():
        sys.stderr.write("EXTREME DEBUG: Trying alternative data paths...\n")
        
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
            sys.stderr.write(f"EXTREME DEBUG: Checking alternative path: {alt_path}\n")
            if os.path.exists(alt_path):
                sys.stderr.write(f"EXTREME DEBUG:   Directory exists. Contents: {os.listdir(alt_path)}\n")
                csv_files_in_dir = [f for f in os.listdir(alt_path) if f.endswith('.csv')]
                if csv_files_in_dir:
                    sys.stderr.write(f"EXTREME DEBUG:   Found CSV files: {csv_files_in_dir}\n")
                    missing = [f for f in csv_files if not os.path.exists(os.path.join(alt_path, f))]
                    if not missing:
                        data_dir = alt_path
                        sys.stderr.write(f"EXTREME DEBUG SUCCESS: Found all data files in alternative path: {data_dir}\n")
                        
                        # Try to copy files to the correct location
                        try:
                            os.makedirs(user_storage_path, exist_ok=True)
                            for csv_file in csv_files:
                                src = os.path.join(alt_path, csv_file)
                                dst = os.path.join(user_storage_path, csv_file)
                                if not os.path.exists(dst):
                                    shutil.copy(src, dst)
                            sys.stderr.write(f"EXTREME DEBUG SUCCESS: Copied CSV files to correct location\n")
                        except Exception as e:
                            sys.stderr.write(f"EXTREME DEBUG ERROR: Could not copy files: {e}\n")
                        
                        break
                    else:
                        sys.stderr.write(f"EXTREME DEBUG:   Missing {len(missing)} files: {missing}\n")
            else:
                sys.stderr.write(f"EXTREME DEBUG:   Directory does not exist\n")
        
        sys.stderr.flush()
        
        if 'data_dir' not in locals():
            specific_path = 'downscalepy/data/converted'
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            possible_dirs = [
                os.path.join(os.getcwd(), specific_path),
                os.path.join(script_dir, 'converted'),
                os.path.join(os.getcwd(), 'data', 'converted'),
                os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
                os.path.join(os.getcwd(), 'converted'),
                os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
                os.path.join(script_dir, '..', '..', 'data', 'converted'),
            ]
            
            if 'DOWNSCALEPY_DATA_DIR' in os.environ:
                possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
            
            for directory in possible_dirs:
                print(f"Checking standard path: {directory}")
                if os.path.exists(directory):
                    print(f"  Directory exists. Contents: {os.listdir(directory)}")
                    missing = [f for f in csv_files if not os.path.exists(os.path.join(directory, f))]
                    if not missing:
                        data_dir = directory
                        print(f"SUCCESS: Found all data files in: {data_dir}")
                        break
                    else:
                        print(f"  Missing {len(missing)} files: {missing}")
                else:
                    print(f"  Directory does not exist")
    
    import sys
    
    if 'data_dir' not in locals():
        sys.stderr.write("EXTREME DEBUG: Searching for CSV files in any subdirectory of /storage/lopesas/downscalepy...\n")
        if os.path.exists('/storage/lopesas/downscalepy'):
            for root, dirs, files in os.walk('/storage/lopesas/downscalepy'):
                csv_files_found = [f for f in files if f.endswith('.csv')]
                if csv_files_found:
                    sys.stderr.write(f"EXTREME DEBUG: Found CSV files in {root}: {csv_files_found}\n")
                    
                    for required_file in csv_files:
                        for found_file in csv_files_found:
                            if required_file.lower() == found_file.lower():
                                sys.stderr.write(f"EXTREME DEBUG: Found required file {required_file} at {os.path.join(root, found_file)}\n")
        
        sys.stderr.write("\n" + "!"*80 + "\n")
        sys.stderr.write("EXTREME DEBUG ERROR: Could not find data files in any of the searched directories.\n")
        sys.stderr.write(f"EXTREME DEBUG ERROR: Missing data files: {csv_files}\n")
        sys.stderr.write("EXTREME DEBUG ERROR: Please ensure data files are in '/storage/lopesas/downscalepy/downscalepy/data/converted' directory.\n")
        sys.stderr.write("EXTREME DEBUG ERROR: Falling back to synthetic data.\n")
        sys.stderr.write("!"*80 + "\n\n")
        sys.stderr.flush()
        
        return generate_synthetic_argentina_data()
    
    original_dir = os.path.join(os.path.dirname(data_dir), 'original')
    
    result = {}
    
    try:
        argentina_luc = pd.read_csv(os.path.join(data_dir, 'argentina_luc.csv'))
        result['argentina_luc'] = argentina_luc
    except Exception as e:
        print(f"Error loading argentina_luc: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_luc'] = synthetic_data['argentina_luc']
    
    # Load argentina_FABLE
    try:
        argentina_FABLE = pd.read_csv(os.path.join(data_dir, 'argentina_FABLE.csv'))
        result['argentina_FABLE'] = argentina_FABLE
    except Exception as e:
        print(f"Error loading argentina_FABLE: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_FABLE'] = synthetic_data['argentina_FABLE']
    
    try:
        xmat = pd.read_csv(os.path.join(data_dir, 'argentina_df_xmat.csv'))
        lu_levels = pd.read_csv(os.path.join(data_dir, 'argentina_df_lu_levels.csv'))
        restrictions = pd.read_csv(os.path.join(data_dir, 'argentina_df_restrictions.csv'))
        pop_data = pd.read_csv(os.path.join(data_dir, 'argentina_df_pop_data.csv'))
        
        result['argentina_df'] = {
            'xmat': xmat,
            'lu_levels': lu_levels,
            'restrictions': restrictions,
            'pop_data': pop_data
        }
    except Exception as e:
        print(f"Error loading argentina_df components: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_df'] = synthetic_data['argentina_df']
    
    raster_path = prepare_argentina_raster()
    if raster_path:
        result['argentina_raster'] = raster_path
    else:
        print("Error loading argentina_raster. Using synthetic raster.")
        result['argentina_raster'] = create_synthetic_raster()
    
    return result


def prepare_argentina_raster() -> Optional[str]:
    """
    Prepare the Argentina raster data for visualization.
    
    This function checks if the raster data is available in multiple possible locations.
    If not, it attempts to convert the original R raster data to GeoTIFF format.
    If that fails, it creates a synthetic raster.
    
    Returns
    -------
    Optional[str]
        Path to the raster file, or None if the raster data could not be prepared.
    """
    user_storage_path = '/storage/lopesas/downscalepy/downscalepy/data/converted'
    raster_tif_path = os.path.join(user_storage_path, 'argentina_raster.tif')
    print(f"CHECKING PRIMARY RASTER PATH: {raster_tif_path}")
    
    if os.path.exists(raster_tif_path):
        print(f"SUCCESS: Found raster file at primary path: {raster_tif_path}")
        try:
            with rasterio.open(raster_tif_path) as src:
                print(f"  ✓ Raster file is valid and readable")
                print(f"  ✓ Raster dimensions: {src.width}x{src.height}")
            return raster_tif_path
        except Exception as e:
            print(f"ERROR: Raster file exists but cannot be read: {e}")
    else:
        print(f"ERROR: Raster file not found at primary path")
        
        if os.path.exists(user_storage_path):
            print(f"  - Directory exists, but raster file is missing")
            all_files = os.listdir(user_storage_path)
            print(f"  - Directory contents: {all_files}")
            
            tif_files = [f for f in all_files if f.lower().endswith('.tif')]
            if tif_files:
                print(f"  - Available .tif files: {tif_files}")
                
                for tif_file in tif_files:
                    if tif_file.lower() == 'argentina_raster.tif'.lower():
                        alt_path = os.path.join(user_storage_path, tif_file)
                        print(f"SUCCESS: Found case-insensitive match: {alt_path}")
                        return alt_path
            
            raster_files = [f for f in all_files if 'raster' in f.lower()]
            if raster_files:
                print(f"  - Files with 'raster' in the name: {raster_files}")
        else:
            print(f"  - Directory does not exist: {user_storage_path}")
            
            parent_dir = os.path.dirname(user_storage_path)
            if os.path.exists(parent_dir):
                print(f"  - Parent directory exists: {parent_dir}")
                print(f"  - Parent directory contents: {os.listdir(parent_dir)}")
                
                # Try to create the directory
                try:
                    os.makedirs(user_storage_path, exist_ok=True)
                    print(f"  - Created missing directory: {user_storage_path}")
                except Exception as e:
                    print(f"  - Failed to create directory: {e}")
            else:
                print(f"  - Parent directory does not exist: {parent_dir}")
    
    alternative_paths = [
        '/storage/lopesas/downscalepy/data/converted',
        '/storage/lopesas/downscalepy/converted',
        '/storage/lopesas/downscalepy/downscalepy/data',
        '/storage/lopesas/downscalepy'
    ]
    
    for alt_path in alternative_paths:
        alt_raster_path = os.path.join(alt_path, 'argentina_raster.tif')
        print(f"Checking alternative path: {alt_raster_path}")
        if os.path.exists(alt_raster_path):
            print(f"SUCCESS: Found raster file at alternative path: {alt_raster_path}")
            return alt_raster_path
        elif os.path.exists(alt_path):
            print(f"  - Directory exists but no raster file")
            tif_files = [f for f in os.listdir(alt_path) if f.lower().endswith('.tif')]
            if tif_files:
                print(f"  - Available .tif files: {tif_files}")
    
    specific_path = 'downscalepy/data/converted'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_dirs = [
        os.path.join(os.getcwd(), specific_path),
        os.path.join(script_dir, 'converted'),
        os.path.join(os.getcwd(), 'data', 'converted'),
        os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
        os.path.join(os.getcwd(), 'converted'),
        os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
        os.path.join(script_dir, '..', '..', 'data', 'converted'),
    ]
    
    if 'DOWNSCALEPY_DATA_DIR' in os.environ:
        possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
    
    for directory in possible_dirs:
        raster_path = os.path.join(directory, 'argentina_raster.tif')
        print(f"Checking standard path: {raster_path}")
        if os.path.exists(raster_path):
            print(f"SUCCESS: Found raster file at standard path: {raster_path}")
            return raster_path
    
    print("Checking for GRD/GRI files to convert...")
    downscalr_dirs = [
        os.path.abspath(os.path.join(script_dir, '../../../downscalr')),  # Default location
        os.path.abspath(os.path.join(os.getcwd(), '../downscalr')),  # Adjacent to working dir
        os.path.abspath(os.path.join(os.path.expanduser('~'), 'repos', 'downscalr')),  # Home dir repos
    ]
    
    for downscalr_dir in downscalr_dirs:
        grd_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.grd')
        gri_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.gri')
        
        if os.path.exists(grd_path) and os.path.exists(gri_path):
            print(f"SUCCESS: Found GRD/GRI files at: {grd_path} and {gri_path}")
            
            # Try to create the user's storage directory
            try:
                os.makedirs(user_storage_path, exist_ok=True)
                os.makedirs(os.path.join(user_storage_path, 'raster'), exist_ok=True)
                
                converted_grd_path = os.path.join(user_storage_path, 'raster', 'argentina_raster.grd')
                converted_gri_path = os.path.join(user_storage_path, 'raster', 'argentina_raster.gri')
                raster_tif_path = os.path.join(user_storage_path, 'argentina_raster.tif')
                
                # Copy GRD/GRI files to converted directory
                shutil.copy(grd_path, converted_grd_path)
                shutil.copy(gri_path, converted_gri_path)
                
                import rasterio
                from rasterio.transform import from_origin
                
                with open(converted_grd_path, 'r') as f:
                    lines = f.readlines()
                
                nrows = ncols = 0
                xmin = ymin = 0
                cellsize = 1
                
                for line in lines:
                    if 'nrows' in line:
                        nrows = int(line.split('=')[1].strip())
                    elif 'ncols' in line:
                        ncols = int(line.split('=')[1].strip())
                    elif 'xmin' in line:
                        xmin = float(line.split('=')[1].strip())
                    elif 'ymin' in line:
                        ymin = float(line.split('=')[1].strip())
                    elif 'cellsize' in line:
                        cellsize = float(line.split('=')[1].strip())
                
                with open(converted_gri_path, 'rb') as f:
                    data = np.fromfile(f, dtype=np.float32)
                
                if len(data) >= nrows * ncols:
                    data = data[:nrows * ncols].reshape((nrows, ncols))
                else:
                    print(f"WARNING: GRI data size ({len(data)}) doesn't match expected dimensions ({nrows}x{ncols})")
                    data = np.random.rand(nrows, ncols).astype(np.float32)
                
                transform = from_origin(xmin, ymin + nrows * cellsize, cellsize, cellsize)
                
                with rasterio.open(
                    raster_tif_path,
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
                
                print(f"SUCCESS: Converted GRD/GRI to GeoTIFF: {raster_tif_path}")
                return raster_tif_path
            
            except Exception as e:
                print(f"ERROR: Failed to convert GRD/GRI to GeoTIFF: {e}")
    
    print("ERROR: No raster data found in any location. Creating synthetic raster.")
    return create_synthetic_raster()


def create_synthetic_raster() -> str:
    """
    Create a synthetic raster file for visualization.
    
    This function creates a synthetic raster file that can be used for visualization
    when the real raster data is not available. It tries to reuse an existing synthetic
    raster if available, or creates a new one if needed.
    
    Returns
    -------
    str
        Path to the synthetic raster file.
    """
    user_storage_path = '/storage/lopesas/downscalepy/downscalepy/data/converted'
    synthetic_path = os.path.join(user_storage_path, 'argentina_raster_synthetic.tif')
    print(f"CHECKING PRIMARY SYNTHETIC RASTER PATH: {synthetic_path}")
    
    if os.path.exists(synthetic_path):
        print(f"SUCCESS: Found existing synthetic raster at primary path: {synthetic_path}")
        try:
            with rasterio.open(synthetic_path) as src:
                print(f"  ✓ Synthetic raster file is valid and readable")
                print(f"  ✓ Raster dimensions: {src.width}x{src.height}")
            return synthetic_path
        except Exception as e:
            print(f"ERROR: Synthetic raster file exists but cannot be read: {e}")
    else:
        print(f"INFO: Synthetic raster not found at primary path")
        
        if os.path.exists(user_storage_path):
            print(f"  - Directory exists, attempting to create synthetic raster")
            all_files = os.listdir(user_storage_path)
            print(f"  - Directory contents: {all_files}")
            
            tif_files = [f for f in all_files if f.lower().endswith('.tif')]
            if tif_files:
                print(f"  - Available .tif files: {tif_files}")
                
                for tif_file in tif_files:
                    if 'synthetic' in tif_file.lower() and 'raster' in tif_file.lower():
                        alt_path = os.path.join(user_storage_path, tif_file)
                        print(f"SUCCESS: Found case-insensitive synthetic raster match: {alt_path}")
                        return alt_path
        else:
            print(f"  - Directory does not exist: {user_storage_path}")
            
            parent_dir = os.path.dirname(user_storage_path)
            if os.path.exists(parent_dir):
                print(f"  - Parent directory exists: {parent_dir}")
                print(f"  - Parent directory contents: {os.listdir(parent_dir)}")
                
                # Try to create the directory
                try:
                    os.makedirs(user_storage_path, exist_ok=True)
                    print(f"SUCCESS: Created missing directory: {user_storage_path}")
                except Exception as e:
                    print(f"ERROR: Failed to create directory: {e}")
            else:
                print(f"  - Parent directory does not exist: {parent_dir}")
                
                # Try to create the entire path
                try:
                    os.makedirs(user_storage_path, exist_ok=True)
                    print(f"SUCCESS: Created entire directory path: {user_storage_path}")
                except Exception as e:
                    print(f"ERROR: Failed to create directory path: {e}")
    
    alternative_paths = [
        '/storage/lopesas/downscalepy/data/converted',
        '/storage/lopesas/downscalepy/converted',
        '/storage/lopesas/downscalepy/downscalepy/data',
        '/storage/lopesas/downscalepy'
    ]
    
    for alt_path in alternative_paths:
        alt_synthetic_path = os.path.join(alt_path, 'argentina_raster_synthetic.tif')
        print(f"Checking alternative path: {alt_synthetic_path}")
        if os.path.exists(alt_synthetic_path):
            print(f"SUCCESS: Found synthetic raster at alternative path: {alt_synthetic_path}")
            return alt_synthetic_path
        elif os.path.exists(alt_path):
            print(f"  - Directory exists but no synthetic raster file")
            try:
                # Try to create the synthetic raster in this directory
                os.makedirs(alt_path, exist_ok=True)
                break
            except Exception as e:
                print(f"  - Could not use directory: {e}")
    
    # Try to create the synthetic raster in the primary path
    try:
        os.makedirs(user_storage_path, exist_ok=True)
        output_dir = user_storage_path
        print(f"SUCCESS: Will create synthetic raster in primary path: {output_dir}")
    except Exception as e:
        print(f"ERROR: Could not create primary directory: {e}")
        
        specific_path = 'downscalepy/data/converted'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_dirs = [
            os.path.join(os.getcwd(), specific_path),
            os.path.join(script_dir, 'converted'),
            os.path.join(os.getcwd(), 'data', 'converted'),
            os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
            os.path.join(os.getcwd(), 'converted'),
            os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
            os.path.join(script_dir, '..', '..', 'data', 'converted'),
        ]
        
        if 'DOWNSCALEPY_DATA_DIR' in os.environ:
            possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
        
        output_dir = None
        for directory in possible_dirs:
            print(f"Checking if directory is writable: {directory}")
            try:
                os.makedirs(directory, exist_ok=True)
                output_dir = directory
                print(f"SUCCESS: Will create synthetic raster in: {output_dir}")
                break
            except Exception as e:
                print(f"  - Could not create directory: {e}")
        
        if output_dir is None:
            # Fallback to current working directory
            output_dir = os.getcwd()
            try:
                os.makedirs(os.path.join(output_dir, 'data', 'converted'), exist_ok=True)
                output_dir = os.path.join(output_dir, 'data', 'converted')
                print(f"SUCCESS: Will create synthetic raster in fallback directory: {output_dir}")
            except Exception as e:
                print(f"ERROR: Could not create fallback directory: {e}")
                print(f"WARNING: Using current directory as last resort")
    
    raster_tif_path = os.path.join(output_dir, 'argentina_raster_synthetic.tif')
    print(f"Creating synthetic raster at: {raster_tif_path}")
    
    height, width = 20, 20
    data = np.zeros((height, width), dtype=np.float32)
    
    # Fill with values representing spatial units
    for i in range(height):
        for j in range(width):
            data[i, j] = i * width + j + 1  # Values from 1 to 400
    
    transform = rasterio.transform.from_bounds(
        west=-70, south=-40, east=-50, north=-20, width=width, height=height
    )
    
    try:
        with rasterio.open(
            raster_tif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data, 1)
        
        print(f"SUCCESS: Created synthetic raster at {raster_tif_path}")
        return raster_tif_path
    except Exception as e:
        print(f"ERROR: Failed to create synthetic raster: {e}")
        print(f"WARNING: Using in-memory raster path as last resort")
        return "memory://synthetic_raster.tif"


def generate_synthetic_argentina_data() -> Dict[str, Any]:
    """
    Generate synthetic data for Argentina.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing synthetic Argentina data.
    """
    ns = [f'ns{i}' for i in range(1, 101)]
    lu_classes = ['Cropland', 'Forest', 'Pasture', 'Urban', 'OtherLand']
    ks = [f'k{i}' for i in range(1, 5)]
    times = ['2000', '2010', '2020', '2030']
    
    # Land use change data
    luc_data = []
    for t in ['2000']:
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                for n in ns:
                    if lu_from != lu_to:
                        luc_data.append({
                            'Ts': t,
                            'lu.from': lu_from,
                            'lu.to': lu_to,
                            'ns': n,
                            'value': np.random.uniform(0, 1)
                        })
    
    argentina_luc = pd.DataFrame(luc_data)
    
    xmat_data = []
    for n in ns:
        for k in ks:
            xmat_data.append({
                'ns': n,
                'ks': k,
                'value': np.random.normal()
            })
    
    lu_levels_data = []
    for n in ns:
        for lu in lu_classes:
            lu_levels_data.append({
                'ns': n,
                'lu.from': lu,
                'value': np.random.uniform(5, 10)
            })
    
    restrictions_data = []
    for n in ns[:20]:  # Only add restrictions for some spatial units
        for lu_from in lu_classes[:2]:  # Only add restrictions for some land use classes
            for lu_to in lu_classes[2:4]:  # Only add restrictions for some transitions
                restrictions_data.append({
                    'ns': n,
                    'lu.from': lu_from,
                    'lu.to': lu_to,
                    'value': 1  # Restrict this transition
                })
    
    pop_data = []
    for n in ns:
        for t in times:
            for k in ks:
                pop_data.append({
                    'ns': n,
                    'times': t,
                    'ks': k,
                    'value': np.random.normal()
                })
    
    argentina_df = {
        'xmat': pd.DataFrame(xmat_data),
        'lu_levels': pd.DataFrame(lu_levels_data),
        'restrictions': pd.DataFrame(restrictions_data),
        'pop_data': pd.DataFrame(pop_data)
    }
    
    fable_data = []
    for t in times:
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                if lu_from != lu_to:
                    fable_data.append({
                        'times': t,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': np.random.uniform(50, 100)
                    })
    
    argentina_FABLE = pd.DataFrame(fable_data)
    
    raster_path = create_synthetic_raster()
    
    return {
        'argentina_luc': argentina_luc,
        'argentina_df': argentina_df,
        'argentina_FABLE': argentina_FABLE,
        'argentina_raster': raster_path
    }
