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
    specific_path = 'downscalepy/data/converted'
    
    if os.path.isabs(specific_path) and os.path.exists(specific_path):
        data_dir = specific_path
        print(f"Using specified absolute path: {data_dir}")
    else:
        # Try relative to current directory
        relative_path = os.path.join(os.getcwd(), specific_path)
        if os.path.exists(relative_path):
            data_dir = relative_path
            print(f"Found data in relative path: {data_dir}")
        else:
            # Try from script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            package_path = os.path.join(script_dir, 'converted')
            if os.path.exists(package_path):
                data_dir = package_path
                print(f"Found data in package path: {data_dir}")
            else:
                possible_dirs = [
                    os.path.join(os.getcwd(), 'data', 'converted'),
                    os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
                    os.path.join(os.getcwd(), 'converted'),
                    os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
                    os.path.join(script_dir, '..', '..', 'data', 'converted'),
                    '/storage/lopesas/downscalepy/downscalepy/data/converted',  # User's specific path
                    '/storage/lopesas/downscalepy/data/converted',
                ]
                
                if 'DOWNSCALEPY_DATA_DIR' in os.environ:
                    possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
                
                csv_files = [
                    'argentina_luc.csv',
                    'argentina_FABLE.csv',
                    'argentina_df_xmat.csv',
                    'argentina_df_lu_levels.csv',
                    'argentina_df_restrictions.csv',
                    'argentina_df_pop_data.csv'
                ]
                
                data_dir = None
                for directory in possible_dirs:
                    print(f"Checking directory: {directory}")
                    if os.path.exists(directory):
                        missing = [f for f in csv_files if not os.path.exists(os.path.join(directory, f))]
                        if not missing:
                            data_dir = directory
                            print(f"Found all data files in: {data_dir}")
                            break
                        else:
                            print(f"  - Missing {len(missing)} files: {missing}")
                    else:
                        print(f"  - Directory does not exist")
    
    if not data_dir:
        print("Could not find data files in any of the searched directories.")
        print("Please ensure data files are in 'downscalepy/data/converted' directory.")
        print("Falling back to synthetic data.")
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
    specific_path = 'downscalepy/data/converted'
    
    if os.path.isabs(specific_path):
        raster_tif_path = os.path.join(specific_path, 'argentina_raster.tif')
        if os.path.exists(raster_tif_path):
            print(f"Found raster file at specific absolute path: {raster_tif_path}")
            return raster_tif_path
    
    # Try relative to current directory
    relative_path = os.path.join(os.getcwd(), specific_path)
    raster_tif_path = os.path.join(relative_path, 'argentina_raster.tif')
    if os.path.exists(raster_tif_path):
        print(f"Found raster file at relative path: {raster_tif_path}")
        return raster_tif_path
    
    # Try from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_path = os.path.join(script_dir, 'converted')
    raster_tif_path = os.path.join(package_path, 'argentina_raster.tif')
    if os.path.exists(raster_tif_path):
        print(f"Found raster file in package path: {raster_tif_path}")
        return raster_tif_path
    
    possible_dirs = [
        os.path.join(os.getcwd(), 'data', 'converted'),
        os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
        os.path.join(os.getcwd(), 'converted'),
        os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
        os.path.join(script_dir, '..', '..', 'data', 'converted'),
        '/storage/lopesas/downscalepy/downscalepy/data/converted',  # User's specific path
        '/storage/lopesas/downscalepy/data/converted',
    ]
    
    if 'DOWNSCALEPY_DATA_DIR' in os.environ:
        possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
    
    for converted_dir in possible_dirs:
        print(f"Checking for raster in: {converted_dir}")
        raster_tif_path = os.path.join(converted_dir, 'argentina_raster.tif')
        if os.path.exists(raster_tif_path):
            print(f"Found raster file at: {raster_tif_path}")
            return raster_tif_path
    
    for converted_dir in possible_dirs:
        original_dir = os.path.join(os.path.dirname(converted_dir), 'original')
        
        r_raster_path = os.path.join(original_dir, 'argentina_raster.RData')
        if os.path.exists(r_raster_path):
            print(f"Found R raster data at: {r_raster_path}")
            print("R raster data conversion not implemented. Creating synthetic raster.")
            return create_synthetic_raster()
        
        downscalr_dirs = [
            os.path.abspath(os.path.join(script_dir, '../../../downscalr')),  # Default location
            os.path.abspath(os.path.join(os.getcwd(), '../downscalr')),  # Adjacent to working dir
            os.path.abspath(os.path.join(os.path.expanduser('~'), 'repos', 'downscalr')),  # Home dir repos
        ]
        
        for downscalr_dir in downscalr_dirs:
            grd_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.grd')
            gri_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.gri')
            
            if os.path.exists(grd_path) and os.path.exists(gri_path):
                print(f"Found GRD/GRI files at: {grd_path} and {gri_path}")
                
                os.makedirs(converted_dir, exist_ok=True)
                os.makedirs(os.path.join(converted_dir, 'raster'), exist_ok=True)
                
                converted_grd_path = os.path.join(converted_dir, 'raster', 'argentina_raster.grd')
                converted_gri_path = os.path.join(converted_dir, 'raster', 'argentina_raster.gri')
                raster_tif_path = os.path.join(converted_dir, 'argentina_raster.tif')
                
                try:
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
                        print(f"Warning: GRI data size ({len(data)}) doesn't match expected dimensions ({nrows}x{ncols})")
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
                    
                    print(f"Successfully converted GRD/GRI to GeoTIFF: {raster_tif_path}")
                    return raster_tif_path
                
                except Exception as e:
                    print(f"Error converting GRD/GRI to GeoTIFF: {e}")
    
    print("No raster data found in any location. Creating synthetic raster.")
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
    specific_path = 'downscalepy/data/converted'
    
    if os.path.isabs(specific_path):
        synthetic_path = os.path.join(specific_path, 'argentina_raster_synthetic.tif')
        if os.path.exists(synthetic_path):
            print(f"Using existing synthetic raster at specific absolute path: {synthetic_path}")
            return synthetic_path
    
    # Try relative to current directory
    relative_path = os.path.join(os.getcwd(), specific_path)
    synthetic_path = os.path.join(relative_path, 'argentina_raster_synthetic.tif')
    if os.path.exists(synthetic_path):
        print(f"Using existing synthetic raster at relative path: {synthetic_path}")
        return synthetic_path
    
    # Try from script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_path = os.path.join(script_dir, 'converted')
    synthetic_path = os.path.join(package_path, 'argentina_raster_synthetic.tif')
    if os.path.exists(synthetic_path):
        print(f"Using existing synthetic raster in package path: {synthetic_path}")
        return synthetic_path
    
    possible_dirs = [
        os.path.join(os.getcwd(), 'data', 'converted'),
        os.path.join(os.getcwd(), 'downscalepy', 'data', 'converted'),
        os.path.join(os.getcwd(), 'converted'),
        os.path.join(os.path.expanduser('~'), 'repos', 'downscalepy', 'downscalepy', 'data', 'converted'),
        os.path.join(script_dir, '..', '..', 'data', 'converted'),
        '/storage/lopesas/downscalepy/downscalepy/data/converted',  # User's specific path
        '/storage/lopesas/downscalepy/data/converted',
    ]
    
    if 'DOWNSCALEPY_DATA_DIR' in os.environ:
        possible_dirs.insert(0, os.environ['DOWNSCALEPY_DATA_DIR'])
    
    for directory in possible_dirs:
        print(f"Checking for synthetic raster in: {directory}")
        synthetic_path = os.path.join(directory, 'argentina_raster_synthetic.tif')
        if os.path.exists(synthetic_path):
            print(f"Using existing synthetic raster at: {synthetic_path}")
            return synthetic_path
    
    output_dir = None
    for directory in possible_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            output_dir = directory
            break
        except Exception as e:
            print(f"Could not create directory {directory}: {e}")
    
    if output_dir is None:
        # Fallback to current working directory
        output_dir = os.getcwd()
        os.makedirs(os.path.join(output_dir, 'data', 'converted'), exist_ok=True)
        output_dir = os.path.join(output_dir, 'data', 'converted')
    
    raster_tif_path = os.path.join(output_dir, 'argentina_raster_synthetic.tif')
    
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
        
        print(f"Created synthetic raster at {raster_tif_path}")
        return raster_tif_path
    except Exception as e:
        print(f"Error creating synthetic raster: {e}")
        print("Creating in-memory raster path as last resort")
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
