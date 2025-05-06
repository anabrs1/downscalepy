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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converted_dir = os.path.join(script_dir, 'converted')
    original_dir = os.path.join(script_dir, 'original')
    
    csv_files = [
        'argentina_luc.csv',
        'argentina_FABLE.csv',
        'argentina_df_xmat.csv',
        'argentina_df_lu_levels.csv',
        'argentina_df_restrictions.csv',
        'argentina_df_pop_data.csv'
    ]
    
    missing_files = [f for f in csv_files if not os.path.exists(os.path.join(converted_dir, f))]
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print("Falling back to synthetic data.")
        return generate_synthetic_argentina_data()
    
    result = {}
    
    try:
        argentina_luc = pd.read_csv(os.path.join(converted_dir, 'argentina_luc.csv'))
        result['argentina_luc'] = argentina_luc
    except Exception as e:
        print(f"Error loading argentina_luc: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_luc'] = synthetic_data['argentina_luc']
    
    # Load argentina_FABLE
    try:
        argentina_FABLE = pd.read_csv(os.path.join(converted_dir, 'argentina_FABLE.csv'))
        result['argentina_FABLE'] = argentina_FABLE
    except Exception as e:
        print(f"Error loading argentina_FABLE: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_FABLE'] = synthetic_data['argentina_FABLE']
    
    try:
        xmat = pd.read_csv(os.path.join(converted_dir, 'argentina_df_xmat.csv'))
        lu_levels = pd.read_csv(os.path.join(converted_dir, 'argentina_df_lu_levels.csv'))
        restrictions = pd.read_csv(os.path.join(converted_dir, 'argentina_df_restrictions.csv'))
        pop_data = pd.read_csv(os.path.join(converted_dir, 'argentina_df_pop_data.csv'))
        
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
    
    This function checks if the raster data is available in the converted directory.
    If not, it attempts to convert the original R raster data to GeoTIFF format.
    
    Returns
    -------
    Optional[str]
        Path to the raster file, or None if the raster data could not be prepared.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converted_dir = os.path.join(script_dir, 'converted')
    original_dir = os.path.join(script_dir, 'original')
    
    raster_tif_path = os.path.join(converted_dir, 'argentina_raster.tif')
    if os.path.exists(raster_tif_path):
        return raster_tif_path
    
    r_raster_path = os.path.join(original_dir, 'argentina_raster.RData')
    if not os.path.exists(r_raster_path):
        downscalr_dir = os.path.abspath(os.path.join(script_dir, '../../../downscalr'))
        grd_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.grd')
        gri_path = os.path.join(downscalr_dir, 'inst/extdata/argentina_raster.gri')
        
        if os.path.exists(grd_path) and os.path.exists(gri_path):
            os.makedirs(os.path.join(converted_dir, 'raster'), exist_ok=True)
            converted_grd_path = os.path.join(converted_dir, 'raster', 'argentina_raster.grd')
            converted_gri_path = os.path.join(converted_dir, 'raster', 'argentina_raster.gri')
            
            shutil.copy(grd_path, converted_grd_path)
            shutil.copy(gri_path, converted_gri_path)
            
            try:
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
                
                return raster_tif_path
            
            except Exception as e:
                print(f"Error converting GRD/GRI to GeoTIFF: {e}")
                return create_synthetic_raster()
        else:
            print("Original raster data not found. Creating synthetic raster.")
            return create_synthetic_raster()
    
    try:
        print("R raster data found but conversion not implemented. Creating synthetic raster.")
        return create_synthetic_raster()
    except Exception as e:
        print(f"Error converting R raster data: {e}")
        return create_synthetic_raster()


def create_synthetic_raster() -> str:
    """
    Create a synthetic raster file for visualization.
    
    Returns
    -------
    str
        Path to the synthetic raster file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converted_dir = os.path.join(script_dir, 'converted')
    os.makedirs(converted_dir, exist_ok=True)
    
    raster_tif_path = os.path.join(converted_dir, 'argentina_raster_synthetic.tif')
    
    height, width = 10, 10
    data = np.zeros((height, width), dtype=np.float32)
    
    # Fill with random values representing spatial units
    for i in range(height):
        for j in range(width):
            data[i, j] = i * width + j + 1  # Values from 1 to 100
    
    transform = rasterio.transform.from_bounds(
        west=0, south=0, east=width, north=height, width=width, height=height
    )
    
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
