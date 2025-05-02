"""
Example script demonstrating the visualization functions in downscalepy.

This script shows how to use the luc_plot function to visualize downscaling results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from downscalepy import downscale, sim_luc, luc_plot, save_luc_plot
from downscalepy.data.load_data import load_argentina_data

def create_dummy_raster(output_path, ns_values, shape=(10, 10)):
    """
    Create a dummy raster file for visualization example.
    
    Parameters
    ----------
    output_path : str
        Path to save the raster file
    ns_values : list
        List of ns values to use in the raster
    shape : tuple, default=(10, 10)
        Shape of the raster
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    height, width = shape
    data = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if idx < len(ns_values):
                data[i, j] = ns_values[idx]
            else:
                data[i, j] = np.nan
    
    transform = rasterio.transform.from_bounds(
        west=0, south=0, east=width, north=height, width=width, height=height
    )
    
    with rasterio.open(
        output_path,
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
    
    print(f"Created dummy raster at {output_path}")
    return output_path

def run_example(use_real_data=True, save_plots=True):
    """
    Run an example of downscaling and visualization.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        Whether to use real Argentina data or synthetic data
    save_plots : bool, default=True
        Whether to save the plots to files
    
    Returns
    -------
    dict
        Dictionary containing the downscaling results and visualization objects
    """
    print(f"Running visualization example (use_real_data={use_real_data})...")
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    if use_real_data:
        data = load_argentina_data(use_real_data=True)
        
        argentina_luc = data['argentina_luc']
        argentina_df = data['argentina_df']
        argentina_FABLE = data['argentina_FABLE']
        
        ns_values = argentina_df['lu_levels']['ns'].unique()
        
        raster_path = os.path.join(output_dir, 'argentina_dummy_raster.tif')
        create_dummy_raster(raster_path, ns_values, shape=(10, 10))
        
        targets_2010 = argentina_FABLE[argentina_FABLE['times'] == '2010']
        
        result = downscale(
            targets=targets_2010,
            start_areas=argentina_df['lu_levels'],
            times=['2010']
        )
    else:
        dgp = sim_luc(100, tt=2)
        
        ns_values = dgp['start_areas']['ns'].unique()
        
        raster_path = os.path.join(output_dir, 'synthetic_dummy_raster.tif')
        create_dummy_raster(raster_path, ns_values, shape=(10, 10))
        
        result = downscale(
            targets=dgp['targets'],
            start_areas=dgp['start_areas'],
            xmat=dgp['xmat'],
            betas=dgp['betas'],
            times=['1', '2']
        )
    
    print("Downscaling complete!")
    
    print("Creating visualizations...")
    
    with rasterio.open(raster_path) as raster:
        all_plot = luc_plot(
            res=result,
            raster_file=raster,
            figsize=(12, 10)
        )
        
        if save_plots:
            save_luc_plot(
                all_plot,
                os.path.join(output_dir, 'all_landuses_times.png')
            )
        
        first_time = result['out_res']['times'].unique()[0]
        first_lu = result['out_res']['lu.to'].unique()[0]
        
        single_plot = luc_plot(
            res=result,
            raster_file=raster,
            year=first_time,
            lu=first_lu,
            cmap='Blues',
            figsize=(8, 6)
        )
        
        if save_plots:
            save_luc_plot(
                single_plot,
                os.path.join(output_dir, f'single_{first_lu}_{first_time}.png')
            )
        
        time_plot = luc_plot(
            res=result,
            raster_file=raster,
            year=first_time,
            cmap='viridis',
            figsize=(12, 6)
        )
        
        if save_plots:
            save_luc_plot(
                time_plot,
                os.path.join(output_dir, f'all_landuses_{first_time}.png')
            )
        
        lu_plot = luc_plot(
            res=result,
            raster_file=raster,
            lu=first_lu,
            cmap='plasma',
            figsize=(12, 6)
        )
        
        if save_plots:
            save_luc_plot(
                lu_plot,
                os.path.join(output_dir, f'all_times_{first_lu}.png')
            )
    
    print("Visualization complete!")
    if save_plots:
        print(f"Plots saved to {output_dir}")
    
    return {
        'result': result,
        'all_plot': all_plot,
        'single_plot': single_plot,
        'time_plot': time_plot,
        'lu_plot': lu_plot
    }

if __name__ == "__main__":
    example_result = run_example(use_real_data=True)
    
    plt.show()
