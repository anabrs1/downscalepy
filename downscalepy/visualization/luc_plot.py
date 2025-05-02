"""
Visualization functions for land-use change results.

This module provides functions for visualizing land-use change results
from the downscaling process, similar to the LUC_plot function in the
original R package.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import rasterio
from rasterio.plot import show
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap

def luc_plot(
    res: Dict[str, Any],
    raster_file: Union[str, rasterio.DatasetReader],
    year: Optional[str] = None,
    lu: Optional[str] = None,
    cmap: str = "Greens",
    label: str = "Area in ha per pixel",
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 100
) -> Dict[str, Any]:
    """
    Plot function to visualize downscale results.
    
    Parameters
    ----------
    res : Dict[str, Any]
        Result from downscale function
    raster_file : Union[str, rasterio.DatasetReader]
        Path to raster file or rasterio dataset
    year : Optional[str], default=None
        Specify dates of results to plot (has to be in res)
    lu : Optional[str], default=None
        Specify land-use of results to plot (has to be in res)
    cmap : str, default="Greens"
        Colormap for the plot
    label : str, default="Area in ha per pixel"
        Label for the colorbar
    figsize : Tuple[int, int], default=(12, 10)
        Figure size
    dpi : int, default=100
        DPI for the figure
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        * 'fig': The matplotlib figure object
        * 'axes': The matplotlib axes objects
        * 'plot_df': The dataframe used for plotting
    """
    if isinstance(raster_file, str):
        raster = rasterio.open(raster_file)
    else:
        raster = raster_file
    
    raster_data = raster.read(1)
    transform = raster.transform
    
    height, width = raster_data.shape
    rows, cols = np.mgrid[0:height, 0:width]
    x, y = rasterio.transform.xy(transform, rows, cols)
    
    x = np.array(x)
    y = np.array(y)
    
    plot_df = pd.DataFrame({
        'x': x.flatten(),
        'y': y.flatten(),
        'ns': raster_data.flatten()
    })
    
    plot_df = plot_df.dropna()
    
    to_plot = res['out_res']
    
    if year is None and lu is None:
        inputs = (to_plot
                 .groupby(['ns', 'lu.to', 'times'])
                 .agg({'value': 'sum'})
                 .reset_index())
    elif year is not None and lu is not None:
        inputs = (to_plot
                 .groupby(['ns', 'lu.to', 'times'])
                 .agg({'value': 'sum'})
                 .reset_index()
                 .query(f"lu.to == '{lu}' and times == '{year}'"))
    elif year is None:
        inputs = (to_plot
                 .groupby(['ns', 'lu.to', 'times'])
                 .agg({'value': 'sum'})
                 .reset_index()
                 .query(f"lu.to == '{lu}'"))
    else:
        inputs = (to_plot
                 .groupby(['ns', 'lu.to', 'times'])
                 .agg({'value': 'sum'})
                 .reset_index()
                 .query(f"times == '{year}'"))
    
    plot_df = pd.merge(plot_df, inputs, on='ns')
    
    if year is None and lu is None:
        unique_times = sorted(plot_df['times'].unique())
        unique_lu = sorted(plot_df['lu.to'].unique())
        
        fig, axes = plt.subplots(
            nrows=len(unique_times),
            ncols=len(unique_lu),
            figsize=figsize,
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        
        if len(unique_times) == 1:
            axes = axes.reshape(1, -1)
        if len(unique_lu) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, t in enumerate(unique_times):
            for j, l in enumerate(unique_lu):
                subset = plot_df[(plot_df['times'] == t) & (plot_df['lu.to'] == l)]
                
                if not subset.empty:
                    scatter = axes[i, j].scatter(
                        subset['x'],
                        subset['y'],
                        c=subset['value'],
                        cmap=cmap,
                        alpha=0.8,
                        s=10
                    )
                
                if i == 0:
                    axes[i, j].set_title(l)
                if j == 0:
                    axes[i, j].set_ylabel(t)
                
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), label=label, orientation='horizontal', pad=0.01)
        
    elif year is not None and lu is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if not plot_df.empty:
            scatter = ax.scatter(
                plot_df['x'],
                plot_df['y'],
                c=plot_df['value'],
                cmap=cmap,
                alpha=0.8,
                s=10
            )
            
            cbar = fig.colorbar(scatter, ax=ax, label=label)
            
            ax.set_title(f"{lu} - {year}")
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        axes = [ax]
        
    elif year is None:
        unique_times = sorted(plot_df['times'].unique())
        
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(unique_times),
            figsize=figsize,
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        
        if len(unique_times) == 1:
            axes = [axes]
        
        for i, t in enumerate(unique_times):
            subset = plot_df[plot_df['times'] == t]
            
            if not subset.empty:
                scatter = axes[i].scatter(
                    subset['x'],
                    subset['y'],
                    c=subset['value'],
                    cmap=cmap,
                    alpha=0.8,
                    s=10
                )
            
            axes[i].set_title(f"{lu} - {t}")
            
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        cbar = fig.colorbar(scatter, ax=axes, label=label, orientation='horizontal', pad=0.01)
        
    else:
        unique_lu = sorted(plot_df['lu.to'].unique())
        
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(unique_lu),
            figsize=figsize,
            dpi=dpi,
            sharex=True,
            sharey=True
        )
        
        if len(unique_lu) == 1:
            axes = [axes]
        
        for i, l in enumerate(unique_lu):
            subset = plot_df[plot_df['lu.to'] == l]
            
            if not subset.empty:
                scatter = axes[i].scatter(
                    subset['x'],
                    subset['y'],
                    c=subset['value'],
                    cmap=cmap,
                    alpha=0.8,
                    s=10
                )
            
            axes[i].set_title(f"{l} - {year}")
            
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        cbar = fig.colorbar(scatter, ax=axes, label=label, orientation='horizontal', pad=0.01)
    
    plt.tight_layout()
    
    return {
        'fig': fig,
        'axes': axes,
        'plot_df': plot_df
    }

def save_luc_plot(
    plot_result: Dict[str, Any],
    filename: str,
    dpi: int = 300
) -> None:
    """
    Save the LUC plot to a file.
    
    Parameters
    ----------
    plot_result : Dict[str, Any]
        Result from luc_plot function
    filename : str
        Filename to save the plot to
    dpi : int, default=300
        DPI for the saved figure
    """
    plot_result['fig'].savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {filename}")
