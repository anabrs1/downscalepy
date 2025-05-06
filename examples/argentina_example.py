"""
Example of downscaling land-use change projections in Argentina using downscalepy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from downscalepy import downscale, mnlogit, luc_plot, save_luc_plot
from downscalepy.data.load_data import load_argentina_data


def run_example(use_real_data=True, debug=True):
    """
    Run the Argentina example.
    
    This example demonstrates how to use downscalepy to downscale land-use change
    projections in Argentina using the FABLE model.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        Whether to use the real data converted from the R package.
        If False, synthetic data will be generated.
    debug : bool, default=True
        Whether to print debug information.
        
    Returns
    -------
    dict
        The result of the downscaling process.
    """
    print(f"Loading Argentina data (use_real_data={use_real_data})...")
    data = load_argentina_data(use_real_data=use_real_data)
    
    argentina_luc = data['argentina_luc']
    argentina_df = data['argentina_df']
    argentina_FABLE = data['argentina_FABLE']
    
    if debug:
        print("\nData summary:")
        print(f"argentina_luc shape: {argentina_luc.shape}")
        print(f"argentina_luc columns: {argentina_luc.columns.tolist()}")
        print(f"argentina_df['xmat'] shape: {argentina_df['xmat'].shape}")
        print(f"argentina_df['lu_levels'] shape: {argentina_df['lu_levels'].shape}")
        print(f"argentina_FABLE shape: {argentina_FABLE.shape}")
    
    required_cols = {
        'argentina_luc': ['ns', 'lu.from', 'lu.to', 'Ts', 'value'],
        'argentina_df_xmat': ['ns', 'ks', 'value'],
        'argentina_df_lu_levels': ['ns', 'lu.from', 'value'],
        'argentina_FABLE': ['times', 'lu.from', 'lu.to', 'value']
    }
    
    for dataset, cols in required_cols.items():
        if dataset == 'argentina_luc':
            missing = [col for col in cols if col not in argentina_luc.columns]
            if missing:
                raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_df_xmat':
            missing = [col for col in cols if col not in argentina_df['xmat'].columns]
            if missing:
                raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_df_lu_levels':
            missing = [col for col in cols if col not in argentina_df['lu_levels'].columns]
            if missing:
                raise ValueError(f"Missing columns in {dataset}: {missing}")
        elif dataset == 'argentina_FABLE':
            missing = [col for col in cols if col not in argentina_FABLE.columns]
            if missing:
                raise ValueError(f"Missing columns in {dataset}: {missing}")
    
    print("\nEstimating coefficients using mnlogit...")
    betas = []
    for lu_from in argentina_luc['lu.from'].unique():
        if debug:
            print(f"Processing lu.from={lu_from}")
        
        mask = (argentina_luc['lu.from'] == lu_from) & ((argentina_luc['Ts'] == '2000') | (argentina_luc['Ts'] == 2000))
        if mask.sum() == 0:
            print(f"No data for lu.from={lu_from}, Ts=2000. Skipping.")
            continue
            
        Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
        
        X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
        X_data = X_data.reindex(Y_data.index)
        
        if X_data.isna().any().any() or Y_data.isna().any().any():
            print(f"Warning: Missing values in data for lu.from={lu_from}. Filling with zeros.")
            X_data = X_data.fillna(0)
            Y_data = Y_data.fillna(0)
        
        baseline = Y_data.columns.get_loc(lu_from) if lu_from in Y_data.columns else None
        
        if debug:
            print(f"  X_data shape: {X_data.shape}")
            print(f"  Y_data shape: {Y_data.shape}")
            print(f"  baseline: {baseline}")
        
        res = mnlogit(
            X=X_data.values,
            Y=Y_data.values,
            baseline=baseline,
            niter=3,
            nburn=2
        )
        
        beta_mean = np.mean(res['postb'], axis=2)
        
        for k_idx, k in enumerate(X_data.columns):
            for lu_idx, lu_to in enumerate(Y_data.columns):
                betas.append({
                    'ks': k,
                    'lu.from': lu_from,
                    'lu.to': lu_to,
                    'value': beta_mean[k_idx, lu_idx]
                })
    
    betas_df = pd.DataFrame(betas)
    
    if debug:
        print(f"\nbetas_df shape: {betas_df.shape}")
        print(f"betas_df columns: {betas_df.columns.tolist() if not betas_df.empty else []}")
    
    if len(betas_df) == 0:
        print("No betas were generated from real data. Using synthetic data instead.")
        return run_example(use_real_data=False, debug=debug)
    
    ns_list = argentina_df['lu_levels']['ns'].unique()
    priors = pd.DataFrame({
        'ns': ns_list,
        'lu.from': 'Cropland',
        'lu.to': 'Forest',
        'value': np.random.uniform(0, 1, size=len(ns_list))
    })
    
    targets_2010 = argentina_FABLE[(argentina_FABLE['times'] == '2010') | (argentina_FABLE['times'] == 2010)]
    if len(targets_2010) == 0:
        print("Warning: No targets for 2010. Using first available time period.")
        first_time = argentina_FABLE['times'].iloc[0]
        targets_2010 = argentina_FABLE[argentina_FABLE['times'] == first_time]
    
    filtered_betas = betas_df[
        ~((betas_df['lu.from'] == 'Cropland') & (betas_df['lu.to'] == 'Forest'))
    ]
    
    print("\nRunning downscaling...")
    result = downscale(
        targets=targets_2010,
        start_areas=argentina_df['lu_levels'],
        xmat=argentina_df['xmat'],
        betas=filtered_betas,
        priors=priors
    )
    
    print("Downscaling complete!")
    if 'out_res' in result:
        print(f"Results shape: {result['out_res'].shape}")
        
        raster_path = data.get('argentina_raster')
        if raster_path:
            print(f"Using raster data from: {raster_path}")
        else:
            print("No raster data available for visualization.")
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_results(result, argentina_df['lu_levels'], raster_path, output_dir)
    else:
        print("Warning: No results returned from downscaling.")
    
    return result


def plot_results(result, start_areas, raster_path=None, output_dir=None):
    """
    Plot the results of the downscaling process.
    
    Parameters
    ----------
    result : dict
        The result of the downscaling process.
    start_areas : pd.DataFrame
        The starting areas.
    raster_path : str, optional
        Path to the raster file for visualization.
    output_dir : str, optional
        Directory to save the output plots.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    out_res = result['out_res']
    
    start_totals = start_areas.groupby('lu.from')['value'].sum().reset_index()
    start_totals = start_totals.rename(columns={'lu.from': 'lu', 'value': 'start_value'})
    
    end_totals = out_res.groupby('lu.to')['value'].sum().reset_index()
    end_totals = end_totals.rename(columns={'lu.to': 'lu', 'value': 'end_value'})
    
    totals = pd.merge(start_totals, end_totals, on='lu', how='outer').fillna(0)
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(totals))
    width = 0.35
    
    plt.bar(x - width/2, totals['start_value'], width, label='Starting Areas')
    plt.bar(x + width/2, totals['end_value'], width, label='Downscaled Areas')
    
    plt.xlabel('Land-Use Class')
    plt.ylabel('Total Area')
    plt.title('Comparison of Starting and Downscaled Areas')
    plt.xticks(x, totals['lu'])
    plt.legend()
    
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'argentina_bar_chart.png')
    plt.savefig(bar_chart_path)
    plt.close()
    
    print(f"Bar chart saved to '{bar_chart_path}'")
    
    if raster_path:
        print(f"Creating raster visualizations using '{raster_path}'...")
        
        try:
            with rasterio.open(raster_path) as raster:
                all_plot = luc_plot(
                    res=result,
                    raster_file=raster,
                    figsize=(12, 10)
                )
                
                all_plot_path = os.path.join(output_dir, 'argentina_all_landuses_times.png')
                save_luc_plot(all_plot, all_plot_path)
                
                first_time = out_res['times'].unique()[0]
                cropland_to_forest = out_res[
                    (out_res['times'] == first_time) & 
                    (out_res['lu.from'] == 'Cropland') & 
                    (out_res['lu.to'] == 'Forest')
                ]
                
                if not cropland_to_forest.empty:
                    cropland_plot = luc_plot(
                        res=result,
                        raster_file=raster,
                        year=first_time,
                        lu='Forest',
                        cmap='Greens',
                        figsize=(8, 6)
                    )
                    
                    cropland_plot_path = os.path.join(output_dir, f'argentina_cropland_to_forest_{first_time}.png')
                    save_luc_plot(cropland_plot, cropland_plot_path)
                
                time_plot = luc_plot(
                    res=result,
                    raster_file=raster,
                    year=first_time,
                    cmap='viridis',
                    figsize=(12, 6)
                )
                
                time_plot_path = os.path.join(output_dir, f'argentina_all_landuses_{first_time}.png')
                save_luc_plot(time_plot, time_plot_path)
                
                print(f"Raster visualizations saved to '{output_dir}'")
        except Exception as e:
            print(f"Error creating raster visualizations: {e}")
    else:
        print("No raster path provided. Skipping raster visualizations.")


if __name__ == "__main__":
    run_example()
