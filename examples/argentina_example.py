"""
Example of downscaling land-use change projections in Argentina using downscalepy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from downscalepy import downscale, mnlogit
from downscalepy.data.load_data import load_argentina_data


def run_example():
    """
    Run the Argentina example.
    
    This example demonstrates how to use downscalepy to downscale land-use change
    projections in Argentina using the FABLE model.
    
    Returns
    -------
    dict
        The result of the downscaling process.
    """
    data = load_argentina_data()
    
    argentina_luc = data['argentina_luc']
    argentina_df = data['argentina_df']
    argentina_FABLE = data['argentina_FABLE']
    
    betas = []
    for lu_from in argentina_luc['lu.from'].unique():
        mask = (argentina_luc['lu.from'] == lu_from) & (argentina_luc['Ts'] == '2000')
        Y_data = argentina_luc[mask].pivot(index='ns', columns='lu.to', values='value')
        
        X_data = argentina_df['xmat'].pivot(index='ns', columns='ks', values='value')
        X_data = X_data.reindex(Y_data.index)
        
        baseline = Y_data.columns.get_loc(lu_from) if lu_from in Y_data.columns else None
        
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
    
    ns_list = argentina_df['lu_levels']['ns'].unique()
    priors = pd.DataFrame({
        'ns': ns_list,
        'lu.from': 'Cropland',
        'lu.to': 'Forest',
        'value': np.random.uniform(0, 1, size=len(ns_list))
    })
    
    targets_2010 = argentina_FABLE[argentina_FABLE['times'] == '2010']
    
    filtered_betas = betas_df[
        ~((betas_df['lu.from'] == 'Cropland') & (betas_df['lu.to'] == 'Forest'))
    ]
    
    result = downscale(
        targets=targets_2010,
        start_areas=argentina_df['lu_levels'],
        xmat=argentina_df['xmat'],
        betas=filtered_betas,
        priors=priors
    )
    
    print("Downscaling complete!")
    print(f"Results shape: {result['out_res'].shape}")
    
    plot_results(result, argentina_df['lu_levels'])
    
    return result


def plot_results(result, start_areas):
    """
    Plot the results of the downscaling process.
    
    Parameters
    ----------
    result : dict
        The result of the downscaling process.
    start_areas : pd.DataFrame
        The starting areas.
    """
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
    plt.savefig('argentina_example_results.png')
    plt.close()
    
    print("Results plot saved to 'argentina_example_results.png'")


if __name__ == "__main__":
    run_example()
