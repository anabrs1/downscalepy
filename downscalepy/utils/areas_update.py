"""
Utility functions for updating areas during downscaling.
"""

import pandas as pd


def areas_sum_to(res, curr_areas, priors, xmat_proj):
    """
    Update areas by summing over lu.to.
    
    Parameters
    ----------
    res : dict
        Result from the solver.
    curr_areas : pd.DataFrame
        Current areas dataframe.
    priors : pd.DataFrame or None
        Priors dataframe.
    xmat_proj : pd.DataFrame or None
        Projected explanatory variables.
        
    Returns
    -------
    pd.DataFrame
        Updated areas dataframe.
    """
    if 'out_res' not in res:
        return curr_areas
    
    new_areas = res['out_res'].groupby(['ns', 'lu.from'])['value'].sum().reset_index()
    
    return new_areas
