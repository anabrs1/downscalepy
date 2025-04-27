"""
Solver for multinomial logit type problems using only prior module projections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

def solve_notarget_mnl(targets: pd.DataFrame, areas: pd.DataFrame, xmat: pd.DataFrame,
                     betas: pd.DataFrame, restrictions: Optional[pd.DataFrame] = None,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Solver for multinomial logit type problems using only prior module projections.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with columns lu.from, lu.to and value (all targets >= 0)
    areas : pd.DataFrame
        A dataframe of areas with columns lu.from, ns and value, with all areas >= 0
        and with sum(areas) >= sum(targets)
    xmat : pd.DataFrame
        A dataframe of explanatory variables with columns ns, ks and value.
    betas : pd.DataFrame
        A dataframe of coefficients with columns ks, lu.from, lu.to & value
    restrictions : pd.DataFrame, optional
        A dataframe with columns ns, lu.from, lu.to and value. Values must be zero or one.
        If restrictions are one, the MNL function is set to zero.
    options : Dict[str, Any], optional
        A dictionary with solver options.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - out_res: A dataframe of area allocations
        - out_solver: A dictionary of the solver output
    """
    if options is None:
        from ..core.downscale import downscale_control
        options = downscale_control()
    
    ns_list = areas['ns'].unique()
    lu_from_list = areas['lu.from'].unique() if 'lu.from' in areas.columns else [None]
    lu_to_list = targets['lu.to'].unique()
    
    n = len(ns_list)
    p = len(lu_to_list)
    
    
    areas_mat = np.zeros(n)
    for i, ns in enumerate(ns_list):
        mask = areas['ns'] == ns
        if 'lu.from' in areas.columns:
            mask = mask & (areas['lu.from'] == lu_from_list[0])
        areas_mat[i] = areas.loc[mask, 'value'].sum()
    
    
    mu = np.ones((n, p)) / p  # Uniform probabilities as fallback
    
    if restrictions is not None:
        restrictions_mat = np.zeros((n, p))
        for i, ns in enumerate(ns_list):
            for j, lu_to in enumerate(lu_to_list):
                mask = (restrictions['ns'] == ns) & (restrictions['lu.to'] == lu_to)
                if 'lu.from' in restrictions.columns:
                    mask = mask & (restrictions['lu.from'] == lu_from_list[0])
                if mask.any():
                    restrictions_mat[i, j] = restrictions.loc[mask, 'value'].iloc[0]
        
        mu = mu * (1 - restrictions_mat)
        
        row_sums = np.sum(mu, axis=1, keepdims=True)
        mu = mu / row_sums
    
    if options['cutoff'] > 0:
        mu = np.maximum(mu, options['cutoff'])
        row_sums = np.sum(mu, axis=1, keepdims=True)
        mu = mu / row_sums
    
    allocations = mu * areas_mat[:, np.newaxis]
    
    out_res = []
    for i, ns in enumerate(ns_list):
        for j, lu_to in enumerate(lu_to_list):
            row = {
                'ns': ns,
                'lu.to': lu_to,
                'value': allocations[i, j]
            }
            if 'lu.from' in areas.columns:
                row['lu.from'] = lu_from_list[0]
            out_res.append(row)
    
    out_res_df = pd.DataFrame(out_res)
    
    return {
        'out_res': out_res_df,
        'out_solver': {
            'mu': mu,
            'areas': areas_mat,
            'allocations': allocations
        }
    }
