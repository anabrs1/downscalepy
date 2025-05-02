"""
Bias correction solver for multinomial logit type problems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize

def mu_mnl(x: np.ndarray, mu: np.ndarray, areas: np.ndarray, 
          restrictions: Optional[np.ndarray] = None, cutoff: float = 0) -> np.ndarray:
    """
    Calculate multinomial logit probabilities.
    
    Parameters
    ----------
    x : np.ndarray
        Bias correction parameters.
    mu : np.ndarray
        Base multinomial logit probabilities.
    areas : np.ndarray
        Areas for each grid cell.
    restrictions : np.ndarray, optional
        Restrictions matrix.
    cutoff : float, default=0
        Cutoff value for numerical stability.
        
    Returns
    -------
    np.ndarray
        Adjusted multinomial logit probabilities.
    """
    n, p = mu.shape
    
    mu_adj = mu.copy()
    
    for j in range(p):
        safe_exp = np.clip(x[j], -100, 100)  # Limit to reasonable range
        mu_adj[:, j] = mu[:, j] * np.exp(safe_exp)
    
    if restrictions is not None:
        mu_adj = mu_adj * (1 - restrictions)
    
    row_sums = np.sum(mu_adj, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # Ensure no zeros
    mu_adj = mu_adj / row_sums
    
    if cutoff > 0:
        mu_adj = np.maximum(mu_adj, cutoff)
        row_sums = np.sum(mu_adj, axis=1, keepdims=True)
        mu_adj = mu_adj / row_sums
    
    return mu_adj

def sqr_diff_mnl(x: np.ndarray, mu: np.ndarray, areas: np.ndarray, targets: np.ndarray,
                restrictions: Optional[np.ndarray] = None, cutoff: float = 0) -> float:
    """
    Calculate squared difference between predictions and targets.
    
    Parameters
    ----------
    x : np.ndarray
        Bias correction parameters.
    mu : np.ndarray
        Base multinomial logit probabilities.
    areas : np.ndarray
        Areas for each grid cell.
    targets : np.ndarray
        Target areas.
    restrictions : np.ndarray, optional
        Restrictions matrix.
    cutoff : float, default=0
        Cutoff value for numerical stability.
        
    Returns
    -------
    float
        Squared difference between predictions and targets.
    """
    mu_adj = mu_mnl(x, mu, areas, restrictions, cutoff)
    
    pred = np.sum(mu_adj * areas[:, np.newaxis], axis=0)
    
    return np.sum((pred - targets) ** 2)

def grad_sqr_diff_mnl(x: np.ndarray, mu: np.ndarray, areas: np.ndarray, targets: np.ndarray,
                     restrictions: Optional[np.ndarray] = None, cutoff: float = 0) -> np.ndarray:
    """
    Calculate gradient of squared difference between predictions and targets.
    
    Parameters
    ----------
    x : np.ndarray
        Bias correction parameters.
    mu : np.ndarray
        Base multinomial logit probabilities.
    areas : np.ndarray
        Areas for each grid cell.
    targets : np.ndarray
        Target areas.
    restrictions : np.ndarray, optional
        Restrictions matrix.
    cutoff : float, default=0
        Cutoff value for numerical stability.
        
    Returns
    -------
    np.ndarray
        Gradient of squared difference.
    """
    n, p = mu.shape
    grad = np.zeros(p)
    eps = 1e-8
    
    for j in range(p):
        x_plus = x.copy()
        x_plus[j] += eps
        f_plus = sqr_diff_mnl(x_plus, mu, areas, targets, restrictions, cutoff)
        
        x_minus = x.copy()
        x_minus[j] -= eps
        f_minus = sqr_diff_mnl(x_minus, mu, areas, targets, restrictions, cutoff)
        
        grad[j] = (f_plus - f_minus) / (2 * eps)
    
    return grad

def solve_biascorr_mnl(targets: pd.DataFrame, areas: pd.DataFrame, xmat: pd.DataFrame,
                     betas: pd.DataFrame, priors: Optional[pd.DataFrame] = None,
                     restrictions: Optional[pd.DataFrame] = None,
                     options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bias correction solver for multinomial logit type problems.
    
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
    priors : pd.DataFrame, optional
        A dataframe of priors (if no betas were supplied) with columns ns, lu.from, lu.to
        (with priors >= 0)
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
    
    targets_mat = np.zeros(p)
    for j, lu_to in enumerate(lu_to_list):
        mask = targets['lu.to'] == lu_to
        if 'lu.from' in targets.columns:
            mask = mask & (targets['lu.from'] == lu_from_list[0])
        targets_mat[j] = targets.loc[mask, 'value'].sum()
    
    
    if priors is not None:
        mu = np.zeros((n, p))
        for i, ns in enumerate(ns_list):
            for j, lu_to in enumerate(lu_to_list):
                mask = (priors['ns'] == ns) & (priors['lu.to'] == lu_to)
                if 'lu.from' in priors.columns:
                    mask = mask & (priors['lu.from'] == lu_from_list[0])
                if mask.any():
                    mu[i, j] = priors.loc[mask, 'value'].iloc[0]
                else:
                    mu[i, j] = 1.0 / p  # Uniform prior if not specified
    else:
        mu = np.ones((n, p)) / p  # Uniform probabilities as fallback
    
    restrictions_mat = None
    if restrictions is not None:
        restrictions_mat = np.zeros((n, p))
        for i, ns in enumerate(ns_list):
            for j, lu_to in enumerate(lu_to_list):
                mask = (restrictions['ns'] == ns) & (restrictions['lu.to'] == lu_to)
                if 'lu.from' in restrictions.columns:
                    mask = mask & (restrictions['lu.from'] == lu_from_list[0])
                if mask.any():
                    restrictions_mat[i, j] = restrictions.loc[mask, 'value'].iloc[0]
    
    x0 = np.zeros(p)
    
    result = minimize(
        sqr_diff_mnl,
        x0,
        args=(mu, areas_mat, targets_mat, restrictions_mat, options['cutoff']),
        method=options['algorithm'],
        jac=grad_sqr_diff_mnl,
        options={
            'maxiter': options['maxeval'],
            'ftol': options['xtol_rel'],
            'disp': False
        }
    )
    
    mu_final = mu_mnl(result.x, mu, areas_mat, restrictions_mat, options['cutoff'])
    
    out_res = []
    for i, ns in enumerate(ns_list):
        for j, lu_to in enumerate(lu_to_list):
            row = {
                'ns': ns,
                'lu.to': lu_to,
                'value': mu_final[i, j] * areas_mat[i]
            }
            if 'lu.from' in areas.columns:
                row['lu.from'] = lu_from_list[0]
            out_res.append(row)
    
    out_res_df = pd.DataFrame(out_res)
    
    return {
        'out_res': out_res_df,
        'out_solver': {
            'result': result,
            'x': result.x,
            'mu': mu,
            'mu_final': mu_final,
            'areas': areas_mat,
            'targets': targets_mat
        }
    }

def solve_biascorr_poisson(targets: pd.DataFrame, xmat: pd.DataFrame, betas: pd.DataFrame,
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bias correction solver for Poisson type problems.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with columns pop.type and value (all targets >= 0)
    xmat : pd.DataFrame
        A dataframe of explanatory variables with columns ns, ks and value.
    betas : pd.DataFrame
        A dataframe of coefficients with columns ks, pop.type & value
    options : Dict[str, Any], optional
        A dictionary with solver options.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - out_res: A dataframe of population allocations
        - out_solver: A dictionary of the solver output
    """
    
    if options is None:
        from ..core.downscale_pop import downscale_control_pop
        options = downscale_control_pop()
    
    ns_list = xmat['ns'].unique()
    pop_type_list = targets['pop.type'].unique() if 'pop.type' in targets.columns else ['pop']
    
    out_res = []
    for ns in ns_list:
        for pop_type in pop_type_list:
            out_res.append({
                'ns': ns,
                'pop.type': pop_type,
                'value': 1.0  # Placeholder value
            })
    
    out_res_df = pd.DataFrame(out_res)
    
    return {
        'out_res': out_res_df,
        'out_solver': {
            'status': 'placeholder'
        }
    }
