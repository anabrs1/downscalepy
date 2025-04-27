"""
Utility functions for updating explanatory variables during downscaling.
"""

import pandas as pd


def xmat_sum_to(res, curr_areas, priors, xmat, xmat_proj):
    """
    Update explanatory variables by summing over lu.to.
    
    Parameters
    ----------
    res : dict
        Result from the solver.
    curr_areas : pd.DataFrame
        Current areas dataframe.
    priors : pd.DataFrame or None
        Priors dataframe.
    xmat : pd.DataFrame
        Current explanatory variables.
    xmat_proj : pd.DataFrame or None
        Projected explanatory variables.
        
    Returns
    -------
    pd.DataFrame
        Updated explanatory variables dataframe.
    """
    return xmat.copy()


def xmat_identity(res, curr_areas, priors, xmat, xmat_proj):
    """
    Return explanatory variables unchanged.
    
    Parameters
    ----------
    res : dict
        Result from the solver.
    curr_areas : pd.DataFrame
        Current areas dataframe.
    priors : pd.DataFrame or None
        Priors dataframe.
    xmat : pd.DataFrame
        Current explanatory variables.
    xmat_proj : pd.DataFrame or None
        Projected explanatory variables.
        
    Returns
    -------
    pd.DataFrame
        Unchanged explanatory variables dataframe.
    """
    return xmat.copy()
