"""
Downscaling of Population Data

This module performs downscaling of population data using specified targets,
explanatory variables, and coefficients.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any

from ..solvers.solve_biascorr import solve_biascorr_poisson
from ..utils.constants import PLCHOLD_T, PLCHOLD_POPT, PLCHOLD_K


def downscale_control_pop(solve_fun: str = "solve_biascorr", algorithm: str = "SLSQP",
                         xtol_rel: float = 1.0e-20, xtol_abs: float = 1.0e-20, maxeval: int = 1600,
                         max_exp: float = np.log(np.finfo(float).max), cutoff: float = 0,
                         max_diff: float = 1.0e-8, ref_class_adjust_threshold: float = 1.0e-8,
                         err_txt: str = "") -> Dict[str, Any]:
    """
    Set options for population downscaling solver.
    
    Parameters
    ----------
    solve_fun : str, default="solve_biascorr"
        Solver function to use. Currently only "solve_biascorr" is supported.
    algorithm : str, default="SLSQP"
        Algorithm to use for optimization.
    xtol_rel : float, default=1.0e-20
        Relative tolerance for optimization.
    xtol_abs : float, default=1.0e-20
        Absolute tolerance for optimization.
    maxeval : int, default=1600
        Maximum number of function evaluations.
    max_exp : float, default=np.log(np.finfo(float).max)
        Maximum exponent for numerical stability.
    cutoff : float, default=0
        Cutoff value for numerical stability.
    max_diff : float, default=1.0e-8
        Maximum difference for convergence.
    ref_class_adjust_threshold : float, default=1.0e-8
        Threshold for reference class adjustment.
    err_txt : str, default=""
        Error text prefix.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of solver options.
    """
    return {
        "solve_fun": solve_fun,
        "algorithm": algorithm,
        "xtol_rel": xtol_rel,
        "xtol_abs": xtol_abs,
        "maxeval": maxeval,
        "MAX_EXP": max_exp,
        "cutoff": cutoff,
        "max_diff": max_diff,
        "ref_class_adjust_threshold": ref_class_adjust_threshold,
        "err_txt": err_txt
    }


def complete_pop_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the population targets dataframe.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with mandatory columns: times, value.
        Optional column: pop.type.
        
    Returns
    -------
    pd.DataFrame
        Completed targets dataframe.
    """
    required_cols = ['times', 'value']
    for col in required_cols:
        if col not in targets.columns:
            raise ValueError(f"Column '{col}' is required in targets dataframe")
    
    if (targets['value'] < 0).any():
        raise ValueError("All values in targets must be non-negative")
    
    if 'pop.type' not in targets.columns:
        targets = targets.copy()
        targets['pop.type'] = PLCHOLD_POPT
    
    return targets


def complete_pop_xmat(xmat: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the population xmat dataframe.
    
    Parameters
    ----------
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
        
    Returns
    -------
    pd.DataFrame
        Completed xmat dataframe.
    """
    required_cols = ['ns', 'ks', 'value']
    for col in required_cols:
        if col not in xmat.columns:
            raise ValueError(f"Column '{col}' is required in xmat dataframe")
    
    return xmat


def complete_pop_betas(betas: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the population betas dataframe.
    
    Parameters
    ----------
    betas : pd.DataFrame
        A dataframe with columns ks, value.
        Optional column: pop.type.
        
    Returns
    -------
    pd.DataFrame
        Completed betas dataframe.
    """
    required_cols = ['ks', 'value']
    for col in required_cols:
        if col not in betas.columns:
            raise ValueError(f"Column '{col}' is required in betas dataframe")
    
    if 'pop.type' not in betas.columns:
        betas = betas.copy()
        betas['pop.type'] = PLCHOLD_POPT
    
    return betas


def complete_pop_xmat_coltypes(xmat_coltypes: Optional[pd.DataFrame], xmat: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the population xmat.coltypes dataframe.
    
    Parameters
    ----------
    xmat_coltypes : pd.DataFrame or None
        A dataframe with columns ks and value.
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
        
    Returns
    -------
    pd.DataFrame
        Completed xmat.coltypes dataframe.
    """
    if xmat_coltypes is None:
        xmat_coltypes = pd.DataFrame({
            'ks': xmat['ks'].unique(),
            'value': 'static'
        })
    else:
        required_cols = ['ks', 'value']
        for col in required_cols:
            if col not in xmat_coltypes.columns:
                raise ValueError(f"Column '{col}' is required in xmat_coltypes dataframe")
        
        if not set(xmat_coltypes['ks']).issubset(set(xmat['ks'])):
            raise ValueError("All ks in xmat_coltypes must be in xmat")
        
        valid_values = ['static', 'dynamic', 'projected']
        if not xmat_coltypes['value'].isin(valid_values).all():
            raise ValueError(f"All values in xmat_coltypes must be one of {valid_values}")
    
    return xmat_coltypes


def complete_pop_xmat_proj(xmat_proj: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the population xmat.proj dataframe.
    
    Parameters
    ----------
    xmat_proj : pd.DataFrame
        A dataframe with columns times, ns, ks, and value.
        
    Returns
    -------
    pd.DataFrame
        Completed xmat.proj dataframe.
    """
    required_cols = ['times', 'ns', 'ks', 'value']
    for col in required_cols:
        if col not in xmat_proj.columns:
            raise ValueError(f"Column '{col}' is required in xmat_proj dataframe")
    
    return xmat_proj


def err_check_pop_inputs(targets: pd.DataFrame, xmat: pd.DataFrame, betas: pd.DataFrame,
                        xmat_coltypes: pd.DataFrame, xmat_proj: Optional[pd.DataFrame],
                        xmat_dyn_fun: Callable, err_txt: str) -> None:
    """
    Check if all population inputs are compatible.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with columns times, pop.type, and value.
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
    betas : pd.DataFrame
        A dataframe with columns ks, pop.type, and value.
    xmat_coltypes : pd.DataFrame
        A dataframe with columns ks and value.
    xmat_proj : pd.DataFrame or None
        A dataframe with columns times, ns, ks, and value.
    xmat_dyn_fun : Callable
        A function providing updates for dynamic xmat columns.
    err_txt : str
        Error text prefix.
        
    Raises
    ------
    ValueError
        If inputs are incompatible.
    """
    if 'projected' in xmat_coltypes['value'].values:
        if xmat_proj is None:
            raise ValueError(f"{err_txt} xmat_proj must be provided for projected columns")
        
        proj_cols = xmat_coltypes.loc[xmat_coltypes['value'] == 'projected', 'ks'].tolist()
        proj_cols_in_xmat_proj = xmat_proj['ks'].unique().tolist()
        
        if not set(proj_cols).issubset(set(proj_cols_in_xmat_proj)):
            raise ValueError(f"{err_txt} All projected columns must have projections in xmat_proj")
    
    if not set(betas['pop.type'].unique()).issubset(set(targets['pop.type'].unique())):
        raise ValueError(f"{err_txt} All pop.type in betas must be in targets")
    
    if not set(betas['ks'].unique()).issubset(set(xmat['ks'].unique())):
        raise ValueError(f"{err_txt} All ks in betas must be in xmat")


def downscale_pop(targets: pd.DataFrame, times: Optional[List[str]] = None,
                 xmat: Optional[pd.DataFrame] = None, betas: Optional[pd.DataFrame] = None,
                 xmat_coltypes: Optional[pd.DataFrame] = None, xmat_proj: Optional[pd.DataFrame] = None,
                 xmat_dyn_fun: Optional[Callable] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Downscaling of population data over specified time steps.

    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with mandatory columns: times (str) and value (float).
        Optional column: pop.type (str). Represents the downscaling targets for each time step.
    times : Optional[List[str]], default=None
        A list of time steps for downscaling. The first time step must be present in targets.
        If None, times are derived from unique values in targets.
    xmat : Optional[pd.DataFrame], default=None
        A dataframe with explanatory variables for econometric priors.
        Includes columns: ns (str), ks (str), and value (float).
    betas : Optional[pd.DataFrame], default=None
        A dataframe of coefficients for econometric priors.
        Includes columns: ks (str) and value (float). Optional column: pop.type (str).
    xmat_coltypes : Optional[pd.DataFrame], default=None
        A dataframe with column types for xmat.
    xmat_proj : Optional[pd.DataFrame], default=None
        A dataframe with projections.
    xmat_dyn_fun : Optional[Callable], default=None
        A function providing updates for dynamic xmat columns.
    options : Optional[Dict[str, Any]], default=None
        A dictionary with solver options.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing three elements:
        - out_res: A dataframe with columns times, ns, pop.type, and value (population allocation).
        - out_solver: A dictionary detailing the solver output.
        - ds_inputs: A dictionary documenting all the inputs used in the downscaling function.
    """
    if options is None:
        options = downscale_control_pop()
    
    err_txt = options['err_txt']
    
    targets = complete_pop_targets(targets)
    
    if xmat is None:
        raise ValueError(f"{err_txt} xmat must be provided")
    
    xmat = complete_pop_xmat(xmat)
    
    if betas is None:
        raise ValueError(f"{err_txt} betas must be provided")
    
    betas = complete_pop_betas(betas)
    xmat_coltypes = complete_pop_xmat_coltypes(xmat_coltypes, xmat)
    
    if xmat_proj is not None:
        xmat_proj = complete_pop_xmat_proj(xmat_proj)
    
    if xmat_dyn_fun is None:
        def xmat_dyn_fun(res, xmat, xmat_proj):
            return xmat.copy()
    
    err_check_pop_inputs(
        targets,
        xmat,
        betas,
        xmat_coltypes,
        xmat_proj,
        xmat_dyn_fun,
        err_txt
    )
    
    proj_colnames = None
    dyn_colnames = None
    
    if 'projected' in xmat_coltypes['value'].values:
        proj_colnames = xmat_coltypes.loc[xmat_coltypes['value'] == 'projected', 'ks'].tolist()
    
    if 'dynamic' in xmat_coltypes['value'].values:
        dyn_colnames = xmat_coltypes.loc[xmat_coltypes['value'] == 'dynamic', 'ks'].tolist()
    
    curr_xmat = xmat.copy()
    
    if times is None:
        times = targets['times'].unique().tolist()
    
    out_solver = {}
    out_res = None
    
    for curr_time in times:
        curr_targets = targets[targets['times'] == curr_time].drop(columns=['times'])
        
        if options['solve_fun'] == "solve_biascorr":
            curr_options = options.copy()
            curr_options['err_txt'] = f"{curr_time} {curr_options['err_txt']}"
            res = solve_biascorr_poisson(
                targets=curr_targets,
                xmat=curr_xmat,
                betas=betas,
                options=curr_options
            )
            out_solver[curr_time] = res['out_solver']
        else:
            raise ValueError(f"{err_txt} Unsupported solver function: {options['solve_fun']}")
        
        if proj_colnames is not None and xmat_proj is not None:
            curr_proj = xmat_proj[xmat_proj['times'] == curr_time].drop(columns=['times'])
            curr_proj = curr_proj.rename(columns={'value': 'proj'})
            
            curr_xmat = pd.merge(
                curr_xmat,
                curr_proj[curr_proj['ks'].isin(proj_colnames)],
                on=['ns', 'ks'],
                how='left'
            )
            
            curr_xmat['value'] = np.where(
                curr_xmat['proj'].notna(),
                curr_xmat['proj'],
                curr_xmat['value']
            )
            
            curr_xmat = curr_xmat.drop(columns=['proj'])
        
        if dyn_colnames is not None:
            tmp_proj = xmat_dyn_fun(res, curr_xmat, xmat_proj)
            tmp_proj = tmp_proj.rename(columns={'value': 'dyn'})
            
            curr_xmat = pd.merge(
                curr_xmat,
                tmp_proj[tmp_proj['ks'].isin(dyn_colnames)],
                on=['ns', 'ks'],
                how='left'
            )
            
            curr_xmat['value'] = np.where(
                curr_xmat['dyn'].notna(),
                curr_xmat['dyn'],
                curr_xmat['value']
            )
            
            curr_xmat = curr_xmat.drop(columns=['dyn'])
        
        res_agg = res['out_res'].copy()
        res_agg['times'] = curr_time
        
        if out_res is None:
            out_res = res_agg
        else:
            out_res = pd.concat([out_res, res_agg], ignore_index=True)
    
    if out_res is not None:
        if PLCHOLD_T in out_res['times'].values:
            out_res = out_res.drop(columns=['times'])
        
        if 'pop.type' in out_res.columns and PLCHOLD_POPT in out_res['pop.type'].values:
            out_res = out_res.drop(columns=['pop.type'])
    
    ret = {
        'out_res': out_res,
        'out_solver': out_solver,
        'ds_inputs': {
            'targets': targets,
            'xmat': xmat,
            'betas': betas,
            'xmat_coltypes': xmat_coltypes,
            'xmat_proj': xmat_proj,
            'xmat_dyn_fun': xmat_dyn_fun,
            'options': options
        }
    }
    
    return ret
