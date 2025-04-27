"""
Downscaling of Land-Use (Change) Data

This module performs downscaling of land-use data over specified time steps using a range
of inputs, including targets, areas, explanatory variables, and priors. It supports both
bias correction and non-targeted downscaling methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any

from ..solvers.solve_biascorr import solve_biascorr_mnl
from ..solvers.solve_notarget import solve_notarget_mnl
from ..utils.areas_update import areas_sum_to
from ..utils.xmat_update import xmat_sum_to
from ..utils.constants import PLCHOLD_T, PLCHOLD_LU, PLCHOLD_K


def complete_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the targets dataframe.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with mandatory columns: times, lu.to, and value.
        Optional column: lu.from.
        
    Returns
    -------
    pd.DataFrame
        Completed targets dataframe.
    """
    required_cols = ['times', 'lu.to', 'value']
    for col in required_cols:
        if col not in targets.columns:
            raise ValueError(f"Column '{col}' is required in targets dataframe")
    
    if (targets['value'] < 0).any():
        raise ValueError("All values in targets must be non-negative")
    
    if 'lu.from' not in targets.columns:
        targets = targets.copy()
        targets['lu.from'] = PLCHOLD_LU
    
    return targets


def complete_areas(start_areas: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the start.areas dataframe.
    
    Parameters
    ----------
    start_areas : pd.DataFrame
        A dataframe with mandatory columns: ns and value.
        Optional column: lu.from.
        
    Returns
    -------
    pd.DataFrame
        Completed start.areas dataframe.
    """
    required_cols = ['ns', 'value']
    for col in required_cols:
        if col not in start_areas.columns:
            raise ValueError(f"Column '{col}' is required in start_areas dataframe")
    
    if (start_areas['value'] < 0).any():
        raise ValueError("All values in start_areas must be non-negative")
    
    if 'lu.from' not in start_areas.columns:
        start_areas = start_areas.copy()
        start_areas['lu.from'] = PLCHOLD_LU
    
    return start_areas


def target_area_check(targets: pd.DataFrame, start_areas: pd.DataFrame) -> None:
    """
    Check if targets and areas are compatible.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with columns times, lu.to, and value.
    start_areas : pd.DataFrame
        A dataframe with columns ns, lu.from, and value.
        
    Raises
    ------
    ValueError
        If targets and areas are incompatible.
    """
    total_area = start_areas['value'].sum()
    total_target = targets['value'].sum()
    
    if total_area < total_target:
        raise ValueError(f"Sum of areas ({total_area}) is less than sum of targets ({total_target})")


def complete_xmat(xmat: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the xmat dataframe.
    
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


def complete_betas(betas: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the betas dataframe.
    
    Parameters
    ----------
    betas : pd.DataFrame
        A dataframe with columns ks, lu.to, and value.
        Optional column: lu.from.
        
    Returns
    -------
    pd.DataFrame
        Completed betas dataframe.
    """
    required_cols = ['ks', 'lu.to', 'value']
    for col in required_cols:
        if col not in betas.columns:
            raise ValueError(f"Column '{col}' is required in betas dataframe")
    
    if 'lu.from' not in betas.columns:
        betas = betas.copy()
        betas['lu.from'] = PLCHOLD_LU
    
    return betas


def complete_xmat_coltypes(xmat_coltypes: Optional[pd.DataFrame], xmat: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the xmat.coltypes dataframe.
    
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


def complete_priors(priors: pd.DataFrame, xmat: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the priors dataframe.
    
    Parameters
    ----------
    priors : pd.DataFrame
        A dataframe with columns ns, lu.to, and value.
        Optional columns: times, lu.from, weight.
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
    targets : pd.DataFrame
        A dataframe with columns times, lu.to, and value.
        
    Returns
    -------
    pd.DataFrame
        Completed priors dataframe.
    """
    required_cols = ['ns', 'lu.to', 'value']
    for col in required_cols:
        if col not in priors.columns:
            raise ValueError(f"Column '{col}' is required in priors dataframe")
    
    if (priors['value'] < 0).any():
        raise ValueError("All values in priors must be non-negative")
    
    priors = priors.copy()
    
    if 'times' not in priors.columns:
        priors['times'] = PLCHOLD_T
    
    if 'lu.from' not in priors.columns:
        priors['lu.from'] = PLCHOLD_LU
    
    if 'weight' not in priors.columns:
        priors['weight'] = 1.0
    else:
        if ((priors['weight'] < 0) | (priors['weight'] > 1)).any():
            raise ValueError("All weights in priors must be between 0 and 1")
    
    if not set(priors['ns']).issubset(set(xmat['ns'])):
        raise ValueError("All ns in priors must be in xmat")
    
    if not set(priors['lu.to']).issubset(set(targets['lu.to'])):
        raise ValueError("All lu.to in priors must be in targets")
    
    return priors


def complete_restrictions(restrictions: pd.DataFrame, xmat: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the restrictions dataframe.
    
    Parameters
    ----------
    restrictions : pd.DataFrame
        A dataframe with columns ns, lu.to, and value.
        Optional column: lu.from.
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
        
    Returns
    -------
    pd.DataFrame
        Completed restrictions dataframe.
    """
    required_cols = ['ns', 'lu.to', 'value']
    for col in required_cols:
        if col not in restrictions.columns:
            raise ValueError(f"Column '{col}' is required in restrictions dataframe")
    
    if not restrictions['value'].isin([0, 1]).all():
        raise ValueError("All values in restrictions must be 0 or 1")
    
    if 'lu.from' not in restrictions.columns:
        restrictions = restrictions.copy()
        restrictions['lu.from'] = PLCHOLD_LU
    
    if not set(restrictions['ns']).issubset(set(xmat['ns'])):
        raise ValueError("All ns in restrictions must be in xmat")
    
    return restrictions


def complete_xmat_proj(xmat_proj: pd.DataFrame) -> pd.DataFrame:
    """
    Complete and validate the xmat.proj dataframe.
    
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


def err_check_inputs(targets: pd.DataFrame, start_areas: pd.DataFrame, xmat: pd.DataFrame,
                    betas: pd.DataFrame, areas_update_fun: Callable, xmat_coltypes: pd.DataFrame,
                    xmat_proj: Optional[pd.DataFrame], xmat_dyn_fun: Callable,
                    priors: Optional[pd.DataFrame], restrictions: Optional[pd.DataFrame],
                    err_txt: str) -> None:
    """
    Check if all inputs are compatible.
    
    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with columns times, lu.to, and value.
    start_areas : pd.DataFrame
        A dataframe with columns ns, lu.from, and value.
    xmat : pd.DataFrame
        A dataframe with columns ns, ks, and value.
    betas : pd.DataFrame
        A dataframe with columns ks, lu.from, lu.to, and value.
    areas_update_fun : Callable
        A function providing an update for dynamic xmat columns.
    xmat_coltypes : pd.DataFrame
        A dataframe with columns ks and value.
    xmat_proj : pd.DataFrame or None
        A dataframe with columns times, ns, ks, and value.
    xmat_dyn_fun : Callable
        A function providing updates for dynamic xmat columns.
    priors : pd.DataFrame or None
        A dataframe with columns ns, lu.to, and value.
    restrictions : pd.DataFrame or None
        A dataframe with columns ns, lu.to, and value.
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
    
    if 'lu.from' in start_areas.columns and 'lu.from' in betas.columns:
        if not set(betas['lu.from'].unique()).issubset(set(start_areas['lu.from'].unique())):
            raise ValueError(f"{err_txt} All lu.from in betas must be in start_areas")
    
    if not set(betas['lu.to'].unique()).issubset(set(targets['lu.to'].unique())):
        raise ValueError(f"{err_txt} All lu.to in betas must be in targets")
    
    if not set(betas['ks'].unique()).issubset(set(xmat['ks'].unique())):
        raise ValueError(f"{err_txt} All ks in betas must be in xmat")


def downscale_control(solve_fun: str = "solve_biascorr", algorithm: str = "SLSQP",
                     xtol_rel: float = 1.0e-20, xtol_abs: float = 1.0e-20, maxeval: int = 1600,
                     max_exp: float = np.log(np.finfo(float).max), cutoff: float = 0,
                     max_diff: float = 1.0e-8, ref_class_adjust_threshold: float = 1.0e-8,
                     err_txt: str = "") -> Dict[str, Any]:
    """
    Set options for downscaling solver.
    
    Parameters
    ----------
    solve_fun : str, default="solve_biascorr"
        Solver function to use. One of "solve_biascorr" or "solve_notarget".
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


def downscale(targets: pd.DataFrame, start_areas: pd.DataFrame, times: Optional[List[str]] = None,
             xmat: Optional[pd.DataFrame] = None, betas: Optional[pd.DataFrame] = None,
             areas_update_fun: Callable = areas_sum_to, xmat_coltypes: Optional[pd.DataFrame] = None,
             xmat_proj: Optional[pd.DataFrame] = None, xmat_dyn_fun: Callable = xmat_sum_to,
             priors: Optional[pd.DataFrame] = None, restrictions: Optional[pd.DataFrame] = None,
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Downscaling of land-use data over specified time steps.

    Parameters
    ----------
    targets : pd.DataFrame
        A dataframe with mandatory columns: times (str), lu.to (str), and value (float).
        Optional column: lu.from (str). Represents the downscaling targets for each time step and land-use change.
    start_areas : pd.DataFrame
        A dataframe with starting areas. Includes mandatory columns: ns (str) and value (float).
        Optional column: lu.from (str).
    times : Optional[List[str]], default=None
        A list of time steps for downscaling. The first time step must be present in targets.
        If None, times are derived from unique values in targets.
    xmat : Optional[pd.DataFrame], default=None
        A dataframe with explanatory variables for econometric priors.
        Includes columns: ns (str), ks (str), and value (float).
    betas : Optional[pd.DataFrame], default=None
        A dataframe of coefficients for econometric priors.
        Includes columns: ks (str), lu.to (str), and value (float). Optional column: lu.from (str).
    areas_update_fun : Callable, default=areas_sum_to
        A function providing an update for dynamic xmat columns.
    xmat_coltypes : Optional[pd.DataFrame], default=None
        A dataframe with column types for xmat.
    xmat_proj : Optional[pd.DataFrame], default=None
        A dataframe with projections.
    xmat_dyn_fun : Callable, default=xmat_sum_to
        A function providing updates for dynamic xmat columns.
    priors : Optional[pd.DataFrame], default=None
        A dataframe with exogenous priors.
    restrictions : Optional[pd.DataFrame], default=None
        A dataframe with restrictions.
    options : Optional[Dict[str, Any]], default=None
        A dictionary with solver options.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing three elements:
        - out_res: A dataframe with columns times, ns, lu.from, lu.to, and value (area allocation).
        - out_solver: A dictionary detailing the solver output.
        - ds_inputs: A dictionary documenting all the inputs used in the downscaling function.
    """
    if options is None:
        options = downscale_control()
    
    err_txt = options['err_txt']
    
    targets = complete_targets(targets)
    start_areas = complete_areas(start_areas)
    target_area_check(targets, start_areas)
    
    if xmat is None:
        xmat = pd.DataFrame({
            'ns': start_areas['ns'].unique(),
            'ks': [PLCHOLD_K],
            'value': 0
        })
    
    xmat = complete_xmat(xmat)
    
    if betas is None:
        betas = pd.DataFrame({
            'lu.from': [PLCHOLD_LU],
            'lu.to': [PLCHOLD_LU],
            'ks': xmat['ks'].unique(),
            'value': 0
        })
    
    betas = complete_betas(betas)
    xmat_coltypes = complete_xmat_coltypes(xmat_coltypes, xmat)
    
    if priors is not None:
        priors = complete_priors(priors, xmat, targets)
    
    if restrictions is not None:
        restrictions = complete_restrictions(restrictions, xmat)
    
    if xmat_proj is not None:
        xmat_proj = complete_xmat_proj(xmat_proj)
    
    err_check_inputs(
        targets,
        start_areas,
        xmat,
        betas,
        areas_update_fun,
        xmat_coltypes,
        xmat_proj,
        xmat_dyn_fun,
        priors,
        restrictions,
        err_txt
    )
    
    proj_colnames = None
    dyn_colnames = None
    
    if 'projected' in xmat_coltypes['value'].values:
        proj_colnames = xmat_coltypes.loc[xmat_coltypes['value'] == 'projected', 'ks'].tolist()
    
    if 'dynamic' in xmat_coltypes['value'].values:
        dyn_colnames = xmat_coltypes.loc[xmat_coltypes['value'] == 'dynamic', 'ks'].tolist()
    
    curr_areas = start_areas.copy()
    curr_xmat = xmat.copy()
    curr_restrictions = restrictions.copy() if restrictions is not None else None
    
    if times is None:
        times = targets['times'].unique().tolist()
    
    out_solver = {}
    out_res = None
    
    for curr_time in times:
        curr_targets = targets[targets['times'] == curr_time].drop(columns=['times'])
        
        curr_priors = None
        if priors is not None and curr_time in priors['times'].values:
            curr_priors = priors[priors['times'] == curr_time].drop(columns=['times'])
        
        missing_luc = None
        if 'lu.from' in curr_areas.columns and 'lu.from' in curr_targets.columns:
            missing_lu_from = set(curr_areas['lu.from'].unique()) - set(curr_targets['lu.from'].unique())
            if missing_lu_from:
                missing_luc = curr_areas[curr_areas['lu.from'].isin(missing_lu_from)].copy()
                missing_luc['lu.to'] = missing_luc['lu.from']
        
        if options['solve_fun'] == "solve_biascorr":
            curr_options = options.copy()
            curr_options['err_txt'] = f"{curr_time} {curr_options['err_txt']}"
            res = solve_biascorr_mnl(
                targets=curr_targets,
                areas=curr_areas,
                xmat=curr_xmat,
                betas=betas,
                priors=curr_priors,
                restrictions=curr_restrictions,
                options=curr_options
            )
            out_solver[curr_time] = res['out_solver']
        elif options['solve_fun'] == "solve_notarget":
            curr_options = options.copy()
            curr_options['err_txt'] = f"{curr_time} {curr_options['err_txt']}"
            res = solve_notarget_mnl(
                targets=curr_targets,
                areas=curr_areas,
                xmat=curr_xmat,
                betas=betas,
                restrictions=curr_restrictions,
                options=curr_options
            )
            out_solver[curr_time] = res['out_solver']
        
        if missing_luc is not None:
            res['out_res'] = pd.concat([res['out_res'], missing_luc], ignore_index=True)
        
        curr_areas = areas_update_fun(res, curr_areas, priors, xmat_proj)
        
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
            tmp_proj = xmat_dyn_fun(res, curr_areas, priors, curr_xmat, xmat_proj)
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
        
        if 'lu.from' in out_res.columns and PLCHOLD_LU in out_res['lu.from'].values:
            out_res = out_res.drop(columns=['lu.from'])
        
        if PLCHOLD_LU in out_res['lu.to'].values:
            out_res = out_res[out_res['lu.to'] != PLCHOLD_LU]
    
    ret = {
        'out_res': out_res,
        'out_solver': out_solver,
        'ds_inputs': {
            'targets': targets,
            'start_areas': start_areas,
            'xmat': xmat,
            'betas': betas,
            'areas_update_fun': areas_update_fun,
            'xmat_coltypes': xmat_coltypes,
            'xmat_proj': xmat_proj,
            'xmat_dyn_fun': xmat_dyn_fun,
            'priors': priors,
            'restrictions': restrictions,
            'options': options
        }
    }
    
    return ret
