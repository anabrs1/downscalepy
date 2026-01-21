"""
Grid search optimization for bias correction.

This module implements an iterated grid search algorithm as a fallback
when standard optimization fails to converge. Based on the R implementation.
"""

import numpy as np
from typing import Callable, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def iterated_grid_search(
    min_param: float,
    max_param: float,
    func: Callable,
    max_iterations: int = 10,
    precision_threshold: float = 1e-3,
    grid_points: int = 20,
    exp_transform: bool = True,
    **func_kwargs
) -> Dict[str, Any]:
    """
    Perform iterated grid search to find optimal parameter value.
    
    This function iteratively refines a grid search to find the parameter
    value that minimizes the objective function. It's used as a fallback
    when standard optimization fails.
    
    Parameters
    ----------
    min_param : float
        Minimum parameter value (in log space if exp_transform=True)
    max_param : float
        Maximum parameter value (in log space if exp_transform=True)
    func : Callable
        Objective function to minimize. Should accept parameter as first
        positional argument followed by **func_kwargs
    max_iterations : int, default=10
        Maximum number of refinement iterations
    precision_threshold : float, default=1e-3
        Stop if parameter range is smaller than this threshold
    grid_points : int, default=20
        Number of grid points to evaluate in each iteration
    exp_transform : bool, default=True
        If True, parameter is in log space and exp transform is applied
    **func_kwargs
        Additional keyword arguments passed to the objective function
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'best_param': Optimal parameter value (transformed if exp_transform=True)
        - 'best_value': Objective function value at optimal parameter
        - 'iterations': Number of iterations performed
        - 'converged': Whether convergence criteria were met
    
    Notes
    -----
    The algorithm works by:
    1. Evaluate function on grid of points
    2. Find best point
    3. Create new, finer grid around best point
    4. Repeat until convergence or max iterations
    
    This is more robust but slower than gradient-based optimization.
    """
    current_min = min_param
    current_max = max_param
    
    best_param = None
    best_value = np.inf
    
    for iteration in range(max_iterations):
        # Create grid of parameter values
        if exp_transform:
            # Grid in log space
            log_params = np.linspace(current_min, current_max, grid_points)
            params = np.exp(log_params)
        else:
            params = np.linspace(current_min, current_max, grid_points)
        
        # Evaluate function at each grid point
        values = np.zeros(grid_points)
        for i, param in enumerate(params):
            try:
                # Call function with scalar parameter
                values[i] = func(np.array([param]), **func_kwargs)
            except Exception as e:
                logger.warning(f"Grid search evaluation failed at param={param}: {e}")
                values[i] = np.inf
        
        # Find best parameter
        best_idx = np.argmin(values)
        current_best_param = params[best_idx]
        current_best_value = values[best_idx]
        
        # Update global best
        if current_best_value < best_value:
            best_value = current_best_value
            best_param = current_best_param
        
        # Check for convergence
        param_range = current_max - current_min
        if param_range < precision_threshold:
            logger.info(f"Grid search converged after {iteration + 1} iterations")
            return {
                'best_param': best_param if not exp_transform else np.exp(current_min + param_range/2),
                'best_value': best_value,
                'iterations': iteration + 1,
                'converged': True
            }
        
        # Refine grid around best parameter
        if exp_transform:
            # Work in log space
            log_best = np.log(current_best_param)
            log_range = (current_max - current_min) / 4  # Narrow by factor of 4
            current_min = max(min_param, log_best - log_range)
            current_max = min(max_param, log_best + log_range)
        else:
            param_range = current_max - current_min
            range_reduction = param_range / 4
            current_min = max(min_param, current_best_param - range_reduction)
            current_max = min(max_param, current_best_param + range_reduction)
        
        logger.debug(
            f"Iteration {iteration + 1}: best_param={best_param:.6f}, "
            f"best_value={best_value:.6f}, range={param_range:.6f}"
        )
    
    logger.warning(
        f"Grid search did not converge after {max_iterations} iterations"
    )
    return {
        'best_param': best_param,
        'best_value': best_value,
        'iterations': max_iterations,
        'converged': False
    }


def multi_parameter_grid_search(
    param_ranges: Dict[str, Tuple[float, float]],
    func: Callable,
    max_iterations: int = 10,
    precision_threshold: float = 1e-3,
    grid_points: int = 10,
    exp_transform: Dict[str, bool] = None,
    **func_kwargs
) -> Dict[str, Any]:
    """
    Grid search for multiple parameters (extension for future use).
    
    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to (min, max) tuples
    func : Callable
        Objective function to minimize
    max_iterations : int, default=10
        Maximum number of iterations
    precision_threshold : float, default=1e-3
        Convergence threshold
    grid_points : int, default=10
        Grid points per dimension
    exp_transform : dict, optional
        Dictionary mapping parameter names to boolean indicating
        whether to use exponential transform
    **func_kwargs
        Additional arguments to objective function
    
    Returns
    -------
    dict
        Results dictionary with best parameters and value
    
    Notes
    -----
    This is a more general version that handles multiple parameters,
    but is exponentially more expensive. Use sparingly.
    """
    if exp_transform is None:
        exp_transform = {name: False for name in param_ranges}
    
    param_names = list(param_ranges.keys())
    n_params = len(param_names)
    
    # Initialize current ranges
    current_ranges = {name: param_ranges[name] for name in param_names}
    
    best_params = None
    best_value = np.inf
    
    for iteration in range(max_iterations):
        # Create grids for each parameter
        grids = []
        for name in param_names:
            min_val, max_val = current_ranges[name]
            if exp_transform.get(name, False):
                log_grid = np.linspace(min_val, max_val, grid_points)
                grid = np.exp(log_grid)
            else:
                grid = np.linspace(min_val, max_val, grid_points)
            grids.append(grid)
        
        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        grid_shape = mesh[0].shape
        
        # Flatten for evaluation
        flat_grids = [m.flatten() for m in mesh]
        n_points = len(flat_grids[0])
        
        # Evaluate function at all grid points
        values = np.zeros(n_points)
        for i in range(n_points):
            params_dict = {name: flat_grids[j][i] for j, name in enumerate(param_names)}
            try:
                values[i] = func(**params_dict, **func_kwargs)
            except Exception as e:
                logger.warning(f"Multi-param grid search failed at {params_dict}: {e}")
                values[i] = np.inf
        
        # Find best
        best_idx = np.argmin(values)
        current_best_params = {name: flat_grids[j][best_idx] for j, name in enumerate(param_names)}
        current_best_value = values[best_idx]
        
        if current_best_value < best_value:
            best_value = current_best_value
            best_params = current_best_params
        
        # Check convergence
        max_range = 0
        for name in param_names:
            min_val, max_val = current_ranges[name]
            max_range = max(max_range, max_val - min_val)
        
        if max_range < precision_threshold:
            logger.info(f"Multi-param grid search converged after {iteration + 1} iterations")
            return {
                'best_params': best_params,
                'best_value': best_value,
                'iterations': iteration + 1,
                'converged': True
            }
        
        # Refine ranges
        for name in param_names:
            min_val, max_val = current_ranges[name]
            best_val = best_params[name]
            
            if exp_transform.get(name, False):
                log_best = np.log(best_val)
                log_range = (max_val - min_val) / 4
                current_ranges[name] = (
                    max(param_ranges[name][0], log_best - log_range),
                    min(param_ranges[name][1], log_best + log_range)
                )
            else:
                param_range = max_val - min_val
                range_reduction = param_range / 4
                current_ranges[name] = (
                    max(param_ranges[name][0], best_val - range_reduction),
                    min(param_ranges[name][1], best_val + range_reduction)
                )
    
    logger.warning(f"Multi-param grid search did not converge after {max_iterations} iterations")
    return {
        'best_params': best_params,
        'best_value': best_value,
        'iterations': max_iterations,
        'converged': False
    }
