"""
Optimization framework for bias correction solver.

This module provides the optimization interface using scipy.optimize,
equivalent to the nloptr functionality in the R implementation.
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from typing import Optional, Dict, Any, Callable
import logging

from .mnl_functions import sqr_diff_mnl, grad_sqr_diff_mnl

logger = logging.getLogger(__name__)


def optimize_scaling_factors(
    initial_x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    targets: np.ndarray,
    restrictions: Optional[np.ndarray] = None,
    cutoff: float = 0.0,
    algorithm: str = "L-BFGS-B",
    max_exp: float = 700.0,
    xtol_rel: float = 1e-4,
    xtol_abs: float = 1e-6,
    maxiter: int = 1000,
    use_gradient: bool = True
) -> OptimizeResult:
    """
    Optimize scaling factors to match targets using MNL model.
    
    This function wraps scipy.optimize.minimize to find optimal scaling
    factors x that minimize the squared difference between projected
    allocations and targets.
    
    Parameters
    ----------
    initial_x : np.ndarray
        Initial guess for scaling factors (shape: p)
    priors_mu : np.ndarray
        Prior probabilities matrix (shape: n × p)
    areas : np.ndarray
        Area values for each pixel (shape: n)
    targets : np.ndarray
        Target values for each class (shape: p)
    restrictions : np.ndarray, optional
        Binary restriction matrix (shape: n × p)
    cutoff : float, default=0.0
        Minimum probability threshold
    algorithm : str, default="L-BFGS-B"
        Optimization algorithm. Options:
        - "L-BFGS-B": Limited-memory BFGS with bounds (gradient-based)
        - "SLSQP": Sequential Least Squares Programming
        - "trust-constr": Trust-region constrained algorithm
        - "Powell": Powell's method (derivative-free)
        - "Nelder-Mead": Nelder-Mead simplex (derivative-free)
    max_exp : float, default=700.0
        Maximum exponent to prevent overflow
    xtol_rel : float, default=1e-4
        Relative tolerance for convergence
    xtol_abs : float, default=1e-6
        Absolute tolerance for convergence
    maxiter : int, default=1000
        Maximum number of iterations
    use_gradient : bool, default=True
        Whether to use analytical gradient (only for gradient-based methods)
    
    Returns
    -------
    OptimizeResult
        Optimization result from scipy.optimize.minimize with fields:
        - x: Optimal scaling factors
        - fun: Objective function value at optimum
        - success: Whether optimization succeeded
        - message: Description of termination
        - nit: Number of iterations
        - nfev: Number of function evaluations
    
    Notes
    -----
    The function uses bounds to constrain scaling factors to prevent
    numerical overflow: exp(-max_exp) <= x <= exp(max_exp)
    
    For gradient-based methods (L-BFGS-B, SLSQP, trust-constr), analytical
    gradients are used if use_gradient=True, which significantly speeds up
    convergence.
    """
    n_classes = len(targets)
    
    # Define bounds to prevent overflow
    lower_bound = np.exp(-max_exp)
    upper_bound = np.exp(max_exp)
    bounds = [(lower_bound, upper_bound) for _ in range(n_classes)]
    
    # Setup gradient function
    gradient_methods = ["L-BFGS-B", "SLSQP", "trust-constr", "CG", "BFGS"]
    if use_gradient and algorithm.upper() in [m.upper() for m in gradient_methods]:
        def gradient_func(x):
            return grad_sqr_diff_mnl(x, priors_mu, areas, targets, restrictions, cutoff)
        jac = gradient_func
    else:
        jac = None
    
    # Define objective function wrapper
    def objective(x):
        return sqr_diff_mnl(x, priors_mu, areas, targets, restrictions, cutoff)
    
    # Setup options
    options = {
        'maxiter': maxiter,
    }
    
    # Add method-specific options
    if algorithm.upper() == "L-BFGS-B":
        options['ftol'] = xtol_rel
        options['gtol'] = xtol_abs
    elif algorithm.upper() == "SLSQP":
        options['ftol'] = xtol_rel
    elif algorithm.upper() == "TRUST-CONSTR":
        options['xtol'] = xtol_abs
        options['gtol'] = xtol_abs
    elif algorithm.upper() in ["NELDER-MEAD", "POWELL"]:
        options['xatol'] = xtol_abs
        options['fatol'] = xtol_rel
    
    # Run optimization
    logger.debug(f"Starting optimization with {algorithm}, maxiter={maxiter}")
    logger.debug(f"Initial x: {initial_x}")
    logger.debug(f"Initial objective: {objective(initial_x):.6e}")
    
    try:
        result = minimize(
            objective,
            initial_x,
            method=algorithm,
            jac=jac,
            bounds=bounds,
            options=options
        )
        
        logger.debug(f"Optimization completed: success={result.success}")
        logger.debug(f"Final x: {result.x}")
        logger.debug(f"Final objective: {result.fun:.6e}")
        logger.debug(f"Iterations: {result.nit}, Function evals: {result.nfev}")
        logger.debug(f"Message: {result.message}")
        
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed with error: {e}")
        # Return a failed result
        return OptimizeResult(
            x=initial_x,
            fun=objective(initial_x),
            success=False,
            message=f"Optimization failed: {e}",
            nit=0,
            nfev=1
        )


def optimize_with_fallback(
    initial_x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    targets: np.ndarray,
    restrictions: Optional[np.ndarray] = None,
    cutoff: float = 0.0,
    max_diff_threshold: float = 1e-4,
    **optimization_kwargs
) -> Dict[str, Any]:
    """
    Optimize with automatic fallback to alternative methods if needed.
    
    This function tries multiple optimization strategies in sequence:
    1. L-BFGS-B with gradient (fast, gradient-based)
    2. SLSQP (alternative gradient-based)
    3. Powell (derivative-free)
    4. Returns best result
    
    Parameters
    ----------
    initial_x : np.ndarray
        Initial guess for scaling factors
    priors_mu : np.ndarray
        Prior probabilities matrix
    areas : np.ndarray
        Area values
    targets : np.ndarray
        Target values
    restrictions : np.ndarray, optional
        Restriction matrix
    cutoff : float, default=0.0
        Probability cutoff
    max_diff_threshold : float, default=1e-4
        Maximum acceptable difference for success
    **optimization_kwargs
        Additional arguments for optimize_scaling_factors
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'result': OptimizeResult from best method
        - 'method': Name of successful method
        - 'success': Whether optimization met threshold
        - 'max_diff': Maximum difference from target
    """
    methods = [
        ("L-BFGS-B", True),   # Fast gradient-based
        ("SLSQP", True),      # Alternative gradient-based
        ("Powell", False),    # Derivative-free fallback
    ]
    
    best_result = None
    best_method = None
    best_diff = np.inf
    
    for method, use_grad in methods:
        logger.info(f"Trying optimization method: {method}")
        
        result = optimize_scaling_factors(
            initial_x,
            priors_mu,
            areas,
            targets,
            restrictions,
            cutoff,
            algorithm=method,
            use_gradient=use_grad,
            **optimization_kwargs
        )
        
        # Calculate maximum difference
        from .mnl_functions import mu_mnl
        mu = mu_mnl(result.x, priors_mu, areas, restrictions, cutoff)
        allocations = mu * areas[:, np.newaxis]
        projected = np.sum(allocations, axis=0)
        max_diff = np.max(np.abs(projected - targets))
        
        logger.info(f"Method {method}: fun={result.fun:.6e}, max_diff={max_diff:.6e}, success={result.success}")
        
        # Update best if improved
        if result.fun < best_diff:
            best_result = result
            best_method = method
            best_diff = result.fun
        
        # Check if good enough
        if result.success and max_diff < max_diff_threshold:
            logger.info(f"Optimization successful with {method}")
            return {
                'result': result,
                'method': method,
                'success': True,
                'max_diff': max_diff
            }
    
    # Return best result even if not perfect
    mu = mu_mnl(best_result.x, priors_mu, areas, restrictions, cutoff)
    allocations = mu * areas[:, np.newaxis]
    projected = np.sum(allocations, axis=0)
    max_diff = np.max(np.abs(projected - targets))
    
    success = max_diff < max_diff_threshold
    
    logger.warning(
        f"Best optimization result from {best_method}: "
        f"max_diff={max_diff:.6e}, success={success}"
    )
    
    return {
        'result': best_result,
        'method': best_method,
        'success': success,
        'max_diff': max_diff
    }
