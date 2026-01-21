"""
Multinomial Logit (MNL) functions for bias correction.

This module implements the core mathematical functions for the MNL
bias correction algorithm, including probability calculations and
objective functions.

Based on the reference R implementation from downscalepy.
"""

import numpy as np
from typing import Optional


def mu_mnl(
    x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    restrictions: Optional[np.ndarray] = None,
    cutoff: float = 0.0
) -> np.ndarray:
    """
    Calculate MNL probabilities for land use allocation.
    
    This function computes the allocation probabilities using the
    multinomial logit formulation:
    
    λ_ij = x_j × priors_mu_ij
    μ_ij = λ_ij / (1 + Σ_j λ_ij)
    
    With optional restrictions and cutoff threshold.
    
    Parameters
    ----------
    x : np.ndarray
        Scaling factors for each target class (shape: p)
    priors_mu : np.ndarray
        Prior probabilities matrix (shape: n × p)
        n = number of pixels, p = number of target classes
    areas : np.ndarray
        Area values for each pixel (shape: n)
    restrictions : np.ndarray, optional
        Binary restriction matrix (shape: n × p)
        1 = forbidden transition, 0 = allowed
    cutoff : float, default=0.0
        Minimum probability threshold. Probabilities below cutoff are set to 0
    
    Returns
    -------
    np.ndarray
        Allocation probabilities matrix (shape: n × p)
    
    Notes
    -----
    The MNL formulation ensures that probabilities sum to less than or equal to 1
    for each pixel, allowing for "no transition" outcomes.
    """
    # Ensure x is positive (scaling factors must be non-negative)
    x = np.maximum(x, 1e-300)
    
    # Calculate λ (lambda): scaled priors
    # lambda_ij = x_j * priors_mu_ij
    lambda_vals = priors_mu * x[np.newaxis, :]
    
    # Calculate denominator: 1 + sum of lambdas across all classes
    denom = 1.0 + np.sum(lambda_vals, axis=1, keepdims=True)
    
    # Calculate MNL probabilities: mu_ij = lambda_ij / (1 + sum_j lambda_ij)
    mu = lambda_vals / denom
    
    # Apply restrictions (set forbidden transitions to 0)
    if restrictions is not None:
        mu = np.where(restrictions == 1, 0.0, mu)
    
    # Apply cutoff threshold
    if cutoff > 0:
        mu = np.where(mu <= cutoff, 0.0, mu)
    
    return mu


def sqr_diff_mnl(
    x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    targets: np.ndarray,
    restrictions: Optional[np.ndarray] = None,
    cutoff: float = 0.0
) -> float:
    """
    Objective function: squared difference between allocations and targets.
    
    This function calculates the sum of squared differences between
    the projected allocations and the target values:
    
    Σ_j (Σ_i μ_ij × areas_i - targets_j)²
    
    This is the function to be minimized by the optimizer.
    
    Parameters
    ----------
    x : np.ndarray
        Scaling factors for each target class (shape: p)
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
    
    Returns
    -------
    float
        Sum of squared differences
    
    Notes
    -----
    The optimizer aims to find scaling factors x that minimize this function,
    effectively matching the spatial allocation to the target distributions.
    """
    # Calculate MNL probabilities
    mu = mu_mnl(x, priors_mu, areas, restrictions, cutoff)
    
    # Calculate allocations: z_ij = mu_ij × areas_i
    allocations = mu * areas[:, np.newaxis]
    
    # Sum allocations by class: sum_i z_ij
    projected = np.sum(allocations, axis=0)
    
    # Calculate squared differences
    diff = projected - targets
    sqr_diff = np.sum(diff ** 2)
    
    return sqr_diff


def grad_sqr_diff_mnl(
    x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    targets: np.ndarray,
    restrictions: Optional[np.ndarray] = None,
    cutoff: float = 0.0
) -> np.ndarray:
    """
    Gradient of the objective function with respect to scaling factors x.
    
    This function computes the analytical gradient of sqr_diff_mnl,
    which can be used by gradient-based optimizers for faster convergence.
    
    The gradient is computed as:
    ∂f/∂x_j = 2 × Σ_k (projected_k - target_k) × ∂projected_k/∂x_j
    
    Parameters
    ----------
    x : np.ndarray
        Scaling factors for each target class (shape: p)
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
    
    Returns
    -------
    np.ndarray
        Gradient vector (shape: p)
    
    Notes
    -----
    Using the analytical gradient significantly speeds up optimization
    compared to numerical gradient estimation.
    """
    # Ensure x is positive
    x = np.maximum(x, 1e-300)
    
    # Calculate lambda and mu
    lambda_vals = priors_mu * x[np.newaxis, :]
    denom = 1.0 + np.sum(lambda_vals, axis=1, keepdims=True)
    mu = lambda_vals / denom
    
    # Apply restrictions
    if restrictions is not None:
        mu = np.where(restrictions == 1, 0.0, mu)
        lambda_vals = np.where(restrictions == 1, 0.0, lambda_vals)
    
    # Apply cutoff
    if cutoff > 0:
        mask = mu <= cutoff
        mu = np.where(mask, 0.0, mu)
        lambda_vals = np.where(mask, 0.0, lambda_vals)
    
    # Calculate allocations and differences
    allocations = mu * areas[:, np.newaxis]
    projected = np.sum(allocations, axis=0)
    diff = projected - targets
    
    # Calculate gradient
    # ∂μ_ij/∂x_j = priors_mu_ij / denom - lambda_ij * sum_k(priors_mu_ik) / denom^2
    n_pixels, n_classes = priors_mu.shape
    gradient = np.zeros(n_classes)
    
    for j in range(n_classes):
        # Derivative of mu with respect to x_j
        dmu_dx = np.zeros((n_pixels, n_classes))
        
        for k in range(n_classes):
            if k == j:
                # ∂μ_ij/∂x_j = priors_mu_ij / denom - lambda_ij * priors_mu_ij / denom^2
                dmu_dx[:, k] = (priors_mu[:, j] / denom.flatten() - 
                               lambda_vals[:, j] * priors_mu[:, j] / (denom.flatten() ** 2))
            else:
                # ∂μ_ik/∂x_j = -lambda_ik * priors_mu_ij / denom^2
                dmu_dx[:, k] = -lambda_vals[:, k] * priors_mu[:, j] / (denom.flatten() ** 2)
        
        # Apply restrictions and cutoff to gradient
        if restrictions is not None:
            dmu_dx = np.where(restrictions == 1, 0.0, dmu_dx)
        if cutoff > 0:
            dmu_dx = np.where(mu <= cutoff, 0.0, dmu_dx)
        
        # Calculate gradient component for x_j
        # gradient_j = 2 * sum_k(diff_k * sum_i(dmu_ik/dx_j * areas_i))
        d_allocations = dmu_dx * areas[:, np.newaxis]
        d_projected = np.sum(d_allocations, axis=0)
        gradient[j] = 2.0 * np.sum(diff * d_projected)
    
    return gradient


def validate_mnl_inputs(
    x: np.ndarray,
    priors_mu: np.ndarray,
    areas: np.ndarray,
    targets: np.ndarray,
    restrictions: Optional[np.ndarray] = None
) -> None:
    """
    Validate inputs to MNL functions.
    
    Parameters
    ----------
    x : np.ndarray
        Scaling factors
    priors_mu : np.ndarray
        Prior probabilities matrix
    areas : np.ndarray
        Area values
    targets : np.ndarray
        Target values
    restrictions : np.ndarray, optional
        Restriction matrix
    
    Raises
    ------
    ValueError
        If inputs have incompatible dimensions or invalid values
    """
    n_pixels = len(areas)
    n_classes = len(targets)
    
    if priors_mu.shape != (n_pixels, n_classes):
        raise ValueError(
            f"priors_mu shape {priors_mu.shape} incompatible with "
            f"areas ({n_pixels}) and targets ({n_classes})"
        )
    
    if len(x) != n_classes:
        raise ValueError(
            f"x length {len(x)} does not match number of classes {n_classes}"
        )
    
    if restrictions is not None:
        if restrictions.shape != (n_pixels, n_classes):
            raise ValueError(
                f"restrictions shape {restrictions.shape} incompatible with "
                f"priors_mu shape {priors_mu.shape}"
            )
        if not np.all(np.isin(restrictions, [0, 1])):
            raise ValueError("restrictions must contain only 0 and 1")
    
    if np.any(areas < 0):
        raise ValueError("areas must be non-negative")
    
    if np.any(targets < 0):
        raise ValueError("targets must be non-negative")
    
    if np.any(priors_mu < 0):
        raise ValueError("priors_mu must be non-negative")
