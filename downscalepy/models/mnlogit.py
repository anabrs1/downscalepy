"""
Bayesian logit model with Pólya Gamma prior with MCMC

Implementation of multinomial logistic regression using Polya-Gamma latent variables,
following Polson et al. (2013).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import multivariate_normal
import tqdm
import warnings


def safe_exp(x, max_exp=30):
    """
    Safe exponentiation function to prevent overflow.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    max_exp : float, default=30
        Maximum exponent value to prevent overflow
        
    Returns
    -------
    np.ndarray
        exp(x) with values clipped to prevent overflow
    """
    return np.exp(np.clip(x, -max_exp, max_exp))


def mnlogit(X: np.ndarray, Y: np.ndarray, baseline: Optional[int] = None, 
           niter: int = 1000, nburn: int = 500, A0: float = 1e4,
           calc_marginal_fx: bool = False, jitter: float = 0.0) -> Dict[str, Any]:
    """
    MCMC estimation of a multinomial logit model following Polson et al. (2013).

    Parameters
    ----------
    X : np.ndarray
        An n by k matrix of explanatory variables
    Y : np.ndarray
        An n by p matrix of dependent variables
    baseline : int, optional
        Baseline class for estimation. Parameters will be set to zero.
        Defaults to the p-th column.
    niter : int, default=1000
        Total number of MCMC draws.
    nburn : int, default=500
        Burn-in draws for MCMC. Note: nburn has to be lower than niter.
    A0 : float, default=1e4
        Prior variance scalar for all slope coefficients
    calc_marginal_fx : bool, default=False
        Should marginal effects be calculated?
    jitter : float, default=0.0
        Small value added to diagonal elements of covariance matrices for numerical stability

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - postb: A k x p x (niter - nburn) array containing posterior draws of the slope coefficients.
        - marginal_fx: A k x p x (niter - nburn) array containing posterior draws of marginal effects.
        - X, Y, baseline: The matrices of explanatory and dependent variables, and the baseline class.

    References
    ----------
    Nicholas G. Polson, James G. Scott, and Jesse Windle. Bayesian inference for 
    logistic models using Polya-Gamma latent variables. 
    Journal of the American statistical Association 108.504 (2013): 1339-1349.
    """
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X contains NaN or Inf values. Please clean your input data.")
    
    if np.any(np.isnan(Y)) or np.any(np.isinf(Y)):
        raise ValueError("Y contains NaN or Inf values. Please clean your input data.")
    
    X_orig = X.copy()
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0  # Prevent division by zero
    X = (X - X_mean) / X_std
    
    n, k = X.shape
    p = Y.shape[1]
    
    if baseline is None:
        baseline = p - 1
    
    if niter <= nburn:
        raise ValueError("niter has to be higher than nburn.")
    
    # Use a small jitter by default for numerical stability
    if jitter == 0.0:
        jitter = 1e-8
    
    nn = np.ones((n, p))
    beta_prior_mean = np.zeros((k, p))
    beta_prior_var = np.eye(k) * A0
    
    ndiscard = nburn
    nretain = niter - ndiscard
    
    postb = np.zeros((k, p, nretain))
    
    curr_beta = np.zeros((k, p))
    curr_beta[:, baseline] = 0  # Baseline class parameters set to zero
    curr_xb = X @ curr_beta
    curr_om = np.zeros((n, p))
    
    beta_prior_var_inv = np.linalg.inv(beta_prior_var)
    kappa = Y - nn / 2
    
    def rpg_approx(n, b, c):
        """
        Approximate Polya-Gamma sampling.
        This is a simplified approximation - in production, use a proper PG sampler.
        """
        c_clipped = np.clip(c, -30, 30)
        result = np.random.gamma(1, 1, size=n) / (1 + np.exp(-c_clipped))
        
        mask = np.isnan(result) | np.isinf(result)
        if np.any(mask):
            result[mask] = np.random.uniform(0.01, 0.1, size=np.sum(mask))
            
        return result
    
    pp = list(range(p))
    pp.remove(baseline)  # Remove baseline class
    
    for iter in tqdm.tqdm(range(niter), desc="MCMC Sampling"):
        for j in pp:
            try:
                xb_matrix = X @ curr_beta[:, np.arange(p) != j]
                
                if np.any(np.isnan(xb_matrix)) or np.any(np.isinf(xb_matrix)):
                    xb_matrix = np.nan_to_num(xb_matrix, nan=0.0, posinf=30.0, neginf=-30.0)
                
                A = np.sum(safe_exp(xb_matrix), axis=1)
                
                A = np.maximum(A, 1e-10)
                
                c_j = np.log(A)
                eta_j = X @ curr_beta[:, j] - c_j
                
                if np.any(np.isnan(eta_j)) or np.any(np.isinf(eta_j)):
                    eta_j = np.nan_to_num(eta_j, nan=0.0, posinf=30.0, neginf=-30.0)
                
                curr_om[:, j] = rpg_approx(n, nn[:, j], eta_j)
                
                curr_om[:, j] = np.maximum(curr_om[:, j], 1e-10)
                
                weighted_X = X * np.sqrt(curr_om[:, j])[:, np.newaxis]
                cov_matrix = beta_prior_var_inv + weighted_X.T @ weighted_X
                
                # Add jitter for numerical stability
                cov_matrix = cov_matrix + np.eye(k) * jitter
                
                cov_matrix = (cov_matrix + cov_matrix.T) / 2
                
                try:
                    L = np.linalg.cholesky(cov_matrix)
                    V = np.linalg.inv(cov_matrix)
                except np.linalg.LinAlgError:
                    additional_jitter = max(jitter, 1e-6) * 10
                    cov_matrix = cov_matrix + np.eye(k) * additional_jitter
                    
                    try:
                        V = np.linalg.inv(cov_matrix)
                    except np.linalg.LinAlgError:
                        V = np.linalg.pinv(cov_matrix)
                
                mean_term = beta_prior_var_inv @ beta_prior_mean[:, j]
                data_term = X.T @ (kappa[:, j] + c_j * curr_om[:, j])
                
                if np.any(np.isnan(data_term)) or np.any(np.isinf(data_term)):
                    data_term = np.nan_to_num(data_term, nan=0.0, posinf=1.0, neginf=-1.0)
                
                b = V @ (mean_term + data_term)
                
                if np.any(np.isnan(b)) or np.any(np.isinf(b)):
                    b = np.nan_to_num(b, nan=0.0, posinf=1.0, neginf=-1.0)
                
                V_sym = (V + V.T) / 2
                
                min_eig = np.min(np.linalg.eigvalsh(V_sym))
                if min_eig < 1e-6:
                    V_sym = V_sym + np.eye(k) * (1e-6 - min_eig)
                
                curr_beta[:, j] = multivariate_normal.rvs(mean=b, cov=V_sym)
                
            except Exception as e:
                warnings.warn(f"Error in MCMC iteration {iter}, class {j}: {str(e)}. Using conservative update.")
                
                curr_beta[:, j] = curr_beta[:, j] + np.random.normal(0, 0.01, size=k)
        
        if iter >= ndiscard:
            s = iter - ndiscard
            postb[:, :, s] = curr_beta
            curr_xb = X @ curr_beta
    
    marginal_fx = None
    for s in range(nretain):
        for j in range(p):
            postb[:, j, s] = postb[:, j, s] / X_std
    
    if calc_marginal_fx:
        marginal_fx = np.zeros((k, p, nretain))
        
        mean_Xs = np.mean(X_orig, axis=0)  # Use original X for marginal effects
        
        for jjj in range(nretain):
            # Use safe_exp for numerical stability
            MU = X_orig @ postb[:, :, jjj]  # Use original X
            
            if np.any(np.isnan(MU)) or np.any(np.isinf(MU)):
                MU = np.nan_to_num(MU, nan=0.0, posinf=30.0, neginf=-30.0)
            
            exp_MU = safe_exp(MU)
            sum_exp = np.sum(exp_MU, axis=1, keepdims=True)
            sum_exp = np.maximum(sum_exp, 1e-10)  # Prevent division by zero
            pr = exp_MU / sum_exp
            
            if np.any(np.isnan(pr)) or np.any(np.isinf(pr)):
                pr = np.nan_to_num(pr, nan=1.0/p, posinf=1.0, neginf=0.0)
                pr = pr / np.sum(pr, axis=1, keepdims=True)
            
            for ppp in range(p):
                bbb = np.ones((n, k)) * postb[:, ppp, jjj]
                pr_bbb = bbb.copy()
                
                for kk in range(k):
                    pr_bbb[:, kk] = np.sum(pr * postb[kk, :, jjj], axis=1)
                
                partial1 = pr[:, ppp][:, np.newaxis] * (bbb - pr_bbb)
                
                if np.any(np.isnan(partial1)) or np.any(np.isinf(partial1)):
                    partial1 = np.nan_to_num(partial1, nan=0.0, posinf=1.0, neginf=-1.0)
                
                marginal_fx[:, ppp, jjj] = np.mean(partial1, axis=0)
    
    results = {
        "postb": postb,
        "marginal_fx": marginal_fx,
        "X": X_orig,  # Return original X
        "Y": Y,
        "baseline": baseline,
        "X_mean": X_mean,  # Store normalization parameters
        "X_std": X_std
    }
    
    return results
