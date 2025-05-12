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
    n, k = X.shape
    p = Y.shape[1]
    
    if baseline is None:
        baseline = p - 1
    
    if niter <= nburn:
        raise ValueError("niter has to be higher than nburn.")
    
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
        return np.random.gamma(1, 1, size=n) / (1 + np.exp(-c))
    
    pp = list(range(p))
    pp.remove(baseline)  # Remove baseline class
    
    for iter in tqdm.tqdm(range(niter), desc="MCMC Sampling"):
        for j in pp:
            A = np.sum(np.exp(X @ curr_beta[:, np.arange(p) != j]), axis=1)
            c_j = np.log(A)
            eta_j = X @ curr_beta[:, j] - c_j
            
            curr_om[:, j] = rpg_approx(n, nn[:, j], eta_j)
            
            cov_matrix = beta_prior_var_inv + X.T @ (X * curr_om[:, j][:, np.newaxis])
            if jitter > 0:
                cov_matrix = cov_matrix + np.eye(k) * jitter
                
            try:
                V = np.linalg.inv(cov_matrix)
                b = V @ (beta_prior_var_inv @ beta_prior_mean[:, j] + 
                         X.T @ (kappa[:, j] + c_j * curr_om[:, j]))
                curr_beta[:, j] = multivariate_normal.rvs(mean=b, cov=V)
            except np.linalg.LinAlgError:
                cov_matrix = cov_matrix + np.eye(k) * max(jitter, 1e-6) * 10
                V = np.linalg.inv(cov_matrix)
                b = V @ (beta_prior_var_inv @ beta_prior_mean[:, j] + 
                         X.T @ (kappa[:, j] + c_j * curr_om[:, j]))
                curr_beta[:, j] = multivariate_normal.rvs(mean=b, cov=V)
        
        if iter >= ndiscard:
            s = iter - ndiscard
            postb[:, :, s] = curr_beta
            curr_xb = X @ curr_beta
    
    marginal_fx = None
    if calc_marginal_fx:
        marginal_fx = np.zeros((k, p, nretain))
        
        mean_Xs = np.mean(X, axis=0)
        
        for jjj in range(nretain):
            MU = X @ postb[:, :, jjj]
            pr = np.exp(MU) / np.sum(np.exp(MU), axis=1)[:, np.newaxis]
            
            for ppp in range(p):
                bbb = np.ones((n, k)) * postb[:, ppp, jjj]
                pr_bbb = bbb.copy()
                
                for kk in range(k):
                    pr_bbb[:, kk] = np.sum(pr * postb[kk, :, jjj], axis=1)
                
                partial1 = pr[:, ppp][:, np.newaxis] * (bbb - pr_bbb)
                marginal_fx[:, ppp, jjj] = np.mean(partial1, axis=0)
    
    results = {
        "postb": postb,
        "marginal_fx": marginal_fx,
        "X": X,
        "Y": Y,
        "baseline": baseline
    }
    
    return results
