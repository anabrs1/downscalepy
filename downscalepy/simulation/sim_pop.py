"""
Simulation of Population Data

This module provides functions for simulating population data for testing and examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


def sim_pop(n: int = 100, p: int = 1, k: int = 3, tt: int = 2, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate population data.
    
    Parameters
    ----------
    n : int, default=100
        Number of spatial units.
    p : int, default=1
        Number of population types.
    k : int, default=3
        Number of explanatory variables.
    tt : int, default=2
        Number of time steps.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing:
        - xmat: A dataframe with columns ns, ks, and value.
        - betas: A dataframe with columns ks, pop.type, and value.
        - targets: A dataframe with columns times, pop.type, and value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    ns_list = [f"ns{i+1}" for i in range(n)]
    
    pop_type_list = [f"pop{i+1}" for i in range(p)]
    
    ks_list = [f"k{i+1}" for i in range(k)]
    
    times_list = [str(i+1) for i in range(tt)]
    
    xmat_data = []
    for ns in ns_list:
        for ks in ks_list:
            xmat_data.append({
                'ns': ns,
                'ks': ks,
                'value': np.random.normal()
            })
    
    xmat = pd.DataFrame(xmat_data)
    
    betas_data = []
    for ks in ks_list:
        for pop_type in pop_type_list:
            betas_data.append({
                'ks': ks,
                'pop.type': pop_type,
                'value': np.random.normal()
            })
    
    betas = pd.DataFrame(betas_data)
    
    targets_data = []
    for t in times_list:
        for pop_type in pop_type_list:
            target_value = n * np.random.uniform(50, 100)
            
            targets_data.append({
                'times': t,
                'pop.type': pop_type,
                'value': target_value
            })
    
    targets = pd.DataFrame(targets_data)
    
    return {
        'xmat': xmat,
        'betas': betas,
        'targets': targets
    }
