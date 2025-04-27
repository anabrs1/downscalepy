"""
Simulation of Land-Use Data

This module provides functions for simulating land-use data for testing and examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


def sim_lu(n: int = 100, p: int = 5, k: int = 3, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate land-use data.
    
    Parameters
    ----------
    n : int, default=100
        Number of spatial units.
    p : int, default=5
        Number of land-use classes.
    k : int, default=3
        Number of explanatory variables.
    seed : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary containing:
        - areas: A dataframe with columns ns, lu.from, and value.
        - xmat: A dataframe with columns ns, ks, and value.
        - betas: A dataframe with columns ks, lu.from, lu.to, and value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    ns_list = [f"ns{i+1}" for i in range(n)]
    
    lu_list = [f"lu{i+1}" for i in range(p)]
    
    ks_list = [f"k{i+1}" for i in range(k)]
    
    areas_data = []
    for ns in ns_list:
        for lu in lu_list:
            areas_data.append({
                'ns': ns,
                'lu.from': lu,
                'value': np.random.uniform(5, 10)
            })
    
    areas = pd.DataFrame(areas_data)
    
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
        for lu_from in lu_list:
            for lu_to in lu_list:
                if lu_from != lu_to:
                    betas_data.append({
                        'ks': ks,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': np.random.normal()
                    })
    
    betas = pd.DataFrame(betas_data)
    
    return {
        'areas': areas,
        'xmat': xmat,
        'betas': betas
    }
