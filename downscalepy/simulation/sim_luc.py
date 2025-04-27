"""
Simulation of Land-Use Change Data

This module provides functions for simulating land-use change data for testing and examples.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any


def sim_luc(n: int = 100, p: int = 5, k: int = 3, tt: int = 2, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulate land-use change data.
    
    Parameters
    ----------
    n : int, default=100
        Number of spatial units.
    p : int, default=5
        Number of land-use classes.
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
        - start_areas: A dataframe with columns ns, lu.from, and value.
        - xmat: A dataframe with columns ns, ks, and value.
        - betas: A dataframe with columns ks, lu.from, lu.to, and value.
        - targets: A dataframe with columns times, lu.from, lu.to, and value.
    """
    if seed is not None:
        np.random.seed(seed)
    
    ns_list = [f"ns{i+1}" for i in range(n)]
    
    lu_list = [f"lu{i+1}" for i in range(p)]
    
    ks_list = [f"k{i+1}" for i in range(k)]
    
    times_list = [str(i+1) for i in range(tt)]
    
    start_areas_data = []
    for ns in ns_list:
        for lu in lu_list:
            start_areas_data.append({
                'ns': ns,
                'lu.from': lu,
                'value': np.random.uniform(5, 10)
            })
    
    start_areas = pd.DataFrame(start_areas_data)
    
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
    
    targets_data = []
    for t in times_list:
        for lu_from in lu_list:
            for lu_to in lu_list:
                if lu_from != lu_to:
                    total_area = start_areas[start_areas['lu.from'] == lu_from]['value'].sum()
                    
                    target_value = total_area * np.random.uniform(0.05, 0.2)
                    
                    targets_data.append({
                        'times': t,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': target_value
                    })
    
    targets = pd.DataFrame(targets_data)
    
    return {
        'start_areas': start_areas,
        'xmat': xmat,
        'betas': betas,
        'targets': targets
    }
