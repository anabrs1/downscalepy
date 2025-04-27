"""
Data loading utilities for downscalepy.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any


def load_argentina_data() -> Dict[str, Any]:
    """
    Load example data for Argentina.
    
    This is a placeholder function that should be replaced with actual data loading
    once the data is converted from the R package.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - argentina_luc: Land use change data
        - argentina_df: Explanatory variables and land use levels
        - argentina_FABLE: Target data from FABLE
    """
    
    ns = [f'ns{i}' for i in range(1, 101)]
    lu_classes = ['Cropland', 'Forest', 'Pasture', 'Urban', 'OtherLand']
    ks = [f'k{i}' for i in range(1, 5)]
    times = ['2000', '2010', '2020', '2030']
    
    luc_data = []
    for t in ['2000']:
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                for n in ns:
                    if lu_from != lu_to:
                        luc_data.append({
                            'Ts': t,
                            'lu.from': lu_from,
                            'lu.to': lu_to,
                            'ns': n,
                            'value': np.random.uniform(0, 1)
                        })
    
    argentina_luc = pd.DataFrame(luc_data)
    
    xmat_data = []
    for n in ns:
        for k in ks:
            xmat_data.append({
                'ns': n,
                'ks': k,
                'value': np.random.normal()
            })
    
    lu_levels_data = []
    for n in ns:
        for lu in lu_classes:
            lu_levels_data.append({
                'ns': n,
                'lu.from': lu,
                'value': np.random.uniform(5, 10)
            })
    
    argentina_df = {
        'xmat': pd.DataFrame(xmat_data),
        'lu_levels': pd.DataFrame(lu_levels_data)
    }
    
    fable_data = []
    for t in times:
        for lu_from in lu_classes:
            for lu_to in lu_classes:
                if lu_from != lu_to:
                    fable_data.append({
                        'times': t,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': np.random.uniform(50, 100)
                    })
    
    argentina_FABLE = pd.DataFrame(fable_data)
    
    return {
        'argentina_luc': argentina_luc,
        'argentina_df': argentina_df,
        'argentina_FABLE': argentina_FABLE
    }
