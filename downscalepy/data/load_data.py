"""
Data loading utilities for downscalepy.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any


def load_argentina_data(use_real_data=True, **kwargs) -> Dict[str, Any]:
    """
    Load example data for Argentina.
    
    Parameters
    ----------
    use_real_data : bool, default=True
        Whether to use the real data converted from the R package.
        If False, synthetic data will be generated.
    **kwargs : dict
        Additional keyword arguments for backward compatibility.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
        - argentina_luc: Land use change data
        - argentina_df: Dictionary containing xmat, lu_levels, restrictions, and pop_data
        - argentina_FABLE: Target data from FABLE
    """
    if 'use_real_data' in kwargs:
        use_real_data = kwargs['use_real_data']
        
    try:
        if use_real_data:
            print("Loading real Argentina data...")
            return load_real_argentina_data()
        else:
            print("Generating synthetic Argentina data...")
            return generate_synthetic_argentina_data()
    except Exception as e:
        print(f"Error loading Argentina data: {e}")
        print("Falling back to synthetic data...")
        return generate_synthetic_argentina_data()


def load_real_argentina_data() -> Dict[str, Any]:
    """
    Load the real Argentina data converted from the R package.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the real Argentina data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    converted_dir = os.path.join(script_dir, 'converted')
    
    csv_files = [
        'argentina_luc.csv',
        'argentina_FABLE.csv',
        'argentina_df_xmat.csv',
        'argentina_df_lu_levels.csv',
        'argentina_df_restrictions.csv',
        'argentina_df_pop_data.csv'
    ]
    
    missing_files = [f for f in csv_files if not os.path.exists(os.path.join(converted_dir, f))]
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print("Falling back to synthetic data.")
        return generate_synthetic_argentina_data()
    
    result = {}
    
    try:
        argentina_luc = pd.read_csv(os.path.join(converted_dir, 'argentina_luc.csv'))
        result['argentina_luc'] = argentina_luc
    except Exception as e:
        print(f"Error loading argentina_luc: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_luc'] = synthetic_data['argentina_luc']
    
    # Load argentina_FABLE
    try:
        argentina_FABLE = pd.read_csv(os.path.join(converted_dir, 'argentina_FABLE.csv'))
        result['argentina_FABLE'] = argentina_FABLE
    except Exception as e:
        print(f"Error loading argentina_FABLE: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_FABLE'] = synthetic_data['argentina_FABLE']
    
    try:
        xmat = pd.read_csv(os.path.join(converted_dir, 'argentina_df_xmat.csv'))
        lu_levels = pd.read_csv(os.path.join(converted_dir, 'argentina_df_lu_levels.csv'))
        restrictions = pd.read_csv(os.path.join(converted_dir, 'argentina_df_restrictions.csv'))
        pop_data = pd.read_csv(os.path.join(converted_dir, 'argentina_df_pop_data.csv'))
        
        result['argentina_df'] = {
            'xmat': xmat,
            'lu_levels': lu_levels,
            'restrictions': restrictions,
            'pop_data': pop_data
        }
    except Exception as e:
        print(f"Error loading argentina_df components: {e}")
        synthetic_data = generate_synthetic_argentina_data()
        result['argentina_df'] = synthetic_data['argentina_df']
    
    return result


def generate_synthetic_argentina_data() -> Dict[str, Any]:
    """
    Generate synthetic data for Argentina.
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing synthetic Argentina data.
    """
    ns = [f'ns{i}' for i in range(1, 101)]
    lu_classes = ['Cropland', 'Forest', 'Pasture', 'Urban', 'OtherLand']
    ks = [f'k{i}' for i in range(1, 5)]
    times = ['2000', '2010', '2020', '2030']
    
    # Land use change data
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
    
    restrictions_data = []
    for n in ns[:20]:  # Only add restrictions for some spatial units
        for lu_from in lu_classes[:2]:  # Only add restrictions for some land use classes
            for lu_to in lu_classes[2:4]:  # Only add restrictions for some transitions
                restrictions_data.append({
                    'ns': n,
                    'lu.from': lu_from,
                    'lu.to': lu_to,
                    'value': 1  # Restrict this transition
                })
    
    pop_data = []
    for n in ns:
        for t in times:
            for k in ks:
                pop_data.append({
                    'ns': n,
                    'times': t,
                    'ks': k,
                    'value': np.random.normal()
                })
    
    argentina_df = {
        'xmat': pd.DataFrame(xmat_data),
        'lu_levels': pd.DataFrame(lu_levels_data),
        'restrictions': pd.DataFrame(restrictions_data),
        'pop_data': pd.DataFrame(pop_data)
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
