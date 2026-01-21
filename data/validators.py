"""
Data validation utilities for bias correction solver.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate input data for bias correction solver."""
    
    @staticmethod
    def validate_areas(df: pd.DataFrame) -> List[str]:
        """Validate areas DataFrame."""
        errors = []
        
        # Check required columns
        required = ['ns', 'value']
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for nulls
        if df['ns'].isnull().any():
            errors.append("Null values found in 'ns' column")
        if df['value'].isnull().any():
            errors.append("Null values found in 'value' column")
        
        # Check for negative values
        if (df['value'] < 0).any():
            errors.append("Negative values found in 'value' column")
        
        # Check for duplicates
        if df['ns'].duplicated().any():
            n_dups = df['ns'].duplicated().sum()
            errors.append(f"Duplicate pixel IDs found: {n_dups}")
        
        return errors
    
    @staticmethod
    def validate_priors(df: pd.DataFrame) -> List[str]:
        """Validate priors DataFrame."""
        errors = []
        
        # Check required columns
        required = ['ns', 'lu.to', 'value']
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for nulls
        for col in required:
            if df[col].isnull().any():
                errors.append(f"Null values found in '{col}' column")
        
        # Check for negative values
        if (df['value'] < 0).any():
            errors.append("Negative values found in 'value' column")
        
        return errors
    
    @staticmethod
    def validate_targets(df: pd.DataFrame) -> List[str]:
        """Validate targets DataFrame."""
        errors = []
        
        # Check required columns
        required = ['lu.to', 'value']
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for nulls
        for col in required:
            if df[col].isnull().any():
                errors.append(f"Null values found in '{col}' column")
        
        # Check for negative values
        if (df['value'] < 0).any():
            errors.append("Negative values found in 'value' column")
        
        return errors
    
    @staticmethod
    def validate_restrictions(df: pd.DataFrame) -> List[str]:
        """Validate restrictions DataFrame."""
        errors = []
        
        # Check required columns
        required = ['ns', 'lu.from', 'lu.to', 'value']
        for col in required:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        if errors:
            return errors
        
        # Check for binary values
        if not df['value'].isin([0, 1]).all():
            errors.append("Restriction values must be 0 or 1")
        
        return errors


def validate_solver_inputs(
    areas: pd.DataFrame,
    priors: pd.DataFrame,
    targets: pd.DataFrame,
    restrictions: Optional[pd.DataFrame] = None
) -> Dict[str, List[str]]:
    """
    Validate all solver inputs.
    
    Returns
    -------
    dict
        Dictionary mapping input names to lists of error messages
    """
    validator = DataValidator()
    
    results = {
        'areas': validator.validate_areas(areas),
        'priors': validator.validate_priors(priors),
        'targets': validator.validate_targets(targets),
    }
    
    if restrictions is not None:
        results['restrictions'] = validator.validate_restrictions(restrictions)
    
    # Check for any errors
    has_errors = any(errors for errors in results.values())
    
    if has_errors:
        logger.warning("Validation errors found")
        for name, errors in results.items():
            if errors:
                logger.warning(f"{name}: {len(errors)} errors")
                for error in errors:
                    logger.warning(f"  - {error}")
    else:
        logger.info("All inputs passed validation")
    
    return results
