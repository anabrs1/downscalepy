"""
Data preprocessing utilities.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for solver."""
    
    @staticmethod
    def normalize_priors(priors: pd.DataFrame) -> pd.DataFrame:
        """Normalize priors to sum to 1 per pixel."""
        logger.info("Normalizing priors...")
        priors = priors.copy()
        
        # Group by pixel and normalize
        priors['value'] = priors.groupby('ns')['value'].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
        )
        
        return priors
    
    @staticmethod
    def filter_zero_targets(
        targets: pd.DataFrame,
        priors: pd.DataFrame
    ) -> tuple:
        """Remove target classes with zero values."""
        logger.info("Filtering zero targets...")
        
        # Identify non-zero targets
        nonzero_targets = targets[targets['value'] > 0].copy()
        
        if len(nonzero_targets) < len(targets):
            removed = len(targets) - len(nonzero_targets)
            logger.info(f"Removed {removed} zero-value targets")
        
        return nonzero_targets, priors
