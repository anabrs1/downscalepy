"""
Multi-format data loader for bias correction solver.

Supports loading CSV, JSON, and Parquet files with automatic format detection
and comprehensive validation.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and preprocess input data for bias correction solver.
    
    Handles multiple file formats:
    - CSV: pandas.read_csv
    - Parquet: pandas.read_parquet
    - JSON: json.load
    """
    
    def __init__(self):
        self.loaded_data = {}
    
    def load_file(self, file_path: str, expected_columns: Optional[list] = None) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Parameters
        ----------
        file_path : str
            Path to file
        expected_columns : list, optional
            Expected column names for validation
        
        Returns
        -------
        pd.DataFrame or dict
            Loaded data
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        logger.info(f"Loading {suffix} file: {file_path}")
        
        try:
            if suffix == '.csv':
                data = pd.read_csv(file_path)
            elif suffix == '.parquet':
                data = pd.read_parquet(file_path)
            elif suffix == '.json':
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                return json_data  # Return dict for JSON
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            logger.info(f"Loaded {len(data):,} rows")
            
            # Validate columns if specified
            if expected_columns:
                missing = set(expected_columns) - set(data.columns)
                if missing:
                    logger.warning(f"Missing expected columns: {missing}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def load_areas(self, file_path: str) -> pd.DataFrame:
        """
        Load areas data.
        
        Expected columns: ns, lu.from, value
        """
        logger.info("Loading areas data...")
        data = self.load_file(file_path, expected_columns=['ns', 'lu.from', 'value'])
        
        # Validate
        required = ['ns', 'value']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Areas file missing required column: {col}")
        
        # Handle lu.from column (might be named differently)
        if 'lu.from' not in data.columns:
            if 'lu_from' in data.columns:
                data.rename(columns={'lu_from': 'lu.from'}, inplace=True)
            else:
                logger.warning("No lu.from column found - will be inferred from data")
        
        self.loaded_data['areas'] = data
        logger.info(f"Areas: {len(data):,} rows, {data['ns'].nunique():,} unique pixels")
        
        return data
    
    def load_priors(self, file_path: str) -> pd.DataFrame:
        """
        Load priors data.
        
        Expected columns: ns, lu.to, value
        """
        logger.info("Loading priors data...")
        data = self.load_file(file_path, expected_columns=['ns', 'lu.to', 'value'])
        
        # Validate
        required = ['ns', 'value']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Priors file missing required column: {col}")
        
        # Handle lu.to column
        if 'lu.to' not in data.columns:
            if 'lu_to' in data.columns:
                data.rename(columns={'lu_to': 'lu.to'}, inplace=True)
            else:
                raise ValueError("Priors file missing lu.to column")
        
        self.loaded_data['priors'] = data
        logger.info(f"Priors: {len(data):,} rows, {data['ns'].nunique():,} unique pixels")
        
        return data
    
    def load_targets(self, file_path: str) -> pd.DataFrame:
        """
        Load targets data.
        
        Expected columns: lu.from, lu.to, value
        """
        logger.info("Loading targets data...")
        data = self.load_file(file_path)
        
        # Handle column names
        if 'lu.from' not in data.columns and 'lu_from' in data.columns:
            data.rename(columns={'lu_from': 'lu.from'}, inplace=True)
        if 'lu.to' not in data.columns and 'lu_to' in data.columns:
            data.rename(columns={'lu_to': 'lu.to'}, inplace=True)
        
        # Validate
        required = ['lu.to', 'value']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Targets file missing required column: {col}")
        
        # If no lu.from, add it (single class)
        if 'lu.from' not in data.columns:
            logger.info("No lu.from column - assuming single source class")
            # Infer from areas or use default
            data['lu.from'] = data['lu.to'].iloc[0] if len(data) > 0 else 0
        
        self.loaded_data['targets'] = data
        logger.info(f"Targets: {len(data):,} rows")
        
        return data
    
    def load_restrictions(
        self, 
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load restrictions data.
        
        Expected columns: ns, lu.from, lu.to, value (0 or 1)
        """
        if not csv_path and not json_path:
            logger.info("No restrictions files provided")
            return None
        
        logger.info("Loading restrictions data...")
        
        # Load JSON metadata if available
        metadata = None
        if json_path:
            metadata = self.load_file(json_path)
            logger.info(f"Restrictions metadata: {metadata.get('total_rules', 'N/A')} rules")
        
        # Load CSV data
        if csv_path:
            data = self.load_file(csv_path)
            
            # Handle column names
            if 'lu.from' not in data.columns and 'lu_from' in data.columns:
                data.rename(columns={'lu_from': 'lu.from'}, inplace=True)
            if 'lu.to' not in data.columns and 'lu_to' in data.columns:
                data.rename(columns={'lu_to': 'lu.to'}, inplace=True)
            
            # Handle 'allowed' vs 'value' column
            if 'value' not in data.columns:
                if 'allowed' in data.columns:
                    # Convert 'allowed' to 'value' (invert logic: allowed=1 -> value=0, allowed=0 -> value=1)
                    data['value'] = (1 - data['allowed']).astype(int)
                    logger.info("Converted 'allowed' column to 'value' (inverted logic)")
                else:
                    raise ValueError("Restrictions file missing 'value' or 'allowed' column")
            
            # Validate
            required = ['ns', 'lu.from', 'lu.to', 'value']
            for col in required:
                if col not in data.columns:
                    raise ValueError(f"Restrictions file missing required column: {col}")
            
            # Ensure binary values
            if not data['value'].isin([0, 1]).all():
                logger.warning("Non-binary values in restrictions - converting to 0/1")
                data['value'] = (data['value'] != 0).astype(int)
            
            self.loaded_data['restrictions'] = data
            self.loaded_data['restrictions_metadata'] = metadata
            
            forbidden = (data['value'] == 1).sum()
            logger.info(f"Restrictions: {len(data):,} rows, {forbidden:,} forbidden transitions")
            
            return data
        
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of loaded data."""
        summary = {}
        
        for name, data in self.loaded_data.items():
            if isinstance(data, pd.DataFrame):
                summary[name] = {
                    'rows': len(data),
                    'columns': list(data.columns),
                    'memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                }
            elif isinstance(data, dict):
                summary[name] = {
                    'type': 'metadata',
                    'keys': list(data.keys())
                }
        
        return summary
    
    def print_summary(self):
        """Print summary of loaded data."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("DATA LOADING SUMMARY")
        print("="*70)
        
        for name, info in summary.items():
            if 'rows' in info:
                print(f"\n{name.upper()}:")
                print(f"  Rows: {info['rows']:,}")
                print(f"  Columns: {', '.join(info['columns'])}")
                print(f"  Memory: {info['memory_mb']:.2f} MB")
            elif 'type' in info:
                print(f"\n{name.upper()}:")
                print(f"  Type: {info['type']}")
                print(f"  Keys: {', '.join(info['keys'])}")
        
        print("\n" + "="*70 + "\n")


def load_all_inputs(paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Convenience function to load all input data.
    
    Parameters
    ----------
    paths : dict
        Dictionary of file paths with keys: areas, priors, targets, 
        restrictions_csv, restrictions_json
    
    Returns
    -------
    tuple
        (areas, priors, targets, restrictions)
    """
    loader = DataLoader()
    
    areas = loader.load_areas(paths['areas'])
    priors = loader.load_priors(paths['priors'])
    targets = loader.load_targets(paths['targets'])
    
    restrictions = None
    if paths.get('restrictions_csv'):
        restrictions = loader.load_restrictions(
            csv_path=paths.get('restrictions_csv'),
            json_path=paths.get('restrictions_json')
        )
    
    loader.print_summary()
    
    return areas, priors, targets, restrictions
