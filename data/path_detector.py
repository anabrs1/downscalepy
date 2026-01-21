"""
Automatic path detection for input data files.

This module provides functionality to automatically detect and locate
required input files for the bias correction solver.
"""

import glob
from pathlib import Path
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class PathDetector:
    """
    Automatically detect input file paths.
    
    Searches for required input files in scenarios directories:
    - areas_*_solver_format.csv
    - priors_solver_format.parquet (or .csv)
    - targets_*_solver_format.csv
    - restrictions_*.csv and restrictions_*.json
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize path detector.
        
        Parameters
        ----------
        base_dir : str
            Base directory to search from (default: current directory)
        """
        self.base_dir = Path(base_dir)
        self.found_paths = {}
    
    def extract_region_from_metadata(self, search_dir: Path) -> Optional[str]:
        """
        Extract region code from metadata.json file.
        
        Parameters
        ----------
        search_dir : Path
            Directory to search for metadata
        
        Returns
        -------
        str or None
            Region code (e.g., 'PT1C') or None if not found
        """
        metadata_file = search_dir / "metadata.json"
        if metadata_file.exists():
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Try to extract region from source raster paths
                rasters = metadata.get('source_data', {}).get('rasters', {})
                for year, path in rasters.items():
                    # Pattern: outputs/PT1C_100m/lum_rasters/...
                    match = re.search(r'outputs/([A-Z0-9]+)_100m', path)
                    if match:
                        region = match.group(1)
                        logger.info(f"Extracted region from metadata: {region}")
                        return region
            except Exception as e:
                logger.warning(f"Could not extract region from metadata: {e}")
        
        return None
    
    def extract_region_from_path(self, search_dir: Path) -> Optional[str]:
        """
        Extract region code from directory structure.
        
        Parameters
        ----------
        search_dir : Path
            Directory path to analyze
        
        Returns
        -------
        str or None
            Region code or None if not found
        """
        # Look for outputs/{REGION}_100m pattern in parent directories
        current = search_dir.resolve()
        while current != current.parent:
            parent = current.parent
            # Check if there's a sibling directory matching {REGION}_100m pattern
            if parent.name == 'outputs':
                for sibling in parent.iterdir():
                    match = re.match(r'([A-Z0-9]+)_100m', sibling.name)
                    if match:
                        region = match.group(1)
                        logger.info(f"Extracted region from path structure: {region}")
                        return region
            current = parent
        
        return None
    
    def detect_all(self, scenario_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Detect all required input files.
        
        Parameters
        ----------
        scenario_dir : str, optional
            Specific scenario directory to search in.
            If None, searches in scenarios/run_*/ directories.
        
        Returns
        -------
        dict
            Dictionary with keys: areas, priors, targets, restrictions_csv, restrictions_json, region
        
        Raises
        ------
        FileNotFoundError
            If required files are not found
        """
        if scenario_dir:
            search_dir = Path(scenario_dir)
        else:
            # Try to find scenario directories in multiple locations
            scenario_dirs = []
            
            # Try outputs/PT1C_100m/scenarios/run_*/
            scenario_dirs.extend(list(self.base_dir.glob("outputs/PT1C_100m/scenarios/run_*/")))
            # Also try old path for backward compatibility
            scenario_dirs.extend(list(self.base_dir.glob("outputs/scenarios/run_*/")))
            
            # Try scenarios/run_*/
            if not scenario_dirs:
                scenario_dirs.extend(list(self.base_dir.glob("scenarios/run_*/")))
            
            # Try run_*/
            if not scenario_dirs:
                scenario_dirs.extend(list(self.base_dir.glob("run_*/")))
            
            if not scenario_dirs:
                raise FileNotFoundError(
                    "No scenario directories found. "
                    "Expected outputs/scenarios/run_*/, scenarios/run_*/, or run_*/ directories."
                )
            
            # Use the most recent or first directory
            search_dir = sorted(scenario_dirs)[-1]
            logger.info(f"Using scenario directory: {search_dir}")
        
        self.found_paths = {}
        
        # Detect areas file
        self.found_paths['areas'] = self._detect_areas(search_dir)
        
        # Detect priors file
        self.found_paths['priors'] = self._detect_priors(search_dir)
        
        # Detect targets file
        self.found_paths['targets'] = self._detect_targets(search_dir)
        
        # Detect restrictions files (optional)
        try:
            self.found_paths['restrictions_csv'] = self._detect_restrictions_csv(search_dir)
        except FileNotFoundError:
            logger.warning("Restrictions CSV file not found (optional)")
            self.found_paths['restrictions_csv'] = None
        
        try:
            self.found_paths['restrictions_json'] = self._detect_restrictions_json(search_dir)
        except FileNotFoundError:
            logger.warning("Restrictions JSON file not found (optional)")
            self.found_paths['restrictions_json'] = None
        
        # Extract region code
        region = self.extract_region_from_metadata(search_dir)
        if not region:
            region = self.extract_region_from_path(search_dir)
        
        if region:
            self.found_paths['region'] = region
            logger.info(f"Detected region: {region}")
        else:
            logger.warning("Could not detect region code")
            self.found_paths['region'] = None
        
        return self.found_paths
    
    def _detect_areas(self, search_dir: Path) -> str:
        """Detect areas file."""
        patterns = [
            "areas_*_solver_format.csv",
            "areas_*_Solver_format.csv",
            "areas_*.csv"
        ]
        
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                path = str(files[0])
                logger.info(f"Found areas file: {path}")
                return path
        
        raise FileNotFoundError(f"Areas file not found in {search_dir}")
    
    def _detect_priors(self, search_dir: Path) -> str:
        """Detect priors file (parquet or csv)."""
        patterns = [
            "priors_solver_format.parquet",
            "priors_solver_format.csv",
            "priors_*.parquet",
            "priors_*.csv"
        ]
        
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                path = str(files[0])
                logger.info(f"Found priors file: {path}")
                return path
        
        raise FileNotFoundError(f"Priors file not found in {search_dir}")
    
    def _detect_targets(self, search_dir: Path) -> str:
        """Detect targets file."""
        patterns = [
            "targets_*_realistic.csv",      # NEW: Prioritize realistic targets
            "targets_2030_realistic.csv",   # NEW: Explicit 2030 realistic targets
            "targets_*_solver_format.csv",
            "targets_*_Solver_format.csv",
            "targets_*.csv"
        ]
        
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                path = str(files[0])
                logger.info(f"Found targets file: {path}")
                return path
        
        raise FileNotFoundError(f"Targets file not found in {search_dir}")
    
    def _detect_restrictions_csv(self, search_dir: Path) -> Optional[str]:
        """Detect restrictions CSV file."""
        patterns = [
            "restrictions_*.csv",
            "restrictions_level*.csv"
        ]
        
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                path = str(files[0])
                logger.info(f"Found restrictions CSV file: {path}")
                return path
        
        raise FileNotFoundError(f"Restrictions CSV file not found in {search_dir}")
    
    def _detect_restrictions_json(self, search_dir: Path) -> Optional[str]:
        """Detect restrictions JSON file."""
        patterns = [
            "restrictions_*.json",
            "restrictions_level*.json"
        ]
        
        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                path = str(files[0])
                logger.info(f"Found restrictions JSON file: {path}")
                return path
        
        raise FileNotFoundError(f"Restrictions JSON file not found in {search_dir}")
    
    def print_summary(self):
        """Print summary of detected paths."""
        print("\n" + "="*70)
        print("DETECTED INPUT PATHS")
        print("="*70)
        for key, path in self.found_paths.items():
            status = "✓" if path else "✗"
            print(f"{status} {key:20s}: {path if path else 'Not found'}")
        print("="*70 + "\n")


def detect_input_paths(base_dir: str = ".", scenario_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to detect all input paths.
    
    Parameters
    ----------
    base_dir : str
        Base directory to search from
    scenario_dir : str, optional
        Specific scenario directory
    
    Returns
    -------
    dict
        Dictionary of detected paths
    """
    detector = PathDetector(base_dir)
    paths = detector.detect_all(scenario_dir)
    detector.print_summary()
    return paths
