"""
Configuration management for bias correction solver.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import json
import yaml
from pathlib import Path


@dataclass
class SolverConfig:
    """
    Configuration for BiasCorrectSolver.
    
    Attributes
    ----------
    algorithm : str
        Optimization algorithm (L-BFGS-B, SLSQP, Powell, etc.)
    max_exp : float
        Maximum exponent to prevent overflow
    xtol_rel : float
        Relative tolerance for convergence
    xtol_abs : float
        Absolute tolerance for convergence
    maxiter : int
        Maximum iterations for optimization
    cutoff : float
        Minimum probability threshold
    max_diff : float
        Maximum acceptable difference for convergence
    use_grid_search_fallback : bool
        Whether to use grid search if optimization fails
    grid_search_iterations : int
        Number of grid search iterations
    verbose : bool
        Whether to enable verbose logging
    """
    
    algorithm: str = "L-BFGS-B"
    max_exp: float = 700.0
    xtol_rel: float = 1e-4
    xtol_abs: float = 1e-6
    maxiter: int = 1000
    cutoff: float = 0.0
    max_diff: float = 1e-4
    use_grid_search_fallback: bool = True
    grid_search_iterations: int = 10
    verbose: bool = True
    
    # Output settings
    output_formats: list = field(default_factory=lambda: ["parquet", "csv"])
    save_intermediate_results: bool = False
    
    # Performance settings
    memory_limit_gb: Optional[float] = None
    use_parallel: bool = False
    n_workers: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SolverConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SolverConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SolverConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save config to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_exp <= 0:
            raise ValueError("max_exp must be positive")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.cutoff < 0:
            raise ValueError("cutoff must be non-negative")
        if self.max_diff <= 0:
            raise ValueError("max_diff must be positive")
        if self.algorithm not in ["L-BFGS-B", "SLSQP", "trust-constr", "Powell", "Nelder-Mead"]:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


def create_default_config() -> SolverConfig:
    """Create default configuration."""
    return SolverConfig()


def load_config(config_path: str) -> SolverConfig:
    """
    Load configuration from file.
    
    Automatically detects format from extension (.json, .yaml, .yml).
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        return SolverConfig.from_json(config_path)
    elif suffix in ['.yaml', '.yml']:
        return SolverConfig.from_yaml(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")
