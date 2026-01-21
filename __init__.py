"""
Bias Correction Solver for Land Use Modeling

A comprehensive Python implementation of the Multinomial Logit (MNL) bias correction
solver for downscaling land use projections. Based on the reference R implementation
from downscalepy.

Main Components:
- BiasCorrectSolver: Core solver class implementing MNL optimization
- DataLoader: Multi-format data loading (CSV, JSON, Parquet)
- PathDetector: Automatic input file detection
- ProjectionGenerator: Generate 2030 projection maps with geospatial metadata

Author: Auto-generated implementation
Version: 1.0.0
Date: 2025-12-01
"""

__version__ = "1.0.0"
__author__ = "Bias Correction Solver Team"

from .core.solver import BiasCorrectSolver
from .data.path_detector import PathDetector
from .data.data_loader import DataLoader
from .output.projector import ProjectionGenerator
from .utils.config import SolverConfig

__all__ = [
    "BiasCorrectSolver",
    "PathDetector",
    "DataLoader",
    "ProjectionGenerator",
    "SolverConfig",
]
