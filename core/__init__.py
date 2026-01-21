"""
Core solver components for bias correction.

This module contains the main solver implementation, MNL functions,
and optimization utilities.
"""

from .solver import BiasCorrectSolver
from .allocation_tracker import AllocationTracker
from .mnl_functions import mu_mnl, sqr_diff_mnl, grad_sqr_diff_mnl
from .optimization import optimize_scaling_factors
from .grid_search import iterated_grid_search

__all__ = [
    "BiasCorrectSolver",
    "AllocationTracker",
    "mu_mnl",
    "sqr_diff_mnl",
    "grad_sqr_diff_mnl",
    "optimize_scaling_factors",
    "iterated_grid_search",
]
