"""
downscalepy: Python package for downscaling land-use and land-use change projections.

This package provides tools for downscaling land-use change projections from models 
like GLOBIOM, CAPRI, and FABLE.
"""

from .core.downscale import downscale, downscale_control
from .core.downscale_pop import downscale_pop, downscale_control_pop
from .models.mnlogit import mnlogit
from .simulation.sim_lu import sim_lu
from .simulation.sim_luc import sim_luc
from .simulation.sim_pop import sim_pop

__all__ = [
    'downscale', 'downscale_control',
    'downscale_pop', 'downscale_control_pop',
    'mnlogit',
    'sim_lu', 'sim_luc', 'sim_pop'
]
