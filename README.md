# downscalepy

Python package for downscaling land-use and land-use change projections from models like GLOBIOM, CAPRI, and FABLE.

## Original R Implementation

This is a Python port of the [downscalr](https://github.com/tkrisztin/downscalr) R package.

## Installation

```bash
pip install -e .
```

## Overview

The `downscalepy` package provides tools for downscaling land-use change projections from coarse resolution models to finer spatial scales. It implements econometric models for prior estimation and various solvers for the downscaling process.

Key features:
- Downscaling of land-use change projections
- Bayesian multinomial logistic regression for prior estimation
- Bias correction and non-targeted downscaling methods
- Simulation functions for generating test data

## Usage

```python
import pandas as pd
import numpy as np
from downscalepy import downscale, sim_luc

# Generate example data
dgp = sim_luc(1000, tt=3)

# Perform downscaling
result = downscale(
    targets=dgp['targets'],
    start_areas=dgp['start_areas'],
    xmat=dgp['xmat'],
    betas=dgp['betas'],
    times=['1', '2', '3']
)
```

## Example

See the `examples` directory for detailed examples, including the Argentina land-use change example.

## Documentation

### Core Functions

- `downscale()`: Downscaling of land-use data over specified time steps
- `downscale_pop()`: Downscaling of population data
- `mnlogit()`: Bayesian multinomial logistic regression using Polya-Gamma latent variables

### Simulation Functions

- `sim_lu()`: Simulate land-use data
- `sim_luc()`: Simulate land-use change data
- `sim_pop()`: Simulate population data


