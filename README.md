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

## Package Structure

The package is organized into several modules, each with specific responsibilities:

```
downscalepy/
├── core/               # Core downscaling algorithms
│   ├── downscale.py    # Main land-use downscaling function
│   └── downscale_pop.py # Population downscaling function
├── solvers/            # Optimization solvers
│   ├── solve_biascorr.py # Bias correction solver
│   └── solve_notarget.py # Non-targeted solver
├── models/             # Statistical models
│   └── mnlogit.py      # Multinomial logit model
├── simulation/         # Data simulation
│   ├── sim_lu.py       # Land-use simulation
│   ├── sim_luc.py      # Land-use change simulation
│   └── sim_pop.py      # Population simulation
├── utils/              # Utility functions
│   ├── areas_update.py # Area updating functions
│   ├── xmat_update.py  # Matrix updating functions
│   └── constants.py    # Shared constants
└── data/               # Data loading and processing
    └── load_data.py    # Functions to load example data
```

## Model Architecture

The downscalepy package implements a modular architecture for downscaling land-use and land-use change projections:

### Core Components

1. **Core Module**: Contains the main downscaling algorithms
   - `downscale.py`: Implements the main land-use downscaling function that orchestrates the entire process
   - `downscale_pop.py`: Implements population downscaling functionality

2. **Solvers Module**: Contains optimization algorithms for the downscaling process
   - `solve_biascorr.py`: Implements bias correction solvers for multinomial logit problems
   - `solve_notarget.py`: Implements non-targeted downscaling solvers

3. **Models Module**: Contains statistical models for prior estimation
   - `mnlogit.py`: Implements Bayesian multinomial logistic regression using Polya-Gamma latent variables

4. **Simulation Module**: Contains functions for generating synthetic data
   - `sim_lu.py`: Simulates land-use data
   - `sim_luc.py`: Simulates land-use change data
   - `sim_pop.py`: Simulates population data

5. **Utils Module**: Contains utility functions used across the package
   - `areas_update.py`: Functions for updating area allocations
   - `xmat_update.py`: Functions for updating explanatory variable matrices
   - `constants.py`: Shared constants used throughout the package

6. **Data Module**: Contains functions for loading and processing data
   - `load_data.py`: Functions for loading example data, including Argentina data

### Data Flow

The typical data flow in the downscaling process is as follows:

1. **Input Data**: The process starts with input data including:
   - Targets: Coarse-resolution land-use change targets
   - Start areas: Initial land-use areas at fine resolution
   - Explanatory variables: Variables that explain land-use patterns
   - Coefficients: Parameters for the statistical model

2. **Prior Estimation**: If coefficients are not provided, they are estimated using the `mnlogit` function, which implements Bayesian multinomial logistic regression.

3. **Downscaling**: The `downscale` function processes the inputs and calls the appropriate solver:
   - For bias correction: `solve_biascorr_mnl`
   - For non-targeted downscaling: `solve_notarget_mnl`

4. **Optimization**: The solvers optimize the allocation of land-use changes to meet the targets while respecting constraints.

5. **Output**: The process returns the downscaled land-use change projections at fine resolution.

### Component Interactions

- The `downscale` function in the core module is the main entry point, which coordinates the entire process.
- It validates and preprocesses the input data, then calls the appropriate solver from the solvers module.
- The solvers use statistical models from the models module to estimate priors if needed.
- Utility functions from the utils module are used throughout the process for data manipulation.
- The simulation module provides functions for generating synthetic data for testing and examples.
- The data module provides functions for loading real-world data for examples and applications.

## Usage

### Basic Usage

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

# Access results
downscaled_areas = result['out_res']
solver_info = result['out_solver']
inputs_used = result['ds_inputs']
```

### Advanced Usage

```python
from downscalepy import downscale, downscale_control, mnlogit
import pandas as pd
import numpy as np

# Load your data
targets = pd.DataFrame(...)  # Coarse-resolution targets
start_areas = pd.DataFrame(...)  # Fine-resolution initial areas
xmat = pd.DataFrame(...)  # Explanatory variables

# Estimate coefficients using mnlogit
X = ...  # Explanatory variables matrix
Y = ...  # Dependent variables matrix
mnlogit_result = mnlogit(X, Y, niter=1000, nburn=500)
betas = ...  # Extract and format coefficients from mnlogit_result

# Configure solver options
options = downscale_control(
    solve_fun="solve_biascorr",
    algorithm="SLSQP",
    maxeval=2000,
    cutoff=1e-6
)

# Perform downscaling with custom options
result = downscale(
    targets=targets,
    start_areas=start_areas,
    xmat=xmat,
    betas=betas,
    options=options,
    restrictions=restrictions  # Optional restrictions on land-use transitions
)
```

## Examples

### Argentina Example

The package includes an example using real data from Argentina:

```python
from examples.argentina_example import run_example

# Run with real data (default)
result = run_example(use_real_data=True)

# Or run with synthetic data
result = run_example(use_real_data=False)
```

This example demonstrates downscaling land-use change projections in Argentina using the FABLE model. It includes:

1. Loading real data converted from the original R package
2. Estimating coefficients using multinomial logit regression
3. Downscaling land-use change projections
4. Visualizing the results

See the `examples` directory for more detailed examples.

## Documentation

### Core Functions

- `downscale()`: Downscaling of land-use data over specified time steps
- `downscale_pop()`: Downscaling of population data
- `mnlogit()`: Bayesian multinomial logistic regression using Polya-Gamma latent variables

### Simulation Functions

- `sim_lu()`: Simulate land-use data
- `sim_luc()`: Simulate land-use change data
- `sim_pop()`: Simulate population data

### Visualization Functions

- `luc_plot()`: Create visualizations of land-use change results
- `save_luc_plot()`: Save land-use change visualizations to files

## Visualization

The package includes visualization functions to help analyze and present downscaling results:

```python
from downscalepy import downscale, luc_plot, save_luc_plot
import rasterio

# Run downscaling
result = downscale(...)

# Open a raster file with spatial information
with rasterio.open('path/to/raster.tif') as raster:
    # Create visualization for all land uses and times
    plot_result = luc_plot(
        res=result,
        raster_file=raster,
        figsize=(12, 10)
    )
    
    # Save the visualization
    save_luc_plot(
        plot_result,
        'output/visualization.png',
        dpi=300
    )
    
    # Create visualization for specific land use and time
    specific_plot = luc_plot(
        res=result,
        raster_file=raster,
        year='2010',
        lu='Forest',
        cmap='Blues'
    )
```

See the `examples/visualization_example.py` script for a complete example of creating and saving visualizations.
