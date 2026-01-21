"""
Main Bias Correction Solver implementation.

This module contains the BiasCorrectSolver class that implements the complete
MNL bias correction algorithm based on the R reference implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import time

from .mnl_functions import mu_mnl, sqr_diff_mnl
from .optimization import optimize_with_fallback
from .grid_search import iterated_grid_search

logger = logging.getLogger(__name__)


class BiasCorrectSolver:
    """
    Multinomial Logit Bias Correction Solver for Land Use Modeling.
    
    This class implements a bias correction algorithm that matches spatial
    allocations to target distributions while respecting prior probabilities
    and transition restrictions.
    
    The solver uses a Multinomial Logit (MNL) framework with optimization
    to find scaling factors that minimize the difference between projected
    and target land use distributions.
    
    Attributes
    ----------
    areas : pd.DataFrame
        Area data with columns: ns (pixel ID), lu.from (current class), value (area)
    priors : pd.DataFrame
        Prior probabilities with columns: ns, lu.to, value
    targets : pd.DataFrame
        Target values with columns: lu.from, lu.to, value
    restrictions : pd.DataFrame, optional
        Binary restrictions with columns: ns, lu.from, lu.to, value (0 or 1)
    config : dict
        Solver configuration parameters
    results : dict
        Solver results after running
    
    Examples
    --------
    >>> solver = BiasCorrectSolver(areas_df, priors_df, targets_df, restrictions_df)
    >>> results = solver.solve(target_year=2030)
    >>> allocations = results['allocations']
    """
    
    def __init__(
        self,
        areas: pd.DataFrame,
        priors: pd.DataFrame,
        targets: pd.DataFrame,
        restrictions: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the BiasCorrectSolver.
        
        Parameters
        ----------
        areas : pd.DataFrame
            Area data with columns: ns, lu.from, value
        priors : pd.DataFrame
            Prior probabilities with columns: ns, lu.to, value
        targets : pd.DataFrame
            Target distributions with columns: lu.from, lu.to, value
        restrictions : pd.DataFrame, optional
            Binary restriction matrix with columns: ns, lu.from, lu.to, value
        config : dict, optional
            Solver configuration. If None, uses default config.
        """
        self.areas = areas.copy()
        self.priors = priors.copy()
        self.targets = targets.copy()
        self.restrictions = restrictions.copy() if restrictions is not None else None
        
        # Default configuration matching R implementation
        self.config = {
            'algorithm': 'L-BFGS-B',
            'max_exp': 700.0,
            'xtol_rel': 1e-4,
            'xtol_abs': 1e-6,
            'maxiter': 1000,
            'cutoff': 0.0,
            'max_diff': 1e-4,
            'use_grid_search_fallback': True,
            'grid_search_iterations': 10,
            'verbose': True
        }
        
        if config:
            self.config.update(config)
        
        self.results = {}
        self._setup_logging()
        
        logger.info("BiasCorrectSolver initialized")
        logger.info(f"Areas: {len(areas)} rows")
        logger.info(f"Priors: {len(priors)} rows")
        logger.info(f"Targets: {len(targets)} rows")
        if restrictions is not None:
            logger.info(f"Restrictions: {len(restrictions)} rows")
    
    def _setup_logging(self):
        """Configure logging based on verbosity setting."""
        if self.config.get('verbose', True):
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
    
    def solve(self, target_year: int = 2030) -> Dict[str, Any]:
        """
        Run the bias correction solver.
        
        This is the main entry point for solving the bias correction problem.
        It iterates over all lu.from classes and optimizes allocations for each.
        
        Parameters
        ----------
        target_year : int, default=2030
            Target year for projections (used for metadata)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'allocations': DataFrame with pixel-level allocations
            - 'summary': Summary statistics
            - 'convergence': Convergence information per lu.from class
            - 'metadata': Solver metadata
        
        Notes
        -----
        The algorithm:
        1. For each lu.from class:
           a. Extract relevant data
           b. Prepare priors and restrictions matrices
           c. Optimize scaling factors
           d. Calculate allocations
           e. Handle convergence failures with grid search
        2. Aggregate results across all classes
        3. Add residual flows (pixels that don't transition)
        """
        start_time = time.time()
        
        logger.info("="*70)
        logger.info("STARTING BIAS CORRECTION SOLVER")
        logger.info("="*70)
        
        # Get unique lu.from classes
        lu_from_classes = sorted(self.targets['lu.from'].unique())
        logger.info(f"Processing {len(lu_from_classes)} lu.from classes: {lu_from_classes}")
        
        all_allocations = []
        convergence_info = {}
        
        for lu_from in lu_from_classes:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing lu.from = {lu_from}")
            logger.info(f"{'='*70}")
            
            try:
                allocation_df, conv_info = self._solve_for_class(lu_from)
                all_allocations.append(allocation_df)
                convergence_info[lu_from] = conv_info
                
            except Exception as e:
                logger.error(f"Failed to solve for lu.from={lu_from}: {e}")
                convergence_info[lu_from] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Combine all allocations
        if all_allocations:
            final_allocations = pd.concat(all_allocations, ignore_index=True)
        else:
            final_allocations = pd.DataFrame()
        
        elapsed = time.time() - start_time
        
        # Create summary
        summary = self._create_summary(final_allocations, convergence_info)
        
        # Store results
        self.results = {
            'allocations': final_allocations,
            'summary': summary,
            'convergence': convergence_info,
            'metadata': {
                'target_year': target_year,
                'solver_version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'elapsed_time': elapsed,
                'config': self.config
            }
        }
        
        logger.info(f"\n{'='*70}")
        logger.info("SOLVER COMPLETED")
        logger.info(f"{'='*70}")
        logger.info(f"Total time: {elapsed:.2f} seconds")
        logger.info(f"Total allocations: {len(final_allocations):,} rows")
        
        return self.results
    
    def _solve_for_class(self, lu_from: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Solve bias correction for a single lu.from class.
        
        Parameters
        ----------
        lu_from : int
            The lu.from class to process
        
        Returns
        -------
        tuple
            (allocation_dataframe, convergence_info)
        """
        # Extract targets for this class
        curr_targets = self.targets[self.targets['lu.from'] == lu_from].copy()
        curr_lu_to = sorted(curr_targets['lu.to'].unique())
        
        logger.info(f"Target classes (lu.to): {curr_lu_to}")
        logger.info(f"Number of targets: {len(curr_targets)}")
        
        # Create target array - aggregate by lu.to to sum all transitions to same destination
        # This ensures that when multiple classes transition to the same lu.to, we sum their targets
        curr_targets_agg = curr_targets.groupby('lu.to', as_index=False)['value'].sum()
        target_dict = dict(zip(curr_targets_agg['lu.to'], curr_targets_agg['value']))
        target_values = np.array([target_dict[lu_to] for lu_to in curr_lu_to])
        
        logger.info(f"Target totals: {target_values}")
        logger.info(f"Sum of targets: {target_values.sum():.2f}")
        
        # Extract areas for this class
        curr_areas = self.areas[self.areas['lu.from'] == lu_from].copy()
        pixel_ids = curr_areas['ns'].values
        area_dict = dict(zip(curr_areas['ns'], curr_areas['value']))
        area_values = np.array([area_dict[ns] for ns in pixel_ids])
        
        n_pixels = len(pixel_ids)
        logger.info(f"Number of pixels: {n_pixels:,}")
        logger.info(f"Total area: {area_values.sum():.2f}")
        
        # Prepare priors matrix (n_pixels × n_classes)
        priors_mu, pixel_order = self._prepare_priors_matrix(
            pixel_ids, curr_lu_to, lu_from
        )
        
        # Reorder areas to match priors
        area_values = np.array([area_dict[ns] for ns in pixel_order])
        
        # Prepare restrictions matrix if available
        restrictions_mat = self._prepare_restrictions_matrix(
            pixel_order, curr_lu_to, lu_from
        )
        
        # Check for all-zero targets
        if np.all(target_values == 0):
            logger.info("All targets are zero - skipping optimization")
            # Return zero allocations
            allocation_data = []
            for ns in pixel_order:
                for lu_to in curr_lu_to:
                    allocation_data.append({
                        'ns': ns,
                        'lu.from': lu_from,
                        'lu.to': lu_to,
                        'value': 0.0
                    })
            return pd.DataFrame(allocation_data), {'success': True, 'method': 'zero_targets'}
        
        # Remove zero-target classes
        nonzero_mask = target_values != 0
        if not np.all(nonzero_mask):
            logger.info(f"Removing {(~nonzero_mask).sum()} zero-target classes")
            target_values = target_values[nonzero_mask]
            priors_mu = priors_mu[:, nonzero_mask]
            curr_lu_to = [lu for lu, nz in zip(curr_lu_to, nonzero_mask) if nz]
            if restrictions_mat is not None:
                restrictions_mat = restrictions_mat[:, nonzero_mask]
        
        # Initial guess: start with 1.0 for all classes
        # This gives the unbiased prior probabilities as starting point
        initial_x = np.ones(len(target_values))
        
        # Alternative: scale by target ratios if that helps
        # total_area = area_values.sum()
        # if total_area > 0:
        #     initial_x = target_values / total_area * len(target_values)
        
        # Run optimization
        opt_result = optimize_with_fallback(
            initial_x,
            priors_mu,
            area_values,
            target_values,
            restrictions_mat,
            self.config['cutoff'],
            self.config['max_diff'],
            max_exp=self.config['max_exp'],
            xtol_rel=self.config['xtol_rel'],
            xtol_abs=self.config['xtol_abs'],
            maxiter=self.config['maxiter']
        )
        
        convergence_info = {
            'success': opt_result['success'],
            'method': opt_result['method'],
            'max_diff': opt_result['max_diff'],
            'objective_value': opt_result['result'].fun
        }
        
        # If optimization failed and grid search is enabled, try grid search
        if not opt_result['success'] and self.config['use_grid_search_fallback']:
            logger.warning("Standard optimization failed - trying grid search fallback")
            opt_result = self._try_grid_search_fallback(
                priors_mu, area_values, target_values, restrictions_mat
            )
            convergence_info['method'] = 'grid_search_fallback'
            convergence_info['grid_search_used'] = True
        
        # Calculate final allocations
        mu = mu_mnl(
            opt_result['result'].x,
            priors_mu,
            area_values,
            restrictions_mat,
            self.config['cutoff']
        )
        
        allocations = mu * area_values[:, np.newaxis]
        
        # Verify results
        projected = allocations.sum(axis=0)
        logger.info(f"Projected totals: {projected}")
        logger.info(f"Target totals: {target_values}")
        logger.info(f"Differences: {projected - target_values}")
        logger.info(f"Max absolute diff: {opt_result['max_diff']:.6e}")
        
        # Convert to DataFrame
        allocation_data = []
        for i, ns in enumerate(pixel_order):
            for j, lu_to in enumerate(curr_lu_to):
                allocation_data.append({
                    'ns': ns,
                    'lu.from': lu_from,
                    'lu.to': lu_to,
                    'value': allocations[i, j]
                })
        
        return pd.DataFrame(allocation_data), convergence_info
    
    def _prepare_priors_matrix(
        self,
        pixel_ids: np.ndarray,
        lu_to_classes: list,
        lu_from: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare priors matrix from priors DataFrame.
        
        Returns
        -------
        tuple
            (priors_matrix, pixel_order)
        """
        # Filter priors for relevant pixels and classes
        relevant_priors = self.priors[
            self.priors['ns'].isin(pixel_ids) &
            self.priors['lu.to'].isin(lu_to_classes)
        ].copy()
        
        # Pivot to wide format
        priors_wide = relevant_priors.pivot_table(
            index='ns',
            columns='lu.to',
            values='value',
            fill_value=0.0
        )
        
        # Ensure all classes are present
        for lu_to in lu_to_classes:
            if lu_to not in priors_wide.columns:
                priors_wide[lu_to] = 0.0
        
        # Order columns
        priors_wide = priors_wide[lu_to_classes]
        
        # Get pixel order
        pixel_order = priors_wide.index.values
        
        # Convert to numpy array
        priors_matrix = priors_wide.values
        
        # Replace any NaN or negative values
        priors_matrix = np.nan_to_num(priors_matrix, nan=0.0, neginf=0.0, posinf=0.0)
        priors_matrix = np.maximum(priors_matrix, 0.0)
        
        logger.info(f"Priors matrix shape: {priors_matrix.shape}")
        logger.info(f"Priors range: [{priors_matrix.min():.6f}, {priors_matrix.max():.6f}]")
        
        return priors_matrix, pixel_order
    
    def _prepare_restrictions_matrix(
        self,
        pixel_ids: np.ndarray,
        lu_to_classes: list,
        lu_from: int
    ) -> Optional[np.ndarray]:
        """Prepare restrictions matrix."""
        if self.restrictions is None:
            return None
        
        # Filter restrictions
        relevant_restrictions = self.restrictions[
            self.restrictions['ns'].isin(pixel_ids) &
            self.restrictions['lu.from'] == lu_from &
            self.restrictions['lu.to'].isin(lu_to_classes)
        ].copy()
        
        if len(relevant_restrictions) == 0:
            return None
        
        # Pivot to wide format
        restrictions_wide = relevant_restrictions.pivot_table(
            index='ns',
            columns='lu.to',
            values='value',
            fill_value=0
        )
        
        # Ensure all pixels and classes are present
        restrictions_wide = restrictions_wide.reindex(
            index=pixel_ids,
            columns=lu_to_classes,
            fill_value=0
        )
        
        restrictions_matrix = restrictions_wide.values.astype(int)
        
        logger.info(f"Restrictions matrix shape: {restrictions_matrix.shape}")
        logger.info(f"Forbidden transitions: {(restrictions_matrix == 1).sum()}")
        
        return restrictions_matrix
    
    def _try_grid_search_fallback(
        self,
        priors_mu: np.ndarray,
        areas: np.ndarray,
        targets: np.ndarray,
        restrictions: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Try individual grid search for each target."""
        # Simplified fallback: use grid search for single scaling parameter
        result = iterated_grid_search(
            min_param=-self.config['max_exp'],
            max_param=self.config['max_exp'],
            func=sqr_diff_mnl,
            max_iterations=self.config['grid_search_iterations'],
            exp_transform=True,
            priors_mu=priors_mu,
            areas=areas,
            targets=targets,
            restrictions=restrictions,
            cutoff=self.config['cutoff']
        )
        
        # Create mock OptimizeResult
        from scipy.optimize import OptimizeResult
        opt_result = OptimizeResult(
            x=np.full(len(targets), result['best_param']),
            fun=result['best_value'],
            success=result['converged'],
            message='Grid search fallback',
            nit=result['iterations']
        )
        
        return {
            'result': opt_result,
            'method': 'grid_search',
            'success': result['converged'],
            'max_diff': 0.0  # Will be recalculated
        }
    
    def _create_summary(
        self,
        allocations: pd.DataFrame,
        convergence_info: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Create summary statistics."""
        if len(allocations) == 0:
            return {'total_allocations': 0}
        
        summary = {
            'total_allocations': len(allocations),
            'total_pixels': allocations['ns'].nunique(),
            'lu_from_classes': sorted(allocations['lu.from'].unique().tolist()),
            'lu_to_classes': sorted(allocations['lu.to'].unique().tolist()),
            'total_area_allocated': allocations['value'].sum(),
            'convergence_summary': {
                'successful': sum(1 for v in convergence_info.values() if v.get('success', False)),
                'failed': sum(1 for v in convergence_info.values() if not v.get('success', False)),
                'total': len(convergence_info)
            }
        }
        
        return summary
