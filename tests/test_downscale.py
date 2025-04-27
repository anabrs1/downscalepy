"""
Tests for the downscale module.
"""

import unittest
import numpy as np
import pandas as pd
from downscalepy import downscale, sim_luc


class TestDownscale(unittest.TestCase):
    """Test cases for the downscale function."""
    
    def setUp(self):
        """Set up test data."""
        self.data = sim_luc(n=10, p=3, k=2, tt=2, seed=42)
    
    def test_downscale_basic(self):
        """Test basic downscaling functionality."""
        result = downscale(
            targets=self.data['targets'],
            start_areas=self.data['start_areas'],
            xmat=self.data['xmat'],
            betas=self.data['betas'],
            times=['1', '2']
        )
        
        self.assertIn('out_res', result)
        self.assertIn('out_solver', result)
        self.assertIn('ds_inputs', result)
        
        expected_cols = ['ns', 'lu.to', 'value', 'lu.from', 'times']
        for col in expected_cols:
            self.assertIn(col, result['out_res'].columns)
        
        n = len(self.data['start_areas']['ns'].unique())
        p = len(self.data['targets']['lu.to'].unique())
        t = len(['1', '2'])
        expected_rows = n * p * t
        self.assertEqual(len(result['out_res']), expected_rows)
        
        for t in ['1', '2']:
            self.assertIn(t, result['out_solver'])
    
    def test_downscale_with_priors(self):
        """Test downscaling with priors."""
        ns_list = self.data['start_areas']['ns'].unique()
        lu_from = self.data['start_areas']['lu.from'].unique()[0]
        lu_to = self.data['targets']['lu.to'].unique()[0]
        
        priors = pd.DataFrame({
            'ns': ns_list,
            'lu.from': lu_from,
            'lu.to': lu_to,
            'value': np.random.uniform(0, 1, size=len(ns_list))
        })
        
        result = downscale(
            targets=self.data['targets'],
            start_areas=self.data['start_areas'],
            xmat=self.data['xmat'],
            betas=self.data['betas'],
            times=['1'],
            priors=priors
        )
        
        self.assertIn('out_res', result)
        self.assertIn('out_solver', result)
        self.assertIn('ds_inputs', result)
        
        self.assertIn('priors', result['ds_inputs'])
        self.assertIsNotNone(result['ds_inputs']['priors'])


if __name__ == '__main__':
    unittest.main()
