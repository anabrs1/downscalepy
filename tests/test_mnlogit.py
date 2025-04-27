"""
Tests for the mnlogit module.
"""

import unittest
import numpy as np
from downscalepy import mnlogit


class TestMnlogit(unittest.TestCase):
    """Test cases for the mnlogit function."""
    
    def setUp(self):
        """Set up test data."""
        n = 20  # Number of observations
        k = 3   # Number of explanatory variables
        p = 4   # Number of classes
        
        np.random.seed(42)
        self.X = np.random.normal(size=(n, k))
        
        probs = np.random.dirichlet(np.ones(p), size=n)
        self.Y = np.zeros((n, p))
        for i in range(n):
            self.Y[i, :] = probs[i, :]
    
    def test_mnlogit_basic(self):
        """Test basic mnlogit functionality."""
        result = mnlogit(
            X=self.X,
            Y=self.Y,
            niter=5,
            nburn=2
        )
        
        self.assertIn('postb', result)
        self.assertIn('X', result)
        self.assertIn('Y', result)
        self.assertIn('baseline', result)
        
        k, p = self.X.shape[1], self.Y.shape[1]
        niter, nburn = 5, 2
        expected_shape = (k, p, niter - nburn)
        self.assertEqual(result['postb'].shape, expected_shape)
    
    def test_mnlogit_with_marginal_fx(self):
        """Test mnlogit with marginal effects calculation."""
        result = mnlogit(
            X=self.X,
            Y=self.Y,
            niter=5,
            nburn=2,
            calc_marginal_fx=True
        )
        
        self.assertIn('marginal_fx', result)
        self.assertIsNotNone(result['marginal_fx'])
        
        k, p = self.X.shape[1], self.Y.shape[1]
        niter, nburn = 5, 2
        expected_shape = (k, p, niter - nburn)
        self.assertEqual(result['marginal_fx'].shape, expected_shape)
    
    def test_mnlogit_with_baseline(self):
        """Test mnlogit with specified baseline."""
        baseline = 0
        result = mnlogit(
            X=self.X,
            Y=self.Y,
            baseline=baseline,
            niter=5,
            nburn=2
        )
        
        self.assertEqual(result['baseline'], baseline)


if __name__ == '__main__':
    unittest.main()
