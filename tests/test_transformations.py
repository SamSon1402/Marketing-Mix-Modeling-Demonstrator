"""
Unit tests for the transformations module.
"""

import unittest
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.transformations import (
    apply_adstock,
    apply_hill_adstock,
    apply_saturation,
    apply_s_curve,
    transform_marketing_data
)

class TestTransformations(unittest.TestCase):
    """Test cases for the transformations module."""
    
    def test_apply_adstock(self):
        """Test the adstock transformation."""
        # Simple test case with a pulse input
        x = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        decay_rate = 0.5
        
        result = apply_adstock(x, decay_rate)
        
        # First value should be unchanged
        self.assertEqual(result[0], x[0])
        
        # Subsequent values should decay
        for i in range(1, len(x)):
            expected = (1 - decay_rate) * result[i-1]
            self.assertAlmostEqual(result[i], expected)
        
        # The sum of the adstock values should be greater than the original input
        # due to the carryover effect
        self.assertGreater(sum(result), sum(x))
    
    def test_apply_hill_adstock(self):
        """Test the Hill adstock transformation."""
        # Simple test case with a pulse input
        x = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        decay_rate = 0.5
        lag = 3
        
        result = apply_hill_adstock(x, decay_rate, lag)
        
        # Peak should be at or near the lag value
        peak_idx = np.argmax(result)
        self.assertTrue(abs(peak_idx - lag) <= 1)  # Allow for slight deviation
        
        # The sum of the adstock values should be greater than the original input
        self.assertGreater(sum(result), sum(x))
    
    def test_apply_saturation(self):
        """Test the saturation transformation."""
        # Linear input
        x = np.linspace(0, 1000, 11)
        k = 0.01
        
        result = apply_saturation(x, k)
        
        # Output should be monotonically increasing but with diminishing returns
        for i in range(1, len(x)):
            # Values should increase
            self.assertGreater(result[i], result[i-1])
            
            if i > 1:
                # But with diminishing marginal returns
                prev_diff = result[i-1] - result[i-2]
                curr_diff = result[i] - result[i-1]
                self.assertGreater(prev_diff, curr_diff)
        
        # Check bounds
        self.assertAlmostEqual(result[0], 0)  # At x=0, output should be 0
        self.assertTrue(result[-1] < x[-1])  # Saturated output should be less than input
    
    def test_apply_s_curve(self):
        """Test the S-curve transformation."""
        # Linear input
        x = np.linspace(0, 1000, 101)
        k = 0.01
        inflection = 500
        
        result = apply_s_curve(x, k, inflection)
        
        # Output should follow S-curve characteristics
        # Around inflection point, the second derivative changes sign
        
        # Calculate first differences
        first_diff = np.diff(result)
        
        # First half should have increasing differences (convex)
        for i in range(1, len(first_diff) // 2 - 5):  # Allow some margin
            self.assertGreaterEqual(first_diff[i], first_diff[i-1])
        
        # Second half should have decreasing differences (concave)
        for i in range(len(first_diff) // 2 + 5, len(first_diff) - 1):  # Allow some margin
            self.assertGreaterEqual(first_diff[i-1], first_diff[i])
        
        # Check bounds
        self.assertAlmostEqual(result[0], 0, delta=1e-10)  # At x=0, output should be near 0
        self.assertTrue(result[-1] < x[-1])  # Output should be less than input at high values
    
    def test_transform_marketing_data(self):
        """Test the transform_marketing_data function."""
        # Create test data
        data = pd.DataFrame({
            'Channel A': [100, 200, 150, 50],
            'Channel B': [50, 75, 100, 25]
        })
        
        channel_cols = ['Channel A', 'Channel B']
        
        # Define transformation parameters
        adstock_params = {
            'Channel A': {'decay_rate': 0.3},
            'Channel B': {'decay_rate': 0.5}
        }
        
        saturation_params = {
            'Channel A': {'k': 0.05},
            'Channel B': {'k': 0.1}
        }
        
        # Transform the data
        transformed_data = transform_marketing_data(
            data,
            channel_cols,
            adstock_params,
            saturation_params
        )
        
        # Check that all expected columns are present
        expected_cols = [
            'Channel A', 'Channel B',
            'Channel A_adstock', 'Channel B_adstock',
            'Channel A_sat', 'Channel B_sat'
        ]
        
        for col in expected_cols:
            self.assertIn(col, transformed_data.columns)
        
        # Check that original data is unchanged
        for channel in channel_cols:
            np.testing.assert_array_equal(data[channel].values, transformed_data[channel].values)

if __name__ == '__main__':
    unittest.main()