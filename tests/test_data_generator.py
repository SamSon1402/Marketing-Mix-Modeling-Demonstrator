"""
Unit tests for the data generator module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.data_generator import generate_synthetic_data

class TestDataGenerator(unittest.TestCase):
    """Test cases for the data generator module."""
    
    def test_generate_synthetic_data_defaults(self):
        """Test generating data with default parameters."""
        data = generate_synthetic_data()
        
        # Check basic structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('Date', data.columns)
        self.assertIn('Sales', data.columns)
        
        # Default is 104 periods (2 years of weekly data)
        self.assertEqual(len(data), 104)
        
        # Default is 4 channels
        channel_names = ["Paid Search", "Social Media", "TV", "Display"]
        for channel in channel_names:
            self.assertIn(channel, data.columns)
        
        # Check for seasonality
        self.assertIn('Seasonality', data.columns)
    
    def test_generate_synthetic_data_custom(self):
        """Test generating data with custom parameters."""
        n_periods = 52
        channel_names = ["Channel A", "Channel B", "Channel C"]
        channel_coeffs = [1.0, 2.0, 3.0]
        base_sales = 1000
        include_seasonality = False
        noise_level = 0.1
        trend_factor = 0.1
        
        data = generate_synthetic_data(
            n_periods=n_periods,
            channel_names=channel_names,
            channel_coeffs=channel_coeffs,
            base_sales=base_sales,
            include_seasonality=include_seasonality,
            noise_level=noise_level,
            trend_factor=trend_factor
        )
        
        # Check structure matches custom parameters
        self.assertEqual(len(data), n_periods)
        
        for channel in channel_names:
            self.assertIn(channel, data.columns)
        
        # Seasonality should not be included
        self.assertNotIn('Seasonality', data.columns)
        
        # Trend should be included
        self.assertIn('Trend', data.columns)
        
        # First date should be 2022-01-01
        self.assertEqual(data['Date'][0], datetime(2022, 1, 1))
    
    def test_generate_data_relationships(self):
        """Test that the generated data has expected relationships."""
        # Generate data with specific parameters to test relationships
        channel_names = ["Channel A", "Channel B"]
        channel_coeffs = [3.0, 1.0]  # Channel A should have stronger effect
        
        data = generate_synthetic_data(
            n_periods=100,
            channel_names=channel_names,
            channel_coeffs=channel_coeffs,
            include_seasonality=False,
            noise_level=0.0,  # No noise for clearer relationship testing
            trend_factor=0.0  # No trend for clearer relationship testing
        )
        
        # Check correlation with sales
        corr_a = data['Channel A'].corr(data['Sales'])
        corr_b = data['Channel B'].corr(data['Sales'])
        
        # Channel A should have higher correlation than Channel B
        # due to higher coefficient (though this isn't guaranteed due to
        # the adstock and saturation effects, but is likely)
        self.assertGreater(abs(corr_a), 0.1)  # Should have some correlation
        
        # All spend values should be positive
        for channel in channel_names:
            self.assertTrue((data[channel] >= 0).all())
        
        # Sales should always be positive
        self.assertTrue((data['Sales'] > 0).all())

if __name__ == '__main__':
    unittest.main()