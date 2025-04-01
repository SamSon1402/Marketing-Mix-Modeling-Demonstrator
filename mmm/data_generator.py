import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(
    n_periods=104,  # 2 years of weekly data by default
    channel_names=None,
    channel_coeffs=None,
    base_sales=500,
    include_seasonality=True,
    noise_level=0.2,
    trend_factor=0.05
):
    """
    Generate synthetic marketing and sales data for MMM demonstration.
    
    Parameters:
    -----------
    n_periods : int
        Number of periods (e.g., weeks) to generate data for
    channel_names : list
        List of channel names
    channel_coeffs : list
        List of channel effectiveness coefficients
    base_sales : float
        Base sales level (non-marketing driven)
    include_seasonality : bool
        Whether to include a seasonality component
    noise_level : float
        Level of random noise to add (0.0 to 1.0)
    trend_factor : float
        Trend factor for sales (can be positive or negative)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with synthetic marketing and sales data
    """
    # Default channel names and coefficients if not provided
    if channel_names is None:
        channel_names = ["Paid Search", "Social Media", "TV", "Display"]
    
    n_channels = len(channel_names)
    
    if channel_coeffs is None:
        channel_coeffs = np.linspace(0.5, 2.0, n_channels)
    
    # Create date range
    start_date = datetime(2022, 1, 1)
    date_range = [start_date + timedelta(weeks=i) for i in range(n_periods)]
    
    # Initialize the DataFrame with dates
    data = pd.DataFrame({"Date": date_range})
    
    # Generate marketing spend for each channel
    for i, channel in enumerate(channel_names):
        # Base spend pattern with some randomness
        spend = np.random.normal(1000, 200, n_periods)
        
        # Add seasonality to some channels
        if i % 2 == 0:  # Every other channel has different seasonality
            seasonal_factor = np.sin(np.linspace(0, 2 * np.pi * (n_periods/52), n_periods))
            spend *= (1 + 0.3 * seasonal_factor)
        else:
            seasonal_factor = np.cos(np.linspace(0, 2 * np.pi * (n_periods/52), n_periods))
            spend *= (1 + 0.2 * seasonal_factor)
        
        # Add some trend
        trend = np.linspace(0, 0.5, n_periods)
        if i % 3 == 0:  # Some channels have positive trend, others negative
            spend *= (1 + trend * 0.5)
        else:
            spend *= (1 - trend * 0.3)
        
        # Add some promotion spikes
        n_spikes = n_periods // 13  # Approximately quarterly promotions
        spike_indices = np.random.choice(range(n_periods), n_spikes, replace=False)
        spike_multiplier = np.random.uniform(1.5, 3.0, n_spikes)
        
        for idx, multiplier in zip(spike_indices, spike_multiplier):
            spend[idx] *= multiplier
        
        # Ensure all spend is positive
        spend = np.maximum(spend, 100)
        
        # Add to DataFrame
        data[channel] = spend
    
    # Generate sales based on marketing spend and other factors
    # Start with base sales
    sales = np.ones(n_periods) * base_sales
    
    # Add trend component
    if trend_factor != 0:
        trend = np.linspace(0, trend_factor * n_periods, n_periods)
        sales *= (1 + trend)
    
    # Add seasonality component
    if include_seasonality:
        seasonality = 0.2 * np.sin(np.linspace(0, 2 * np.pi * (n_periods/52), n_periods))
        sales *= (1 + seasonality)
        data["Seasonality"] = seasonality
    
    # Add marketing contribution
    for i, channel in enumerate(channel_names):
        # Apply carryover effect (simple adstock)
        decay_rate = 0.3 + 0.4 * np.random.random()  # Random between 0.3 and 0.7
        adstock = np.zeros(n_periods)
        
        for t in range(n_periods):
            if t == 0:
                adstock[t] = data[channel].values[t]
            else:
                adstock[t] = data[channel].values[t] + (1 - decay_rate) * adstock[t-1]
        
        # Apply saturation effect (diminishing returns)
        saturation_k = 0.0001 + 0.0003 * np.random.random()  # Random parameter
        saturation = 1 - np.exp(-saturation_k * adstock)
        
        # Add to sales with coefficient effect
        contribution = channel_coeffs[i] * saturation * base_sales
        sales += contribution
    
    # Add noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * base_sales, n_periods)
        sales += noise
        data["Noise"] = noise
    
    # Add trend to the dataframe for reference
    if trend_factor != 0:
        data["Trend"] = trend
    
    # Add sales to DataFrame
    data["Sales"] = sales
    
    return data

if __name__ == "__main__":
    # Example usage
    data = generate_synthetic_data()
    print(data.head())
    print(f"Data shape: {data.shape}")