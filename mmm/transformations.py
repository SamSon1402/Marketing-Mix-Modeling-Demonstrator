import numpy as np
import pandas as pd

def apply_adstock(x, decay_rate=0.5, normalize=True):
    """
    Apply adstock transformation to a marketing time series.
    
    The adstock transformation models the carryover effect of marketing
    activities, where the impact of marketing decays over time.
    
    Parameters:
    -----------
    x : array-like
        Original marketing time series
    decay_rate : float, default=0.5
        Rate at which marketing effect decays (between 0 and 1)
        Higher values mean faster decay
    normalize : bool, default=True
        Whether to normalize the resulting adstock values
        
    Returns:
    --------
    numpy.ndarray
        Transformed time series with adstock effect
    """
    # Convert to numpy array if not already
    x = np.array(x)
    n = len(x)
    
    # Initialize adstock array
    adstock = np.zeros(n)
    
    # Apply recursive formula for adstock
    for t in range(n):
        if t == 0:
            adstock[t] = x[t]
        else:
            adstock[t] = x[t] + (1 - decay_rate) * adstock[t-1]
    
    # Normalize if requested
    if normalize and np.sum(adstock) > 0:
        adstock = adstock / np.max(adstock) * np.max(x)
    
    return adstock

def apply_hill_adstock(x, decay_rate=0.5, lag=1, normalize=True):
    """
    Apply Hill adstock transformation with lag to a marketing time series.
    
    A more flexible adstock transformation that allows for delayed peak effect.
    
    Parameters:
    -----------
    x : array-like
        Original marketing time series
    decay_rate : float, default=0.5
        Rate at which marketing effect decays (between 0 and 1)
    lag : int, default=1
        Number of periods before peak effect
    normalize : bool, default=True
        Whether to normalize the resulting adstock values
        
    Returns:
    --------
    numpy.ndarray
        Transformed time series with Hill adstock effect
    """
    # Convert to numpy array if not already
    x = np.array(x)
    n = len(x)
    
    # Initialize weights array (for the lag effect)
    if lag <= 1:
        weights = np.array([1.0])
    else:
        # Create weights that peak at the lag value
        theta = np.linspace(0, 1, lag)
        weights = theta * (1 - theta)**(decay_rate)
        weights = weights / np.sum(weights)
    
    # Initialize adstock array
    adstock = np.zeros(n)
    
    # Apply convolution for lagged effect
    for t in range(n):
        for l, w in enumerate(weights):
            if t - l >= 0:
                adstock[t] += w * x[t - l]
    
    # Apply decay after the peak effect
    for t in range(lag, n):
        adstock[t] = adstock[t] + (1 - decay_rate) * adstock[t - 1]
    
    # Normalize if requested
    if normalize and np.sum(adstock) > 0:
        adstock = adstock / np.max(adstock) * np.max(x)
    
    return adstock

def apply_saturation(x, k=0.1, exponent=1.0, max_value=None):
    """
    Apply saturation transformation to model diminishing returns.
    
    Uses a modified exponential function to model how marketing effectiveness
    diminishes with increased spend.
    
    Parameters:
    -----------
    x : array-like
        Original marketing time series
    k : float, default=0.1
        Saturation rate parameter
    exponent : float, default=1.0
        Exponent parameter for more flexible curves
    max_value : float, optional
        Maximum value for the saturation curve
        
    Returns:
    --------
    numpy.ndarray
        Transformed time series with saturation effect
    """
    # Convert to numpy array if not already
    x = np.array(x)
    
    # Apply saturation transformation
    if exponent == 1.0:
        # Standard saturation curve
        transformed = 1 - np.exp(-k * x)
    else:
        # Hill-type saturation curve for more flexibility
        transformed = 1 - np.exp(-(k * x)**exponent)
    
    # Scale to original range
    if max_value is None:
        max_value = np.max(x)
    
    transformed = transformed * max_value
    
    return transformed

def apply_s_curve(x, k=0.1, inflection=None, max_value=None):
    """
    Apply S-curve transformation to model diminishing returns with inflection point.
    
    Uses a logistic function to model how marketing effectiveness has
    an inflection point where ROI changes.
    
    Parameters:
    -----------
    x : array-like
        Original marketing time series
    k : float, default=0.1
        Steepness parameter
    inflection : float, optional
        Point of inflection (defaults to median of x)
    max_value : float, optional
        Maximum value for the saturation curve
        
    Returns:
    --------
    numpy.ndarray
        Transformed time series with S-curve effect
    """
    # Convert to numpy array if not already
    x = np.array(x)
    
    # Set default inflection point if not provided
    if inflection is None:
        inflection = np.median(x)
    
    # Apply S-curve transformation (logistic function)
    transformed = 1 / (1 + np.exp(-k * (x - inflection)))
    
    # Scale to original range
    if max_value is None:
        max_value = np.max(x)
    
    transformed = transformed * max_value
    
    return transformed

def transform_marketing_data(
    data, 
    channel_cols, 
    adstock_params=None, 
    saturation_params=None
):
    """
    Apply both adstock and saturation transformations to marketing data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing marketing time series
    channel_cols : list
        List of column names for marketing channels
    adstock_params : dict, optional
        Dictionary mapping channel names to adstock parameters
    saturation_params : dict, optional
        Dictionary mapping channel names to saturation parameters
        
    Returns:
    --------
    pandas.DataFrame
        Transformed marketing data
    """
    # Create a copy of the data
    transformed_data = data.copy()
    
    # Initialize dictionaries if not provided
    if adstock_params is None:
        adstock_params = {channel: {'decay_rate': 0.5} for channel in channel_cols}
    
    if saturation_params is None:
        saturation_params = {channel: {'k': 0.1} for channel in channel_cols}
    
    # Apply transformations to each channel
    for channel in channel_cols:
        # Get original values
        x = data[channel].values
        
        # Apply adstock transformation
        adstock_params_ch = adstock_params.get(channel, {'decay_rate': 0.5})
        adstock_col = f"{channel}_adstock"
        transformed_data[adstock_col] = apply_adstock(
            x, 
            decay_rate=adstock_params_ch.get('decay_rate', 0.5)
        )
        
        # Apply saturation transformation to adstock values
        saturation_params_ch = saturation_params.get(channel, {'k': 0.1})
        saturation_col = f"{channel}_sat"
        transformed_data[saturation_col] = apply_saturation(
            transformed_data[adstock_col],
            k=saturation_params_ch.get('k', 0.1)
        )
    
    return transformed_data

if __name__ == "__main__":
    # Example usage
    x = np.array([0, 10, 5, 0, 0, 0, 0, 0, 0, 0])
    
    adstock = apply_adstock(x, decay_rate=0.5)
    print("Adstock transformation:")
    print(adstock)
    
    saturated = apply_saturation(x, k=0.1)
    print("\nSaturation transformation:")
    print(saturated)
    
    # Example with both transformations
    adstock_saturated = apply_saturation(apply_adstock(x, decay_rate=0.5), k=0.1)
    print("\nAdstock + Saturation:")
    print(adstock_saturated)