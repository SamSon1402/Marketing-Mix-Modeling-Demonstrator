"""
Reusable visualization components for the Marketing Mix Modeling Streamlit application.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_time_series(data, x_col, y_cols, title=None, x_axis_title=None, y_axis_title=None):
    """
    Plot time series data for multiple columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to plot
    x_col : str
        Column name for the x-axis (typically a date column)
    y_cols : list
        List of column names for the y-axis
    title : str, optional
        Plot title
    x_axis_title : str, optional
        X-axis title
    y_axis_title : str, optional
        Y-axis title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    for y_col in y_cols:
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[y_col],
            name=y_col,
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title or x_col,
        yaxis_title=y_axis_title or y_cols[0],
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    return fig

def plot_correlation_matrix(data, columns=None):
    """
    Plot correlation matrix for selected columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data to plot
    columns : list, optional
        List of column names to include in the correlation matrix
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    corr = data[columns].corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation"),
        x=columns,
        y=columns
    )
    
    fig.update_layout(
        title="Correlation Matrix",
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(title=None),
        yaxis=dict(title=None)
    )
    
    return fig

def plot_model_coefficients(model, feature_names):
    """
    Plot model coefficients.
    
    Parameters:
    -----------
    model : fitted model
        Trained model with coefficients
    feature_names : list
        List of feature names
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Get coefficients
    if hasattr(model, 'coef_'):
        coefficients = model.coef_
    else:
        raise ValueError("Model does not have 'coef_' attribute")
    
    # Create a DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
    
    # Create color mapping based on coefficient value
    coef_df['Color'] = np.where(coef_df['Coefficient'] > 0, 'Positive', 'Negative')
    
    # Create the plot
    fig = px.bar(
        coef_df,
        x='Feature',
        y='Coefficient',
        color='Color',
        color_discrete_map={'Positive': 'rgba(100, 200, 255, 0.7)', 'Negative': 'rgba(255, 100, 200, 0.7)'},
        labels={'Feature': 'Feature', 'Coefficient': 'Coefficient Value'}
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=len(coef_df) - 0.5,
        y0=0,
        y1=0,
        line=dict(color='rgba(255, 255, 255, 0.5)', width=1, dash='dash')
    )
    
    fig.update_layout(
        title="Model Coefficients",
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    return fig

def plot_actual_vs_predicted(dates, y_true, y_pred):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    dates : array-like
        Date values for the x-axis
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines',
        name='Actual',
        line=dict(color='rgba(100, 200, 255, 0.7)', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='rgba(200, 100, 255, 0.7)', width=2)
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    return fig

def plot_contributions(contributions, channel_names):
    """
    Plot stacked area chart of sales decomposition.
    
    Parameters:
    -----------
    contributions : pandas.DataFrame
        DataFrame with decomposed contributions
    channel_names : list
        List of channel names
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add area traces for each channel with a neon color palette
    colors = px.colors.sequential.Plasma
    
    # Start with intercept
    if 'Intercept' in contributions.columns:
        fig.add_trace(go.Scatter(
            x=contributions['Date'],
            y=contributions['Intercept'],
            mode='lines',
            stackgroup='one',
            name='Base Sales',
            line=dict(width=0.5, color=colors[0]),
            fillcolor=colors[0]
        ))
    
    # Add channels
    for i, col in enumerate(channel_names):
        if col in contributions.columns:
            color_idx = (i + 1) % len(colors)
            fig.add_trace(go.Scatter(
                x=contributions['Date'],
                y=contributions[col],
                mode='lines',
                stackgroup='one',
                name=col,
                line=dict(width=0.5, color=colors[color_idx]),
                fillcolor=colors[color_idx]
            ))
    
    # Add seasonality and trend if they exist
    for col, name in [('Seasonality', 'Seasonal Effects'), ('Trend', 'Trend Component')]:
        if col in contributions.columns:
            color_idx = (len(channel_names) + 2) % len(colors)
            fig.add_trace(go.Scatter(
                x=contributions['Date'],
                y=contributions[col],
                mode='lines',
                stackgroup='one',
                name=name,
                line=dict(width=0.5, color=colors[color_idx]),
                fillcolor=colors[color_idx]
            ))
    
    # Add line trace for actual sales
    fig.add_trace(go.Scatter(
        x=contributions['Date'],
        y=contributions['Actual_Sales'],
        mode='lines',
        name='Actual Sales',
        line=dict(color='white', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title="Sales Decomposition by Channel",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        hovermode="x unified"
    )
    
    return fig

def plot_roi_bar_chart(roi_df):
    """
    Plot ROI bar chart.
    
    Parameters:
    -----------
    roi_df : pandas.DataFrame
        DataFrame with ROI calculations
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Sort by ROI
    roi_df = roi_df.sort_values('ROI', ascending=False).copy()
    
    # Create the plot
    fig = px.bar(
        roi_df,
        x='Channel',
        y='ROI',
        text='ROI',
        labels={'Channel': 'Marketing Channel', 'ROI': 'Return on Investment'},
        color='ROI',
        color_continuous_scale='viridis'
    )
    
    # Format text labels
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    
    fig.update_layout(
        title='ROI by Channel',
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(100, 100, 255, 0.1)')
    )
    
    return fig