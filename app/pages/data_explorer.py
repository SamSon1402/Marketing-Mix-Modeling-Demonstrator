import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.components.visualizations import plot_time_series, plot_correlation_matrix

def render_data_explorer():
    """
    Render the data exploration page.
    """
    st.title("Data Explorer")
    
    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("No data available. Please generate or upload data first.")
        return
    
    data = st.session_state.data
    
    # Data summary
    st.subheader("Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Time Periods", len(data))
        st.metric("Sales Mean", f"${data['Sales'].mean():.2f}")
        
    with col2:
        channel_names = [col for col in data.columns if col not in ['Date', 'Sales', 'Seasonality', 'Trend', 'Noise']]
        st.metric("Marketing Channels", len(channel_names))
        st.metric("Total Marketing Spend", f"${data[channel_names].sum().sum():.2f}")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(data.head(10))
    
    # Time series visualization
    st.subheader("Time Series Visualization")
    
    # Channel selection
    selected_channels = st.multiselect(
        "Select marketing channels to display:",
        channel_names,
        default=channel_names[:3] if len(channel_names) > 3 else channel_names
    )
    
    if not selected_channels:
        st.warning("Please select at least one channel to display.")
    else:
        # Marketing spend over time
        fig = plot_time_series(
            data, 
            x_col='Date', 
            y_cols=selected_channels, 
            title='Marketing Spend Over Time',
            y_axis_title='Spend'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sales over time
    fig = plot_time_series(
        data, 
        x_col='Date', 
        y_cols=['Sales'], 
        title='Sales Over Time',
        y_axis_title='Sales'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Correlation heatmap
    correlation_cols = channel_names + ['Sales']
    if 'Seasonality' in data.columns:
        correlation_cols.append('Seasonality')
    if 'Trend' in data.columns:
        correlation_cols.append('Trend')
    
    fig = plot_correlation_matrix(data, correlation_cols)
    st.plotly_chart(fig, use_container_width=True)
    
    # Channel-sales scatter plots
    st.subheader("Channel-Sales Relationships")
    
    selected_channel = st.selectbox(
        "Select a marketing channel to analyze its relationship with sales:",
        channel_names
    )
    
    fig = px.scatter(
        data,
        x=selected_channel,
        y='Sales',
        trendline='ols',
        labels={selected_channel: f"{selected_channel} Spend", 'Sales': 'Sales'},
        title=f"Relationship between {selected_channel} and Sales"
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(
            title=f"{selected_channel} Spend",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100, 100, 255, 0.1)'
        ),
        yaxis=dict(
            title="Sales",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100, 100, 255, 0.1)'
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Lag analysis
    st.subheader("Lag Analysis")
    
    max_lag = min(12, len(data) // 4)  # Limit maximum lag to 12 periods or 1/4 of data length
    selected_lag = st.slider("Select lag periods to analyze:", 0, max_lag, 2)
    
    if selected_lag > 0:
        # Create lagged data
        lagged_data = data.copy()
        lagged_data[f'{selected_channel}_lag_{selected_lag}'] = lagged_data[selected_channel].shift(selected_lag)
        lagged_data = lagged_data.dropna()
        
        fig = px.scatter(
            lagged_data,
            x=f'{selected_channel}_lag_{selected_lag}',
            y='Sales',
            trendline='ols',
            labels={f'{selected_channel}_lag_{selected_lag}': f"{selected_channel} Spend (Lag {selected_lag})", 'Sales': 'Sales'},
            title=f"Relationship between {selected_channel} (Lag {selected_lag}) and Sales"
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            xaxis=dict(
                title=f"{selected_channel} Spend (Lag {selected_lag})",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100, 100, 255, 0.1)'
            ),
            yaxis=dict(
                title="Sales",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100, 100, 255, 0.1)'
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display correlation for different lags
        lag_correlations = []
        
        for lag in range(1, max_lag + 1):
            lagged_series = data[selected_channel].shift(lag)
            correlation = data['Sales'].corr(lagged_series)
            lag_correlations.append({
                'Lag': lag,
                'Correlation': correlation
            })
        
        lag_corr_df = pd.DataFrame(lag_correlations)
        
        fig = px.bar(
            lag_corr_df,
            x='Lag',
            y='Correlation',
            labels={'Lag': 'Lag Periods', 'Correlation': f'Correlation with Sales'},
            title=f"Correlation between {selected_channel} and Sales at Different Lags"
        )
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(25, 25, 44, 0.0)',
            paper_bgcolor='rgba(25, 25, 44, 0.0)',
            xaxis=dict(
                title="Lag Periods",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100, 100, 255, 0.1)'
            ),
            yaxis=dict(
                title="Correlation with Sales",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(100, 100, 255, 0.1)'
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution analysis
    st.subheader("Distribution Analysis")
    
    dist_col = st.selectbox(
        "Select a column to visualize its distribution:",
        channel_names + ['Sales']
    )
    
    fig = px.histogram(
        data,
        x=dist_col,
        nbins=30,
        labels={dist_col: dist_col},
        title=f"Distribution of {dist_col}"
    )
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor='rgba(25, 25, 44, 0.0)',
        paper_bgcolor='rgba(25, 25, 44, 0.0)',
        xaxis=dict(
            title=dist_col,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100, 100, 255, 0.1)'
        ),
        yaxis=dict(
            title="Frequency",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(100, 100, 255, 0.1)'
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(data[channel_names + ['Sales']].describe())
    
    # Download data
    st.download_button(
        "Download Data as CSV",
        data.to_csv(index=False),
        "mmm_data.csv",
        "text/csv",
        key='download-csv'
    )

if __name__ == "__main__":
    st.set_page_config(
        page_title="Data Explorer",
        page_icon="📊",
        layout="wide"
    )
    render_data_explorer()