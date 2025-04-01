import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time

# Add the project root to the path so we can import the mmm package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.data_generator import generate_synthetic_data
from mmm.transformations import apply_adstock, apply_saturation
from mmm.models import train_linear_model, train_ridge_model
from mmm.evaluation import calculate_metrics, decompose_contributions
from mmm.optimization import optimize_budget

# Apply futuristic UI with serif fonts
st.set_page_config(
    page_title="Quantum MMM • 2040 Edition",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for futuristic UI and serif fonts
st.markdown("""
<style>
    /* Futuristic background and gradients */
    .stApp {
        background: linear-gradient(to bottom, #0a0a20, #1a1a35);
    }
    
    /* Serif fonts for text */
    html, body, p, div, h1, h2, h3, h4, h5, h6, ol, ul, li, span, label, input, button {
        font-family: 'Georgia', 'Times New Roman', serif !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(to bottom, #151530, #252545);
    }
    
    /* Futuristic cards for sections */
    div.stBlock {
        border-radius: 15px;
        padding: 1.5rem;
        background: rgba(30, 30, 70, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100, 100, 255, 0.3);
        box-shadow: 0 4px 25px rgba(0, 0, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Headers in futuristic style */
    h1, h2, h3 {
        background: linear-gradient(to right, #64c8ff, #c864ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Buttons styling */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(90deg, #3a3af7, #7a3af7);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(58, 58, 247, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(58, 58, 247, 0.5);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div {
        background-color: rgba(100, 100, 255, 0.2);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #64c8ff, #c864ff);
    }
    
    /* Tables with neon borders */
    .stDataFrame {
        border: 1px solid #5a5af7;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stDataFrame thead {
        background: linear-gradient(90deg, #151530, #252545);
    }
    
    /* Success message */
    .element-container div[data-testid="stAlert"] {
        background: linear-gradient(90deg, rgba(0, 255, 170, 0.1), rgba(0, 200, 255, 0.1));
        border: 1px solid rgba(0, 255, 200, 0.3);
        color: #00ffc8;
        border-radius: 10px;
    }
    
    /* Animations for transitions */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease forwards;
    }
    
    /* Custom metric containers */
    .metric-container {
        background: rgba(30, 30, 70, 0.6);
        border: 1px solid rgba(100, 100, 255, 0.3);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 100, 0.2);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-label {
        color: #aaaaff;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }
    
    .metric-value {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .metric-delta {
        color: #00ffc8;
        font-size: 0.9rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(30, 30, 70, 0.3);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(100, 100, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Helper function for animated section headers
def futuristic_header(title, icon="✧"):
    st.markdown(f"""
    <div class="animate-fade-in">
        <h2>{icon} {title}</h2>
    </div>
    """, unsafe_allow_html=True)

# Define animated container
def futuristic_container():
    return st.container()

# Define sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="margin-bottom: 0;">🔮 Quantum MMM</h1>
        <p style="color: #aaaaff; font-style: italic;">Advanced Marketing Analytics • 2040 Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "System Navigation",
        ["Introduction", "Data Generation", "Model Training", "Results Analysis", "Budget Optimization"]
    )
    
    st.divider()
    
    st.markdown("""
    <div style="position: absolute; bottom: 20px; padding: 10px; width: calc(100% - 40px);">
        <p style="color: #aaaaff; text-align: center; font-size: 0.8rem;">
            Designed for fifty-five<br>
            MMM Internship Application<br>
            © 2040 Neural Dynamics
        </p>
    </div>
    """, unsafe_allow_html=True)

# Create session state for storing data and models
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'contributions' not in st.session_state:
    st.session_state.contributions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Introduction page
if page == "Introduction":
    futuristic_header("QUANTUM MARKETING MIX MODELING", "🔮")
    
    with futuristic_container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="animate-fade-in" style="line-height: 1.8;">
                <p>Welcome to the advanced Quantum MMM platform, where traditional marketing analytics meets quantum computing principles to decode the complex relationships between marketing activities and business outcomes.</p>
                
                <p>This cutting-edge system demonstrates the evolution of Marketing Mix Modeling (MMM) in the year 2040, allowing marketers to understand multi-dimensional marketing effects with unprecedented precision.</p>
                
                <h3>Core Capabilities</h3>
                <ul>
                    <li><strong>Neural Data Synthesis</strong>: Generate synthetic marketing and sales data with realistic patterns</li>
                    <li><strong>Quantum Adstock Transformations</strong>: Model intricate carryover effects across time dimensions</li>
                    <li><strong>Hyperbolic Saturation Curves</strong>: Visualize diminishing returns with mathematical precision</li>
                    <li><strong>Multi-Model Analysis</strong>: Apply multiple algorithmic approaches simultaneously</li>
                    <li><strong>Dimensional Decomposition</strong>: Break down revenue streams into channel contributions</li>
                    <li><strong>Predictive Optimization</strong>: Leverage quantum algorithms for budget allocation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container animate-fade-in" style="height: 360px; display: flex; flex-direction: column; justify-content: center; margin-top: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem; background: linear-gradient(to right, #64c8ff, #c864ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">MMM</div>
                <p style="color: #aaaaff; font-size: 1.1rem;">Marketing Mix Modeling</p>
                <p style="color: white; font-size: 0.9rem; margin-top: 2rem; font-style: italic;">The science of quantifying marketing impact across dimensions</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    futuristic_header("KEY CONCEPTUAL FRAMEWORK", "✧")
    
    with futuristic_container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-container animate-fade-in">
                <h3 style="color: #64c8ff;">Adstock Dynamics</h3>
                <p>Quantum modeling of marketing effects persistence over time, capturing the complex decay patterns of advertising impact beyond traditional exponential decay.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-container animate-fade-in">
                <h3 style="color: #7a64ff;">Saturation Curves</h3>
                <p>Hyperbolic representation of diminishing returns as marketing investment increases, incorporating multi-dimensional response patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-container animate-fade-in">
                <h3 style="color: #c864ff;">Channel Attribution</h3>
                <p>Advanced decomposition of business outcomes into precise channel contributions using quantum-inspired attribution algorithms.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    st.info("This system represents a conceptual demonstration of advanced marketing analytics techniques. Navigate through the modules using the sidebar controls.")
    
    # Add futuristic animation at the bottom
    st.markdown("""
    <div style="margin-top: 2rem; text-align: center;">
        <svg width="200" height="50" viewBox="0 0 200 50">
            <rect x="0" y="20" width="200" height="10" fill="rgba(100, 100, 255, 0.1)" rx="5"></rect>
            <rect x="0" y="20" width="50" height="10" fill="url(#grad)" rx="5">
                <animate attributeName="x" from="0" to="150" dur="2s" repeatCount="indefinite" />
            </rect>
            <defs>
                <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" style="stop-color:#64c8ff;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#c864ff;stop-opacity:1" />
                </linearGradient>
            </defs>
        </svg>
        <p style="color: #aaaaff; font-size: 0.8rem; margin-top: 1rem;">NEURAL SYSTEMS ACTIVE • QUANTUM PROCESSING ENABLED</p>
    </div>
    """, unsafe_allow_html=True)

# Data Generation page
elif page == "Data Generation":
    futuristic_header("NEURAL DATA SYNTHESIS", "⚡")
    
    with futuristic_container():
        st.markdown("""
        <div class="animate-fade-in">
            <p>Generate synthetic marketing data through our advanced neural synthesis engine. This module creates realistic simulations of marketing activities and corresponding business outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("data_generation_form"):
            st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Data Parameters Configuration</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_periods = st.slider("Temporal Dimension (Weeks)", 52, 260, 104, help="Number of time periods to generate")
                n_channels = st.slider("Marketing Channel Dimensions", 2, 8, 4, help="Number of marketing channels to include")
                seasonality = st.checkbox("Include Seasonality Effects", True, help="Add cyclical patterns to the data")
                
            with col2:
                noise_level = st.slider("Stochastic Variance Level", 0.0, 1.0, 0.2, help="Random noise magnitude")
                trend_factor = st.slider("Long-term Trend Coefficient", -0.2, 0.2, 0.05, help="Directional trend in the data")
                base_sales = st.slider("Baseline Revenue Parameter", 100, 1000, 500, help="Non-marketing driven sales")
            
            st.divider()
            
            # Channel names and effectiveness (coefficients)
            st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Channel Effectiveness Matrix</h3>", unsafe_allow_html=True)
            
            channel_cols = st.columns(n_channels)
            channel_names = []
            channel_coeffs = []
            
            for i, col in enumerate(channel_cols):
                with col:
                    default_name = f"Channel {i+1}"
                    default_names = ["Quantum Search", "Neural Social", "Holographic TV", "4D Display", "Neural Print", "Thought Email", "Neuro Display", "Dream Influencer"]
                    default_name = default_names[i] if i < len(default_names) else default_name
                    
                    name = st.text_input(f"Channel {i+1} Identity", default_name)
                    coeff = st.slider(f"{name} Coefficient", 0.0, 5.0, 1.0 + i*0.5, help="Channel effectiveness parameter")
                    
                    channel_names.append(name)
                    channel_coeffs.append(coeff)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_button = st.form_submit_button("GENERATE SYNTHETIC DATA")
                
                if generate_button:
                    with st.spinner("Neural synthesis in progress..."):
                        # Add a small delay for effect
                        time.sleep(1.5)
                        
                        # Generate the data
                        data = generate_synthetic_data(
                            n_periods=n_periods,
                            channel_names=channel_names,
                            channel_coeffs=channel_coeffs,
                            base_sales=base_sales,
                            include_seasonality=seasonality,
                            noise_level=noise_level,
                            trend_factor=trend_factor
                        )
                        st.session_state.data = data
                        
                        # Reset models when new data is generated
                        st.session_state.models = {}
                        st.session_state.contributions = None
                        st.session_state.metrics = {}
                        
                        st.success("Quantum synthesis complete! Data manifested successfully.")
        
        # Display the data if it exists
        if st.session_state.data is not None:
            data = st.session_state.data
            
            st.divider()
            futuristic_header("SYNTHESIZED DATA VISUALIZATION", "📊")
            
            tab1, tab2, tab3 = st.tabs(["Data Preview", "Marketing Dynamics", "Correlation Matrix"])
            
            with tab1:
                st.markdown("<p style='color: #aaaaff;'>Neural synthesis output sample:</p>", unsafe_allow_html=True)
                st.dataframe(data.head(10), use_container_width=True)
                
                st.download_button(
                    label="Download Quantum Dataset",
                    data=data.to_csv(index=False),
                    file_name="quantum_mmm_synthetic_data.csv",
                    mime="text/csv",
                )
            
            with tab2:
                st.markdown("<p style='color: #aaaaff;'>Marketing channel dynamics over time:</p>", unsafe_allow_html=True)
                
                # Create a more futuristic Plotly figure for marketing spend
                fig1 = go.Figure()
                
                for channel in channel_names:
                    fig1.add_trace(go.Scatter(
                        x=data['Date'],
                        y=data[channel],
                        name=channel,
                        mode='lines',
                        line=dict(width=2),
                        hoverinfo='x+y+name'
                    ))
                
                fig1.update_layout(
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
                    xaxis=dict(
                        title="Temporal Dimension",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    yaxis=dict(
                        title="Investment Intensity",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=500
                )
                
                st.plotly_chart(fig1, use_container_width=True)
                
                st.markdown("<p style='color: #aaaaff;'>Revenue outcomes over time:</p>", unsafe_allow_html=True)
                
                # Create a futuristic sales visualization
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=data['Date'],
                    y=data['Sales'],
                    name='Revenue',
                    line=dict(color='rgb(100, 200, 255)', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(100, 200, 255, 0.1)'
                ))
                
                # Add trend line if exists
                if 'Trend' in data.columns:
                    trend_values = data['Trend'] * data['Sales'].mean()
                    fig2.add_trace(go.Scatter(
                        x=data['Date'],
                        y=trend_values,
                        name='Trend Component',
                        line=dict(color='rgb(200, 100, 255)', width=2, dash='dash'),
                        opacity=0.7
                    ))
                
                # Add seasonality if exists
                if 'Seasonality' in data.columns:
                    seasonality_values = data['Seasonality'] * data['Sales'].mean() + data['Sales'].mean()
                    fig2.add_trace(go.Scatter(
                        x=data['Date'],
                        y=seasonality_values,
                        name='Seasonal Component',
                        line=dict(color='rgb(255, 100, 200)', width=2, dash='dot'),
                        opacity=0.7
                    ))
                
                fig2.update_layout(
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
                    xaxis=dict(
                        title="Temporal Dimension",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    yaxis=dict(
                        title="Revenue Quantum",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab3:
                st.markdown("<p style='color: #aaaaff;'>Inter-dimensional correlation matrix:</p>", unsafe_allow_html=True)
                
                # Create correlation matrix with a futuristic theme
                corr = data[channel_names + ['Sales']].corr()
                
                fig3 = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale=[[0, "rgb(0, 0, 50)"], 
                                            [0.5, "rgb(40, 40, 80)"], 
                                            [1, "rgb(100, 200, 255)"]],
                    labels=dict(color="Correlation"),
                    x=[name.split()[0] for name in corr.columns],
                    y=[name.split()[0] for name in corr.columns]
                )
                
                fig3.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(25, 25, 44, 0.0)',
                    paper_bgcolor='rgba(25, 25, 44, 0.0)',
                    xaxis=dict(side="top"),
                    margin=dict(l=20, r=20, t=60, b=20),
                    height=500
                )
                
                st.plotly_chart(fig3, use_container_width=True)

# Model Training page
elif page == "Model Training":
    # Add implementation for Model Training page
    futuristic_header("QUANTUM MODEL CALIBRATION", "🧠")
    
    if st.session_state.data is None:
        st.warning("Please generate quantum data first on the Neural Data Synthesis module.")
    else:
        with futuristic_container():
            st.markdown("""
            <div class="animate-fade-in">
                <p>Calibrate quantum models on the synthesized marketing data. This module applies advanced transformations to model complex marketing effects.</p>
            </div>
            """, unsafe_allow_html=True)
            
            data = st.session_state.data
            channel_names = [col for col in data.columns if col not in ['Date', 'Sales', 'Seasonality', 'Trend', 'Noise']]
            
            with st.form("model_training_form"):
                st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Quantum Transformation Parameters</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div style='text-align: center;'><h4>Adstock Dynamics</h4><p style='color: #aaaaff;'>Models how marketing effects persist over time</p></div>", unsafe_allow_html=True)
                    
                    apply_adstock_transform = st.checkbox("Apply Quantum Adstock", True)
                    
                    # Different adstock rates for each channel
                    if apply_adstock_transform:
                        st.markdown("<p style='color: #aaaaff; margin-top: 10px;'>Temporal Decay Coefficients</p>", unsafe_allow_html=True)
                        
                        adstock_rates = {}
                        cols = st.columns(len(channel_names))
                        
                        for i, channel in enumerate(channel_names):
                            with cols[i]:
                                adstock_rates[channel] = st.slider(
                                    f"{channel.split()[0]}",
                                    min_value=0.1,
                                    max_value=0.9,
                                    value=0.3,
                                    key=f"adstock_{channel}"
                                )
                
                with col2:
                    st.markdown("<div style='text-align: center;'><h4>Saturation Curves</h4><p style='color: #aaaaff;'>Models diminishing returns on marketing spend</p></div>", unsafe_allow_html=True)
                    
                    apply_saturation_transform = st.checkbox("Apply Hyperbolic Saturation", True)
                    
                    # Different saturation parameters for each channel
                    if apply_saturation_transform:
                        st.markdown("<p style='color: #aaaaff; margin-top: 10px;'>Diminishing Returns Coefficients</p>", unsafe_allow_html=True)
                        
                        saturation_rates = {}
                        cols = st.columns(len(channel_names))
                        
                        for i, channel in enumerate(channel_names):
                            with cols[i]:
                                saturation_rates[channel] = st.slider(
                                    f"{channel.split()[0]}",
                                    min_value=0.01,
                                    max_value=0.5,
                                    value=0.1,
                                    key=f"saturation_{channel}"
                                )
                
                st.divider()
                
                st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Quantum Algorithm Selection</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    train_linear = st.checkbox("Linear Quantum Regression", True)
                    
                with col2:
                    train_ridge = st.checkbox("Regularized Dimensional Regression", True)
                    if train_ridge:
                        alpha = st.slider("Dimensional Regulation Coefficient", 0.1, 10.0, 1.0)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    train_button = st.form_submit_button("CALIBRATE QUANTUM MODELS")
                    
                    if train_button:
                        with st.spinner("Initializing quantum calibration sequence..."):
                            # Add a small delay for effect
                            time.sleep(1.5)
                            
                            # Create transformed features based on settings
                            transformed_data = data.copy()
                            X_cols = []
                            
                            # Apply adstock transformation if selected
                            if apply_adstock_transform:
                                for channel in channel_names:
                                    adstock_col = f"{channel}_adstock"
                                    transformed_data[adstock_col] = apply_adstock(
                                        transformed_data[channel],
                                        decay_rate=adstock_rates[channel]
                                    )
                                    X_cols.append(adstock_col)
                            else:
                                X_cols = channel_names.copy()
                                
                            # Apply saturation transformation if selected
                            if apply_saturation_transform:
                                # If we're using adstock, apply saturation to the adstock columns
                                columns_to_transform = X_cols
                                
                                for channel_idx, channel in enumerate(channel_names):
                                    col_to_transform = columns_to_transform[channel_idx]
                                    sat_col = f"{channel}_sat"
                                    
                                    transformed_data[sat_col] = apply_saturation(
                                        transformed_data[col_to_transform],
                                        k=saturation_rates[channel]
                                    )
                                    
                                    # Replace the column in X_cols
                                    X_cols[X_cols.index(col_to_transform)] = sat_col
                            
                            # Add seasonality if it exists in the data
                            if 'Seasonality' in transformed_data.columns:
                                X_cols.append('Seasonality')
                                
                            # Add trend if it exists in the data
                            if 'Trend' in transformed_data.columns:
                                X_cols.append('Trend')
                            
                            # Train the models
                            X = transformed_data[X_cols]
                            y = transformed_data['Sales']
                            
                            st.session_state.models = {}
                            st.session_state.metrics = {}
                            st.session_state.X_cols = X_cols
                            st.session_state.transformed_data = transformed_data
                            
                            if train_linear:
                                linear_model, linear_preds = train_linear_model(X, y)
                                st.session_state.models['Linear Quantum Regression'] = linear_model
                                metrics = calculate_metrics(y, linear_preds)
                                st.session_state.metrics['Linear Quantum Regression'] = metrics
                                
                            if train_ridge:
                                ridge_model, ridge_preds = train_ridge_model(X, y, alpha=alpha)
                                st.session_state.models['Regularized Dimensional Regression'] = ridge_model
                                metrics = calculate_metrics(y, ridge_preds)
                                st.session_state.metrics['Regularized Dimensional Regression'] = metrics
                            
                            # Calculate contributions
                            if len(st.session_state.models) > 0:
                                # Use the last trained model for contributions
                                model_name = list(st.session_state.models.keys())[-1]
                                model = st.session_state.models[model_name]
                                
                                contributions = decompose_contributions(
                                    model, 
                                    transformed_data[X_cols], 
                                    original_channels=channel_names
                                )
                                
                                # Add date and actual sales for plotting
                                contributions['Date'] = transformed_data['Date']
                                contributions['Actual_Sales'] = transformed_data['Sales']
                                
                                st.session_state.contributions = contributions
                            
                            st.success("Quantum calibration complete! Models have been synchronized to the data patterns.")
            
            # Display model results if they exist
            if st.session_state.models:
                st.divider()
                futuristic_header("MODEL PERFORMANCE METRICS", "📈")
                
                metrics_df = pd.DataFrame(st.session_state.metrics).T
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("<p style='color: #aaaaff;'>Quantum Algorithm Performance:</p>", unsafe_allow_html=True)
                    st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
                
                with col2:
                    # Plot actual vs predicted
                    if 'Linear Quantum Regression' in st.session_state.models:
                        model = st.session_state.models['Linear Quantum Regression']
                        X = st.session_state.transformed_data[st.session_state.X_cols]
                        y = st.session_state.transformed_data['Sales']
                        
                        y_pred = model.predict(X)
                        
                        st.markdown("<p style='color: #aaaaff;'>Actual vs. Quantum Predicted Revenue:</p>", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=st.session_state.transformed_data['Date'], 
                            y=y, 
                            mode='lines', 
                            name='Actual Revenue',
                            line=dict(color='rgb(100, 200, 255)', width=3)
                        ))
                        fig.add_trace(go.Scatter(
                            x=st.session_state.transformed_data['Date'], 
                            y=y_pred, 
                            mode='lines', 
                            name='Predicted Revenue',
                            line=dict(color='rgb(200, 100, 255)', width=3, dash='dash')
                        ))
                        
                        fig.update_layout(
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
                            xaxis=dict(
                                title="Temporal Dimension",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(100, 100, 255, 0.1)'
                            ),
                            yaxis=dict(
                                title="Revenue Quantum",
                                showgrid=True,
                                gridwidth=1,
                                gridcolor='rgba(100, 100, 255, 0.1)'
                            ),
                            margin=dict(l=20, r=20, t=20, b=20),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                futuristic_header("DIMENSIONAL COEFFICIENTS", "🔍")
                
                # Display coefficients
                all_coefs = {}
                for model_name, model in st.session_state.models.items():
                    coefs = pd.Series(model.coef_, index=st.session_state.X_cols)
                    if hasattr(model, 'intercept_'):
                        coefs['intercept'] = model.intercept_
                    all_coefs[model_name] = coefs
                
                coefs_df = pd.DataFrame(all_coefs)
                
                # Plot coefficients as bar chart with futuristic styling
                fig = go.Figure()
                
                for model_name in coefs_df.columns:
                    fig.add_trace(go.Bar(
                        x=coefs_df.index,
                        y=coefs_df[model_name],
                        name=model_name,
                        marker_line_width=0
                    ))
                
                fig.update_layout(
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
                    xaxis=dict(
                        title="Dimensional Parameter",
                        categoryorder='total descending'
                    ),
                    yaxis=dict(
                        title="Coefficient Magnitude",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    barmode='group',
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the coefficients as a table as well
                st.markdown("<p style='color: #aaaaff;'>Quantum Coefficient Matrix:</p>", unsafe_allow_html=True)
                st.dataframe(coefs_df.style.highlight_max(axis=0), use_container_width=True)

# Results Analysis page
elif page == "Results Analysis":
    # Add implementation for Results Analysis page
    futuristic_header("DIMENSIONAL DECOMPOSITION ANALYSIS", "🔬")
    
    if st.session_state.contributions is None:
        st.warning("Please calibrate quantum models first on the Quantum Model Calibration module.")
    else:
        with futuristic_container():
            st.markdown("""
            <div class="animate-fade-in">
                <p>Explore the multi-dimensional decomposition of business outcomes. This module breaks down revenue into precise channel contributions and analyzes marketing effectiveness.</p>
            </div>
            """, unsafe_allow_html=True)
            
            contributions = st.session_state.contributions
            data = st.session_state.data
            
            # Extract channel columns (exclude Date and Actual_Sales)
            channel_cols = [col for col in contributions.columns 
                           if col not in ['Date', 'Actual_Sales', 'Intercept', 'Seasonality', 'Trend', 'Unexplained']]
            
            tab1, tab2, tab3 = st.tabs(["Revenue Decomposition", "Channel Analysis", "ROI Matrix"])
            
            with tab1:
                st.markdown("<p style='color: #aaaaff;'>Multi-dimensional Revenue Stream Analysis:</p>", unsafe_allow_html=True)
                
                # Stacked area chart of contributions with futuristic styling
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
                        name='Base Revenue',
                        line=dict(width=0.5, color=colors[0]),
                        fillcolor=colors[0]
                    ))
                
                # Add channels
                for i, col in enumerate(channel_cols):
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
                        color_idx = (len(channel_cols) + 2) % len(colors)
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
                    name='Actual Revenue',
                    line=dict(color='white', width=2, dash='dot')
                ))
                
                fig.update_layout(
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
                    xaxis=dict(
                        title="Temporal Dimension",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    yaxis=dict(
                        title="Revenue Quantum",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    hovermode="x unified",
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("<p style='color: #aaaaff;'>Channel Contribution Analysis:</p>", unsafe_allow_html=True)
                
                # Calculate total contribution per channel
                channel_totals = {}
                for col in channel_cols:
                    if col in contributions.columns:
                        channel_totals[col] = contributions[col].sum()
                
                # Calculate total contribution and percentage
                total_contrib = sum(channel_totals.values())
                
                # Convert to DataFrame for visualization
                channel_df = pd.DataFrame({
                    'Channel': list(channel_totals.keys()),
                    'Total Contribution': list(channel_totals.values())
                })
                
                channel_df['Contribution %'] = (channel_df['Total Contribution'] / total_contrib * 100).round(2)
                channel_df = channel_df.sort_values('Total Contribution', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display as a table
                    st.dataframe(channel_df.style.background_gradient(subset=['Contribution %'], cmap='viridis'), use_container_width=True)
                
                with col2:
                    # Pie chart of channel contributions with futuristic styling
                    fig = go.Figure(data=[go.Pie(
                        labels=channel_df['Channel'],
                        values=channel_df['Total Contribution'],
                        hole=0.4,
                        marker=dict(
                            colors=px.colors.sequential.Plasma,
                            line=dict(color='rgba(0, 0, 0, 0)', width=2)
                        ),
                        textinfo='label+percent',
                        textfont=dict(size=12, color='white'),
                        insidetextorientation='radial'
                    )])
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(25, 25, 44, 0.0)',
                        paper_bgcolor='rgba(25, 25, 44, 0.0)',
                        title="Channel Contribution Share",
                        title_font=dict(size=16, color='white'),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        margin=dict(l=20, r=20, t=50, b=20),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.markdown("<p style='color: #aaaaff;'>Return on Investment Analysis Matrix:</p>", unsafe_allow_html=True)
                
                # Calculate spend by channel
                spend_by_channel = {}
                for channel in channel_cols:
                    if channel in data.columns:
                        spend_by_channel[channel] = data[channel].sum()
                
                # Create ROI dataframe
                roi_df = pd.DataFrame({
                    'Channel': list(channel_totals.keys()),
                    'Total Contribution': list(channel_totals.values()),
                    'Total Spend': [spend_by_channel.get(channel, 0) for channel in channel_totals.keys()]
                })
                
                roi_df['ROI'] = (roi_df['Total Contribution'] / roi_df['Total Spend']).round(2)
                roi_df = roi_df.sort_values('ROI', ascending=False)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Display ROI table with futuristic styling
                    st.dataframe(roi_df.style.background_gradient(subset=['ROI'], cmap='viridis'), use_container_width=True)
                
                with col2:
                    # Bar chart of ROI by channel with futuristic styling
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=roi_df['Channel'],
                        y=roi_df['ROI'],
                        text=roi_df['ROI'].round(2),
                        textposition='outside',
                        marker=dict(
                            color=roi_df['ROI'],
                            colorscale='Plasma',
                            line=dict(color='rgba(0, 0, 0, 0)', width=2)
                        )
                    ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(25, 25, 44, 0.0)',
                        paper_bgcolor='rgba(25, 25, 44, 0.0)',
                        title="ROI Quantum by Channel",
                        title_font=dict(size=16, color='white'),
                        xaxis=dict(
                            title="Channel",
                            tickangle=-45,
                            tickfont=dict(size=10)
                        ),
                        yaxis=dict(
                            title="ROI Factor",
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(100, 100, 255, 0.1)'
                        ),
                        margin=dict(l=20, r=20, t=50, b=100),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Spend vs. Contribution analysis
                st.markdown("<p style='color: #aaaaff;'>Investment-Outcome Relationship Matrix:</p>", unsafe_allow_html=True)
                
                fig = px.scatter(
                    roi_df,
                    x='Total Spend',
                    y='Total Contribution',
                    size='ROI',
                    color='ROI',
                    hover_name='Channel',
                    color_continuous_scale='Plasma',
                    size_max=50
                )
                
                fig.update_traces(
                    marker=dict(
                        line=dict(width=2, color='rgba(255, 255, 255, 0.5)')
                    ),
                    textposition='top center'
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(25, 25, 44, 0.0)',
                    paper_bgcolor='rgba(25, 25, 44, 0.0)',
                    title="Investment-Outcome Relationship",
                    title_font=dict(size=16, color='white'),
                    xaxis=dict(
                        title="Total Investment Quantum",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    yaxis=dict(
                        title="Total Contribution Quantum",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    coloraxis_colorbar=dict(
                        title="ROI Factor"
                    ),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Budget Optimization page
elif page == "Budget Optimization":
    # Add implementation for Budget Optimization page
    futuristic_header("QUANTUM BUDGET OPTIMIZATION", "⚛️")
    
    if st.session_state.models is None or len(st.session_state.models) == 0:
        st.warning("Please calibrate quantum models first on the Quantum Model Calibration module.")
    else:
        with futuristic_container():
            st.markdown("""
            <div class="animate-fade-in">
                <p>Optimize marketing budget allocation using quantum algorithms. This module simulates multiple budget scenarios and identifies optimal allocations to maximize business outcomes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            data = st.session_state.data
            
            # Get channel names for optimization
            channel_names = [col for col in data.columns 
                             if col not in ['Date', 'Sales', 'Seasonality', 'Trend', 'Noise']]
            
            # Current allocation
            current_spend = {}
            for channel in channel_names:
                current_spend[channel] = data[channel].mean()
            
            total_current_budget = sum(current_spend.values())
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<div style='text-align: center;'><h3>Current Allocation Matrix</h3></div>", unsafe_allow_html=True)
                
                # Create current allocation dataframe
                current_df = pd.DataFrame({
                    'Channel': list(current_spend.keys()),
                    'Average Spend': list(current_spend.values()),
                    'Allocation %': [(spend / total_current_budget * 100).round(2) for spend in current_spend.values()]
                })
                
                # Display current allocation
                st.dataframe(current_df.style.background_gradient(subset=['Allocation %'], cmap='Blues'), use_container_width=True)
            
            with col2:
                # Pie chart of current allocation with futuristic styling
                fig = go.Figure(data=[go.Pie(
                    labels=current_df['Channel'],
                    values=current_df['Average Spend'],
                    hole=0.4,
                    marker=dict(
                        colors=px.colors.sequential.Blues,
                        line=dict(color='rgba(0, 0, 0, 0)', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=12, color='white'),
                    insidetextorientation='radial'
                )])
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(25, 25, 44, 0.0)',
                    paper_bgcolor='rgba(25, 25, 44, 0.0)',
                    title="Current Investment Distribution",
                    title_font=dict(size=16, color='white'),
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Budget optimization section
            futuristic_header("QUANTUM OPTIMIZATION ENGINE", "📊")
            
            with st.form("budget_optimization_form"):
                st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Optimization Parameters</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Budget constraint
                    budget_constraint = st.slider(
                        "Total Budget Quantum (% of current)",
                        min_value=50,
                        max_value=150,
                        value=100,
                        help="Total budget available for allocation"
                    )
                    
                    # Calculate the actual budget amount
                    target_budget = total_current_budget * (budget_constraint / 100)
                    
                    st.markdown(f"""
                    <div class="metric-container" style="margin-top: 1rem;">
                        <div class="metric-label">Target Budget Quantum</div>
                        <div class="metric-value">{target_budget:.2f}</div>
                        <div class="metric-delta">{budget_constraint}% of current allocation</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Select optimization objective
                    objective = st.selectbox(
                        "Optimization Objective Function",
                        ["Maximize Revenue", "Maximize ROI"],
                        format_func=lambda x: x.replace("Revenue", "Revenue Quantum").replace("ROI", "ROI Factor")
                    )
                    
                    # Map to actual values for the backend
                    objective_map = {
                        "Maximize Revenue Quantum": "sales",
                        "Maximize ROI Factor": "roi"
                    }
                    
                    # Description of the selected objective
                    if objective == "Maximize Revenue":
                        st.markdown("""
                        <div style="background: rgba(30, 30, 70, 0.4); padding: 10px; border-radius: 10px; margin-top: 1rem;">
                            <p style="color: #aaaaff; margin: 0;">Optimizes allocation to generate maximum total revenue quantum, regardless of efficiency.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: rgba(30, 30, 70, 0.4); padding: 10px; border-radius: 10px; margin-top: 1rem;">
                            <p style="color: #aaaaff; margin: 0;">Optimizes allocation to achieve maximum return per investment unit, prioritizing efficiency.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()
                
                # Channel constraints
                st.markdown("<h3 style='text-align: center; margin-bottom: 1.5rem;'>Channel Constraint Parameters</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color: #aaaaff; text-align: center;'>Define minimum and maximum allocation limits for each channel (% of current allocation)</p>", unsafe_allow_html=True)
                
                min_constraints = {}
                max_constraints = {}
                
                for i, channel in enumerate(channel_names):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        min_constraints[channel] = st.slider(
                            f"{channel} Minimum",
                            min_value=0,
                            max_value=100,
                            value=50,
                            help="Minimum allocation required for this channel",
                            key=f"min_{channel}"
                        )
                    
                    with col2:
                        max_constraints[channel] = st.slider(
                            f"{channel} Maximum",
                            min_value=100,
                            max_value=200,
                            value=150,
                            help="Maximum allocation allowed for this channel",
                            key=f"max_{channel}"
                        )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    optimize_button = st.form_submit_button("EXECUTE QUANTUM OPTIMIZATION")
                    
                    if optimize_button:
                        with st.spinner("Initializing quantum optimization algorithms..."):
                            # Add a small delay for effect
                            time.sleep(1.5)
                            
                            # Select the model to use (use the first one for simplicity)
                            model_name = list(st.session_state.models.keys())[0]
                            model = st.session_state.models[model_name]
                            
                            # Get transformed data and columns
                            transformed_data = st.session_state.transformed_data
                            X_cols = st.session_state.X_cols
                            
                            # Convert constraints to actual values
                            min_values = {channel: current_spend[channel] * (min_constraints[channel] / 100) 
                                        for channel in channel_names}
                            
                            max_values = {channel: current_spend[channel] * (max_constraints[channel] / 100) 
                                        for channel in channel_names}
                            
                            # Map the objective to the backend value
                            backend_objective = "sales" if "Revenue" in objective else "roi"
                            
                            # Optimize budget
                            optimized_allocation, expected_sales = optimize_budget(
                                model=model,
                                data=transformed_data,
                                channel_names=channel_names,
                                total_budget=target_budget,
                                min_values=min_values,
                                max_values=max_values,
                                objective=backend_objective
                            )
                            
                            # Store optimization results
                            st.session_state.optimized_allocation = optimized_allocation
                            st.session_state.expected_sales = expected_sales
                            
                            st.success("Quantum optimization complete! Optimal allocation matrix generated.")
            
            # Display optimization results if they exist
            if hasattr(st.session_state, 'optimized_allocation') and st.session_state.optimized_allocation is not None:
                st.divider()
                futuristic_header("OPTIMIZED ALLOCATION MATRIX", "🔄")
                
                opt_allocation = st.session_state.optimized_allocation
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Channel': channel_names,
                    'Current Spend': [current_spend[channel] for channel in channel_names],
                    'Optimized Spend': [opt_allocation[channel] for channel in channel_names]
                })
                
                comparison_df['Change %'] = ((comparison_df['Optimized Spend'] / comparison_df['Current Spend'] - 1) * 100).round(2)
                
                # Calculate allocation percentages
                comparison_df['Current Allocation %'] = (comparison_df['Current Spend'] / comparison_df['Current Spend'].sum() * 100).round(2)
                comparison_df['Optimized Allocation %'] = (comparison_df['Optimized Spend'] / comparison_df['Optimized Spend'].sum() * 100).round(2)
                comparison_df['Allocation Shift'] = (comparison_df['Optimized Allocation %'] - comparison_df['Current Allocation %']).round(2)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display comparison table with futuristic styling
                    st.dataframe(
                        comparison_df.style.background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-50, vmax=50)
                        .background_gradient(subset=['Allocation Shift'], cmap='coolwarm', vmin=-20, vmax=20),
                        use_container_width=True
                    )
                
                with col2:
                    # Expected performance improvement
                    if hasattr(st.session_state, 'expected_sales'):
                        # Calculate current sales prediction
                        model_name = list(st.session_state.models.keys())[0]
                        model = st.session_state.models[model_name]
                        current_sales = data['Sales'].mean()
                        
                        # Calculate improvement
                        pct_improvement = ((st.session_state.expected_sales / current_sales - 1) * 100).round(2)
                        
                        st.markdown("""
                        <div style="text-align: center; margin-top: 1rem;">
                            <h3>Performance Impact Projection</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Current Revenue</div>
                                <div class="metric-value">{current_sales:.2f}</div>
                                <div class="metric-delta">Baseline</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Optimized Revenue</div>
                                <div class="metric-value">{st.session_state.expected_sales:.2f}</div>
                                <div class="metric-delta">+{pct_improvement:.2f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add some extra metrics for futuristic feel
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Calculate ROI improvement
                            current_roi = current_sales / total_current_budget
                            opt_roi = st.session_state.expected_sales / target_budget
                            roi_improvement = ((opt_roi / current_roi) - 1) * 100
                            
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">ROI Improvement</div>
                                <div class="metric-value">{roi_improvement:.2f}%</div>
                                <div class="metric-delta">Efficiency Gain</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Calculate efficiency score (made up metric for futuristic feel)
                            efficiency_score = min(100, (pct_improvement + roi_improvement) / 2 + 50)
                            
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-label">Optimization Score</div>
                                <div class="metric-value">{efficiency_score:.1f}</div>
                                <div class="metric-delta">Quantum Efficiency Rating</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.divider()
                
                # Bar chart comparing current vs optimized with futuristic styling
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=channel_names,
                    y=comparison_df['Current Allocation %'],
                    name='Current Allocation %',
                    marker=dict(
                        color='rgba(100, 200, 255, 0.7)',
                        line=dict(color='rgba(100, 200, 255, 1.0)', width=2)
                    )
                ))
                
                fig.add_trace(go.Bar(
                    x=channel_names,
                    y=comparison_df['Optimized Allocation %'],
                    name='Optimized Allocation %',
                    marker=dict(
                        color='rgba(200, 100, 255, 0.7)',
                        line=dict(color='rgba(200, 100, 255, 1.0)', width=2)
                    )
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(25, 25, 44, 0.0)',
                    paper_bgcolor='rgba(25, 25, 44, 0.0)',
                    title="Allocation Transformation Matrix",
                    title_font=dict(size=16, color='white'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    xaxis=dict(
                        title="Channel",
                        tickangle=-45
                    ),
                    yaxis=dict(
                        title="Allocation %",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(100, 100, 255, 0.1)'
                    ),
                    barmode='group',
                    margin=dict(l=20, r=20, t=50, b=60),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a radar chart for a more futuristic visualization
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=comparison_df['Current Allocation %'],
                    theta=comparison_df['Channel'],
                    fill='toself',
                    name='Current Allocation',
                    line=dict(color='rgba(100, 200, 255, 1)'),
                    fillcolor='rgba(100, 200, 255, 0.2)'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=comparison_df['Optimized Allocation %'],
                    theta=comparison_df['Channel'],
                    fill='toself',
                    name='Optimized Allocation',
                    line=dict(color='rgba(200, 100, 255, 1)'),
                    fillcolor='rgba(200, 100, 255, 0.2)'
                ))
                
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(25, 25, 44, 0.0)',
                    paper_bgcolor='rgba(25, 25, 44, 0.0)',
                    title="Dimensional Allocation Comparison",
                    title_font=dict(size=16, color='white'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.1,
                        xanchor="center",
                        x=0.5
                    ),
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(comparison_df['Optimized Allocation %'].max(), comparison_df['Current Allocation %'].max()) * 1.2]
                        ),
                        bgcolor='rgba(25, 25, 44, 0.0)'
                    ),
                    margin=dict(l=80, r=80, t=50, b=50),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Add futuristic footer
st.markdown("""
<div style="margin-top: 2rem; border-top: 1px solid rgba(100, 100, 255, 0.3); padding-top: 1rem;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <p style="color: #aaaaff; font-size: 0.8rem; margin: 0;">Quantum MMM Platform • Neural Dynamics Interactive Interface</p>
        </div>
        <div>
            <p style="color: #aaaaff; font-size: 0.8rem; margin: 0;">Created for fifty-five MMM Internship Application • © 2040</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)