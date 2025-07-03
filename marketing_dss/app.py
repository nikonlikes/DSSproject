import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Marketing Campaign Decision Support System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #666;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load data with caching for performance
@st.cache_data
def load_data():
    """Load and preprocess marketing campaign data"""
    try:
        df = pd.read_csv("data/marketing_campaign_dataset.csv")
        
        # Clean data
        df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace(r"[\$,]", "", regex=True).astype(float)
        df["Duration_days"] = df["Duration"].str.extract(r"(\d+)").astype(int)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Initialize data
df = load_data()

# App header
st.title("📊 Marketing Campaign Decision Support System")
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/250x100/1f77b4/ffffff?text=Marketing+DSS", 
             caption="Campaign Analytics Platform")
    
    st.markdown("## Navigation")
    st.markdown("""
    - **🏠 Dashboard**: Overview of campaign performance
    - **🎯 Campaign Simulator**: Predict conversion rates for new campaigns
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This Decision Support System helps optimize marketing campaigns by:
    - Analyzing historical performance
    - Identifying key success drivers
    - Predicting conversion rates
    - Supporting data-driven decisions
    """)

# Main content area
if not df.empty:
    # Quick stats in main area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Campaigns",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        avg_conversion = df['Conversion_Rate'].mean()
        st.metric(
            label="Avg Conversion Rate",
            value=f"{avg_conversion:.2%}",
            delta=None
        )
    
    with col3:
        avg_roi = df['ROI'].mean()
        st.metric(
            label="Avg ROI",
            value=f"{avg_roi:.2f}",
            delta=None
        )
    
    with col4:
        total_clicks = df['Clicks'].sum()
        st.metric(
            label="Total Clicks",
            value=f"{total_clicks:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Navigation message
    st.info("""
    👈 **Navigate using the sidebar** or select a page:
    - **Dashboard**: Comprehensive analytics and filtering
    - **Campaign Simulator**: Predict performance for new campaigns
    """)
    
    # Quick preview of data
    with st.expander("🔍 Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Date Range:**", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        
        with col2:
            st.write("**Unique Companies:**", df['Company'].nunique())
            st.write("**Campaign Types:**", df['Campaign_Type'].nunique())

else:
    st.error("Failed to load data. Please check if the data file exists.")
    st.stop()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Marketing Decision Support System | Built with Streamlit | Optimized for M2 Mac mini (8GB RAM)
</div>
""", unsafe_allow_html=True) 