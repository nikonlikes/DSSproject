import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
import datetime

# Configure page
st.set_page_config(
    page_title="Campaign Simulator - Marketing DSS",
    page_icon="🎯",
    layout="wide"
)

# Load data for reference values
@st.cache_data
def load_data():
    """Load marketing campaign data for reference"""
    try:
        # Try multiple possible paths for the data file
        possible_paths = [
            "data/marketing_campaign_dataset.csv",
            "../data/marketing_campaign_dataset.csv",
            "../../data/marketing_campaign_dataset.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find marketing_campaign_dataset.csv")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load all trained models and feature names"""
    try:
        # Try multiple possible paths for the model files
        possible_model_dirs = [
            "models/",
            "../models/",
            "../../models/"
        ]
        
        feature_names = None
        models = {}
        
        # Define model names
        model_names = {
            'conversion_rate': 'conversion_rate_model.pkl',
            'acquisition_cost': 'acquisition_cost_model.pkl',
            'clicks': 'clicks_model.pkl',
            'impressions': 'impressions_model.pkl',
            'engagement_score': 'engagement_score_model.pkl',
        }
        
        # Try to load from each directory
        for model_dir in possible_model_dirs:
            try:
                # Load feature names
                feature_path = f"{model_dir}feature_names_v2.pkl"
                if os.path.exists(feature_path):
                    feature_names = joblib.load(feature_path)
                    
                    # Load all models
                    for name, filename in model_names.items():
                        model_path = f"{model_dir}{filename}"
                        if os.path.exists(model_path):
                            models[name] = joblib.load(model_path)
                    
                    if len(models) == 5:  # All models loaded
                        break
            except Exception as e:
                continue
        
        return feature_names, models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, {}

# Initialize data and models
df = load_data()
feature_names, models = load_model_artifacts()

# Page header
st.title("🎯 Campaign Performance Simulator")
st.markdown("Multi-target forecasting for comprehensive campaign analysis")
st.markdown("---")

# Check if models are loaded
if len(models) != 5:
    st.error("""
    ⚠️ **Models not found!** 
    
    Please run the Jupyter notebook `01_explore_marketing.ipynb` first to train and save all models.
    """)
    st.stop()

st.success(f"✅ All 5 models loaded successfully!")

# Input Section
st.subheader("📝 Campaign Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    # Campaign Type
    campaign_types = sorted(df['Campaign_Type'].unique().tolist())
    campaign_type = st.selectbox(
        "Campaign Type",
        campaign_types,
        help="Select the type of marketing campaign"
    )
    
    # Target Audience
    audiences = sorted(df['Target_Audience'].unique().tolist())
    target_audience = st.selectbox(
        "Target Audience",
        audiences,
        help="Select the target demographic"
    )
    
    # Channel Used
    channels = sorted(df['Channel_Used'].unique().tolist())
    channel_used = st.selectbox(
        "Channel Used",
        channels,
        help="Select the marketing channel"
    )

with col2:
    # Duration
    duration_options = ['15 days', '30 days', '45 days', '60 days']
    duration = st.selectbox(
        "Duration",
        duration_options,
        index=1,  # Default to 30 days
        help="Campaign duration"
    )
    
    # Campaign Start Month
    month_options = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    # Default to current month
    current_month = datetime.datetime.now().strftime('%B')
    default_month_idx = month_options.index(current_month) if current_month in month_options else 0
    
    campaign_month = st.selectbox(
        "Campaign Start Month",
        month_options,
        index=default_month_idx,
        help="Select the month when the campaign will begin"
    )
    
    # Acquisition Cost
    acquisition_cost = st.number_input(
        "Acquisition Cost ($)",
        min_value=0.0,
        value=2000.0,
        step=100.0,
        help="Budget per acquisition"
    )

with col3:
    # Location
    locations = sorted(df['Location'].unique().tolist())
    location = st.selectbox(
        "Location",
        locations,
        help="Geographic target location"
    )
    
    # Language
    languages = sorted(df['Language'].unique().tolist())
    language = st.selectbox(
        "Language",
        languages,
        help="Campaign language"
    )
    
    # Customer Segment
    segments = sorted(df['Customer_Segment'].unique().tolist())
    customer_segment = st.selectbox(
        "Customer Segment",
        segments,
        help="Target customer segment"
    )

# Prediction Section
st.markdown("---")
st.subheader("🔮 Prediction Results")

# Create prediction button
if st.button("🚀 Predict Campaign Performance", type="primary"):
    # Create base input data with only numeric features initially
    input_data = {
        'Acquisition_Cost': acquisition_cost
    }
    
    # Add all categorical features with one-hot encoding
    if feature_names:
        for feature in feature_names:
            if feature in input_data:
                continue
            
            # Initialize to 0 (for one-hot encoded features)
            input_data[feature] = 0
            
            # Set appropriate categorical features to 1
            if feature == f'Campaign_Type_{campaign_type}':
                input_data[feature] = 1
            elif feature == f'Target_Audience_{target_audience}':
                input_data[feature] = 1
            elif feature == f'Channel_Used_{channel_used}':
                input_data[feature] = 1
            elif feature == f'Location_{location}':
                input_data[feature] = 1
            elif feature == f'Language_{language}':
                input_data[feature] = 1
            elif feature == f'Customer_Segment_{customer_segment}':
                input_data[feature] = 1
            elif feature == f'Month_{campaign_month}':
                input_data[feature] = 1
            elif feature == f'Duration_{duration}':
                input_data[feature] = 1
    
    # Create DataFrame with correct column order
    X_input = pd.DataFrame([input_data])[feature_names]
    
    # Make predictions for all targets
    results = {}
    for name, model in models.items():
        try:
            prediction = model.predict(X_input)[0]
            results[name] = prediction
        except Exception as e:
            st.error(f"Error predicting {name}: {e}")
            results[name] = None
    
    # Display results table
    if all(v is not None for v in results.values()):
        st.success("🎉 Predictions generated successfully!")
        
        # Create results dataframe for display
        results_df = pd.DataFrame([
            ["Conversion Rate", f"{results['conversion_rate']:.3f}"],
            ["Predicted Acquisition Cost", f"${results['acquisition_cost']:,.2f}"],
            ["Clicks", f"{results['clicks']:,.0f}"],
            ["Impressions", f"{results['impressions']:,.0f}"],
            ["Engagement Score", f"{results['engagement_score']:.2f}"]
        ], columns=["Metric", "Prediction"])
        
        # Display as table
        st.table(results_df)
        
        # Additional summary
        st.markdown("---")
        st.markdown("### 📊 Summary Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("💰 Cost per Click", f"${results['acquisition_cost'] / max(1, results['clicks']):.2f}")
        
        with col2:
            st.metric("📈 Click-through Rate", f"{results['clicks'] / max(1, results['impressions']) * 100:.2f}%")
    else:
        st.error("Failed to generate predictions. Please check your inputs.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    🎯 Multi-Target Campaign Simulator | Powered by HistGradientBoostingRegressor | 
    Results are ML predictions based on historical data
</div>
""", unsafe_allow_html=True) 