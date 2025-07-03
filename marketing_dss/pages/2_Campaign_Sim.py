import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Campaign Simulator - Marketing DSS",
    page_icon="🎯",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    """Load marketing campaign data for reference"""
    try:
        df = pd.read_csv("../data/marketing_campaign_dataset.csv")
        df["Acquisition_Cost"] = df["Acquisition_Cost"].str.replace(r"[\$,]", "", regex=True).astype(float)
        df["Duration_days"] = df["Duration"].str.extract(r"(\d+)").astype(int)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessor"""
    try:
        import joblib
        
        # Try to load model artifacts from the main project directory
        model_path = "../models/conversion_rate_model.pkl"
        preprocess_path = "../models/preprocess.pkl"
        metadata_path = "../models/model_metadata.pkl"
        
        # Check if files exist
        if not (os.path.exists(model_path) and os.path.exists(preprocess_path)):
            return None, None, None
            
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocess_path)
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}
        
        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Initialize data and model
df = load_data()
model, preprocessor, metadata = load_model_artifacts()

# Page header
st.title("🎯 Campaign Performance Simulator")
st.markdown("Predict conversion rates for new marketing campaigns using AI")
st.markdown("---")

if df.empty:
    st.error("Failed to load data. Please check if the data file exists.")
    st.stop()

# Model status
if model is None:
    st.warning("""
    ⚠️ **Model not found!** 
    
    Please run the Jupyter notebook `00_explore_marketing.ipynb` first to train and save the model.
    
    For demo purposes, this simulator will use statistical estimates based on historical data.
    """)
    use_model = False
else:
    st.success(f"✅ Model loaded successfully! Using {metadata.get('model_type', 'trained')} model.")
    use_model = True

st.markdown("---")

# Input Section
st.subheader("📝 Campaign Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎯 Campaign Setup**")
    
    # Campaign Type
    campaign_types = sorted(df['Campaign_Type'].unique().tolist())
    campaign_type = st.selectbox(
        "Campaign Type",
        campaign_types,
        help="Select the type of marketing campaign"
    )
    
    # Channel Used
    channels = sorted(df['Channel_Used'].unique().tolist())
    channel_used = st.selectbox(
        "Channel Used",
        channels,
        help="Select the marketing channel"
    )
    
    # Target Audience
    audiences = sorted(df['Target_Audience'].unique().tolist())
    target_audience = st.selectbox(
        "Target Audience",
        audiences,
        help="Select the target demographic"
    )

with col2:
    st.markdown("**💰 Budget & Duration**")
    
    # Duration
    duration_days = st.slider(
        "Duration (days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Campaign duration in days"
    )
    
    # Acquisition Cost
    min_cost = float(df['Acquisition_Cost'].min())
    max_cost = float(df['Acquisition_Cost'].max())
    avg_cost = float(df['Acquisition_Cost'].mean())
    
    acquisition_cost = st.number_input(
        "Acquisition Cost ($)",
        min_value=min_cost,
        max_value=max_cost,
        value=avg_cost,
        step=100.0,
        help=f"Budget per acquisition (${min_cost:,.0f} - ${max_cost:,.0f})"
    )
    
    # Engagement Score
    engagement_score = st.slider(
        "Expected Engagement Score",
        min_value=1,
        max_value=10,
        value=5,
        help="Expected engagement level (1-10 scale)"
    )

with col3:
    st.markdown("**🌍 Location & Segment**")
    
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

# Additional inputs for realistic simulation
st.markdown("**📊 Additional Metrics**")
col1, col2 = st.columns(2)

with col1:
    # Estimated clicks
    avg_clicks = df['Clicks'].mean()
    clicks = st.number_input(
        "Expected Clicks",
        min_value=0,
        value=int(avg_clicks),
        step=10,
        help="Expected number of clicks"
    )

with col2:
    # Estimated impressions
    avg_impressions = df['Impressions'].mean()
    impressions = st.number_input(
        "Expected Impressions",
        min_value=0,
        value=int(avg_impressions),
        step=100,
        help="Expected number of impressions"
    )

# Prediction Section
st.markdown("---")
st.subheader("🔮 Prediction Results")

# Create prediction button
predict_button = st.button("🚀 Predict Campaign Performance", type="primary")

if predict_button:
    # Prepare input data
    user_dict = {
        'Campaign_Type': campaign_type,
        'Channel_Used': channel_used,
        'Target_Audience': target_audience,
        'Duration_days': duration_days,
        'Acquisition_Cost': acquisition_cost,
        'Engagement_Score': engagement_score,
        'Location': location,
        'Language': language,
        'Customer_Segment': customer_segment,
        'Clicks': clicks,
        'Impressions': impressions,
        'Month': datetime.now().month  # Current month
    }
    
    if use_model:
        try:
            # Use trained model for prediction
            X_new = pd.DataFrame([user_dict])
            X_prep = preprocessor.transform(X_new)
            y_pred = model.predict(X_prep)[0]
            
            # Get confidence interval if available
            if hasattr(model, 'predict_proba'):
                confidence = "High"
            else:
                confidence = "Medium"
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            y_pred = None
    else:
        # Use statistical estimation based on historical data
        similar_campaigns = df[
            (df['Campaign_Type'] == campaign_type) &
            (df['Channel_Used'] == channel_used) &
            (df['Customer_Segment'] == customer_segment)
        ]
        
        if len(similar_campaigns) > 0:
            # Base prediction on similar campaigns
            base_conversion = similar_campaigns['Conversion_Rate'].mean()
            
            # Adjust for engagement score
            engagement_factor = (engagement_score - 5) * 0.02  # ±2% per point from baseline
            
            # Adjust for duration (longer campaigns might have lower conversion)
            duration_factor = -0.001 * max(0, duration_days - 30)  # Penalty for long campaigns
            
            y_pred = base_conversion + engagement_factor + duration_factor
            y_pred = max(0, min(1, y_pred))  # Bound between 0 and 1
            confidence = "Estimated"
        else:
            # Fallback to overall average
            y_pred = df['Conversion_Rate'].mean()
            confidence = "Low"
    
    if y_pred is not None:
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Predicted Conversion Rate",
                value=f"{y_pred:.2%}",
                delta=f"{y_pred - df['Conversion_Rate'].mean():.2%} vs avg"
            )
        
        with col2:
            # Estimate ROI based on conversion rate
            estimated_roi = df[df['Conversion_Rate'].between(y_pred-0.05, y_pred+0.05)]['ROI'].mean()
            if pd.isna(estimated_roi):
                estimated_roi = df['ROI'].mean()
            
            st.metric(
                label="💰 Estimated ROI",
                value=f"{estimated_roi:.2f}",
                delta=f"{estimated_roi - df['ROI'].mean():.2f} vs avg"
            )
        
        with col3:
            st.metric(
                label="📊 Confidence Level",
                value=confidence,
                delta=None
            )
        
        # Performance category
        performance_percentile = (y_pred > df['Conversion_Rate']).mean() * 100
        
        if performance_percentile >= 75:
            performance_category = "🟢 Excellent"
            category_color = "success"
        elif performance_percentile >= 50:
            performance_category = "🟡 Good"
            category_color = "warning"
        else:
            performance_category = "🔴 Needs Improvement"
            category_color = "error"
        
        st.markdown(f"**Performance Category:** :{category_color}[{performance_category}]")
        st.markdown(f"**Percentile Rank:** {performance_percentile:.0f}% (better than {performance_percentile:.0f}% of historical campaigns)")

# Comparison Section
st.markdown("---")
st.subheader("📊 Historical Comparison")

if predict_button and y_pred is not None:
    # Create comparison visualization
    fig = go.Figure()
    
    # Historical distribution
    fig.add_trace(go.Histogram(
        x=df['Conversion_Rate'],
        name='Historical Campaigns',
        opacity=0.7,
        nbinsx=30,
        histnorm='probability'
    ))
    
    # Predicted value
    fig.add_vline(
        x=y_pred,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Your Prediction: {y_pred:.2%}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Conversion Rate Distribution: Your Campaign vs Historical Data",
        xaxis_title="Conversion Rate",
        yaxis_title="Probability Density",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Optimization Suggestions
st.markdown("---")
st.subheader("💡 Optimization Suggestions")

if predict_button:
    # Find best performing combinations
    best_performers = df.groupby(['Campaign_Type', 'Channel_Used'])['Conversion_Rate'].mean().nlargest(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Top Performing Combinations**")
        for i, ((camp_type, channel), conv_rate) in enumerate(best_performers.items(), 1):
            st.write(f"{i}. {camp_type} + {channel}: {conv_rate:.1%}")
    
    with col2:
        st.markdown("**🎯 Recommendations**")
        
        # Find best channel for selected campaign type
        best_channel_for_type = df[df['Campaign_Type'] == campaign_type].groupby('Channel_Used')['Conversion_Rate'].mean().idxmax()
        best_rate_for_type = df[df['Campaign_Type'] == campaign_type].groupby('Channel_Used')['Conversion_Rate'].mean().max()
        
        if channel_used != best_channel_for_type:
            st.write(f"💡 Consider switching to {best_channel_for_type} channel ({best_rate_for_type:.1%} avg conversion)")
        
        # Engagement optimization
        if engagement_score < 7:
            st.write("💡 Increase engagement through interactive content")
        
        # Duration optimization
        optimal_duration = df.groupby('Duration_days')['Conversion_Rate'].mean().idxmax()
        if abs(duration_days - optimal_duration) > 10:
            st.write(f"💡 Consider {optimal_duration}-day duration for optimal results")

# Model Information
if use_model and metadata:
    st.markdown("---")
    st.subheader("🤖 Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Model Type",
            value=metadata.get('model_type', 'Unknown')
        )
    
    with col2:
        st.metric(
            label="Test R² Score",
            value=f"{metadata.get('test_r2', 0):.3f}"
        )
    
    with col3:
        st.metric(
            label="Training Samples",
            value=f"{metadata.get('training_samples', 0):,}"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    🎯 Campaign Simulator | Predictions based on historical data analysis | 
    Results are estimates and actual performance may vary
</div>
""", unsafe_allow_html=True) 