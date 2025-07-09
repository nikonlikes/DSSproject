import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Dashboard - Marketing DSS",
    page_icon="📊",
    layout="wide"
)

# Load data with caching
@st.cache_data
def load_data():
    """Load and preprocess marketing campaign data"""
    try:
        df = pd.read_csv("../data/marketing_campaign_dataset.csv")
        
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

if df.empty:
    st.error("Failed to load data. Please check if the data file exists.")
    st.stop()

# Page header
st.title("📊 Marketing Campaign Dashboard")
st.markdown("Comprehensive analytics and performance insights")
st.markdown("---")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    
    # Company filter
    companies = ['All'] + sorted(df['Company'].unique().tolist())
    selected_companies = st.multiselect(
        "Select Companies",
        companies,
        default=['All']
    )
    
    # Channel filter
    channels = ['All'] + sorted(df['Channel_Used'].unique().tolist())
    selected_channel = st.selectbox(
        "Select Channel",
        channels
    )
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Campaign type filter
    campaign_types = ['All'] + sorted(df['Campaign_Type'].unique().tolist())
    selected_campaign_type = st.selectbox(
        "Select Campaign Type",
        campaign_types
    )
    
    # Apply filters button
    apply_filters = st.button("🔄 Update Dashboard", type="primary")

# Apply filters to data
filtered_df = df.copy()

# Filter by companies
if 'All' not in selected_companies and selected_companies:
    filtered_df = filtered_df[filtered_df['Company'].isin(selected_companies)]

# Filter by channel
if selected_channel != 'All':
    filtered_df = filtered_df[filtered_df['Channel_Used'] == selected_channel]

# Filter by campaign type
if selected_campaign_type != 'All':
    filtered_df = filtered_df[filtered_df['Campaign_Type'] == selected_campaign_type]

# Filter by date range
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= start_date) & 
        (filtered_df['Date'].dt.date <= end_date)
    ]

# Display filter summary
st.info(f"📈 Showing {len(filtered_df):,} campaigns (filtered from {len(df):,} total)")

# KPI Cards
st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_conversion = filtered_df['Conversion_Rate'].mean()
    conversion_delta = avg_conversion - df['Conversion_Rate'].mean()
    st.metric(
        label="Average Conversion Rate",
        value=f"{avg_conversion:.2%}",
        delta=f"{conversion_delta:.2%}"
    )

with col2:
    avg_roi = filtered_df['ROI'].mean()
    roi_delta = avg_roi - df['ROI'].mean()
    st.metric(
        label="Average ROI",
        value=f"{avg_roi:.2f}",
        delta=f"{roi_delta:.2f}"
    )

with col3:
    total_clicks = filtered_df['Clicks'].sum()
    st.metric(
        label="Total Clicks",
        value=f"{total_clicks:,}",
        delta=None
    )

with col4:
    total_impressions = filtered_df['Impressions'].sum()
    st.metric(
        label="Total Impressions",
        value=f"{total_impressions:,}",
        delta=None
    )

st.markdown("---")

# Visualizations
st.subheader("📈 Performance Analytics")

# Row 1: Conversion Rate and Dynamic Time Series Analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Conversion Rate by Channel**")
    channel_conv = filtered_df.groupby('Channel_Used')['Conversion_Rate'].mean().reset_index()
    fig_conv = px.bar(
        channel_conv,
        x='Channel_Used',
        y='Conversion_Rate',
        title="Average Conversion Rate by Channel",
        color='Conversion_Rate',
        color_continuous_scale='viridis'
    )
    fig_conv.update_layout(showlegend=False, height=400)
    fig_conv.update_traces(texttemplate='%{y:.1%}', textposition='outside')
    st.plotly_chart(fig_conv, use_container_width=True)

with col2:
    # Feature selector for time series
    st.markdown("**Performance Metrics Over Time**")
    
    # Define available metrics with their display names and formatting
    metric_options = {
        'ROI': {'label': 'ROI', 'format': ':.2f'},
        'Conversion_Rate': {'label': 'Conversion Rate', 'format': ':.2%'},
        'Acquisition_Cost': {'label': 'Acquisition Cost ($)', 'format': ':,.2f'},
        'Clicks': {'label': 'Clicks', 'format': ':,'},
        'Impressions': {'label': 'Impressions', 'format': ':,'},
        'Engagement_Score': {'label': 'Engagement Score', 'format': ':.1f'}
    }
    
    selected_metric = st.selectbox(
        "Select metric to display:",
        options=list(metric_options.keys()),
        format_func=lambda x: metric_options[x]['label'],
        index=0  # Default to ROI
    )
    
    # Create time series data
    time_series_data = filtered_df.groupby('Date')[selected_metric].mean().reset_index()
    
    # Create the dynamic chart
    fig_time = px.line(
        time_series_data,
        x='Date',
        y=selected_metric,
        title=f"{metric_options[selected_metric]['label']} Trend Over Time"
    )
    
    # Customize the chart based on metric type
    if selected_metric == 'Conversion_Rate':
        fig_time.update_traces(
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{metric_options[selected_metric]['label']}:</b> %{{y:.2%}}<extra></extra>"
        )
    elif selected_metric in ['Clicks', 'Impressions']:
        fig_time.update_traces(
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{metric_options[selected_metric]['label']}:</b> %{{y:,}}<extra></extra>"
        )
    elif selected_metric == 'Acquisition_Cost':
        fig_time.update_traces(
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{metric_options[selected_metric]['label']}:</b> $%{{y:,.2f}}<extra></extra>"
        )
    else:
        fig_time.update_traces(
            hovertemplate=f"<b>Date:</b> %{{x}}<br><b>{metric_options[selected_metric]['label']}:</b> %{{y:.2f}}<extra></extra>"
        )
    
    # Update y-axis label
    fig_time.update_yaxes(title_text=metric_options[selected_metric]['label'])
    fig_time.update_layout(height=400)
    
    st.plotly_chart(fig_time, use_container_width=True)

# Row 2: Engagement Analysis
st.markdown("**Engagement vs Performance Analysis (Random Sample from 1000)**")
col1, col2 = st.columns(2)

with col1:
    # Impressions vs Clicks bubble chart
    fig_bubble = px.scatter(
        filtered_df.sample(n=min(1000, len(filtered_df)), random_state=42),  # Fixed seed for consistent sampling
        x='Impressions',
        y='Clicks',
        size='Engagement_Score',
        color='Conversion_Rate',
        hover_data=['Campaign_Type', 'Channel_Used'],
        title="Impressions vs Clicks (size = Engagement Score)",
        color_continuous_scale='viridis'
    )
    fig_bubble.update_layout(height=400)
    st.plotly_chart(fig_bubble, use_container_width=True)

with col2:
    # Dynamic Categorical Performance Analysis
    st.markdown("**Performance Analysis by Category**")
    
    # Define available categorical variables
    categorical_options = {
        'Campaign_Type': 'Campaign Type',
        'Channel_Used': 'Channel Used',
        'Customer_Segment': 'Customer Segment',
        'Target_Audience': 'Target Audience',
        'Language': 'Language',
        'Location': 'Location'
    }
    
    # Filter options to only include columns that exist in the dataframe
    available_categories = {k: v for k, v in categorical_options.items() if k in filtered_df.columns}
    
    selected_category = st.selectbox(
        "Select category to analyze:",
        options=list(available_categories.keys()),
        format_func=lambda x: available_categories[x],
        index=0  # Default to first available option
    )
    
    # Create performance data for selected category
    category_perf = filtered_df.groupby(selected_category).agg({
        'Conversion_Rate': 'mean',
        'ROI': 'mean',
        'Campaign_ID': 'count'
    }).reset_index()
    category_perf.columns = [selected_category, 'Avg_Conversion_Rate', 'Avg_ROI', 'Count']
    
    # Create the dynamic scatter plot
    fig_category = px.scatter(
        category_perf,
        x='Avg_Conversion_Rate',
        y='Avg_ROI',
        size='Count',
        text=selected_category,
        title=f"{available_categories[selected_category]} Performance (Conversion vs ROI)",
        hover_data={'Count': True}
    )
    
    # Customize hover template
    fig_category.update_traces(
        textposition="top center",
        hovertemplate="<b>%{text}</b><br>" +
                     "Conversion Rate: %{x:.2%}<br>" +
                     "ROI: %{y:.2f}<br>" +
                     "Campaign Count: %{customdata[0]}<br>" +
                     "<extra></extra>"
    )
    
    # Update axis labels
    fig_category.update_xaxes(title_text="Average Conversion Rate", tickformat=".1%")
    fig_category.update_yaxes(title_text="Average ROI")
    fig_category.update_layout(height=400)
    
    st.plotly_chart(fig_category, use_container_width=True)

# Row 3: Detailed Analysis
st.markdown("---")
st.subheader("🎯 Detailed Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    # Top performing campaigns
    st.markdown("**🏆 Top 10 Campaigns by Conversion Rate**")
    top_campaigns = filtered_df.nlargest(10, 'Conversion_Rate')[
        ['Campaign_ID', 'Company', 'Campaign_Type', 'Channel_Used', 'Conversion_Rate', 'ROI']
    ]
    st.dataframe(top_campaigns, use_container_width=True)

with col2:
    # Performance by customer segment
    st.markdown("**👥 Performance by Customer Segment**")
    segment_perf = filtered_df.groupby('Customer_Segment').agg({
        'Conversion_Rate': ['mean', 'count'],
        'ROI': 'mean'
    }).round(3)
    segment_perf.columns = ['Avg_Conversion_Rate', 'Campaign_Count', 'Avg_ROI']
    segment_perf = segment_perf.reset_index()
    st.dataframe(segment_perf, use_container_width=True)

# Correlation Analysis
st.markdown("---")
st.subheader("🔗 Correlation Analysis")

# Select numeric columns for correlation
numeric_cols = ['Conversion_Rate', 'ROI', 'Clicks', 'Impressions', 
                'Engagement_Score', 'Acquisition_Cost', 'Duration_days']

correlation_matrix = filtered_df[numeric_cols].corr()

fig_corr = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale='RdBu',
    title="Feature Correlation Matrix"
)
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# Summary insights
st.markdown("---")
st.subheader("💡 Key Insights")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("**📊 Performance Highlights:**")
    best_channel = filtered_df.groupby('Channel_Used')['Conversion_Rate'].mean().idxmax()
    best_channel_rate = filtered_df.groupby('Channel_Used')['Conversion_Rate'].mean().max()
    
    best_campaign = filtered_df.groupby('Campaign_Type')['ROI'].mean().idxmax()
    best_campaign_roi = filtered_df.groupby('Campaign_Type')['ROI'].mean().max()
    
    st.write(f"• **Best performing channel:** {best_channel} ({best_channel_rate:.1%} conversion)")
    st.write(f"• **Highest ROI campaign type:** {best_campaign} ({best_campaign_roi:.2f} ROI)")
    st.write(f"• **Total investment:** ${filtered_df['Acquisition_Cost'].sum():,.0f}")

with insights_col2:
    st.markdown("**🎯 Optimization Opportunities:**")
    
    # Find underperforming segments
    low_performing = filtered_df.groupby('Channel_Used')['Conversion_Rate'].mean().nsmallest(1)
    worst_channel = low_performing.index[0]
    worst_rate = low_performing.iloc[0]
    
    st.write(f"• **Focus area:** {worst_channel} channel needs improvement ({worst_rate:.1%} conversion)")
    
    # High engagement, low conversion opportunities
    high_eng_low_conv = filtered_df[
        (filtered_df['Engagement_Score'] > filtered_df['Engagement_Score'].median()) &
        (filtered_df['Conversion_Rate'] < filtered_df['Conversion_Rate'].median())
    ]
    
    st.write(f"• **Conversion opportunity:** {len(high_eng_low_conv)} campaigns with high engagement but low conversion")
    st.write(f"• **Average engagement score:** {filtered_df['Engagement_Score'].mean():.1f}/10")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    📊 Dashboard last updated: Real-time | Data points: {data_points:,} campaigns
</div>
""".format(data_points=len(filtered_df)), unsafe_allow_html=True) 