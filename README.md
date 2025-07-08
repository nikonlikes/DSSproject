# 📊 Marketing Campaign Decision Support System (DSS)

A comprehensive machine learning-powered decision support system for optimizing marketing campaigns. This project combines exploratory data analysis, predictive modeling, and an interactive web dashboard to help marketing teams make data-driven decisions.

## 🎯 Project Overview

This Decision Support System (DSS) provides:

- **📊 Interactive Dashboard**: Real-time analytics and performance insights
- **🔮 Campaign Simulator**: AI-powered predictions for new campaigns
- **📈 Data Analysis**: Comprehensive exploratory data analysis of marketing campaigns
- **🤖 Machine Learning Models**: Multi-target regression models for campaign optimization
- **📋 Performance Tracking**: KPI monitoring and comparative analysis

## 🏗️ Project Structure

```
DSSproject/
├── 📄 README.md                              # This file
├── 📊 00_explore_marketing.ipynb             # Initial data exploration
├── 📊 01_explore_marketing.ipynb             # Advanced analysis and modeling
├── 📊 02_explore_marketing.ipynb             # Additional analysis
├── 📊 dataExplor.ipynb                       # Data exploration notebook
├── 📊 models.ipynb                           # Model training and evaluation
├── 📂 data/
│   ├── marketing_campaign_dataset.csv        # Raw marketing campaign data
│   ├── df_encoded_for_training.csv          # Processed training data
│   └── df_encoded_v2.csv                    # Enhanced encoded data
├── 📂 models/                               # Trained ML models
│   ├── conversion_rate_model_v2.pkl         # Conversion rate predictor
│   ├── acquisition_cost_model_v2.pkl        # Cost prediction model
│   ├── clicks_model_v2.pkl                  # Click prediction model
│   ├── impressions_model_v2.pkl             # Impression prediction model
│   ├── engagement_score_model_v2.pkl        # Engagement score predictor
│   ├── feature_names_v2.pkl                 # Feature engineering metadata
│   └── *.pkl                                # Other model artifacts
├── 📂 marketing_dss/                        # Streamlit web application
│   ├── app.py                               # Main application entry point
│   ├── 📂 pages/
│   │   ├── 1_Dashboard.py                   # Analytics dashboard
│   │   └── 2_Campaign_Sim.py                # Campaign simulator
│   ├── 📂 data/
│   │   └── marketing_campaign_dataset.csv   # Data for web app
│   ├── requirements.txt                     # Python dependencies
│   └── README.md                            # Web app documentation
└── 📂 catboost_info/                        # CatBoost training artifacts
    └── (training logs and temporary files)
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+** (recommended: 3.9-3.11)
- **8GB RAM minimum** (optimized for M2 Mac mini)
- **500MB storage** for data and models
- **Git** (for cloning the repository)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd DSSproject

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for data analysis
pip install jupyter pandas numpy scikit-learn matplotlib seaborn plotly

# Install additional dependencies for web app
pip install streamlit joblib
```

### 3. Running the Project

#### Option A: Data Analysis & Model Training

```bash
# 1. Start with data exploration
jupyter notebook 00_explore_marketing.ipynb

# 2. Run advanced analysis and model training
jupyter notebook 01_explore_marketing.ipynb

# 3. Additional analysis (optional)
jupyter notebook 02_explore_marketing.ipynb
```

#### Option B: Web Application

```bash
# Navigate to the web app directory
cd marketing_dss

# Install web app dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

The web application will open in your browser at `http://localhost:8501`

## 🎮 Usage Guide

### 📊 Data Analysis (Jupyter Notebooks)

1. **Start with `00_explore_marketing.ipynb`**:
   - Initial data exploration and visualization
   - Data cleaning and preprocessing
   - Basic statistical analysis

2. **Run `01_explore_marketing.ipynb`**:
   - Advanced feature engineering
   - Model training and evaluation
   - Performance metrics and validation

3. **Use `models.ipynb`**:
   - Model comparison and selection
   - Hyperparameter tuning
   - Final model deployment

### 🌐 Web Application

1. **Home Page**: Overview of the system with key metrics
2. **Dashboard**: Interactive analytics with filtering and visualization
3. **Campaign Simulator**: Predict performance for new campaigns

#### Dashboard Features:
- **Filters**: Company, channel, date range, campaign type
- **KPIs**: Conversion rates, ROI, clicks, impressions
- **Visualizations**: Performance trends, correlation analysis
- **Insights**: Top performers and optimization opportunities

#### Campaign Simulator Features:
- **Input Configuration**: Set campaign parameters
- **AI Predictions**: Get conversion rate, cost, and engagement forecasts
- **Performance Metrics**: Detailed prediction results
- **Optimization Tips**: Recommendations for improvement

## 🤖 Machine Learning Models

The system includes 5 trained regression models:

1. **Conversion Rate Model**: Predicts campaign conversion rates
2. **Acquisition Cost Model**: Forecasts cost per acquisition
3. **Clicks Model**: Estimates expected clicks
4. **Impressions Model**: Predicts impression volume
5. **Engagement Score Model**: Forecasts engagement metrics

### Model Training Process:

```python
# 1. Data preprocessing
- Feature engineering and encoding
- Handling missing values
- Scaling and normalization

# 2. Model training
- Multiple algorithm testing
- Cross-validation
- Hyperparameter optimization

# 3. Model evaluation
- Performance metrics (R², RMSE, MAE)
- Feature importance analysis
- Validation on test set
```

## 📊 Data Description

### Marketing Campaign Dataset

The dataset contains **2,000 marketing campaigns** with the following features:

| Column | Type | Description |
|--------|------|-------------|
| Campaign_ID | int | Unique campaign identifier |
| Company | str | Company name |
| Campaign_Type | str | Email, Display, Search, Influencer |
| Channel_Used | str | Google Ads, Facebook, YouTube, etc. |
| Target_Audience | str | Demographics (e.g., "Men 18-24") |
| Duration | str | Campaign duration (e.g., "30 days") |
| Acquisition_Cost | str | Cost with $ and commas |
| Conversion_Rate | float | 0-1 conversion rate |
| ROI | float | Return on investment |
| Clicks | int | Number of clicks |
| Impressions | int | Number of impressions |
| Engagement_Score | int | 1-10 engagement rating |
| Location | str | Geographic location |
| Language | str | Campaign language |
| Customer_Segment | str | Target segment |
| Date | str | Campaign date (YYYY-MM-DD) |

## 🔧 Requirements

### Python Dependencies

```
# Core libraries
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.4.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0

# Web application
streamlit>=1.28.0
joblib>=1.3.0

# Jupyter notebooks
jupyter>=1.0.0
ipykernel>=6.25.0
```

### System Requirements

- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 1GB free space
- **CPU**: Modern processor (M2 Mac mini optimized)
- **Browser**: Chrome/Firefox/Safari for web app

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Error**
```
Solution: Run the Jupyter notebooks first to train and save models
```

**2. Data File Not Found**
```
Solution: Ensure data/marketing_campaign_dataset.csv exists
```

**3. Memory Issues**
```
Solution: Close other applications, reduce data sample size
```

**4. Package Import Errors**
```
Solution: Install missing packages with pip install <package-name>
```

### Performance Optimization

- **Use virtual environment** to avoid conflicts
- **Close unnecessary browser tabs** when running web app
- **Reduce sample size** in visualizations for better performance
- **Clear Jupyter kernel** if memory issues occur

## 🚀 Getting Started Workflow

1. **Clone the repository and setup environment**
2. **Run data exploration notebooks** (00_explore_marketing.ipynb)
3. **Train models** (01_explore_marketing.ipynb or models.ipynb)
4. **Launch web application** (cd marketing_dss && streamlit run app.py)
5. **Explore the dashboard** and try the campaign simulator

## 📈 Future Enhancements

- [ ] Real-time data integration
- [ ] Advanced deep learning models
- [ ] A/B testing framework
- [ ] Multi-user authentication
- [ ] Database integration
- [ ] REST API development
- [ ] Mobile-responsive design

## 📝 License

This project is for educational purposes (MSE436 coursework).

## 👥 Support

For questions or issues:
- Review the troubleshooting section
- Check Jupyter notebook outputs for error messages
- Ensure all dependencies are installed correctly

---

**🎯 Happy Analyzing! Ready to optimize your marketing campaigns? 🚀**
