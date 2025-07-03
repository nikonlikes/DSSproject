# 📊 Marketing Campaign Decision Support System

A comprehensive Streamlit-based Decision Support System for optimizing marketing campaigns using machine learning and data analytics.

## 🎯 Features

### 📈 Dashboard
- **Interactive Filters**: Company, channel, date range, campaign type
- **KPI Cards**: Real-time conversion rates, ROI, clicks, impressions  
- **Advanced Visualizations**: 
  - Conversion rate by channel
  - ROI trends over time
  - Engagement vs performance analysis
  - Campaign type performance matrices
  - Feature correlation heatmaps

### 🔮 Campaign Simulator
- **AI-Powered Predictions**: Predict conversion rates for new campaigns
- **Interactive Input Interface**: Configure campaign parameters
- **Performance Optimization**: Get recommendations for improvement
- **Statistical Fallback**: Works even without trained ML models
- **Historical Comparison**: Compare predictions against historical data

## 🛠️ System Requirements

### Hardware (Optimized for M2 Mac mini)
- **RAM**: 8GB (optimized memory usage with sparse matrices)
- **Storage**: 500MB for app + data
- **CPU**: M2 processor or equivalent

### Software
- **Python**: 3.8+ 
- **Operating System**: macOS, Linux, Windows
- **Browser**: Chrome, Firefox, Safari (for Streamlit interface)

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have completed the machine learning model training:

```bash
# Run the exploratory notebook first (in parent directory)
jupyter notebook 00_explore_marketing.ipynb
```

This creates the required model artifacts:
- `models/conversion_rate_model.pkl`
- `models/preprocess.pkl` 
- `models/model_metadata.pkl`

### 2. Installation

```bash
# Clone or navigate to the marketing_dss directory
cd marketing_dss

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
marketing_dss/
│
├── 📄 app.py                     # Main Streamlit application
├── 📂 pages/
│   ├── 1_Dashboard.py            # Analytics dashboard  
│   └── 2_Campaign_Sim.py         # Campaign simulator
├── 📂 data/
│   └── marketing_campaign_dataset.csv  # Historical campaign data
├── 📂 models/                    # ML model artifacts (created by notebook)
│   ├── conversion_rate_model.pkl
│   ├── preprocess.pkl
│   └── model_metadata.pkl
├── 📄 requirements.txt           # Python dependencies
└── 📄 README.md                  # This file
```

## 🎮 Usage Guide

### Dashboard Page
1. **Apply Filters**: Use sidebar to filter by company, channel, dates, campaign type
2. **Monitor KPIs**: View key metrics with comparison to averages
3. **Analyze Trends**: Explore visualizations for insights
4. **Identify Opportunities**: Review correlation matrix and top performers

### Campaign Simulator
1. **Configure Campaign**: Set type, channel, audience, duration, budget
2. **Set Expectations**: Input expected engagement, clicks, impressions
3. **Get Prediction**: Click "Predict Campaign Performance"
4. **Review Results**: Analyze conversion rate, ROI, and confidence level
5. **Optimize**: Follow recommendations for improvement

## 🔧 Configuration

### Memory Optimization (8GB RAM)
The app includes several optimizations for constrained environments:

- **Data Sampling**: Large datasets sampled for visualizations
- **Caching**: `@st.cache_data` for data loading
- **Sparse Matrices**: OneHotEncoder with sparse output
- **Efficient Libraries**: Plotly for interactive visualizations

### Performance Settings
Edit these parameters in the code for your hardware:

```python
# In Dashboard.py and Campaign_Sim.py
SAMPLE_SIZE = 1000  # Reduce for lower memory usage
CACHE_TTL = 3600    # Cache timeout in seconds
```

## 🤖 Machine Learning Integration

### Model Requirements
The simulator works with scikit-learn compatible models:
- **Supported**: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting, LightGBM
- **Format**: Pickled model with separate preprocessor
- **Features**: Numeric + categorical with OneHotEncoding

### Fallback Mode
If models aren't available, the simulator uses statistical estimation:
- Historical campaign averages
- Engagement score adjustments  
- Duration and channel factors
- Confidence levels based on data availability

## 📊 Data Requirements

### Input Format
The `marketing_campaign_dataset.csv` should contain:

| Column | Type | Description |
|--------|------|-------------|
| Campaign_ID | int | Unique campaign identifier |
| Company | str | Company name |
| Campaign_Type | str | Email, Display, Search, Influencer |
| Channel_Used | str | Google Ads, Facebook, YouTube, etc. |
| Target_Audience | str | Demographics (e.g., "Men 18-24") |
| Duration | str | Campaign duration (e.g., "30 days") |
| Acquisition_Cost | str | Cost with $ and commas (e.g., "$1,234.00") |
| Conversion_Rate | float | 0-1 conversion rate |
| ROI | float | Return on investment |
| Clicks | int | Number of clicks |
| Impressions | int | Number of impressions |
| Engagement_Score | int | 1-10 engagement rating |
| Location | str | Geographic location |
| Language | str | Campaign language |
| Customer_Segment | str | Target segment |
| Date | str | Campaign date (YYYY-MM-DD) |

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Error**
```
Error loading model: [Errno 2] No such file or directory
```
**Solution**: Run the Jupyter notebook `00_explore_marketing.ipynb` first to create model files.

**2. Data Loading Error**
```
Error loading data: [Errno 2] No such file or directory
```  
**Solution**: Ensure `data/marketing_campaign_dataset.csv` exists in the correct location.

**3. Memory Issues**
```
MemoryError or slow performance
```
**Solution**: 
- Reduce `SAMPLE_SIZE` in visualization code
- Close other applications
- Consider upgrading RAM

**4. Import Errors**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

### Performance Optimization
- **Reduce Data Size**: Sample larger datasets before loading
- **Clear Cache**: Use Streamlit's cache clearing options
- **Browser**: Use Chrome for best Plotly performance
- **Close Tabs**: Limit browser tabs when running

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time campaign monitoring
- [ ] A/B testing framework  
- [ ] Advanced ML models (deep learning)
- [ ] Export/import campaign configurations
- [ ] Multi-user authentication
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] REST API for programmatic access

### Contribution Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

For support and questions:
- 📧 Email: support@marketingdss.com
- 💬 GitHub Issues: [Create an issue](https://github.com/yourrepo/marketing-dss/issues)
- 📚 Documentation: [Wiki](https://github.com/yourrepo/marketing-dss/wiki)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for rapid web app development
- Visualizations powered by [Plotly](https://plotly.com/)
- Machine learning with [scikit-learn](https://scikit-learn.org/)
- Optimized for Apple M2 architecture

---

**📊 Happy Analyzing! 🚀** 