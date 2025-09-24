# JP Morgan Chase & Co. Software Engineering Job Simulation

This repository contains my complete work from the JP Morgan Chase & Co. Software Engineering Job Simulation on Forage. The project demonstrates practical applications of quantitative analysis, machine learning, and financial modeling across four key areas of investment banking and financial services.

## 🎯 Project Overview

The simulation covered essential skills in:
- **Financial Data Analysis**: Processing and analyzing large datasets
- **Machine Learning**: Implementing predictive models for credit risk assessment
- **Quantitative Finance**: Pricing complex financial instruments
- **Data Engineering**: Building robust data processing pipelines

## 📜 Certificate

**[View My Completion Certificate](https://www.theforage.com/completion-certificates/Sj7temL583QAYpHXD/bWqaecPDbYAwSDqJy_Sj7temL583QAYpHXD_4RvGKXtXQPW8SbrRa_1758718019542_completion_certificate.pdf)**

---

## 🚀 Project Sections

### 1. Natural Gas Price Prediction 📈
**Location**: `Natural Gas Prediction/`

**Objective**: Develop a time series forecasting model to predict natural gas prices using historical data and harmonic regression.

**What I Accomplished**:
- **Data Analysis**: Performed comprehensive exploratory data analysis on 4 years of natural gas pricing data
- **Pattern Recognition**: Identified seasonal and cyclical patterns in commodity pricing using visualization techniques
- **FFT Analysis**: Applied Fast Fourier Transform to decompose the price signal into frequency components
- **Harmonic Regression**: Implemented a sophisticated forecasting model combining:
  - Linear drift component to capture long-term trends
  - Harmonic regression with dominant frequencies for cyclical patterns
  - Detrending techniques to isolate periodic components
- **Model Validation**: Achieved strong predictive performance with detailed R² and RMSE metrics
- **Production Function**: Created `predict()` function for real-time price forecasting

**Key Technical Skills**:
- Time series analysis
- Fourier analysis and signal processing
- Regression modeling
- Data visualization with matplotlib/seaborn

**Files**:
- `Nat_gas.ipynb`: Complete analysis and model development
- `nat_gas.py`: Production-ready forecasting module
- `Nat_Gas.csv`: Historical price dataset

---

### 2. Commodity Storage Contract Pricing 💰
**Location**: `Commodity Storage Contract Pricing/`

**Objective**: Build a pricing engine for natural gas storage contracts, integrating predictive models with financial contract valuation.

**What I Accomplished**:
- **Contract Modeling**: Designed a comprehensive storage contract pricing framework considering:
  - Injection and withdrawal schedules
  - Storage capacity constraints
  - Rate limitations (injection/withdrawal speeds)
  - Daily storage costs
  - Market price dynamics
- **Integration**: Connected the natural gas prediction model to real-time contract valuation
- **Optimization Logic**: Implemented decision algorithms for optimal storage utilization
- **Financial Calculations**: Developed present value calculations accounting for:
  - Time value of money
  - Storage cost accruals
  - Market timing strategies

**Key Technical Skills**:
- Financial derivatives pricing
- Optimization algorithms  
- Contract valuation methodologies
- Cross-module integration

**Files**:
- `pricing.ipynb`: Contract pricing analysis and testing
- `pricing.py`: Modular pricing engine

---

### 3. Credit Risk Analysis & Machine Learning 🎯
**Location**: `Credit Risk Analysis/`

**Objective**: Develop a machine learning model to predict loan defaults and estimate expected losses for credit risk management.

**What I Accomplished**:
- **Data Preprocessing**: Cleaned and prepared loan portfolio data including:
  - Customer demographics
  - Credit scores (FICO)
  - Debt-to-income ratios
  - Employment history
  - Outstanding obligations
- **Feature Engineering**: Selected and engineered predictive features for default modeling
- **Model Development**: Implemented Random Forest Classifier with:
  - 200 estimators for robust predictions
  - Maximum depth of 10 to prevent overfitting
  - Cross-validation for model selection
- **Risk Quantification**: Built `loss_estimated()` function incorporating:
  - Default probability predictions
  - Recovery rate assumptions (10%)
  - Expected loss calculations
- **Model Performance**: Achieved strong classification performance with comprehensive evaluation metrics

**Key Technical Skills**:
- Machine learning (scikit-learn)
- Credit risk modeling
- Feature selection and engineering
- Model evaluation and validation

**Files**:
- `credit_risk_analysis.ipynb`: Complete ML pipeline and analysis
- `credit_risk_analysis.py`: Production model implementation
- `Task 3 and 4_Loan_Data.csv`: Loan portfolio dataset

---

### 4. FICO Score Bucketing & Quantization 📊
**Location**: `Bucket FICO scores/`

**Objective**: Create a rating system that maps FICO scores to letter grades using optimal quantization techniques for credit assessment automation.

**What I Accomplished**:
- **Quantization Theory**: Implemented multiple bucketing approaches:
  - **K-Means Clustering**: Used unsupervised learning to find natural score groupings
  - **Mean Squared Error Optimization**: Minimized within-bucket variance
  - **Log-Likelihood Maximization**: Optimized for default rate separation
  - **Dynamic Programming**: Applied advanced algorithms for optimal bucket boundaries
- **Rating System Design**: Created A-F letter rating system where:
  - Lower letters (A, B) represent better credit quality
  - Higher letters (E, F) represent higher risk profiles
  - Systematic mapping from continuous scores to discrete ratings
- **Statistical Analysis**: Comprehensive bucket evaluation including:
  - Population distribution across ratings
  - Mean FICO scores per bucket
  - Standard deviation within buckets
  - Default rate analysis by rating
- **Production Implementation**: Built reusable functions for:
  - Rating assignment to new applicants
  - Boundary optimization
  - Model validation and performance tracking

**Key Technical Skills**:
- Quantization algorithms
- Dynamic programming
- Statistical optimization
- Credit scoring methodologies
- Data segmentation techniques

**Files**:
- `bucket_fico_score.ipynb`: Complete quantization analysis and rating system

---

## 🛠️ Technical Stack

**Languages**: Python
**Libraries**: 
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Signal Processing**: scipy
- **Statistical Modeling**: scipy.optimize

**Development Environment**: Jupyter Notebooks for interactive development and analysis

---

## 🎓 Key Learning Outcomes

1. **Quantitative Finance**: Applied mathematical models to real-world financial problems
2. **Risk Management**: Implemented industry-standard credit risk assessment techniques  
3. **Time Series Forecasting**: Developed sophisticated predictive models for commodity markets
4. **Machine Learning in Finance**: Built production-ready ML pipelines for default prediction
5. **Financial Engineering**: Created complex pricing algorithms for derivative instruments
6. **Data Science Workflow**: End-to-end project development from data exploration to production deployment

---

## 🔍 Project Structure
```
JP MORGAN/
├── README.md                          # This comprehensive overview
├── certificate.pdf                    # Forage completion certificate
├── Natural Gas Prediction/            # Time series forecasting
│   ├── Nat_gas.ipynb                 # Analysis and model development
│   ├── nat_gas.py                     # Production forecasting module
│   └── Nat_Gas.csv                    # Historical price data
├── Commodity Storage Contract Pricing/ # Financial derivatives pricing
│   ├── pricing.ipynb                  # Contract valuation analysis  
│   └── pricing.py                     # Pricing engine module
├── Credit Risk Analysis/              # Machine learning for credit risk
│   ├── credit_risk_analysis.ipynb     # ML pipeline and evaluation
│   ├── credit_risk_analysis.py        # Production risk model
│   └── Task 3 and 4_Loan_Data.csv    # Loan portfolio dataset
└── Bucket FICO scores/               # Credit score quantization
    └── bucket_fico_score.ipynb       # Rating system development
```

---

This project demonstrates proficiency in applying data science and machine learning techniques to solve complex problems in investment banking, risk management, and quantitative finance - core competencies for software engineering roles at JP Morgan Chase & Co.