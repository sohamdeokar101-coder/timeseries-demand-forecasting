# 📈 Time-Series Demand Forecasting Model

[![Standard Readme Compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg?style=flat-square)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-blue.svg?style=flat-square)](https://www.statsmodels.org/)
[![PyTest](https://img.shields.io/badge/PyTest-Passed-brightgreen.svg?style=flat-square)](https://docs.pytest.org/)

A multi-product time-series forecasting framework designed to project 30-day out-of-sample demand for German retail and e-commerce markets. The pipeline performs time-series stationarity verification using the Augmented Dickey-Fuller (ADF) test, executes temporal lag feature engineering, trains independent Random Forest Regressors per product SKU, and outputs predictions bounded by 95% confidence intervals.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Generator](#generator)

---

## Background

Predicting daily sales volume across multiple product SKUs requires accounting for non-linear temporal trends, day-of-week seasonality (such as high weekend shopping volume and Sunday retail closures in Germany), and seasonal demand surges (Oktoberfest / Q4 Christmas holidays).

This project implements an operationalized **Multi-Product Demand Forecasting Engine**:
- **German Retail Dataset Ingestion:** Ingests daily transaction logs across specific named German retail products (*Adidas Ultraboost*, *Bosch Espresso Machine*, *Puma Football Jersey*, *Siemens Washing Machine*).
- **Stationarity Testing:** Evaluates structural trends and time-series stationarity using the Augmented Dickey-Fuller (ADF) test.
- **Temporal Lag Feature Engineering:** Constructs historical lag predictors ($y_{t-1}, y_{t-7}$), 7-day rolling window statistics (mean and standard deviation), and calendar attributes (`day_of_week`, `day_of_year`, `month`).
- **Random Forest Machine Learning Engine:** Trains dedicated `RandomForestRegressor` models per product SKU to capture non-linear demand interactions.
- **95% Out-of-Sample Confidence Bounds:** Calculates root mean squared error bounds to derive upper and lower safety stock margins for inventory planning.

---

## Install

### Prerequisites
- Python 3.11 or higher
- Git

### Setup

Clone the repository and set up your Python virtual environment:

```bash
# Clone the repository
git clone [https://github.com/sohamdeokar101-coder/timeseries-demand-forecasting.git](https://github.com/sohamdeokar101-coder/timeseries-demand-forecasting.git)
cd timeseries-demand-forecasting

# Initialize and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

Usage
1. Execute Master Forecasting Pipeline
Run the master script to process product records, engineer lag features, train the Random Forest models, and display terminal forecast tables:
python main.py

2. Run Automated Unit Tests
Verify stationarity testing and feature extraction mechanics using PyTest:
python -m pytest

Generator
The forecasting engine executes via main.py and displays individual product metrics alongside 30-day out-of-sample predictions:

1. Evaluation & Per-Product Metrics Output
======================================================================
🇩🇪 GERMAN PRODUCT DEMAND FORECASTING EVALUATION (PER PRODUCT)
======================================================================
 • Adidas Ultraboost 5 Sneakers        | RMSE:  24.49 units | MAPE:  2.41%
 • Bosch Series 6 Espresso Machine     | RMSE:  29.48 units | MAPE:  7.26%
 • Puma German Football Jersey         | RMSE:  26.96 units | MAPE:  6.19%
 • Siemens iQ500 Washing Machine       | RMSE:  20.88 units | MAPE:  2.11%
======================================================================

2. 30-Day Out-of-Sample Product Demand Forecast Table
🔮 30-DAY GERMAN PRODUCT DEMAND FORECAST (PRODUCT BREAKDOWN)  
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃    Date    ┃ Product Name                      ┃ Predicted Sales (Units) ┃ Lower 95% CI ┃ Upper 95% CI ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 2026-08-02 │ Adidas Ultraboost 5 Sneakers      │                  607.15 │       597.55 │       616.75 │
│ 2026-08-02 │ Bosch Series 6 Espresso Machine   │                  299.66 │       288.10 │       311.22 │
│ 2026-08-02 │ Puma German Football Jersey       │                  236.85 │       226.28 │       247.41 │
│ 2026-08-02 │ Siemens iQ500 Washing Machine     │                  596.75 │       588.57 │       604.93 │
│ 2026-08-03 │ Adidas Ultraboost 5 Sneakers      │                  932.80 │       923.19 │       942.40 │
│ 2026-08-03 │ Bosch Series 6 Espresso Machine   │                  414.48 │       402.92 │       426.04 │
│ 2026-08-03 │ Puma German Football Jersey       │                  311.46 │       300.89 │       322.02 │
│ 2026-08-03 │ Siemens iQ500 Washing Machine     │                  888.25 │       880.07 │       896.44 │
└────────────┴───────────────────────────────────┴─────────────────────────┴──────────────┴──────────────┘


