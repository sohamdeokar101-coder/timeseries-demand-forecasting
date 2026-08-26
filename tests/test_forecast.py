import pandas as pd
from src.data_loader import generate_german_retail_data
from src.decomposition import analyze_stationarity
from src.forecaster import train_and_forecast_demand

def test_forecasting_pipeline():
    df = generate_german_retail_data(n_days=100)
    
    stationarity = analyze_stationarity(df)
    assert "adf_statistic" in stationarity
    assert "p_value" in stationarity
    
    forecast_df, metrics = train_and_forecast_demand(df, forecast_horizon_days=7)
    assert len(forecast_df) == 7
    assert "forecast_demand" in forecast_df.columns
    assert "lower_ci_95" in forecast_df.columns
    assert metrics["mape_pct"] >= 0.0