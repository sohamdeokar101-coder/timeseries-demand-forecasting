import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Tuple, Dict, Any

def create_product_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers temporal lag features per product group."""
    df_list = []
    
    for product, group in df.groupby("product_name"):
        g = group.sort_values("ds").copy()
        g["day_of_week"] = g["ds"].dt.dayofweek
        g["month"] = g["ds"].dt.month
        g["day_of_year"] = g["ds"].dt.dayofyear
        
        # Product Lags
        g["lag_1"] = g["y"].shift(1)
        g["lag_7"] = g["y"].shift(7)
        g["rolling_mean_7"] = g["y"].shift(1).rolling(window=7).mean()
        g["rolling_std_7"] = g["y"].shift(1).rolling(window=7).std()
        
        df_list.append(g)
        
    return pd.concat(df_list, ignore_index=True).dropna().reset_index(drop=True)

def train_and_forecast_by_product(
    df: pd.DataFrame, 
    forecast_horizon_days: int = 30
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Trains forecasting models for each named product and projects 30-day demand."""
    featured_df = create_product_lag_features(df)
    feature_cols = ["day_of_week", "month", "day_of_year", "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7"]
    
    forecast_results = []
    metrics_summary = {}
    
    for product, p_data in featured_df.groupby("product_name"):
        train_size = len(p_data) - forecast_horizon_days
        train_df = p_data.iloc[:train_size]
        test_df = p_data.iloc[train_size:]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(train_df[feature_cols], train_df["y"])
        
        preds = rf.predict(test_df[feature_cols])
        y_true = test_df["y"].values
        
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mape = mean_absolute_percentage_error(y_true, preds) * 100
        
        res = test_df[["ds", "product_name"]].copy()
        res["forecast_demand"] = np.round(preds, 2)
        res["lower_ci_95"] = np.round(np.clip(preds - (1.96 * rmse * 0.2), 0, None), 2)
        res["upper_ci_95"] = np.round(preds + (1.96 * rmse * 0.2), 2)
        
        forecast_results.append(res)
        metrics_summary[product] = {
            "rmse": round(float(rmse), 2),
            "mape_pct": round(float(mape), 2)
        }
        
    final_forecast_df = pd.concat(forecast_results, ignore_index=True)
    return final_forecast_df, metrics_summary