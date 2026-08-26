import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any

def analyze_stationarity(df: pd.DataFrame) -> Dict[str, Any]:
    """Executes Augmented Dickey-Fuller (ADF) test to evaluate stationarity."""
    result = adfuller(df["y"].dropna())
    
    adf_statistic = float(result[0])
    p_value = float(result[1])
    is_stationary = p_value < 0.05
    
    return {
        "adf_statistic": round(adf_statistic, 4),
        "p_value": round(p_value, 5),
        "is_stationary": is_stationary,
        "critical_values": {k: round(v, 4) for k, v in result[4].items()}
    }

def decompose_time_series(df: pd.DataFrame, period: int = 7) -> Dict[str, pd.Series]:
    """Decomposes time-series into trend, seasonal, and residual components."""
    ts = df.set_index("ds")["y"]
    decomp = seasonal_decompose(ts, model="additive", period=period)
    
    return {
        "trend": decomp.trend,
        "seasonal": decomp.seasonal,
        "resid": decomp.resid
    }