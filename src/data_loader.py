import os
import numpy as np
import pandas as pd

# Define specific product names sold across German retail stores
GERMAN_PRODUCTS = [
    "Bosch Series 6 Espresso Machine",
    "Siemens iQ500 Washing Machine",
    "Adidas Ultraboost 5 Sneakers",
    "Puma German Football Jersey"
]

def generate_german_retail_data(n_days: int = 730) -> pd.DataFrame:
    """Generates daily historical demand records for multiple specific German products."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-09-01", periods=n_days, freq="D")
    
    records = []
    
    for product in GERMAN_PRODUCTS:
        # Base daily demand per product
        base_demand = np.random.randint(150, 600)
        trend = np.linspace(base_demand, base_demand * 1.8, n_days)
        
        # German Weekly Pattern (High Friday/Saturday sales, Sunday low/closed)
        day_of_week = dates.dayofweek
        weekly = np.where(day_of_week == 6, -base_demand * 0.6, np.where(day_of_week == 5, base_demand * 0.4, 0))
        
        # Q4 Surge (Oktoberfest / Christmas)
        day_of_year = dates.dayofyear
        seasonality = 80 * np.sin(2 * np.pi * day_of_year / 365) + np.where((day_of_year >= 335) & (day_of_year <= 355), 200, 0)
        
        noise = np.random.normal(0, 20, n_days)
        demand = np.clip(trend + weekly + seasonality + noise, 10, None)
        
        for d, y in zip(dates, demand):
            records.append({
                "ds": d,
                "product_name": product,
                "y": np.round(y, 2),
                "region": "Germany_DE"
            })
            
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/german_product_demand.csv", index=False)
    return df

def load_demand_data(filepath: str = "data/german_product_demand.csv") -> pd.DataFrame:
    """Loads product time-series dataset."""
    if not os.path.exists(filepath):
        return generate_german_retail_data()
    df = pd.read_csv(filepath)
    df["ds"] = pd.to_datetime(df["ds"])
    return df