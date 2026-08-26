import logging
from rich.console import Console
from rich.table import Table
from src.data_loader import load_demand_data
from src.forecaster import train_and_forecast_by_product

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Demand_Forecasting_Pipeline")
console = Console()

def print_multi_product_forecast_table(forecast_df):
    """Renders named product predictions in the terminal using Rich."""
    table = Table(title="🔮 30-DAY GERMAN PRODUCT DEMAND FORECAST (PRODUCT BREAKDOWN)", show_header=True, header_style="bold magenta")
    table.add_column("Date", style="cyan", justify="center")
    table.add_column("Product Name", style="bold yellow")
    table.add_column("Predicted Sales (Units)", style="bold green", justify="right")
    table.add_column("Lower 95% CI", style="dim", justify="right")
    table.add_column("Upper 95% CI", style="dim", justify="right")

    # Sample top 2 records for each product
    sample_df = forecast_df.groupby("product_name").head(2).sort_values("ds")

    for _, row in sample_df.iterrows():
        table.add_row(
            row["ds"].strftime("%Y-%m-%d"),
            row["product_name"],
            f"{row['forecast_demand']:,.2f}",
            f"{row['lower_ci_95']:,.2f}",
            f"{row['upper_ci_95']:,.2f}"
        )

    console.print(table)

def main():
    logger.info("Starting Product-Level German Demand Forecasting Pipeline...")
    
    df = load_demand_data()
    logger.info(f"Loaded dataset with {len(df):,} total records across products.")
    
    forecast_df, metrics = train_and_forecast_by_product(df, forecast_horizon_days=30)
    logger.info("Multi-product model training complete.")
    
    print("\n" + "=" * 70)
    print("🇩🇪 GERMAN PRODUCT DEMAND FORECASTING EVALUATION (PER PRODUCT)")
    print("=" * 70)
    for prod, m in metrics.items():
        print(f" • {prod:<35} | RMSE: {m['rmse']:>6.2f} units | MAPE: {m['mape_pct']:>5.2f}%")
    print("=" * 70 + "\n")
    
    print_multi_product_forecast_table(forecast_df)

if __name__ == "__main__":
    main()