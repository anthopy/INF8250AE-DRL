import yfinance as yf
import pandas as pd
from pathlib import Path
import time


# ==========================
# Parameters to adjust here
# ==========================

# List of tickers for your investment universe
TICKERS = [
    "SPY",      # S&P 500 ETF
    "XLF",      # Financials sector ETF
    "XLK",      # Technology sector ETF
    "XLE",      # Energy sector ETF
    "XLV",      # Health Care sector ETF
    "XLI",      # Industrials sector ETF
    "XLP",      # Consumer Staples sector ETF
    "XLY",      # Consumer Discretionary sector ETF
    "XLB",      # Materials sector ETF
    "XLRE",     # Real Estate sector ETF  CONSTANT DATA BEFORE 2015-10-09
    "XLU",      # Utilities sector ETF
    "BTC-USD",  # Bitcoin   CONSTANT DATA  BEFORE 2014-09-18
]

# Volatility index
VIX_TICKER = "^VIX"

# Period
START_DATE = "2015-10-09"
END_DATE = "2025-01-01"

# Rolling volatility window (in days)
ROLLING_VOLATILITY_WINDOW = 20

# Output folder for CSV files
OUTPUT_DIR = Path("./data")


# ==========================
# Utility functions
# ==========================

def fill_missing_adj_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame using linear interpolation.
    """
    df = df.copy()
    df_interpolated = df.interpolate(method="linear", limit_direction="both")
    return df_interpolated


def download_data(tickers, start, end):
    """
    Download Yahoo Finance market data for the specified tickers.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=True,
        threads=False
    )
    return data


def main():
    # Create output directory if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading market data for tickers:")
    print(TICKERS)
    data = download_data(TICKERS, START_DATE, END_DATE)

    # Extract only adjusted prices
    if "Adj Close" not in data.columns.get_level_values(0):
        raise ValueError("'Adj Close' column is missing from the downloaded data.")
    
    adj_close = data["Adj Close"]
    print("\nPreview of raw Adj Close:")
    print(adj_close.head())

    # Fill missing values column by column
    adj_close_filled = fill_missing_adj_close(adj_close)

    # Save adjusted prices
    prices_path = OUTPUT_DIR / "prices.csv"
    adj_close_filled.to_csv(prices_path)
    print(f"\nprices.csv saved in: {prices_path.resolve()}")

    # Compute daily returns (simple returns)
    returns = adj_close_filled.pct_change().dropna()
    returns_path = OUTPUT_DIR / "returns.csv"
    returns.to_csv(returns_path)
    print(f"returns.csv saved in: {returns_path.resolve()}")

    # Download VIX data
    print(f"\nDownloading VIX ({VIX_TICKER}) ...")
    vix_data = download_data(VIX_TICKER, START_DATE, END_DATE)

    if "Adj Close" not in vix_data.columns:
        raise ValueError("'Adj Close' column is missing from VIX data.")

    vix_adj = vix_data["Adj Close"]
    vix = vix_adj.rename(columns={vix_adj.columns[0]: "VIX"})

    # Fill any missing VIX values
    vix_filled = fill_missing_adj_close(vix)
    vix_path = OUTPUT_DIR / "vix.csv"
    vix_filled.to_csv(vix_path)
    print(f"vix.csv saved in: {vix_path.resolve()}")
    
    # Create dataset_rl: returns + VIX + rolling volatility + regime indicator
    print(f"\nBuilding RL dataset with rolling volatility (window={ROLLING_VOLATILITY_WINDOW} days)...")
    
    # Start with returns
    dataset_rl = returns.copy()
    
    # Add VIX
    dataset_rl = dataset_rl.join(vix_filled, how="inner")
    
    # Calculate rolling volatility for each ticker (annualized)
    # Rolling std of returns * sqrt(252) for annualization
    for ticker in returns.columns:
        vol_col_name = f"{ticker}_vol_{ROLLING_VOLATILITY_WINDOW}d"
        dataset_rl[vol_col_name] = (
            returns[ticker].rolling(window=ROLLING_VOLATILITY_WINDOW).std() * (252 ** 0.5)
        )
    
    # Create volatility regime indicator based on VIX
    # Low regime: VIX <= median, High regime: VIX > median
    vix_median = dataset_rl['VIX'].median()
    dataset_rl['volatility_regime'] = (dataset_rl['VIX'] > vix_median).astype(int)
    # 0 = low volatility, 1 = high volatility
    
    # Drop rows with NaN (from rolling window calculation)
    dataset_rl = dataset_rl.dropna()
    
    # Save dataset_rl
    dataset_rl_path = OUTPUT_DIR / "dataset_rl.csv"
    dataset_rl.to_csv(dataset_rl_path)
    print(f"✅ dataset_rl.csv saved in: {dataset_rl_path.resolve()}")
    print(f"   - Shape: {dataset_rl.shape}")
    print(f"   - VIX median threshold: {vix_median:.2f}")
    print(f"   - Low volatility regime (0): {(dataset_rl['volatility_regime'] == 0).sum()} days")
    print(f"   - High volatility regime (1): {(dataset_rl['volatility_regime'] == 1).sum()} days")

    # Merge prices and VIX into a single DataFrame
    dataset_full = adj_close_filled.join(vix_filled, how="inner")
    dataset_full_path = OUTPUT_DIR / "dataset_full.csv"
    dataset_full.to_csv(dataset_full_path)
    print(f"\ndataset_full.csv saved in: {dataset_full_path.resolve()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
