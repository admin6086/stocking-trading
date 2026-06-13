import argparse
from io import StringIO
from pathlib import Path
import time

import pandas as pd
import requests


REQUIRED_COLUMNS = {"High", "Low", "Close", "Volume"}
OUTPUT_COLUMNS = ["name", "price", "Variation", "volume_change", "target"]
DEFAULT_OUTPUT_FILE = "data/stock_data.csv"
DEFAULT_OUTPUT_NAME = "stock_data.csv"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


def download_stock_data(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    if not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty.")
    if years <= 0:
        raise ValueError("Years must be a positive number.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "The yfinance package is required. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    data = pd.DataFrame()
    last_error = None

    for attempt in range(1, 4):
        try:
            data = yf.download(
                ticker,
                period=f"{years}y",
                interval="1d",
                progress=False,
                auto_adjust=False,
                timeout=30,
            )
            if not data.empty:
                break
        except Exception as exc:
            last_error = exc

        if attempt < 3:
            wait_seconds = attempt * 2
            print(f"Could not download {ticker} on attempt {attempt}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    if data.empty:
        if last_error is not None:
            raise ValueError(f"No data downloaded for ticker '{ticker}': {last_error}")
        raise ValueError(
            f"No data downloaded for ticker '{ticker}'. Check the ticker symbol."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Downloaded data is missing required columns: {missing_columns}")

    return data


def transform_stock_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create model-ready features and the next-day movement target."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    records = pd.DataFrame(index=data.index)
    records["name"] = ticker
    previous_close = data["Close"].shift(1)

    records["price"] = (data["Close"] - previous_close) / previous_close
    records["Variation"] = (data["High"] - data["Low"]) / previous_close
    records["volume_change"] = data["Volume"].pct_change()
    next_close = data["Close"].shift(-1)
    records["target"] = (next_close > data["Close"]).astype(float)
    records.loc[next_close.isna(), "target"] = pd.NA

    dataset = records[OUTPUT_COLUMNS].dropna().copy()
    dataset["target"] = dataset["target"].astype(int)
    return dataset


def save_dataset(dataset: pd.DataFrame, out_path: str) -> None:
    """Save the transformed dataset as CSV."""
    if dataset.empty:
        raise ValueError("Final dataset is empty after dropping missing values.")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index_label="Date")


def build_dataset(ticker: str, out_path: str, years: int = 5) -> pd.DataFrame:
    """Download, transform, and save stock data."""
    dataset = build_ticker_dataset(ticker=ticker, years=years)
    save_dataset(dataset, out_path)
    return dataset


def build_ticker_dataset(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download and transform stock data for one ticker."""
    raw_data = download_stock_data(ticker=ticker, years=years)
    return transform_stock_data(raw_data, ticker=ticker)


def yahoo_ticker(symbol: str) -> str:
    """Convert exchange symbols to Yahoo Finance ticker format."""
    return symbol.strip().replace(".", "-")


def output_csv_path(out_path: str) -> Path:
    """Treat paths without a .csv suffix as output directories."""
    path = Path(out_path)
    if path.suffix.lower() != ".csv":
        path = path / DEFAULT_OUTPUT_NAME
    return path


def read_symbol_file(url: str) -> pd.DataFrame:
    """Download a Nasdaq Trader symbol file with retries."""
    last_error = None

    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text), sep="|")
        except requests.RequestException as exc:
            last_error = exc
            wait_seconds = attempt * 2
            print(f"Could not download {url} on attempt {attempt}; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)

    raise ValueError(f"Could not download ticker list from {url}: {last_error}")


def load_all_stock_tickers() -> list[str]:
    """Load US-listed stock tickers from Nasdaq Trader symbol directories."""
    nasdaq = read_symbol_file(NASDAQ_LISTED_URL)
    other = read_symbol_file(OTHER_LISTED_URL)

    nasdaq = nasdaq[
        (nasdaq["Test Issue"] == "N")
        & (nasdaq["ETF"] == "N")
        & nasdaq["Symbol"].notna()
    ]
    other = other[
        (other["Test Issue"] == "N")
        & (other["ETF"] == "N")
        & other["ACT Symbol"].notna()
    ]

    symbols = set(nasdaq["Symbol"].astype(str)) | set(other["ACT Symbol"].astype(str))
    return sorted(yahoo_ticker(symbol) for symbol in symbols)


def build_all_datasets(out_path: str, years: int = 5, limit: int | None = None) -> pd.DataFrame:
    """Download all discovered stock tickers and save them in one CSV."""
    tickers = load_all_stock_tickers()
    if limit is not None:
        tickers = tickers[:limit]

    if not tickers:
        raise ValueError("No stock tickers were found.")

    datasets = []
    successful = 0
    failed = 0

    for index, ticker in enumerate(tickers, start=1):
        try:
            dataset = build_ticker_dataset(ticker=ticker, years=years)
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(tickers)}] Skipped {ticker}: {exc}")
            continue

        datasets.append(dataset)
        successful += 1
        print(f"[{index}/{len(tickers)}] Added {ticker}: {len(dataset)} rows")

    if not datasets:
        raise ValueError("No datasets were created.")

    combined = pd.concat(datasets).sort_index()
    output_path = output_csv_path(out_path)
    save_dataset(combined, str(output_path))
    print(f"Done. Saved {len(combined)} rows for {successful} tickers to {output_path}.")
    print(f"Skipped {failed} tickers.")
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Yahoo Finance stock data and create a next-day movement dataset."
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker symbol, e.g. AAPL. If omitted, all discovered stock tickers are downloaded.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_FILE,
        help="Output CSV path. Default: data/stock_data.csv",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of past years to download, default: 5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of tickers to download when running all tickers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.ticker:
            out_path = output_csv_path(args.out)

            dataset = build_dataset(
                ticker=args.ticker,
                out_path=str(out_path),
                years=args.years,
            )
            print(f"Saved {len(dataset)} rows to {out_path}")
        else:
            build_all_datasets(out_path=args.out, years=args.years, limit=args.limit)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
