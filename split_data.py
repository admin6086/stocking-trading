import argparse
import json
from pathlib import Path

import pandas as pd


# PyCharm users: you can click Run with these defaults, or override them in
# Run > Edit Configurations > Parameters using the same --name value format.
DEFAULT_DATA_DIR = "data"
DEFAULT_SPLIT_DIR = "Split"
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_LIMIT_TICKERS = None

DATE_COLUMN = "Date"
SPLIT_NAMES = ("train", "validation", "test")
TICKER_MAP_FILE = "ticker_to_id.csv"
METADATA_FILE = "metadata.json"


def list_ticker_files(data_dir: Path, limit_tickers: int | None = None) -> list[Path]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")
    if limit_tickers is not None:
        files = files[:limit_tickers]
    return files


def load_sorted_ticker_file(file_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(file_path, parse_dates=[DATE_COLUMN])
    if DATE_COLUMN not in frame.columns:
        raise ValueError(f"{file_path.name} is missing the {DATE_COLUMN} column.")
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def ticker_name_from_frame(file_path: Path, frame: pd.DataFrame) -> str:
    if "name" in frame.columns and not frame.empty:
        return str(frame["name"].iloc[0])
    return file_path.stem


def split_frame(
    frame: pd.DataFrame, train_ratio: float, val_ratio: float
) -> dict[str, pd.DataFrame]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    train_end = int(len(frame) * train_ratio)
    validation_end = int(len(frame) * (train_ratio + val_ratio))
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("Not enough rows to create non-empty train, validation, and test splits.")

    return {
        "train": frame.iloc[:train_end].copy(),
        "validation": frame.iloc[train_end:validation_end].copy(),
        "test": frame.iloc[validation_end:].copy(),
    }


def save_split_file(split_dir: Path, split_name: str, file_name: str, frame: pd.DataFrame) -> Path:
    output_dir = split_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name
    frame.to_csv(output_path, index=False)
    return output_path


def save_ticker_mapping(ticker_to_id: dict[str, int], split_dir: Path) -> None:
    pd.DataFrame(
        [{"ticker": ticker, "ticker_id": ticker_id} for ticker, ticker_id in sorted(ticker_to_id.items())]
    ).to_csv(split_dir / TICKER_MAP_FILE, index=False)


def save_metadata(
    split_dir: Path,
    data_dir: Path,
    train_ratio: float,
    val_ratio: float,
    ticker_summaries: list[dict[str, object]],
) -> None:
    metadata = {
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "date_column": DATE_COLUMN,
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "ticker_count": len(ticker_summaries),
        "tickers": ticker_summaries,
    }
    with (split_dir / METADATA_FILE).open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def split_and_save_datasets(
    data_dir: Path,
    split_dir: Path,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    limit_tickers: int | None = DEFAULT_LIMIT_TICKERS,
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        (split_dir / split_name).mkdir(parents=True, exist_ok=True)

    ticker_files = list_ticker_files(data_dir, limit_tickers)
    ticker_to_id: dict[str, int] = {}
    split_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    ticker_summaries: list[dict[str, object]] = []

    for file_path in ticker_files:
        try:
            frame = load_sorted_ticker_file(file_path)
            splits = split_frame(frame, train_ratio=train_ratio, val_ratio=val_ratio)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        ticker = ticker_name_from_frame(file_path, frame)
        ticker_to_id[ticker] = len(ticker_to_id)

        summary: dict[str, object] = {"ticker": ticker, "file": file_path.name}
        for split_name, split_frame_data in splits.items():
            output_path = save_split_file(split_dir, split_name, file_path.name, split_frame_data)
            split_counts[split_name] += len(split_frame_data)
            summary[f"{split_name}_rows"] = len(split_frame_data)
            summary[f"{split_name}_file"] = str(output_path)
            summary[f"{split_name}_start"] = split_frame_data[DATE_COLUMN].iloc[0].strftime("%Y-%m-%d")
            summary[f"{split_name}_end"] = split_frame_data[DATE_COLUMN].iloc[-1].strftime("%Y-%m-%d")
        ticker_summaries.append(summary)

    if not ticker_to_id:
        raise ValueError("No ticker files were split.")

    save_ticker_mapping(ticker_to_id, split_dir)
    save_metadata(split_dir, data_dir, train_ratio, val_ratio, ticker_summaries)
    return split_counts


def split_files_exist(split_dir: Path) -> bool:
    has_split_csvs = all(
        (split_dir / split_name).is_dir() and any((split_dir / split_name).glob("*.csv"))
        for split_name in SPLIT_NAMES
    )
    return has_split_csvs and (split_dir / TICKER_MAP_FILE).exists()


def load_ticker_mapping(split_dir: Path) -> dict[str, int]:
    mapping_path = split_dir / TICKER_MAP_FILE
    if not mapping_path.exists():
        raise ValueError(f"Ticker mapping not found: {mapping_path}")

    mapping = pd.read_csv(mapping_path)
    return {str(row.ticker): int(row.ticker_id) for row in mapping.itertuples(index=False)}


def load_split_frames(
    split_dir: Path, split_name: str, limit_tickers: int | None = None
) -> dict[str, pd.DataFrame]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split name: {split_name}")

    split_path = split_dir / split_name
    files = sorted(split_path.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {split_path}")
    if limit_tickers is not None:
        files = files[:limit_tickers]

    frames: dict[str, pd.DataFrame] = {}
    for file_path in files:
        frame = load_sorted_ticker_file(file_path)
        frames[ticker_name_from_frame(file_path, frame)] = frame
    return frames


def load_all_split_frames(
    split_dir: Path, limit_tickers: int | None = None
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        split_name: load_split_frames(split_dir, split_name, limit_tickers)
        for split_name in SPLIT_NAMES
    }


def load_ticker_frames(
    data_dir: Path, limit_tickers: int | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    frames: dict[str, pd.DataFrame] = {}
    for file_path in list_ticker_files(data_dir, limit_tickers):
        frame = load_sorted_ticker_file(file_path)
        frames[ticker_name_from_frame(file_path, frame)] = frame

    ticker_to_id = {ticker: index for index, ticker in enumerate(sorted(frames))}
    return frames, ticker_to_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split each ticker CSV chronologically into train/validation/test.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument(
        "--limit-tickers",
        type=int,
        default=DEFAULT_LIMIT_TICKERS,
        help="Optional ticker limit for quick local runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        split_counts = split_and_save_datasets(
            data_dir=Path(args.data_dir),
            split_dir=Path(args.split_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            limit_tickers=args.limit_tickers,
        )
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(f"Saved chronological splits under {args.split_dir}.")
    print(
        f"Rows: train={split_counts['train']}, "
        f"validation={split_counts['validation']}, "
        f"test={split_counts['test']}"
    )


if __name__ == "__main__":
    main()
