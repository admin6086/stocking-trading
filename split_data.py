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
DEFAULT_STANDARDIZE_FEATURES = True

DATE_COLUMN = "Date"
FEATURE_COLUMNS = ["price", "Variation", "volume_change"]
TARGET_COLUMN = "target"
SPLIT_NAMES = ("train", "validation", "test")
TICKER_MAP_FILE = "ticker_to_id.csv"
METADATA_FILE = "metadata.json"
SPLIT_STRATEGY = "global_date_cutoff"
SCALER_FIT_SPLIT = "train"


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


def prepare_stock_frame(frame: pd.DataFrame, standardize_features: bool) -> pd.DataFrame:
    required_columns = {DATE_COLUMN, TARGET_COLUMN}
    if standardize_features:
        required_columns.update(FEATURE_COLUMNS)

    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")

    clean_frame = frame.copy()
    if standardize_features:
        clean_frame[FEATURE_COLUMNS] = clean_frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        clean_frame[FEATURE_COLUMNS] = clean_frame[FEATURE_COLUMNS].replace(
            [float("inf"), float("-inf")], pd.NA
        )
        clean_frame = clean_frame.dropna(subset=FEATURE_COLUMNS)

    clean_frame = clean_frame.dropna(subset=[TARGET_COLUMN])
    if clean_frame.empty:
        raise ValueError("no usable rows after cleaning")
    return clean_frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def compute_global_cutoffs(
    frames: list[pd.DataFrame], train_ratio: float, val_ratio: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    dates = sorted({date for frame in frames for date in frame[DATE_COLUMN]})
    train_cutoff_index = int(len(dates) * train_ratio)
    test_cutoff_index = int(len(dates) * (train_ratio + val_ratio))
    if train_cutoff_index <= 0 or test_cutoff_index <= train_cutoff_index or test_cutoff_index >= len(dates):
        raise ValueError("Not enough dates to create non-empty train, validation, and test periods.")

    return dates[train_cutoff_index], dates[test_cutoff_index]


def split_frame_by_dates(
    frame: pd.DataFrame, train_cutoff: pd.Timestamp, test_cutoff: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    train = frame[frame[DATE_COLUMN] < train_cutoff].copy()
    validation = frame[(frame[DATE_COLUMN] >= train_cutoff) & (frame[DATE_COLUMN] < test_cutoff)].copy()
    test = frame[frame[DATE_COLUMN] >= test_cutoff].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("date cutoffs produced an empty train, validation, or test split")

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def fit_feature_scaler(split_frames: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, float]]:
    train_frames = [splits["train"] for splits in split_frames.values()]
    train_data = pd.concat(train_frames, ignore_index=True)
    scaler: dict[str, dict[str, float]] = {}

    for feature in FEATURE_COLUMNS:
        mean = float(train_data[feature].mean())
        std = float(train_data[feature].std(ddof=0))
        if pd.isna(std) or std == 0.0:
            std = 1.0
        scaler[feature] = {"mean": mean, "std": std}

    return scaler


def apply_feature_scaler(frame: pd.DataFrame, scaler: dict[str, dict[str, float]]) -> pd.DataFrame:
    scaled = frame.copy()
    for feature, stats in scaler.items():
        scaled[feature] = (scaled[feature] - stats["mean"]) / stats["std"]
    return scaled


def clear_existing_split_csvs(split_dir: Path) -> None:
    for split_name in SPLIT_NAMES:
        output_dir = split_dir / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in output_dir.glob("*.csv"):
            csv_file.unlink()


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
    train_cutoff: pd.Timestamp,
    test_cutoff: pd.Timestamp,
    standardize_features: bool,
    scaler: dict[str, dict[str, float]] | None,
    ticker_summaries: list[dict[str, object]],
) -> None:
    metadata = {
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "date_column": DATE_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "split_strategy": SPLIT_STRATEGY,
        "train_ratio": train_ratio,
        "validation_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "train_period": f"dates before {train_cutoff.strftime('%Y-%m-%d')}",
        "validation_period": (
            f"{train_cutoff.strftime('%Y-%m-%d')} to before {test_cutoff.strftime('%Y-%m-%d')}"
        ),
        "test_period": f"dates from {test_cutoff.strftime('%Y-%m-%d')} onward",
        "train_cutoff": train_cutoff.strftime("%Y-%m-%d"),
        "test_cutoff": test_cutoff.strftime("%Y-%m-%d"),
        "standardization": {
            "enabled": standardize_features,
            "fit_on": SCALER_FIT_SPLIT if standardize_features else None,
            "method": "z_score" if standardize_features else None,
            "scaler": scaler or {},
        },
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
    standardize_features: bool = DEFAULT_STANDARDIZE_FEATURES,
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_split_csvs(split_dir)

    ticker_files = list_ticker_files(data_dir, limit_tickers)
    ticker_to_id: dict[str, int] = {}
    split_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    ticker_summaries: list[dict[str, object]] = []
    source_frames: list[tuple[Path, str, pd.DataFrame]] = []

    for file_path in ticker_files:
        try:
            frame = prepare_stock_frame(load_sorted_ticker_file(file_path), standardize_features)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        ticker = ticker_name_from_frame(file_path, frame)
        source_frames.append((file_path, ticker, frame))

    if not source_frames:
        raise ValueError("No ticker files were loaded.")

    train_cutoff, test_cutoff = compute_global_cutoffs(
        [frame for _, _, frame in source_frames],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    split_frames: dict[str, dict[str, pd.DataFrame]] = {}
    for file_path, ticker, frame in source_frames:
        try:
            split_frames[ticker] = split_frame_by_dates(frame, train_cutoff, test_cutoff)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

    if not split_frames:
        raise ValueError("No ticker files were split.")

    scaler = fit_feature_scaler(split_frames) if standardize_features else None
    for file_path, ticker, _ in source_frames:
        if ticker not in split_frames:
            continue

        ticker_to_id[ticker] = len(ticker_to_id)
        summary: dict[str, object] = {"ticker": ticker, "file": file_path.name}
        splits = split_frames[ticker]
        for split_name, split_frame_data in splits.items():
            output_frame = (
                apply_feature_scaler(split_frame_data, scaler)
                if standardize_features and scaler is not None
                else split_frame_data
            )
            output_path = save_split_file(split_dir, split_name, file_path.name, output_frame)
            split_counts[split_name] += len(split_frame_data)
            summary[f"{split_name}_rows"] = len(split_frame_data)
            summary[f"{split_name}_file"] = str(output_path)
            summary[f"{split_name}_start"] = split_frame_data[DATE_COLUMN].iloc[0].strftime("%Y-%m-%d")
            summary[f"{split_name}_end"] = split_frame_data[DATE_COLUMN].iloc[-1].strftime("%Y-%m-%d")
        ticker_summaries.append(summary)

    save_ticker_mapping(ticker_to_id, split_dir)
    save_metadata(
        split_dir=split_dir,
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        train_cutoff=train_cutoff,
        test_cutoff=test_cutoff,
        standardize_features=standardize_features,
        scaler=scaler,
        ticker_summaries=ticker_summaries,
    )
    return split_counts


def split_files_exist(split_dir: Path) -> bool:
    metadata_path = split_dir / METADATA_FILE
    if not metadata_path.exists():
        return False

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return False

    if metadata.get("split_strategy") != SPLIT_STRATEGY or "standardization" not in metadata:
        return False

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
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Store raw feature values instead of train-fitted z-score standardized features.",
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
            standardize_features=not args.no_standardize,
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
