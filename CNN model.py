import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# PyCharm users: you can click Run with these defaults, or override them in
# Run > Edit Configurations > Parameters using the same --name value format.
DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_OUT = "model/stock_cnn.pt"
DEFAULT_PREDICTIONS_OUT = "predictions/latest_predictions.csv"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_LIMIT_TICKERS = None

FEATURE_COLUMNS = ["price", "Variation", "volume_change"]
TARGET_COLUMN = "target"
DATE_COLUMN = "Date"
WINDOW_SIZE = 60


@dataclass(frozen=True)
class SplitData:
    train: list[tuple[pd.Timestamp, int, torch.Tensor, float]]
    val: list[tuple[pd.Timestamp, int, torch.Tensor, float]]
    test: list[tuple[pd.Timestamp, int, torch.Tensor, float]]


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class StockWindowDataset(Dataset):
    def __init__(self, samples: list[tuple[pd.Timestamp, int, torch.Tensor, float]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, ticker_id, features, target = self.samples[index]
        # Conv1d expects channels first: (features, window).
        x = features.transpose(0, 1).contiguous()
        ticker = torch.tensor(ticker_id, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.float32)
        return x, ticker, y


class StockCNN(nn.Module):
    def __init__(self, num_tickers: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(64 + embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(x).squeeze(-1)
        ticker_features = self.ticker_embedding(ticker_id)
        combined = torch.cat([cnn_features, ticker_features], dim=1)
        return self.classifier(combined).squeeze(1)


def load_ticker_frames(
    data_dir: Path, limit_tickers: int | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")
    if limit_tickers is not None:
        files = files[:limit_tickers]

    frames: dict[str, pd.DataFrame] = {}
    for file_path in files:
        frame = pd.read_csv(file_path, parse_dates=[DATE_COLUMN])
        required_columns = {DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
        missing = required_columns - set(frame.columns)
        if missing:
            print(f"Skipping {file_path.name}: missing columns {sorted(missing)}")
            continue

        ticker = str(frame["name"].iloc[0]) if "name" in frame.columns and not frame.empty else file_path.stem
        frame = frame[[DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].replace(
            [float("inf"), float("-inf")], pd.NA
        )
        frame = frame.dropna().copy()
        frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
        if len(frame) <= WINDOW_SIZE:
            print(f"Skipping {ticker}: only {len(frame)} usable rows")
            continue
        frames[ticker] = frame

    if not frames:
        raise ValueError("No ticker files had enough usable rows to create windows.")

    ticker_to_id = {ticker: index for index, ticker in enumerate(sorted(frames))}
    return frames, ticker_to_id


def make_window_samples(
    frames: dict[str, pd.DataFrame], ticker_to_id: dict[str, int]
) -> list[tuple[pd.Timestamp, int, torch.Tensor, float]]:
    samples: list[tuple[pd.Timestamp, int, torch.Tensor, float]] = []
    for ticker, frame in frames.items():
        values = frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
        targets = frame[TARGET_COLUMN].to_numpy(dtype="float32")
        dates = frame[DATE_COLUMN].to_list()

        for row_index in range(WINDOW_SIZE - 1, len(frame)):
            start = row_index - WINDOW_SIZE + 1
            window = torch.from_numpy(values[start : row_index + 1].copy())
            prediction_date = dates[row_index]
            target = float(targets[row_index])
            samples.append((prediction_date, ticker_to_id[ticker], window, target))

    return sorted(samples, key=lambda sample: sample[0])


def chronological_split(
    samples: list[tuple[pd.Timestamp, int, torch.Tensor, float]],
    train_ratio: float,
    val_ratio: float,
) -> SplitData:
    if not samples:
        raise ValueError("No samples were created.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    dates = sorted({sample[0] for sample in samples})
    train_cutoff = dates[int(len(dates) * train_ratio)]
    val_cutoff = dates[int(len(dates) * (train_ratio + val_ratio))]

    train = [sample for sample in samples if sample[0] < train_cutoff]
    val = [sample for sample in samples if train_cutoff <= sample[0] < val_cutoff]
    test = [sample for sample in samples if sample[0] >= val_cutoff]
    if not train or not val or not test:
        raise ValueError("Chronological split produced an empty train, validation, or test set.")

    return SplitData(train=train, val=val, test=test)


def make_latest_inference_windows(
    frames: dict[str, pd.DataFrame], ticker_to_id: dict[str, int]
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    inference_samples: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for ticker, frame in sorted(frames.items()):
        latest_window = frame[FEATURE_COLUMNS].tail(WINDOW_SIZE).to_numpy(dtype="float32").copy()
        x = torch.from_numpy(latest_window).transpose(0, 1).unsqueeze(0).contiguous()
        ticker_id = torch.tensor([ticker_to_id[ticker]], dtype=torch.long)
        inference_samples.append((ticker, x, ticker_id))
    return inference_samples


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> EvaluationMetrics:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for x, ticker_id, y in loader:
            x = x.to(device)
            ticker_id = ticker_id.to(device)
            y = y.to(device)

            logits = model(x, ticker_id)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += (predictions == y).sum().item()
            true_positive += ((predictions == 1) & (y == 1)).sum().item()
            false_positive += ((predictions == 1) & (y == 0)).sum().item()
            false_negative += ((predictions == 0) & (y == 1)).sum().item()
            total += y.size(0)

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return EvaluationMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, ticker_id, y in train_loader:
            x = x.to(device)
            ticker_id = ticker_id.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, ticker_id)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = total_loss / total
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"val_accuracy={val_metrics.accuracy:.4f} | "
            f"val_precision={val_metrics.precision:.4f} | "
            f"val_recall={val_metrics.recall:.4f} | "
            f"val_f1={val_metrics.f1_score:.4f}"
        )


def predict_latest(
    model: nn.Module,
    frames: dict[str, pd.DataFrame],
    ticker_to_id: dict[str, int],
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    model.eval()

    with torch.no_grad():
        for ticker, x, ticker_id in make_latest_inference_windows(frames, ticker_to_id):
            probability = torch.sigmoid(model(x.to(device), ticker_id.to(device))).item()
            rows.append({"ticker": ticker, "next_day_probability": probability})

    return pd.DataFrame(rows).sort_values("next_day_probability", ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D CNN on per-ticker stock CSV files.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder containing one CSV file per ticker.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument(
        "--limit-tickers",
        type=int,
        default=DEFAULT_LIMIT_TICKERS,
        help="Optional ticker limit for quick local runs.",
    )
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Path for the saved model checkpoint.")
    parser.add_argument("--predictions-out", default=DEFAULT_PREDICTIONS_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames, ticker_to_id = load_ticker_frames(Path(args.data_dir), args.limit_tickers)

    samples = make_window_samples(frames, ticker_to_id)
    split = chronological_split(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    train_loader = DataLoader(StockWindowDataset(split.train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(StockWindowDataset(split.val), batch_size=args.batch_size)
    test_loader = DataLoader(StockWindowDataset(split.test), batch_size=args.batch_size)

    model = StockCNN(num_tickers=len(ticker_to_id)).to(device)
    print(
        f"Loaded {len(frames)} tickers and {len(samples)} samples. "
        f"Train/val/test: {len(split.train)}/{len(split.val)}/{len(split.test)}. "
        f"Device: {device}."
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"Test loss={test_metrics.loss:.4f} | "
        f"test_accuracy={test_metrics.accuracy:.4f} | "
        f"test_precision={test_metrics.precision:.4f} | "
        f"test_recall={test_metrics.recall:.4f} | "
        f"test_f1={test_metrics.f1_score:.4f}"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ticker_to_id": ticker_to_id,
        "feature_columns": FEATURE_COLUMNS,
        "window_size": WINDOW_SIZE,
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)
    print(f"Saved model checkpoint to {model_out}")

    predictions = predict_latest(model, frames, ticker_to_id, device)
    predictions_out = Path(args.predictions_out)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)
    print(f"Saved latest predictions to {predictions_out}")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
