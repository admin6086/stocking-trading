import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from split_data import (
    DATE_COLUMN,
    DEFAULT_DATA_DIR,
    DEFAULT_LIMIT_TICKERS,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    apply_saved_feature_scaler,
    load_all_split_frames,
    load_ticker_frames,
    load_ticker_mapping,
    split_and_save_datasets,
    split_files_exist,
)


DEFAULT_MODEL_OUT = "model/stock_vit.pt"
DEFAULT_PREDICTIONS_OUT = "predictions/vit_latest_predictions.csv"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3
FEATURE_COLUMNS = ["price", "Variation", "volume_change"]
TARGET_COLUMN = "target"
WINDOW_SIZE = 60
PATCH_SIZE = 5
WindowData = dict[str, torch.Tensor | list[str]]


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


class StockWindowDataset(Dataset):
    def __init__(self, split: WindowData) -> None:
        self.x = split["x"]
        self.ticker_ids = split["ticker_id"]
        self.y = split["y"]

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Transformer model receives the same channels-first shape as CNN/ResNet.
        x = self.x[index].transpose(0, 1).contiguous()
        return x, self.ticker_ids[index], self.y[index]


class StockViT(nn.Module):
    def __init__(
        self,
        num_tickers: int,
        embedding_dim: int = 16,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if WINDOW_SIZE % PATCH_SIZE != 0:
            raise ValueError("WINDOW_SIZE must be divisible by PATCH_SIZE.")

        self.num_patches = WINDOW_SIZE // PATCH_SIZE
        self.patch_embedding = nn.Linear(PATCH_SIZE * len(FEATURE_COLUMNS), model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim + embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        sequence = x.transpose(1, 2).contiguous()
        patches = sequence.reshape(batch_size, self.num_patches, PATCH_SIZE * len(FEATURE_COLUMNS))
        patch_tokens = self.patch_embedding(patches)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        tokens = tokens + self.position_embedding

        encoded = self.transformer(tokens)
        stock_features = self.norm(encoded[:, 0])
        ticker_features = self.ticker_embedding(ticker_id)
        combined = torch.cat([stock_features, ticker_features], dim=1)
        return self.classifier(combined).squeeze(1)


def clean_vit_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame | None:
    required_columns = {DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required_columns - set(frame.columns)
    if missing:
        print(f"Skipping {ticker}: missing columns {sorted(missing)}")
        return None

    clean_frame = frame[[DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    clean_frame = clean_frame.dropna().sort_values(DATE_COLUMN).reset_index(drop=True)
    if len(clean_frame) < WINDOW_SIZE:
        print(f"Skipping {ticker}: only {len(clean_frame)} usable rows")
        return None
    return clean_frame


def make_vit_window_data(
    frames: dict[str, pd.DataFrame], ticker_to_id: dict[str, int]
) -> WindowData:
    windows: list[torch.Tensor] = []
    ticker_ids: list[int] = []
    targets: list[float] = []
    dates: list[str] = []
    tickers: list[str] = []

    for ticker, frame in sorted(frames.items()):
        if ticker not in ticker_to_id:
            continue

        clean_frame = clean_vit_frame(ticker, frame)
        if clean_frame is None:
            continue

        values = clean_frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
        target_values = clean_frame[TARGET_COLUMN].to_numpy(dtype="float32")
        date_values = clean_frame[DATE_COLUMN].to_list()

        for row_index in range(WINDOW_SIZE - 1, len(clean_frame)):
            start = row_index - WINDOW_SIZE + 1
            windows.append(torch.from_numpy(values[start : row_index + 1].copy()))
            ticker_ids.append(ticker_to_id[ticker])
            targets.append(float(target_values[row_index]))
            dates.append(date_values[row_index].strftime("%Y-%m-%d"))
            tickers.append(ticker)

    if not windows:
        raise ValueError("No ViT windows were created from this split.")

    return {
        "x": torch.stack(windows).float(),
        "ticker_id": torch.tensor(ticker_ids, dtype=torch.long),
        "y": torch.tensor(targets, dtype=torch.float32),
        "date": dates,
        "ticker": tickers,
    }


def make_latest_inference_windows(
    frames: dict[str, pd.DataFrame], ticker_to_id: dict[str, int]
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    inference_samples: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for ticker, frame in sorted(frames.items()):
        if ticker not in ticker_to_id:
            continue

        clean_frame = clean_vit_frame(ticker, frame)
        if clean_frame is None:
            continue

        latest_window = clean_frame[FEATURE_COLUMNS].tail(WINDOW_SIZE).to_numpy(dtype="float32").copy()
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
    parser = argparse.ArgumentParser(description="Train a ViT-style Transformer on per-ticker stock CSV files.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder containing one CSV file per ticker.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Recreate Split/train, Split/validation, and Split/test before training.",
    )
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

    data_dir = Path(args.data_dir)
    split_dir = Path(args.split_dir)
    if args.rebuild_splits or not split_files_exist(split_dir):
        split_counts = split_and_save_datasets(
            data_dir=data_dir,
            split_dir=split_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            limit_tickers=args.limit_tickers,
        )
        print(
            f"Created chronological CSV splits under {split_dir}: "
            f"train={split_counts['train']}, "
            f"validation={split_counts['validation']}, "
            f"test={split_counts['test']} rows."
        )
    else:
        print(f"Loaded existing chronological CSV splits from {split_dir}.")

    split_frames = load_all_split_frames(split_dir, args.limit_tickers)
    ticker_to_id = load_ticker_mapping(split_dir)
    train_windows = make_vit_window_data(split_frames["train"], ticker_to_id)
    validation_windows = make_vit_window_data(split_frames["validation"], ticker_to_id)
    test_windows = make_vit_window_data(split_frames["test"], ticker_to_id)

    train_loader = DataLoader(StockWindowDataset(train_windows), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(StockWindowDataset(validation_windows), batch_size=args.batch_size)
    test_loader = DataLoader(StockWindowDataset(test_windows), batch_size=args.batch_size)

    model = StockViT(num_tickers=len(ticker_to_id)).to(device)
    print(
        f"Loaded {len(ticker_to_id)} ticker IDs. "
        f"ViT windows train/validation/test: "
        f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}. "
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
        "patch_size": PATCH_SIZE,
        "model_type": "StockViT",
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_out)
    print(f"Saved model checkpoint to {model_out}")

    frames, _ = load_ticker_frames(data_dir, args.limit_tickers)
    frames = apply_saved_feature_scaler(frames, split_dir)
    predictions = predict_latest(model, frames, ticker_to_id, device)
    predictions_out = Path(args.predictions_out)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_out, index=False)
    print(f"Saved latest predictions to {predictions_out}")
    print(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
