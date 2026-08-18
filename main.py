import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from split_data import (
    DATE_COLUMN,
    DEFAULT_DATA_DIR,
    DEFAULT_LIMIT_TICKERS,
    DEFAULT_SPLIT_DIR,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_saved_feature_scaler,
    load_all_split_frames,
    load_ticker_frames,
    load_ticker_mapping,
    split_and_save_datasets,
    split_files_exist,
)


# PyCharm users: you can click Run with these defaults, or override them in
# Run > Edit Configurations > Parameters using the same --name value format.
DEFAULT_MODEL_DIR = "model"
DEFAULT_PREDICTIONS_DIR = "predictions"
DEFAULT_CNN_CHECKPOINT = "model/stock_cnn.pt"
DEFAULT_RESNET_CHECKPOINT = "model/stock_resnet.pt"
DEFAULT_VIT_CHECKPOINT = "model/stock_vit.pt"
DEFAULT_LATEST_OUT = "predictions/ensemble_latest_predictions.csv"
DEFAULT_VALIDATION_OUT = "predictions/ensemble_validation_predictions.csv"
DEFAULT_TEST_OUT = "predictions/ensemble_test_predictions.csv"
DEFAULT_METRICS_OUT = "predictions/ensemble_metrics.csv"
DEFAULT_WEIGHTS_OUT = "predictions/ensemble_weights.csv"
DEFAULT_BATCH_SIZE = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODE = "latest"
WINDOW_SIZE = 60


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module_path: str
    class_name: str
    checkpoint_path: str


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float


WindowData = dict[str, torch.Tensor | list[str]]


def log(message: str) -> None:
    print(message, flush=True)


def load_python_module(module_name: str, module_path: Path) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Model code file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model code from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Train this base model first."
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    return checkpoint


def normalize_ticker_mapping(mapping: object) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise ValueError("Checkpoint does not contain a valid ticker_to_id mapping.")
    return {str(ticker): int(ticker_id) for ticker, ticker_id in mapping.items()}


def load_trained_model(
    spec: ModelSpec,
    ticker_to_id: dict[str, int],
    device: torch.device,
) -> nn.Module:
    module = load_python_module(f"{spec.name}_module", Path(spec.module_path))
    model_class = getattr(module, spec.class_name)
    checkpoint = load_checkpoint(Path(spec.checkpoint_path), device)

    checkpoint_ticker_to_id = normalize_ticker_mapping(checkpoint.get("ticker_to_id"))
    if checkpoint_ticker_to_id != ticker_to_id:
        raise ValueError(
            f"{spec.name} checkpoint ticker_to_id does not match {DEFAULT_SPLIT_DIR}/ticker_to_id.csv. "
            "Use one shared Split folder and retrain all base models from that same split."
        )

    checkpoint_window_size = int(checkpoint.get("window_size", WINDOW_SIZE))
    if checkpoint_window_size != WINDOW_SIZE:
        raise ValueError(
            f"{spec.name} checkpoint uses window_size={checkpoint_window_size}, "
            f"but the ensemble expects {WINDOW_SIZE}."
        )

    checkpoint_features = list(checkpoint.get("feature_columns", FEATURE_COLUMNS))
    if checkpoint_features != FEATURE_COLUMNS:
        raise ValueError(
            f"{spec.name} checkpoint features {checkpoint_features} do not match {FEATURE_COLUMNS}."
        )

    model = model_class(num_tickers=len(ticker_to_id)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def clean_frame_for_training(
    ticker: str, frame: pd.DataFrame, show_skipped: bool = False
) -> pd.DataFrame | None:
    required_columns = {DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}
    missing = required_columns - set(frame.columns)
    if missing:
        if show_skipped:
            log(f"Skipping {ticker}: missing columns {sorted(missing)}")
        return None

    clean_frame = frame[[DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    clean_frame = clean_frame.dropna(subset=[*FEATURE_COLUMNS, TARGET_COLUMN])
    clean_frame = clean_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    if len(clean_frame) < WINDOW_SIZE:
        if show_skipped:
            log(f"Skipping {ticker}: only {len(clean_frame)} usable rows")
        return None
    return clean_frame


def clean_frame_for_latest(
    ticker: str, frame: pd.DataFrame, show_skipped: bool = False
) -> pd.DataFrame | None:
    required_columns = {DATE_COLUMN, *FEATURE_COLUMNS}
    missing = required_columns - set(frame.columns)
    if missing:
        if show_skipped:
            log(f"Skipping {ticker}: missing columns {sorted(missing)}")
        return None

    clean_frame = frame[[DATE_COLUMN, *FEATURE_COLUMNS]].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    clean_frame = clean_frame.dropna(subset=FEATURE_COLUMNS)
    clean_frame = clean_frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    if len(clean_frame) < WINDOW_SIZE:
        if show_skipped:
            log(f"Skipping {ticker}: only {len(clean_frame)} usable rows")
        return None
    return clean_frame


def make_window_data(
    frames: dict[str, pd.DataFrame],
    ticker_to_id: dict[str, int],
    show_skipped: bool = False,
) -> WindowData:
    windows: list[torch.Tensor] = []
    ticker_ids: list[int] = []
    targets: list[float] = []
    dates: list[str] = []
    tickers: list[str] = []

    for ticker, frame in sorted(frames.items()):
        if ticker not in ticker_to_id:
            continue

        clean_frame = clean_frame_for_training(ticker, frame, show_skipped)
        if clean_frame is None:
            continue

        values = clean_frame[FEATURE_COLUMNS].to_numpy(dtype="float32")
        target_values = clean_frame[TARGET_COLUMN].to_numpy(dtype="float32")
        date_values = pd.to_datetime(clean_frame[DATE_COLUMN]).dt.strftime("%Y-%m-%d").to_list()

        for row_index in range(WINDOW_SIZE - 1, len(clean_frame)):
            start = row_index - WINDOW_SIZE + 1
            windows.append(torch.from_numpy(values[start : row_index + 1].copy()))
            ticker_ids.append(ticker_to_id[ticker])
            targets.append(float(target_values[row_index]))
            dates.append(date_values[row_index])
            tickers.append(ticker)

    if not windows:
        raise ValueError("No windows were created. Check the split files and required columns.")

    return {
        "x": torch.stack(windows).float(),
        "ticker_id": torch.tensor(ticker_ids, dtype=torch.long),
        "y": torch.tensor(targets, dtype=torch.float32),
        "date": dates,
        "ticker": tickers,
    }


def make_latest_window_data(
    frames: dict[str, pd.DataFrame],
    ticker_to_id: dict[str, int],
    show_skipped: bool = False,
) -> WindowData:
    windows: list[torch.Tensor] = []
    ticker_ids: list[int] = []
    dates: list[str] = []
    tickers: list[str] = []

    for ticker, frame in sorted(frames.items()):
        if ticker not in ticker_to_id:
            continue

        clean_frame = clean_frame_for_latest(ticker, frame, show_skipped)
        if clean_frame is None:
            continue

        latest_window = clean_frame[FEATURE_COLUMNS].tail(WINDOW_SIZE).to_numpy(dtype="float32").copy()
        latest_date = pd.to_datetime(clean_frame[DATE_COLUMN].iloc[-1]).strftime("%Y-%m-%d")
        windows.append(torch.from_numpy(latest_window))
        ticker_ids.append(ticker_to_id[ticker])
        dates.append(latest_date)
        tickers.append(ticker)

    if not windows:
        raise ValueError("No latest inference windows were created from data.")

    return {
        "x": torch.stack(windows).float(),
        "ticker_id": torch.tensor(ticker_ids, dtype=torch.long),
        "date": dates,
        "ticker": tickers,
    }


def predict_probabilities(
    model: nn.Module,
    data: WindowData,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    x_values = data["x"]
    ticker_ids = data["ticker_id"]
    if not isinstance(x_values, torch.Tensor) or not isinstance(ticker_ids, torch.Tensor):
        raise TypeError("Window data has invalid tensor fields.")

    model.eval()
    probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x_values.size(0), batch_size):
            end = start + batch_size
            x_batch = x_values[start:end].transpose(1, 2).contiguous().to(device)
            ticker_batch = ticker_ids[start:end].to(device)
            logits = model(x_batch, ticker_batch)
            probabilities.append(torch.sigmoid(logits).cpu())

    return torch.cat(probabilities)


def evaluate_probabilities(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> EvaluationMetrics:
    probabilities = probabilities.float().clamp(1e-7, 1 - 1e-7)
    targets = targets.float()
    predictions = (probabilities >= threshold).float()

    true_positive = ((predictions == 1) & (targets == 1)).sum().item()
    false_positive = ((predictions == 1) & (targets == 0)).sum().item()
    false_negative = ((predictions == 0) & (targets == 1)).sum().item()
    correct = (predictions == targets).sum().item()
    total = int(targets.numel())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return EvaluationMetrics(
        loss=float(F.binary_cross_entropy(probabilities, targets).item()),
        accuracy=correct / total if total else 0.0,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )


def calculate_weights(
    validation_metrics: dict[str, EvaluationMetrics],
    weight_metric: str,
) -> dict[str, float]:
    if weight_metric == "equal":
        raw_weights = {name: 1.0 for name in validation_metrics}
    elif weight_metric == "inverse_loss":
        raw_weights = {
            name: 1.0 / max(metrics.loss, 1e-7)
            for name, metrics in validation_metrics.items()
        }
    elif weight_metric == "accuracy":
        raw_weights = {
            name: max(metrics.accuracy, 0.0)
            for name, metrics in validation_metrics.items()
        }
    elif weight_metric == "f1":
        raw_weights = {
            name: max(metrics.f1_score, 0.0)
            for name, metrics in validation_metrics.items()
        }
    else:
        raise ValueError(f"Unknown weight metric: {weight_metric}")

    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raw_weights = {name: 1.0 for name in validation_metrics}
        total_weight = sum(raw_weights.values())

    return {name: weight / total_weight for name, weight in raw_weights.items()}


def equal_model_weights(model_names: list[str]) -> dict[str, float]:
    if not model_names:
        raise ValueError("No model names were provided for weighting.")

    weight = 1.0 / len(model_names)
    return {model_name: weight for model_name in model_names}


def normalize_weights(weights: dict[str, float], model_names: list[str]) -> dict[str, float]:
    missing = sorted(set(model_names) - set(weights))
    extra = sorted(set(weights) - set(model_names))
    if missing or extra:
        raise ValueError(
            f"Weight model names do not match loaded models. Missing={missing}, extra={extra}."
        )

    clipped_weights = {name: max(float(weights[name]), 0.0) for name in model_names}
    total_weight = sum(clipped_weights.values())
    if total_weight <= 0:
        return equal_model_weights(model_names)
    return {name: weight / total_weight for name, weight in clipped_weights.items()}


def load_weights_file(weights_path: Path, model_names: list[str]) -> dict[str, float] | None:
    if not weights_path.exists():
        return None

    frame = pd.read_csv(weights_path)
    required_columns = {"model", "weight"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{weights_path} is missing columns {sorted(missing)}")

    weights = {str(row.model): float(row.weight) for row in frame.itertuples(index=False)}
    return normalize_weights(weights, model_names)


def save_weights_file(
    weights_path: Path,
    weights: dict[str, float],
    weight_metric: str,
) -> None:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"model": model_name, "weight": weight, "source": weight_metric}
        for model_name, weight in sorted(weights.items())
    ]
    pd.DataFrame(rows).to_csv(weights_path, index=False)


def weighted_average_probabilities(
    model_probabilities: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    ensemble_probability: torch.Tensor | None = None
    for model_name, probabilities in model_probabilities.items():
        weighted_probabilities = probabilities * weights[model_name]
        if ensemble_probability is None:
            ensemble_probability = weighted_probabilities
        else:
            ensemble_probability = ensemble_probability + weighted_probabilities

    if ensemble_probability is None:
        raise ValueError("No model probabilities were provided.")
    return ensemble_probability


def build_prediction_frame(
    data: WindowData,
    model_probabilities: dict[str, torch.Tensor],
    ensemble_probabilities: torch.Tensor,
    threshold: float,
    include_target: bool,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ticker": data["ticker"],
            "date": data["date"],
            "ensemble_probability": ensemble_probabilities.numpy(),
            "ensemble_prediction": (ensemble_probabilities >= threshold).int().numpy(),
        }
    )

    if include_target:
        targets = data["y"]
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Target data has an invalid tensor field.")
        frame["target"] = targets.numpy()

    for model_name, probabilities in model_probabilities.items():
        frame[f"{model_name}_probability"] = probabilities.numpy()

    return frame.sort_values("ensemble_probability", ascending=False).reset_index(drop=True)


def metrics_to_rows(
    validation_metrics: dict[str, EvaluationMetrics],
    test_metrics: dict[str, EvaluationMetrics],
    weights: dict[str, float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    model_names = sorted(set(validation_metrics) | set(test_metrics))

    for model_name in model_names:
        validation = validation_metrics.get(model_name)
        test = test_metrics.get(model_name)
        if validation is not None:
            rows.append(
                {
                    "split": "validation",
                    "model": model_name,
                    "weight": weights.get(model_name, 0.0),
                    "loss": validation.loss,
                    "accuracy": validation.accuracy,
                    "precision": validation.precision,
                    "recall": validation.recall,
                    "f1_score": validation.f1_score,
                }
            )
        if test is not None:
            rows.append(
                {
                    "split": "test",
                    "model": model_name,
                    "weight": weights.get(model_name, 0.0),
                    "loss": test.loss,
                    "accuracy": test.accuracy,
                    "precision": test.precision,
                    "recall": test.recall,
                    "f1_score": test.f1_score,
                }
            )

    return rows


def print_metrics(label: str, model_name: str, metrics: EvaluationMetrics) -> None:
    log(
        f"{label} {model_name}: "
        f"loss={metrics.loss:.4f} | "
        f"accuracy={metrics.accuracy:.4f} | "
        f"precision={metrics.precision:.4f} | "
        f"recall={metrics.recall:.4f} | "
        f"f1={metrics.f1_score:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a weighted soft-voting stock model ensemble.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit-tickers", type=int, default=DEFAULT_LIMIT_TICKERS)
    parser.add_argument("--cnn-checkpoint", default=DEFAULT_CNN_CHECKPOINT)
    parser.add_argument("--resnet-checkpoint", default=DEFAULT_RESNET_CHECKPOINT)
    parser.add_argument("--vit-checkpoint", default=DEFAULT_VIT_CHECKPOINT)
    parser.add_argument("--latest-out", default=DEFAULT_LATEST_OUT)
    parser.add_argument("--validation-out", default=DEFAULT_VALIDATION_OUT)
    parser.add_argument("--test-out", default=DEFAULT_TEST_OUT)
    parser.add_argument("--metrics-out", default=DEFAULT_METRICS_OUT)
    parser.add_argument("--weights-in", default=DEFAULT_WEIGHTS_OUT)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_OUT)
    parser.add_argument(
        "--mode",
        choices=("latest", "evaluate", "all"),
        default=DEFAULT_MODE,
        help="latest predicts current tickers, evaluate refreshes metrics/weights, all does both.",
    )
    parser.add_argument(
        "--weight-metric",
        choices=("f1", "accuracy", "inverse_loss", "equal"),
        default="f1",
        help="Validation metric used to set soft-voting weights in evaluate/all mode.",
    )
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="Recreate Split/train, Split/validation, and Split/test before running the ensemble.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only calculate validation weights and latest predictions.",
    )
    parser.add_argument(
        "--show-skipped-tickers",
        action="store_true",
        help="Print tickers skipped because they have missing columns or fewer than 60 usable rows.",
    )
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
        log(
            f"Created chronological CSV splits under {split_dir}: "
            f"train={split_counts['train']}, "
            f"validation={split_counts['validation']}, "
            f"test={split_counts['test']} rows."
        )
    else:
        log(f"Loaded existing chronological CSV splits from {split_dir}.")

    ticker_to_id = load_ticker_mapping(split_dir)
    model_specs = [
        ModelSpec("cnn", "CNN model.py", "StockCNN", args.cnn_checkpoint),
        ModelSpec("resnet", "ResNet.py", "StockResNet", args.resnet_checkpoint),
        ModelSpec("vit", "Vit.py", "StockViT", args.vit_checkpoint),
    ]
    models = {
        spec.name: load_trained_model(spec, ticker_to_id, device)
        for spec in model_specs
    }
    log(f"Loaded {len(models)} base models for {len(ticker_to_id)} ticker IDs. Device: {device}.")
    model_names = list(models)

    weights: dict[str, float] | None = None
    validation_metrics: dict[str, EvaluationMetrics] = {}
    test_metrics: dict[str, EvaluationMetrics] = {}

    if args.mode in ("evaluate", "all"):
        log("Loading validation/test split files.")
        split_frames = load_all_split_frames(split_dir, args.limit_tickers)

        log("Building validation windows.")
        validation_data = make_window_data(
            split_frames["validation"],
            ticker_to_id,
            show_skipped=args.show_skipped_tickers,
        )
        validation_x = validation_data["x"]
        if not isinstance(validation_x, torch.Tensor):
            raise TypeError("Validation window data has an invalid tensor field.")
        log(f"Validation windows: {validation_x.size(0)}")

        validation_probabilities: dict[str, torch.Tensor] = {}
        for model_name, model in models.items():
            log(f"Predicting validation probabilities with {model_name}.")
            validation_probabilities[model_name] = predict_probabilities(
                model, validation_data, args.batch_size, device
            )

        validation_targets = validation_data["y"]
        if not isinstance(validation_targets, torch.Tensor):
            raise TypeError("Validation target data has an invalid tensor field.")

        validation_metrics = {
            name: evaluate_probabilities(probabilities, validation_targets, args.threshold)
            for name, probabilities in validation_probabilities.items()
        }
        for model_name, metrics in validation_metrics.items():
            print_metrics("Validation", model_name, metrics)

        weights = calculate_weights(validation_metrics, args.weight_metric)
        weights_text = ", ".join(f"{name}={weight:.4f}" for name, weight in weights.items())
        log(f"Ensemble weights from validation {args.weight_metric}: {weights_text}")
        save_weights_file(Path(args.weights_out), weights, args.weight_metric)
        log(f"Saved ensemble weights to {args.weights_out}")

        validation_ensemble = weighted_average_probabilities(validation_probabilities, weights)
        ensemble_validation_metrics = evaluate_probabilities(
            validation_ensemble, validation_targets, args.threshold
        )
        print_metrics("Validation", "ensemble", ensemble_validation_metrics)

        validation_output = build_prediction_frame(
            validation_data,
            validation_probabilities,
            validation_ensemble,
            args.threshold,
            include_target=True,
        )
        validation_out = Path(args.validation_out)
        validation_out.parent.mkdir(parents=True, exist_ok=True)
        validation_output.to_csv(validation_out, index=False)
        log(f"Saved validation ensemble predictions to {validation_out}")

        if not args.skip_test:
            log("Building test windows.")
            test_data = make_window_data(
                split_frames["test"],
                ticker_to_id,
                show_skipped=args.show_skipped_tickers,
            )
            test_x = test_data["x"]
            if not isinstance(test_x, torch.Tensor):
                raise TypeError("Test window data has an invalid tensor field.")
            log(f"Test windows: {test_x.size(0)}")

            test_probabilities: dict[str, torch.Tensor] = {}
            for model_name, model in models.items():
                log(f"Predicting test probabilities with {model_name}.")
                test_probabilities[model_name] = predict_probabilities(
                    model, test_data, args.batch_size, device
                )

            test_targets = test_data["y"]
            if not isinstance(test_targets, torch.Tensor):
                raise TypeError("Test target data has an invalid tensor field.")

            test_metrics = {
                name: evaluate_probabilities(probabilities, test_targets, args.threshold)
                for name, probabilities in test_probabilities.items()
            }
            for model_name, metrics in test_metrics.items():
                print_metrics("Test", model_name, metrics)

            test_ensemble = weighted_average_probabilities(test_probabilities, weights)
            test_metrics["ensemble"] = evaluate_probabilities(test_ensemble, test_targets, args.threshold)
            print_metrics("Test", "ensemble", test_metrics["ensemble"])

            test_output = build_prediction_frame(
                test_data,
                test_probabilities,
                test_ensemble,
                args.threshold,
                include_target=True,
            )
            test_out = Path(args.test_out)
            test_out.parent.mkdir(parents=True, exist_ok=True)
            test_output.to_csv(test_out, index=False)
            log(f"Saved test ensemble predictions to {test_out}")

        validation_metrics["ensemble"] = ensemble_validation_metrics
        metrics_rows = metrics_to_rows(validation_metrics, test_metrics, weights)
        metrics_out = Path(args.metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(metrics_out, index=False)
        log(f"Saved ensemble metrics to {metrics_out}")

    if weights is None:
        weights = load_weights_file(Path(args.weights_in), model_names)
        if weights is None:
            weights = equal_model_weights(model_names)
            log(f"No saved ensemble weights found at {args.weights_in}; using equal weights.")
        else:
            log(f"Loaded ensemble weights from {args.weights_in}.")

        weights_text = ", ".join(f"{name}={weight:.4f}" for name, weight in weights.items())
        log(f"Ensemble weights: {weights_text}")

    if args.mode in ("latest", "all"):
        log("Loading latest raw ticker data.")
        latest_frames, _ = load_ticker_frames(data_dir, args.limit_tickers)
        latest_frames = apply_saved_feature_scaler(latest_frames, split_dir)
        latest_data = make_latest_window_data(
            latest_frames,
            ticker_to_id,
            show_skipped=args.show_skipped_tickers,
        )
        latest_x = latest_data["x"]
        if not isinstance(latest_x, torch.Tensor):
            raise TypeError("Latest window data has an invalid tensor field.")
        log(f"Latest prediction windows: {latest_x.size(0)}")

        latest_probabilities: dict[str, torch.Tensor] = {}
        for model_name, model in models.items():
            log(f"Predicting latest probabilities with {model_name}.")
            latest_probabilities[model_name] = predict_probabilities(
                model, latest_data, args.batch_size, device
            )

        latest_ensemble = weighted_average_probabilities(latest_probabilities, weights)
        latest_output = build_prediction_frame(
            latest_data,
            latest_probabilities,
            latest_ensemble,
            args.threshold,
            include_target=False,
        )
        latest_output = latest_output.rename(columns={"date": "latest_date"})

        latest_out = Path(args.latest_out)
        latest_out.parent.mkdir(parents=True, exist_ok=True)
        latest_output.to_csv(latest_out, index=False)
        log(f"Saved latest ensemble predictions to {latest_out}")
        log(latest_output.head(20).to_string(index=False))
    else:
        log("Evaluation complete. Run main.py with --mode latest to create current predictions.")


if __name__ == "__main__":
    main()
