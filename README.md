# Stock Movement Prediction with Deep Learning

This project builds a stock movement prediction pipeline using historical stock data and three deep learning models: a 1D CNN, a 1D ResNet, and a ViT-inspired time-series model. The trained models are combined with a weighted soft-voting ensemble to produce next-day stock movement predictions for each ticker.

The project is intended for research and educational use. It is not financial advice and should not be used as a standalone trading system.

## Project Overview

The pipeline has five main stages:

1. Fetch historical stock data from Yahoo Finance.
2. Engineer stock features and save one CSV file per ticker.
3. Split the data chronologically into training, validation, and test periods.
4. Train CNN, ResNet, and ViT-inspired models using rolling 60-day windows.
5. Combine the trained models with an ensemble system and generate latest predictions.

Each model receives:

- a 60-day feature window with shape `(60, 3)`
- a scalar `ticker_id`
- a next-day binary target during training

For PyTorch model input, the feature window is transposed to:

```text
(batch_size, 3, 60)
```

## Features

The raw OHLCV data are transformed into three model features:

| Feature | Description |
|---|---|
| `price` | Daily close-to-close return |
| `Variation` | Daily high-low range divided by previous close |
| `volume_change` | Daily percentage change in volume |

The training target is binary:

```text
target = 1 if next day's close is higher than current close
target = 0 otherwise
```

## Project Structure

```text
.
├── get_stock_data.py          # Fetches Yahoo Finance data and creates per-ticker CSV files
├── split_data.py              # Creates chronological train/validation/test splits
├── CNN model.py               # Trains the 1D CNN model
├── ResNet.py                  # Trains the 1D ResNet model
├── Vit.py                     # Trains the ViT-inspired time-series model
├── main.py                    # Runs the weighted ensemble system
├── requirements.txt           # Python dependencies
├── data/                      # Raw processed ticker CSV files
├── Split/                     # Chronological train/validation/test files and metadata
├── model/                     # Saved PyTorch model checkpoints
├── new_data/                  # Newest inference-ready ticker data
└── predictions/               # Prediction and evaluation output files
```

Large generated folders such as `data`, `Split`, `predictions`, and virtual environments are ignored by Git. Model checkpoint files in `model` may also be large, so check file sizes before publishing them.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `pandas`
- `requests`
- `torch`
- `yfinance`

## Usage

### 1. Fetch Stock Data

Fetch data for all discovered US-listed stocks:

```bash
python get_stock_data.py
```

Fetch data for one ticker:

```bash
python get_stock_data.py --ticker AAPL
```

Limit the number of tickers for a quick test:

```bash
python get_stock_data.py --limit 20
```

By default, processed ticker files are saved under `data/`.

### 2. Split the Dataset

Create chronological train, validation, and test splits:

```bash
python split_data.py
```

The split files are saved under:

```text
Split/train/
Split/validation/
Split/test/
```

The script also saves:

- `Split/ticker_to_id.csv`
- `Split/metadata.json`

Feature standardisation is fitted on the training period only, then applied to validation and test data to avoid future-data leakage.

### 3. Train the Base Models

Train the CNN model:

```bash
python "CNN model.py"
```

Train the ResNet model:

```bash
python ResNet.py
```

Train the ViT-inspired model:

```bash
python Vit.py
```

By default, checkpoints are saved to:

```text
model/stock_cnn.pt
model/stock_resnet.pt
model/stock_vit.pt
```

Training settings such as epochs, batch size, learning rate, and output path can be changed through command-line arguments. In PyCharm, these can be set in:

```text
Run > Edit Configurations > Parameters
```

Example:

```bash
python ResNet.py --epochs 20 --batch-size 128 --learning-rate 0.0005
```

### 4. Run the Ensemble

Evaluate the three trained models and calculate ensemble weights:

```bash
python main.py --mode evaluate
```

Run latest prediction only:

```bash
python main.py
```

The default mode is `latest`, so running `main.py` loads the trained checkpoints and creates latest ensemble predictions.

Run both evaluation and latest prediction:

```bash
python main.py --mode all
```

The ensemble uses weighted soft voting. By default, model weights are calculated from validation F1 scores:

```text
ensemble_probability = w_cnn * p_cnn + w_resnet * p_resnet + w_vit * p_vit
```

The final class prediction uses a threshold of `0.5`.

## New Data for Inference

For newest prediction, the ensemble expects recent processed ticker files under:

```text
new_data/
```

Each ticker file should contain at least 60 usable rows with these columns:

```text
Date, name, price, Variation, volume_change
```

During inference, `main.py` applies the saved feature scaler from `Split/metadata.json`, selects the latest 60 rows for each ticker, and generates one prediction per ticker.

If you want to use another folder for newest data:

```bash
python main.py --latest-data-dir path/to/new_data
```

## Outputs

The project saves model checkpoints in:

```text
model/
```

Prediction and evaluation outputs are saved in:

```text
predictions/
```

Common output files include:

| File | Description |
|---|---|
| `ensemble_latest_predictions.csv` | Latest one-row-per-ticker ensemble predictions |
| `ensemble_validation_predictions.csv` | Validation-period ensemble predictions |
| `ensemble_test_predictions.csv` | Test-period ensemble predictions |
| `ensemble_metrics.csv` | Accuracy, precision, recall, F1 score, and loss |
| `ensemble_weights.csv` | Saved soft-voting weights |

## Evaluation

The models are evaluated using:

- loss
- accuracy
- precision
- recall
- F1 score

Chronological splitting is used instead of random row splitting because stock data are time-dependent. This helps reduce look-ahead bias.

## Notes

- The models are global models trained across many tickers.
- Ticker IDs are passed through trainable embedding layers.
- The same `ticker_to_id` mapping must be used during training, evaluation, and inference.
- New tickers that were not present during training will not have learned ticker embeddings.
- The prediction task is difficult because next-day stock movement is noisy and affected by many external factors not included in this dataset.

## Disclaimer

This project is for academic and experimental purposes only. The predictions should not be interpreted as financial advice, investment recommendations, or guaranteed trading signals.
