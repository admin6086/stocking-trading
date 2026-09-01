# Stock Movement Prediction with Deep Learning

This project implements a deep learning pipeline for next-day stock movement prediction. It downloads historical stock data, creates ticker-level datasets, splits the data chronologically, trains three neural network models, and combines them with a weighted soft-voting ensemble.

## Overview

The workflow is:

1. Fetch historical stock data from Yahoo Finance.
2. Create one processed CSV file per ticker under `data/`.
3. Split the ticker files chronologically into `train`, `validation`, and `test` sets.
4. Train three base models: 1D CNN, 1D ResNet, and ViT-inspired time-series model.
5. Combine the trained models using a weighted ensemble.
6. Generate latest ticker-level predictions from the processed files in `data/`.

Each training sample represents one ticker at one prediction date. The model receives a 60-day window of three engineered features and a ticker ID:

```text
X shape before PyTorch transpose: (60, 3)
X shape used by the models:       (batch_size, 3, 60)
ticker_id:                        scalar
y:                                next-day binary target
```

## Features and Target

The raw OHLCV data are converted into three features:

| Feature | Description |
|---|---|
| `price` | Daily close-to-close return |
| `Variation` | Daily high-low range divided by previous close |
| `volume_change` | Daily percentage change in trading volume |

The prediction target is:

```text
target = 1 if the next trading day's close is higher than the current close
target = 0 otherwise
```

## Project Structure

```text
.
├── get_stock_data.py          # Downloads Yahoo Finance data and saves one CSV per ticker
├── split_data.py              # Creates chronological train/validation/test splits
├── CNN model.py               # Trains the 1D CNN model
├── ResNet.py                  # Trains the 1D ResNet model
├── Vit.py                     # Trains the ViT-inspired time-series model
├── main.py                    # Runs the weighted ensemble system
├── requirements.txt           # Python dependencies
├── data/                      # Processed ticker CSV files
├── Split/                     # Split datasets, ticker mapping, and metadata
├── model/                     # Saved PyTorch checkpoints
└── predictions/               # Prediction and evaluation outputs
```

The generated folders `data/`, `Split/`, and `predictions/` are ignored by Git. The `model/` folder may contain large checkpoint files, so decide whether to commit them, ignore them, or store them with Git LFS before publishing.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Dependencies:

- `pandas`
- `requests`
- `torch`
- `yfinance`

## How to Run

### 1. Fetch Stock Data

Download and process data for all discovered US-listed stocks:

```bash
python get_stock_data.py
```

Download data for one ticker:

```bash
python get_stock_data.py --ticker AAPL
```

Run a smaller test download:

```bash
python get_stock_data.py --limit 20
```

By default, files are saved under `data/`, with one CSV file per ticker.

### 2. Split the Dataset

Create chronological training, validation, and test files:

```bash
python split_data.py
```

This creates:

```text
Split/train/
Split/validation/
Split/test/
Split/ticker_to_id.csv
Split/metadata.json
```

Feature standardisation is fitted on the training period only and then applied to validation and test data.

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

Default checkpoints are saved as:

```text
model/stock_cnn.pt
model/stock_resnet.pt
model/stock_vit.pt
```

You can change parameters such as epochs, batch size, and learning rate:

```bash
python ResNet.py --epochs 20 --batch-size 128 --learning-rate 0.0005
```

In PyCharm, set these parameters in:

```text
Run > Edit Configurations > Parameters
```

### 4. Run the Ensemble

Evaluate the trained models and calculate ensemble weights:

```bash
python main.py --mode evaluate
```

Generate latest predictions:

```bash
python main.py
```

The default mode is `latest`, so `python main.py` loads the saved model checkpoints and predicts from the latest processed rows in `data/`.

Run evaluation and latest prediction together:

```bash
python main.py --mode all
```

## Ensemble Method

The ensemble uses weighted soft voting. Each model outputs a probability, and the final ensemble probability is calculated as:

```text
ensemble_probability = w_cnn * p_cnn + w_resnet * p_resnet + w_vit * p_vit
```

By default, the weights are based on validation F1 scores. The final binary prediction uses a threshold of `0.5`.

## Outputs

Prediction and evaluation files are saved under `predictions/`.

| File | Description |
|---|---|
| `ensemble_latest_predictions.csv` | Latest one-row-per-ticker ensemble predictions |
| `ensemble_validation_predictions.csv` | Validation-period predictions |
| `ensemble_test_predictions.csv` | Test-period predictions |
| `ensemble_metrics.csv` | Loss, accuracy, precision, recall, and F1 score |
| `ensemble_weights.csv` | Saved ensemble model weights |

## Evaluation Metrics

The models are evaluated using:

- loss
- accuracy
- precision
- recall
- F1 score

Chronological splitting is used instead of random row splitting to reduce look-ahead bias in the time-series setting.

## Notes

- The CNN, ResNet, and ViT-inspired models are global models trained across many tickers.
- Each ticker is mapped to a numerical ID and passed through an embedding layer.
- The same `ticker_to_id` mapping must be used during training, evaluation, and inference.
- New tickers that were not seen during training will not have learned ticker embeddings.
- To refresh predictions, update the ticker CSV files in `data/` and rerun `main.py`.

## Disclaimer

This repository is for research and learning purposes only. The outputs should not be interpreted as investment advice, trading recommendations, or guaranteed market signals.
