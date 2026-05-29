# Mastercard B2B Transaction Classifier

## Overview

This project is a machine learning pipeline for detecting business-oriented transaction behavior using банковские транзакционные данные.

The system aggregates card transaction activity and trains a binary classifier that distinguishes:

* business/commercial transaction patterns
* regular consumer transaction patterns

The project is designed for large-scale parquet datasets and optimized for memory-efficient processing using Polars.

---

# Project Structure

```text
mastercard_ml/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── business_cards_MDQ.parquet
│   │   ├── consumer_cards_MDQ.parquet
│   │   └── merchants_reference.parquet
│   └── processed/
├── logs/
├── models/
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── features.py
│   ├── train.py
│   └── explain.py
├── main.py
├── requirements.txt
├── check_model.py
└── README.md
```

---

# Main Components

## Feature Engineering (`src/features.py`)

Responsible for:

* loading parquet datasets
* transaction aggregation
* MCC-based behavioral features
* temporal statistics
* customer-level feature generation

Processing is implemented with Polars for better performance on large datasets.

---

## Model Training (`src/train.py`)

Responsible for:

* train/validation split
* cross-validation
* LightGBM training
* metric calculation
* threshold optimization
* model serialization

Current model type:

* LightGBM Classifier

Main evaluation metric:

* ROC-AUC

---

## Explainability (`src/explain.py`)

Contains utilities for:

* SHAP value generation
* feature importance analysis
* model interpretation

---

## Logging (`src/logger.py`)

Centralized logging configuration for pipeline execution.

Logs are written into the `logs/` directory.

---

# Dataset Requirements

Place source parquet files into:

```text
data/raw/
```

Required files:

```text
business_cards_MDQ.parquet
consumer_cards_MDQ.parquet
```

---

# Installation

Python version:

```text
Python 3.10+
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Configuration

Main configuration file:

```text
configs/config.yaml
```

Contains:

* dataset paths
* model hyperparameters
* feature settings
* training configuration
* output paths

---

# Running the Pipeline

## Run Full Pipeline

```bash
python main.py --step all
```

This command will:

1. generate features
2. train the model
3. evaluate metrics
4. save trained artifacts

---

## Expected Outputs

### Processed Features

```text
data/processed/
```

### Trained Models

```text
models/
```

### Logs

```text
logs/
```

---

# Current Results

Latest cross-validation result:

```text
ROC-AUC: 1.0000
```

The result should be additionally validated for possible feature leakage or deterministic behavioral rules inside the dataset.

---

# Technical Notes

## Why Polars

Polars was selected instead of pandas because:

* lower memory consumption
* faster parquet scanning
* lazy execution support
* streaming aggregations for large datasets

The current pipeline processes datasets with more than 13 million rows on standard hardware.

---

# Possible Improvements

## Hyperparameter Optimization

Integrate Optuna into the training pipeline.

---

## Model Serving

Wrap the model into an API service:

* FastAPI
* gRPC

---

## Monitoring

Add:

* data drift monitoring
* feature distribution tracking
* prediction monitoring

---

# Development Notes

The repository is separated into independent modules to simplify collaboration.

Typical workflow:

* data engineering → `features.py`
* model training → `train.py`
* configuration tuning → `config.yaml`
* deployment integration → `models/`

---

# Example Workflow

```bash
# install dependencies
pip install -r requirements.txt

# run full pipeline
python main.py --step all
```

---

# Dependencies

Main libraries:

* polars
* lightgbm
* scikit-learn
* shap
* joblib
* pyarrow
* pyyaml

---

# Author Notes

This repository contains an experimental behavioral classification pipeline intended for research and internal analytical usage.
