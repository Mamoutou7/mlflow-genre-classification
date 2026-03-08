# Genre Classification

A concise MLOps project for music genre classification using audio features and text metadata.

## Overview

This project builds an end-to-end machine learning pipeline with:

- **MLflow** for orchestration and model tracking
- **Hydra** for configuration
- **Weights & Biases (W&B)** for artifact tracking
- **scikit-learn** for preprocessing and modeling

The model predicts a song's **genre** from numeric, categorical, and text features.

## Pipeline

1. **Download** raw dataset
2. **Preprocess** data and create `text_feature`
3. **Validate** data with tests
4. **Split** into train/test sets
5. **Train** a Random Forest model
6. **Evaluate** on the test set

## Project Structure

```text
genre-classification/
├── config.yaml
├── conda.yml
├── main.py
├── MLproject
├── download/
├── preprocess/
├── check_data/
├── segregate/
├── random_forest/
└── evaluate/
```

## Main Features

- Audio + text feature pipeline
- Stratified train/test split
- Data quality checks with `pytest`
- TF-IDF for text processing
- Random Forest classification
- MLflow model export

## Setup

```bash
conda env create -f conda.yml
conda activate download_data
wandb login
```

Or install dependencies manually:

```bash
pip install mlflow hydra-core wandb pandas scikit-learn scipy matplotlib pyarrow requests omegaconf pytest
```

## Run

From the project root:

```bash
python main.py
```

Depending on the project configuration, you can also run it with MLflow:

```bash
mlflow run .
```

## Output

The pipeline produces:

- tracked datasets in W&B
- trained model artifacts
- evaluation metrics
- confusion matrix plots
- MLflow exported model

## Target

The target column is:

- `genre`
