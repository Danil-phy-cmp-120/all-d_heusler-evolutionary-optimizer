# Random Forest Surrogate Models

This folder contains scripts for training and refining random forest models for DFT-based property prediction.

## 🔍 Overview

Each script implements an **active learning** loop, where a surrogate model (random forest) is trained on known DFT data and used to suggest new candidates for evaluation.

## 📂 Files

- `active_learning_mag_aust.py` – trains a model for magnetic austenite properties
- `active_learning_mag_mart.py` – trains a model for magnetic martensite properties
- `active_learning_tetr.py` – targets tetragonality
- `active_learning_vol.py` – targets volume

## ⚙️ How It Works

1. Load existing dataset (e.g., `df.csv`)
2. Train a random forest model using scikit-learn
3. Select new data points using uncertainty or score-based criteria
4. Iterate to improve model performance and data coverage

## 🚀 Usage

Example:

```bash
python active_learning_tetr.py
```

## 📄 License

See the root `LICENSE` file for licensing information.

