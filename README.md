# 🏁 F1 Pit Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CatBoost](https://img.shields.io/badge/CatBoost-Enabled-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Style-blueviolet)

A polished Kaggle-style F1 pit-stop prediction repository designed for rapid cloning, training, and submission.

---

## 🚀 Project Overview

This project predicts whether an F1 car will pit on the next lap using telemetry, tyre condition, race progress, and lap performance features.

The core model is built with `CatBoostClassifier`, using GroupKFold cross-validation and threshold tuning to maximize F1 score.

---

## 📦 What’s Included

- `model.py` - main training and inference script
- `train.csv` - training dataset
- `test.csv` - test dataset
- `sample_submission.csv` - Kaggle-format submission example
- `submission.csv` - generated output file after running the model
- `catboost_info/` - CatBoost training metadata and logs
- `requirements/requirements.txt` - dependency list for the project
- `.gitignore` - project cleanup rules for Python and Kaggle artifacts

---

## 🧠 Modeling Highlights

This solution includes:

- Fast data loading and preprocessing
- Numeric downcasting for memory efficiency
- Feature engineering with race-aware derived features
- Group-aware cross-validation by `Race`
- CatBoost training with early stopping and F1 evaluation
- Threshold search to choose the best binary decision cut-off
- Averaged fold predictions for stable test output

### Key derived features

- `Deg_Per_Lap` — degradation rate per lap
- `Pace_Degradation` — current pace impact scaled by tyre life
- `TrafficRisk` — position and lap delta interaction
- `Race_Pressure` — race progress weighted by position

---

## 🛠 Installation

Install the required dependencies from the `requirements` folder:

```bash
pip install -r requirements/requirements.txt
```

If you prefer a direct install, use:

```bash
pip install pandas numpy scikit-learn catboost
```

---

## ▶️ Run the project

From the `F1-Pit` folder:

```bash
python model.py
```

This will train the model, print cross-validation scores, and create `submission.csv`.

---

## 📌 Clone this repository

```bash
git clone https://github.com/MaheshReddy-ML/F1-Pit.git
```

---

## 📈 Notes

- The model logs data shapes, null checks, column types, fold performance, and final prediction distribution.
- Final submission is saved with `id` and `PitNextLap` columns.

---

## 🧹 Repository Hygiene

The `.gitignore` file excludes:

- Python caches and compiled files
- virtual environments
- log files and temporary files
- macOS metadata
- `catboost_info/` cache files
- generated `submission.csv`
