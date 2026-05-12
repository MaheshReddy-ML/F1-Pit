"""
Formula 1 Pit Stop Prediction
Kaggle Playground Series - Season 6 Episode 5

Objective
---------
Predict whether a Formula 1 driver will pit on the next lap.

Pipeline Overview
-----------------
- Data preprocessing
- Memory optimization
- Feature engineering
- Group-aware cross validation
- CatBoost classification
- Threshold optimization
- Submission generation

Validation Strategy
-------------------
GroupKFold grouped by Race to avoid race-level leakage.

Author
------
Mahesh
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier


"""
Load Data
---------

Training data contains:
- race telemetry
- tyre information
- degradation metrics
- positional information

Test data contains the same feature structure
without target labels.
"""

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)


"""
Rename Columns
--------------

Renaming improves:
- readability
- cleaner feature handling
- code consistency
"""

train_df.rename(
    columns={"LapTime (s)": "LapTime_s"},
    inplace=True
)

test_df.rename(
    columns={"LapTime (s)": "LapTime_s"},
    inplace=True
)


"""
Target Separation
-----------------

PitNextLap:
1 -> Driver pits next lap
0 -> Driver does not pit next lap
"""

TARGET = "PitNextLap"

X = train_df.drop(columns=[TARGET, "id"])
y = train_df[TARGET]

X_test = test_df.drop(columns=["id"])


"""
Column Type Identification
--------------------------

CatBoost handles categorical features natively,
making it highly effective for mixed tabular data.
"""

cat_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)


"""
Memory Optimization
-------------------

Dataset size is relatively large (~400k rows).

Downcasting helps:
- reduce RAM usage
- speed up training
- improve notebook efficiency

Care must be taken to avoid precision loss.
"""

for col in num_cols:

    if X[col].dtype == "int64":

        X[col] = pd.to_numeric(
            X[col],
            downcast="integer"
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            downcast="integer"
        )

    elif X[col].dtype == "float64":

        X[col] = pd.to_numeric(
            X[col],
            downcast="float"
        )

        X_test[col] = pd.to_numeric(
            X_test[col],
            downcast="float"
        )


"""
Feature Engineering
-------------------

Goal:
Capture race strategy behavior and tyre degradation dynamics.

Pit stop decisions are highly nonlinear and depend on:
- tyre wear
- pace collapse
- race progression
- traffic conditions

These engineered features attempt to model
those racing conditions.
"""


"""
Degradation Per Lap
-------------------

Normalized tyre degradation relative to tyre life.

Higher values may indicate tyres approaching
unstable performance states.
"""

X["Deg_Per_Lap"] = (
    X["Cumulative_Degradation"] /
    (X["TyreLife"] + 1)
)

X_test["Deg_Per_Lap"] = (
    X_test["Cumulative_Degradation"] /
    (X_test["TyreLife"] + 1)
)


"""
Pace Degradation
----------------

Models how tyre wear impacts lap pace.

Drivers experiencing increasing pace loss
are more likely to pit.
"""

X["Pace_Degradation"] = (
    X["LapTime_Delta"] *
    X["TyreLife"]
)

X_test["Pace_Degradation"] = (
    X_test["LapTime_Delta"] *
    X_test["TyreLife"]
)


"""
Traffic Risk
------------

Attempts to capture how track position
and pace interact under traffic pressure.
"""

X["TrafficRisk"] = (
    X["Position"] *
    X["LapTime_Delta"]
)

X_test["TrafficRisk"] = (
    X_test["Position"] *
    X_test["LapTime_Delta"]
)


"""
Race Pressure
-------------

Models late-race strategic pressure.

Drivers under positional stress later in races
may adopt alternative pit strategies.
"""

X["Race_Pressure"] = (
    X["RaceProgress"] *
    X["Position"]
)

X_test["Race_Pressure"] = (
    X_test["RaceProgress"] *
    X_test["Position"]
)


"""
Data Validation Checks
----------------------

Basic sanity checks before model training.
"""

print("\nTrain Nulls:", X.isnull().sum().sum())
print("Test Nulls:", X_test.isnull().sum().sum())

print("\nProcessed Train Shape:", X.shape)
print("Processed Test Shape:", X_test.shape)

print("\nTarget Distribution:")
print(y.value_counts(normalize=True))


"""
Group-Aware Cross Validation
----------------------------

Race telemetry contains strong intra-race correlations.

Using standard random train-validation splits can
introduce race-level leakage because laps from the
same race may appear in both datasets.

GroupKFold grouped by Race prevents leakage and
produces more realistic validation estimates.
"""

groups = train_df["Race"]

gkf = GroupKFold(n_splits=5)


"""
Storage Initialization
----------------------

oof_probs:
Out-of-fold validation probabilities.

test_probs:
Averaged test predictions across folds.
"""

oof_probs = np.zeros(len(X))
test_probs = np.zeros(len(X_test))

scores = []
best_thresholds = []


"""
Training Loop
-------------

Each fold:
- trains on unseen races
- validates on completely different races
- tunes threshold for best F1 performance
"""

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(X, y, groups)
):

    print(f"\n========== Fold {fold+1} ==========")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]


    """
    CatBoost Configuration
    ----------------------

    CatBoost chosen because:
    - excellent tabular performance
    - native categorical handling
    - robust boosting behavior
    - minimal preprocessing requirements
    """

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=10,
        eval_metric="TotalF1",
        loss_function="Logloss",
        random_seed=42,
        verbose=200
    )


    """
    Model Training
    --------------

    Early stopping prevents unnecessary boosting
    rounds once validation performance stops improving.
    """

    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=300
    )


    """
    Validation Probabilities
    ------------------------

    CatBoost outputs probabilities which are later
    converted into binary predictions using
    threshold optimization.
    """

    val_probs = model.predict_proba(X_val)[:, 1]

    oof_probs[val_idx] = val_probs


    """
    Threshold Optimization
    ----------------------

    Optimal F1 threshold is rarely 0.5.

    We search for the threshold producing
    the best fold-wise F1 score.
    """

    best_threshold = 0.5
    best_score = 0

    thresholds = np.arange(0.30, 0.70, 0.01)

    for threshold in thresholds:

        preds = (
            val_probs > threshold
        ).astype(int)

        score = f1_score(y_val, preds)

        if score > best_score:

            best_score = score
            best_threshold = threshold

    print(f"Best Threshold: {best_threshold}")
    print(f"Fold F1 Score: {best_score}")

    scores.append(best_score)
    best_thresholds.append(best_threshold)


    """
    Test Predictions
    ----------------

    Fold predictions are averaged to:
    - improve robustness
    - reduce variance
    - stabilize final predictions
    """

    fold_test_probs = model.predict_proba(
        X_test
    )[:, 1]

    test_probs += (
        fold_test_probs / gkf.n_splits
    )


"""
Final Validation Score
----------------------

Mean F1 across all folds.
"""

print("\n========================")
print("Mean F1:", np.mean(scores))


"""
Final Threshold
---------------

Average threshold across folds used for
final prediction generation.
"""

final_threshold = np.mean(best_thresholds)

print("Final Threshold:", final_threshold)


"""
Final Predictions
-----------------
"""

final_preds = (
    test_probs > final_threshold
).astype(int)


"""
Submission File
---------------

Generate Kaggle-compatible submission file.
"""

submission = pd.DataFrame({
    "id": test_df["id"],
    "PitNextLap": final_preds
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nSubmission file created!")


"""
Prediction Distribution
-----------------------

Useful sanity check to ensure predictions
are not heavily collapsed toward one class.
"""

print("\nPrediction Distribution:")

print(
    submission["PitNextLap"]
    .value_counts(normalize=True)
)