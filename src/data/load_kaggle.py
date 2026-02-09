"""
Kaggle Credit Card Fraud Dataset Loader
=========================================
Loads the real-world credit card fraud dataset from Kaggle (MLG-ULB).
Selects the top-k most discriminative features for quantum encoding.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - 284,807 transactions (492 frauds = 0.17%)
    - 30 features: Time, V1–V28 (PCA), Amount
    - Binary label: 0 = normal, 1 = fraud
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_selection import f_classif


def load_creditcard_data(
    csv_path: str = "data/creditcard.csv",
    n_features: int = 4,
    max_normal: int = None,
    max_fraud: int = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """
    Load and prepare the Kaggle credit card fraud dataset.

    Selects the top `n_features` most discriminative features based on
    the absolute difference in means between normal and fraud classes.

    Parameters
    ----------
    csv_path : str
        Path to the creditcard.csv file.
    n_features : int
        Number of features to select (must match qubit count).
    max_normal : int or None
        Cap on normal samples (None = use all). Useful for faster
        quantum simulation since circuit count drives runtime.
    max_fraud : int or None
        Cap on fraud samples (None = use all).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    X : np.ndarray
        Feature array of shape (n_samples, n_features).
    labels : np.ndarray
        Binary labels (0=normal, 1=fraud).
    selected_features : list of str
        Names of the selected features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            "Download it with:\n"
            "  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip\n"
            "Or visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(csv_path)
    labels_all = df["Class"].values
    features_df = df.drop(columns=["Class"])

    f_scores, p_values = f_classif(features_df.values, labels_all)
    f_score_series = pd.Series(f_scores, index=features_df.columns)
    top_features = f_score_series.sort_values(ascending=False).head(n_features).index.tolist()

    print(f"  Selected features (top {n_features} by ANOVA F-score):")
    for feat in top_features:
        idx = features_df.columns.get_loc(feat)
        print(f"    {feat}: F={f_scores[idx]:.2f}, p={p_values[idx]:.2e}")

    X_all = features_df[top_features].values
    labels_all = labels_all.astype(int)

    rng = np.random.RandomState(seed)

    normal_idx = np.where(labels_all == 0)[0]
    fraud_idx = np.where(labels_all == 1)[0]

    if max_normal is not None and len(normal_idx) > max_normal:
        normal_idx = rng.choice(normal_idx, size=max_normal, replace=False)

    if max_fraud is not None and len(fraud_idx) > max_fraud:
        fraud_idx = rng.choice(fraud_idx, size=max_fraud, replace=False)

    selected_idx = np.concatenate([normal_idx, fraud_idx])
    rng.shuffle(selected_idx)

    X = X_all[selected_idx]
    labels = labels_all[selected_idx]

    print(f"  Dataset: {len(labels)} samples "
          f"(normal: {(labels == 0).sum()}, fraud: {(labels == 1).sum()})")

    return X, labels, top_features, selected_idx
