"""
Synthetic Financial Transaction Dataset Generator
==================================================
Generates realistic transaction data with injected fraud patterns.
Normal transactions follow learned multivariate distributions;
fraudulent transactions exhibit anomalous correlations.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_normal_transactions(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate normal (legitimate) transaction feature vectors.

    Features (4-dimensional):
        0 - transaction_amount:  Log-normal, mean ~$50
        1 - time_since_last_txn: Exponential, mean ~2 hours
        2 - distance_from_home:  Gamma-distributed, mean ~10 km
        3 - merchant_risk_score: Beta-distributed, skewed low (safe merchants)

    Parameters
    ----------
    n_samples : int
        Number of normal transactions to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 4) with raw feature values.
    """
    rng = np.random.RandomState(seed)

    amount = rng.lognormal(mean=3.9, sigma=0.8, size=n_samples)
    time_gap = rng.exponential(scale=2.0, size=n_samples)
    distance = rng.gamma(shape=2.0, scale=5.0, size=n_samples)
    merchant_risk = rng.beta(a=2.0, b=8.0, size=n_samples)

    return np.column_stack([amount, time_gap, distance, merchant_risk])


def generate_fraud_transactions(n_samples: int, seed: int = 99) -> np.ndarray:
    """
    Generate fraudulent transaction feature vectors.

    Fraud signatures:
        - Unusually high transaction amounts
        - Very short time gaps (rapid successive transactions)
        - Large distance from cardholder's home
        - High merchant risk scores

    Parameters
    ----------
    n_samples : int
        Number of fraudulent transactions to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 4) with raw feature values.
    """
    rng = np.random.RandomState(seed)

    amount = rng.lognormal(mean=5.5, sigma=1.0, size=n_samples)
    time_gap = rng.exponential(scale=0.3, size=n_samples)
    distance = rng.gamma(shape=5.0, scale=20.0, size=n_samples)
    merchant_risk = rng.beta(a=7.0, b=2.0, size=n_samples)

    return np.column_stack([amount, time_gap, distance, merchant_risk])


def create_dataset(
    n_normal: int = 5000,
    n_fraud: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create a labeled transaction dataset.

    Parameters
    ----------
    n_normal : int
        Number of normal transactions.
    n_fraud : int
        Number of fraudulent transactions.
    seed : int
        Random seed.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns [transaction_amount, time_since_last_txn,
        distance_from_home, merchant_risk_score, label].
    labels : np.ndarray
        Binary labels (0 = normal, 1 = fraud).
    """
    normal = generate_normal_transactions(n_normal, seed=seed)
    fraud = generate_fraud_transactions(n_fraud, seed=seed + 1)

    features = np.vstack([normal, fraud])
    labels = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_fraud, dtype=int),
    ])

    # Shuffle
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(labels))
    features = features[idx]
    labels = labels[idx]

    df = pd.DataFrame(features, columns=[
        "transaction_amount",
        "time_since_last_txn",
        "distance_from_home",
        "merchant_risk_score",
    ])
    df["label"] = labels

    return df, labels


if __name__ == "__main__":
    df, labels = create_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Normal: {(labels == 0).sum()}, Fraud: {(labels == 1).sum()}")
    print(df.describe())
