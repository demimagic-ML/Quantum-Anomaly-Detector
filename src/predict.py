"""
Prediction on Unseen Data
===========================
Loads a trained Quantum Autoencoder and evaluates it on a fresh,
completely unseen batch of transactions from the Kaggle dataset.

The key guarantee: the samples used here were NEVER seen during
training or the original train/test split.
"""

import os
import pickle
import numpy as np
import pandas as pd
from src.model.quantum_autoencoder import QuantumAutoencoder, ALL_QUBITS
from src.data.preprocessing import prepare_quantum_data
from src.evaluate import evaluate


def load_unseen_kaggle_data(
    csv_path: str,
    selected_features: list,
    train_indices: np.ndarray,
    predict_n_normal: int = 1000,
    predict_n_fraud: int = 200,
    predict_seed: int = 999,
):
    """
    Robustly loads unseen data by using Pandas Index Exclusion.
    Guarantees ZERO overlap with training data.

    Parameters
    ----------
    csv_path : str
        Path to creditcard.csv.
    selected_features : list
        Feature names selected during training.
    train_indices : np.ndarray
        Exact row indices used during training (loaded from disk).
    predict_n_normal, predict_n_fraud : int
        How many unseen normal/fraud samples to use for prediction.
    predict_seed : int
        Random seed for sampling the unseen data.

    Returns
    -------
    X : np.ndarray
        Feature array (n_samples, n_features).
    labels : np.ndarray
        Binary labels (0=normal, 1=fraud).
    """
    df = pd.read_csv(csv_path)

    df_unseen_pool = df.drop(index=train_indices)

    print(f"  Total rows: {len(df)}")
    print(f"  Training rows excluded: {len(train_indices)}")
    print(f"  Available unseen pool: {len(df_unseen_pool)}")

    unseen_normal = df_unseen_pool[df_unseen_pool["Class"] == 0]
    unseen_fraud = df_unseen_pool[df_unseen_pool["Class"] == 1]

    n_norm = min(predict_n_normal, len(unseen_normal))
    n_fr = min(predict_n_fraud, len(unseen_fraud))

    print(f"  Unseen available: {len(unseen_normal)} normal, {len(unseen_fraud)} fraud")
    print(f"  Sampling {n_norm} normal + {n_fr} fraud for prediction...")

    final_normal = unseen_normal.sample(n=n_norm, random_state=predict_seed)
    final_fraud = unseen_fraud.sample(n=n_fr, random_state=predict_seed)

    df_final = pd.concat([final_normal, final_fraud])
    df_final = df_final.sample(frac=1, random_state=predict_seed)  # shuffle

    X = df_final[selected_features].values
    labels = df_final["Class"].values.astype(int)

    print(f"  Prediction set: {len(labels)} samples "
          f"(normal: {(labels == 0).sum()}, fraud: {(labels == 1).sum()})")

    return X, labels


def predict_unseen(
    checkpoint: str = "results/model_params.npy",
    results_dir: str = "results",
    csv_path: str = "data/creditcard.csv",
    predict_n_normal: int = 1000,
    predict_n_fraud: int = 200,
    predict_seed: int = 999,
):
    """
    Full prediction pipeline on unseen Kaggle data.

    1. Loads the trained model + scaler + metadata
    2. Samples transactions that were NEVER seen during training
    3. Scores them and produces evaluation metrics + plots
    """
    meta_path = os.path.join(results_dir, "train_meta.pkl")
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    indices_path = os.path.join(results_dir, "train_indices.npy")

    for path in [checkpoint, meta_path, scaler_path, indices_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run training first: python main.py --mode train --data-source kaggle"
            )

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    train_indices = np.load(indices_path)

    print("=" * 60)
    print("PREDICT: Scoring unseen transactions")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Training config: depth={meta['depth']}, seed={meta['seed']}, "
          f"n_normal={meta['n_normal']}, n_fraud={meta['n_fraud']}")

    if meta.get("data_source") != "kaggle":
        raise ValueError(
            "Predict mode requires a model trained on Kaggle data "
            "(--data-source kaggle). Retrain with Kaggle data first."
        )

    selected_features = meta["selected_features"]
    print(f"  Features: {selected_features}")

    model = QuantumAutoencoder(depth=meta["depth"])
    model.load_params(checkpoint)
    print(f"  Model loaded: {model.n_params} var params, "
          f"{model.n_data_params} data params")

    print("\n  Loading unseen data (excluding all training rows by index)...")
    X_unseen, y_unseen = load_unseen_kaggle_data(
        csv_path=csv_path,
        selected_features=selected_features,
        train_indices=train_indices,
        predict_n_normal=predict_n_normal,
        predict_n_fraud=predict_n_fraud,
        predict_seed=predict_seed,
    )

    X_scaled = scaler.transform(X_unseen)
    X_scaled = np.clip(X_scaled, 0.0, np.pi)
    q_unseen = prepare_quantum_data(X_scaled, ALL_QUBITS)

    print(f"  Prepared {len(q_unseen)} quantum circuits for unseen data")

    predict_results_dir = os.path.join(results_dir, "unseen")
    data_bundle = {
        "q_test": q_unseen,
        "y_test": y_unseen,
        "history": None,
    }
    results = evaluate(model, data_bundle, results_dir=predict_results_dir)

    print("\n" + "=" * 60)
    print("PREDICT COMPLETE — Unseen Data Results")
    print("=" * 60)
    print(f"  Samples:   {len(y_unseen)} (normal: {(y_unseen==0).sum()}, "
          f"fraud: {(y_unseen==1).sum()})")
    print(f"  AUROC:     {results['auroc']:.4f}")
    print(f"  F1 Score:  {results['f1']:.4f}")
    print(f"  Threshold: {results['threshold']:.4f}")
    print(f"  Results saved to: {predict_results_dir}/")

    return results
