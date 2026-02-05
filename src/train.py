"""
Training Pipeline
==================
Trains the Quantum Autoencoder on normal transactions only.
The model learns to compress normal patterns into the latent register,
causing trash qubits to collapse to |0⟩. Anomalous (fraud) patterns
will fail this compression, yielding high anomaly scores at inference.
"""

import os
import numpy as np

from src.data.generate_data import create_dataset
from src.data.load_kaggle import load_creditcard_data
from src.data.preprocessing import scale_features, prepare_quantum_data, split_data
from src.model.quantum_autoencoder import (
    ALL_QUBITS,
    QuantumAutoencoder,
)


def train(
    n_normal: int = 5000,
    n_fraud: int = 500,
    depth: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.02,
    results_dir: str = "results",
    seed: int = 42,
    verbose: int = 1,
    data_source: str = "synthetic",
    csv_path: str = "data/creditcard.csv",
):
    """
    End-to-end training of the Quantum Autoencoder for fraud detection.

    Parameters
    ----------
    n_normal : int
        Number of normal transactions in synthetic dataset.
    n_fraud : int
        Number of fraud transactions (used only for evaluation).
    depth : int
        Variational ansatz depth.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size.
    learning_rate : float
        Parameter-shift gradient step size.
    results_dir : str
        Directory to save model weights and training history.
    seed : int
        Random seed.
    verbose : int
        Verbosity level.
    data_source : str
        'synthetic' for generated data, 'kaggle' for real credit card fraud.
    csv_path : str
        Path to creditcard.csv (only used when data_source='kaggle').

    Returns
    -------
    model : QuantumAutoencoder
        Trained model.
    history : dict
        Training history with 'loss' key.
    data_bundle : dict
        Dictionary containing test data and labels for evaluation.
    """
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(seed)

    print("=" * 60)
    if data_source == "kaggle":
        print("STEP 1: Loading Kaggle Credit Card Fraud dataset")
        print("=" * 60)
        X, labels, selected_features = load_creditcard_data(
            csv_path=csv_path,
            n_features=len(ALL_QUBITS),
            max_normal=n_normal,
            max_fraud=n_fraud,
            seed=seed,
        )
    else:
        print("STEP 1: Generating synthetic transaction data")
        print("=" * 60)
        df, labels = create_dataset(n_normal=n_normal, n_fraud=n_fraud, seed=seed)
        X = df.drop(columns=["label"]).values

    print(f"  Total samples: {len(labels)}")
    print(f"  Normal: {(labels == 0).sum()}, Fraud: {(labels == 1).sum()}")

    print("\nSTEP 2: Splitting data (train=normal only, test=normal+fraud)")

    X_train, X_test, y_train, y_test = split_data(X, labels, seed=seed)
    print(f"  Train: {len(y_train)} (all normal)")
    print(f"  Test:  {len(y_test)} (normal: {(y_test == 0).sum()}, fraud: {(y_test == 1).sum()})")

    print("\nSTEP 3: Scaling features → [0, π] for angle encoding")

    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.clip(X_test_scaled, 0.0, np.pi)

    print("\nSTEP 4: Converting to quantum state-preparation circuits")

    qubits = ALL_QUBITS
    q_train = prepare_quantum_data(X_train_scaled, qubits)
    q_test = prepare_quantum_data(X_test_scaled, qubits)

    print(f"  Train circuits: {len(q_train)}")
    print(f"  Test circuits:  {len(q_test)}")

    print("\nSTEP 5: Building Quantum Autoencoder model")

    model = QuantumAutoencoder(depth=depth)
    print(f"  Ansatz depth: {depth}")
    print(f"  Trainable parameters: {model.n_params}")
    print(f"  Trash qubits: {len(model.readouts)}")

    print("\n" + "=" * 60)
    print("STEP 6: Training the Quantum Autoencoder")
    print("=" * 60)

    history = model.fit(
        q_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=10,
        verbose=verbose,
    )

    params_path = os.path.join(results_dir, "model_params.npy")
    model.save_params(params_path)
    print(f"\nModel parameters saved to {params_path}")

    data_bundle = {
        "q_test": q_test,
        "y_test": y_test,
        "X_test": X_test,
        "scaler": scaler,
    }

    return model, history, data_bundle
