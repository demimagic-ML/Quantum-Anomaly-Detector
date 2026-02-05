"""
Data Preprocessing & Quantum State Preparation
================================================
Scales classical features to [0, π] range for angle encoding,
then converts each sample into a Cirq circuit that prepares
the corresponding quantum state.
"""

import numpy as np
import cirq
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


def scale_features(X: np.ndarray, feature_range: Tuple[float, float] = (0.0, np.pi)) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Scale features to [0, π] for angle encoding via Ry rotations.

    Parameters
    ----------
    X : np.ndarray
        Raw feature array of shape (n_samples, n_features).
    feature_range : tuple
        Target range for scaling.

    Returns
    -------
    X_scaled : np.ndarray
        Scaled features in [0, π].
    scaler : MinMaxScaler
        Fitted scaler (save for inference).
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def create_state_preparation_circuit(features: np.ndarray, qubits: List[cirq.GridQubit]) -> cirq.Circuit:
    """
    Create a state preparation circuit using angle encoding.

    Each feature x_i is encoded as Ry(x_i)|0⟩ on the i-th qubit.

    Parameters
    ----------
    features : np.ndarray
        1-D array of scaled feature values, length == len(qubits).
    qubits : list of cirq.GridQubit
        Qubits to encode features onto.

    Returns
    -------
    cirq.Circuit
        State preparation circuit.
    """
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(features[i])(qubit))
    return circuit


def prepare_quantum_data(
    X_scaled: np.ndarray,
    qubits: List[cirq.GridQubit],
) -> List[cirq.Circuit]:
    """
    Convert an entire dataset of scaled features into a list of
    state-preparation circuits.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature array of shape (n_samples, n_features).
    qubits : list of cirq.GridQubit
        Data qubits (len must equal n_features).

    Returns
    -------
    list of cirq.Circuit
        One state-preparation circuit per sample.
    """
    circuits = []
    for sample in X_scaled:
        circuit = create_state_preparation_circuit(sample, qubits)
        circuits.append(circuit)
    return circuits


def split_data(
    X: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets, stratified.

    For the autoencoder, training uses ONLY normal samples.
    Test set contains both normal and fraud for evaluation.

    Parameters
    ----------
    X : np.ndarray
        Feature array (n_samples, n_features).
    labels : np.ndarray
        Binary labels (0=normal, 1=fraud).
    train_ratio : float
        Fraction of normal samples used for training.
    seed : int
        Random seed.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    rng = np.random.RandomState(seed)

    normal_idx = np.where(labels == 0)[0]
    fraud_idx = np.where(labels == 1)[0]

    rng.shuffle(normal_idx)
    split = int(len(normal_idx) * train_ratio)

    train_idx = normal_idx[:split]
    test_idx = np.concatenate([normal_idx[split:], fraud_idx])
    rng.shuffle(test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    return X_train, X_test, y_train, y_test
