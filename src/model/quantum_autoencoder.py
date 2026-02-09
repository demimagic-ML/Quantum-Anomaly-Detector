"""
Quantum Autoencoder Model
==========================
Implements a Hybrid Quantum-Classical Autoencoder using Cirq + SciPy.

Architecture:
    - 4 total qubits: 2 latent + 2 trash
    - Hardware-efficient ansatz with Ry rotations and CNOT entanglement
    - Trash qubits are measured; cost = 1 - ⟨Z⟩_trash (fidelity proxy)
    - Classical optimizer (L-BFGS-B / COBYLA) updates circuit parameters
"""

import os
import cirq
import sympy
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional


# ---------------------------------------------------------------------------
# Qubit layout
# ---------------------------------------------------------------------------
NUM_DATA_QUBITS = 4       # Total qubits (data register)
NUM_LATENT_QUBITS = 2     # Qubits that retain compressed info
NUM_TRASH_QUBITS = NUM_DATA_QUBITS - NUM_LATENT_QUBITS

ALL_QUBITS = cirq.GridQubit.rect(1, NUM_DATA_QUBITS)
TRASH_QUBITS = ALL_QUBITS[:NUM_TRASH_QUBITS]   # q0, q1 → trash
LATENT_QUBITS = ALL_QUBITS[NUM_TRASH_QUBITS:]  # q2, q3 → latent

# Simulator (statevector — exact, no sampling noise)
SIMULATOR = cirq.Simulator()


def _single_qubit_rotation_layer(qubits: List[cirq.GridQubit],
                                  params_ry: List[sympy.Symbol],
                                  params_rz: List[sympy.Symbol] = None) -> cirq.Circuit:
    """Apply Ry(θ) and optionally Rz(φ) to each qubit."""
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(params_ry[i])(qubit))
        if params_rz is not None:
            circuit.append(cirq.rz(params_rz[i])(qubit))
    return circuit


def _entangling_layer(qubits: List[cirq.GridQubit]) -> cirq.Circuit:
    """Circular CNOT entanglement (includes wrap-around)."""
    circuit = cirq.Circuit()
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    if len(qubits) > 2:
        circuit.append(cirq.CNOT(qubits[-1], qubits[0]))
    return circuit


def create_variational_ansatz(
    qubits: List[cirq.GridQubit],
    depth: int = 3,
    prefix: str = "theta",
) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
    """
    Build a hardware-efficient variational ansatz.

    Structure per layer:
        1. Ry(θ) on every qubit
        2. Linear CNOT entanglement chain

    Parameters
    ----------
    qubits : list of cirq.GridQubit
        Qubits the ansatz acts on.
    depth : int
        Number of variational layers.
    prefix : str
        Symbol name prefix.

    Returns
    -------
    circuit : cirq.Circuit
        The variational ansatz circuit.
    symbols : list of sympy.Symbol
        Trainable parameters.
    """
    n_qubits = len(qubits)
    symbols = []
    circuit = cirq.Circuit()

    for layer in range(depth):
        ry_params = sympy.symbols(
            f"{prefix}_Ry_L{layer}_0:{n_qubits}"
        )
        rz_params = sympy.symbols(
            f"{prefix}_Rz_L{layer}_0:{n_qubits}"
        )
        symbols.extend(ry_params)
        symbols.extend(rz_params)
        circuit += _single_qubit_rotation_layer(qubits, ry_params, rz_params)
        circuit += _entangling_layer(qubits)

    final_ry = sympy.symbols(f"{prefix}_Ry_final_0:{n_qubits}")
    final_rz = sympy.symbols(f"{prefix}_Rz_final_0:{n_qubits}")
    symbols.extend(final_ry)
    symbols.extend(final_rz)
    circuit += _single_qubit_rotation_layer(qubits, final_ry, final_rz)

    return circuit, symbols


def create_data_reupload_ansatz(
    qubits: List[cirq.GridQubit],
    depth: int = 3,
    prefix: str = "theta",
) -> Tuple[cirq.Circuit, List[sympy.Symbol], List[sympy.Symbol]]:
    """
    Build an ansatz with data re-uploading.

    Structure per layer:
        1. Data encoding: Rx(w_i * x_i) on each qubit (w_i are trainable scaling params)
        2. Variational: Ry(θ) + Rz(φ) on each qubit
        3. Circular CNOT entanglement

    The data encoding symbols are separate from the variational symbols
    so that data values can be injected at runtime.

    Returns
    -------
    circuit, var_symbols, data_symbols
    """
    n_qubits = len(qubits)
    var_symbols = []
    data_symbols = []
    circuit = cirq.Circuit()

    for layer in range(depth):
        layer_data = sympy.symbols(f"data_L{layer}_0:{n_qubits}")
        data_symbols.extend(layer_data)
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(layer_data[i])(qubit))

        ry_params = sympy.symbols(f"{prefix}_Ry_L{layer}_0:{n_qubits}")
        rz_params = sympy.symbols(f"{prefix}_Rz_L{layer}_0:{n_qubits}")
        var_symbols.extend(ry_params)
        var_symbols.extend(rz_params)
        circuit += _single_qubit_rotation_layer(qubits, ry_params, rz_params)
        circuit += _entangling_layer(qubits)

    final_ry = sympy.symbols(f"{prefix}_Ry_final_0:{n_qubits}")
    final_rz = sympy.symbols(f"{prefix}_Rz_final_0:{n_qubits}")
    var_symbols.extend(final_ry)
    var_symbols.extend(final_rz)
    circuit += _single_qubit_rotation_layer(qubits, final_ry, final_rz)

    return circuit, var_symbols, data_symbols


def create_quantum_autoencoder_circuit(
    depth: int = 3,
) -> Tuple[cirq.Circuit, List[sympy.Symbol], List[cirq.ops.PauliString]]:
    """
    Construct the full Quantum Autoencoder circuit with data re-uploading.

    The circuit consists of:
        1. Interleaved data encoding + variational layers
        2. Measurements of the trash qubits in the Z basis

    The cost is minimized when trash qubits are in |0⟩ (⟨Z⟩ = +1),
    meaning information is fully compressed into the latent register.

    Parameters
    ----------
    depth : int
        Depth of the variational ansatz.

    Returns
    -------
    circuit : cirq.Circuit
        Parameterized autoencoder circuit (without measurements).
    symbols : list of sympy.Symbol
        All symbols (variational + data re-upload).
    readouts : list of cirq.PauliString
        Z-observables on each trash qubit.
    """
    ansatz, var_symbols, data_symbols = create_data_reupload_ansatz(
        ALL_QUBITS, depth=depth
    )

    # Observables: measure ⟨Z⟩ on each trash qubit
    readouts = [cirq.Z(q) for q in TRASH_QUBITS]

    all_symbols = var_symbols + data_symbols
    return ansatz, all_symbols, readouts, var_symbols, data_symbols


# ---------------------------------------------------------------------------
# QuantumAutoencoder class — wraps circuit + simulator + optimizer
# ---------------------------------------------------------------------------
class QuantumAutoencoder:
    """
    Hybrid Quantum-Classical Autoencoder using cirq.Simulator.

    Trains a PQC to compress normal transaction data into latent qubits
    by minimizing the fidelity loss on the trash register.

    Parameters
    ----------
    depth : int
        Number of variational ansatz layers.
    """

    def __init__(self, depth: int = 3):
        self.depth = depth
        self.ansatz, self.all_symbols, self.readouts, self.var_symbols, self.data_symbols = \
            create_quantum_autoencoder_circuit(depth=depth)
        self.symbols = self.all_symbols  # for backward compat
        self.n_var_params = len(self.var_symbols)
        self.n_data_params = len(self.data_symbols)
        self.n_params = self.n_var_params  # only variational params are trainable
        self.params = np.random.uniform(0, 2 * np.pi, size=self.n_params)
        # Trainable scaling weights for data re-uploading
        self.data_weights = np.ones(self.n_data_params)
        self.n_total_trainable = self.n_params + self.n_data_params
        self.history: Dict[str, list] = {"loss": []}
        self._adam_m = np.zeros(self.n_total_trainable)
        self._adam_v = np.zeros(self.n_total_trainable)
        self._adam_t = 0

    def _get_all_param_values(self, features: np.ndarray = None) -> np.ndarray:
        """Combine variational params + data re-upload values."""
        if features is not None:
            # Tile features for each re-upload layer
            n_layers = self.n_data_params // NUM_DATA_QUBITS
            tiled = np.tile(features, n_layers)
            data_vals = self.data_weights * tiled
        else:
            data_vals = np.zeros(self.n_data_params)
        return np.concatenate([self.params, data_vals])

    def _resolve_circuit(self, prep_circuit: cirq.Circuit,
                         param_values: np.ndarray) -> cirq.Circuit:
        """Combine state-preparation circuit with resolved ansatz."""
        resolver = cirq.ParamResolver(
            {s: v for s, v in zip(self.all_symbols, param_values)}
        )
        resolved_ansatz = cirq.resolve_parameters(self.ansatz, resolver)
        return prep_circuit + resolved_ansatz

    def _evaluate_trash_expectation(
        self,
        prep_circuits: List[cirq.Circuit],
        param_values: np.ndarray,
        data_weights: np.ndarray = None,
        features_list: List[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ⟨Z⟩ for each trash qubit across a batch of circuits.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_trash), expectation values in [-1, +1].
        """
        if data_weights is None:
            data_weights = self.data_weights

        results = np.zeros((len(prep_circuits), NUM_TRASH_QUBITS))
        for i, prep in enumerate(prep_circuits):
            if features_list is not None:
                feat = features_list[i]
            else:
                feat = np.zeros(NUM_DATA_QUBITS)
            n_layers = self.n_data_params // NUM_DATA_QUBITS
            tiled = np.tile(feat, n_layers)
            data_vals = data_weights * tiled
            all_vals = np.concatenate([param_values, data_vals])

            resolver = cirq.ParamResolver(
                {s: v for s, v in zip(self.all_symbols, all_vals)}
            )
            resolved_ansatz = cirq.resolve_parameters(self.ansatz, resolver)
            full_circuit = prep + resolved_ansatz

            for j, readout in enumerate(self.readouts):
                exp_val = SIMULATOR.simulate_expectation_values(
                    full_circuit, observables=[readout]
                )
                results[i, j] = exp_val[0].real
        return results

    def fidelity_loss(
        self,
        param_values: np.ndarray,
        prep_circuits: List[cirq.Circuit],
        data_weights: np.ndarray = None,
        features_list: List[np.ndarray] = None,
    ) -> float:
        """
        Fidelity-based loss: L = 1 - mean(⟨Z⟩_trash).

        Minimized when all trash qubits are in |0⟩ for normal data.
        """
        expectations = self._evaluate_trash_expectation(
            prep_circuits, param_values, data_weights, features_list
        )
        mean_z = np.mean(expectations)
        return 1.0 - mean_z

    def fit(
        self,
        prep_circuits: List[cirq.Circuit],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.02,
        patience: int = 10,
        verbose: int = 1,
    ) -> Dict[str, list]:
        """
        Train the autoencoder using mini-batch gradient descent
        with parameter-shift rule.

        Parameters
        ----------
        prep_circuits : list of cirq.Circuit
            State-preparation circuits for normal training data.
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.
        learning_rate : float
            Step size for parameter updates.
        patience : int
            Early stopping patience (epochs without improvement).
        verbose : int
            Print progress every epoch if > 0.

        Returns
        -------
        dict
            Training history with 'loss' key.
        """
        n_samples = len(prep_circuits)
        best_loss = float("inf")
        wait = 0
        best_params = self.params.copy()
        best_data_weights = self.data_weights.copy()

        features_list = self._extract_features(prep_circuits)

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch = [prep_circuits[i] for i in batch_idx]
                batch_features = [features_list[i] for i in batch_idx]

                grads = self._parameter_shift_gradient(
                    batch, features_list=batch_features
                )
                self._adam_update(grads, learning_rate)

                batch_loss = self.fidelity_loss(
                    self.params, batch,
                    self.data_weights, batch_features
                )
                epoch_losses.append(batch_loss)

            epoch_loss = np.mean(epoch_losses)
            self.history["loss"].append(epoch_loss)

            if verbose:
                print(f"  Epoch {epoch + 1:3d}/{epochs} — loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_params = self.params.copy()
                best_data_weights = self.data_weights.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self.params = best_params
        self.data_weights = best_data_weights
        return self.history

    def _extract_features(self, prep_circuits: List[cirq.Circuit]) -> List[np.ndarray]:
        """
        Extract encoded feature values from state-preparation circuits.
        Each prep circuit encodes features as Ry(x_i) rotations.
        """
        features_list = []
        for circuit in prep_circuits:
            features = np.zeros(NUM_DATA_QUBITS)
            for moment in circuit.moments:
                for op in moment.operations:
                    if isinstance(op.gate, cirq.ops.common_gates.YPowGate):
                        qubit_idx = ALL_QUBITS.index(op.qubits[0])
                        features[qubit_idx] = float(op.gate.exponent) * np.pi
                    elif hasattr(op.gate, '_exponent'):
                        qubit_idx = ALL_QUBITS.index(op.qubits[0])
                        features[qubit_idx] = float(op.gate._exponent) * np.pi
            features_list.append(features)
        return features_list

    def _adam_update(
        self,
        grads: np.ndarray,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        """Apply Adam optimizer update to all trainable parameters."""
        self._adam_t += 1
        self._adam_m = beta1 * self._adam_m + (1 - beta1) * grads
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * grads ** 2
        m_hat = self._adam_m / (1 - beta1 ** self._adam_t)
        v_hat = self._adam_v / (1 - beta2 ** self._adam_t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)

        self.params -= update[:self.n_params]
        self.data_weights -= update[self.n_params:]

    def _parameter_shift_gradient(
        self,
        prep_circuits: List[cirq.Circuit],
        shift: float = np.pi / 2,
        features_list: List[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute gradients via the parameter-shift rule.

        ∂L/∂θ_k = [L(θ_k + π/2) - L(θ_k - π/2)] / 2

        Computes gradients for both variational params and data weights.
        """
        grads = np.zeros(self.n_total_trainable)

        for k in range(self.n_params):
            params_plus = self.params.copy()
            params_minus = self.params.copy()
            params_plus[k] += shift
            params_minus[k] -= shift

            loss_plus = self.fidelity_loss(
                params_plus, prep_circuits, self.data_weights, features_list
            )
            loss_minus = self.fidelity_loss(
                params_minus, prep_circuits, self.data_weights, features_list
            )
            grads[k] = (loss_plus - loss_minus) / 2.0

        for k in range(self.n_data_params):
            dw_plus = self.data_weights.copy()
            dw_minus = self.data_weights.copy()
            dw_plus[k] += shift
            dw_minus[k] -= shift

            loss_plus = self.fidelity_loss(
                self.params, prep_circuits, dw_plus, features_list
            )
            loss_minus = self.fidelity_loss(
                self.params, prep_circuits, dw_minus, features_list
            )
            grads[self.n_params + k] = (loss_plus - loss_minus) / 2.0

        return grads

    def predict(self, prep_circuits: List[cirq.Circuit]) -> np.ndarray:
        """
        Compute ⟨Z⟩ on trash qubits for each input circuit.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_trash).
        """
        features_list = self._extract_features(prep_circuits)
        return self._evaluate_trash_expectation(
            prep_circuits, self.params, self.data_weights, features_list
        )

    def compute_anomaly_scores(self, prep_circuits: List[cirq.Circuit]) -> np.ndarray:
        """
        Compute anomaly scores for a dataset.

        Score = 1 - mean(⟨Z⟩_trash).
        Normal → low score (~0), Fraud → high score (~1).

        Parameters
        ----------
        prep_circuits : list of cirq.Circuit
            State-preparation circuits.

        Returns
        -------
        np.ndarray
            Anomaly scores in [0, 1], shape (n_samples,).
        """
        expectations = self.predict(prep_circuits)
        mean_z = np.mean(expectations, axis=-1)
        scores = 1.0 - mean_z
        return np.clip(scores, 0.0, 1.0)

    def save_params(self, path: str):
        """Save trained parameters and data weights to .npy files."""
        np.save(path, self.params)
        np.save(path.replace('.npy', '_data_weights.npy'), self.data_weights)

    def load_params(self, path: str):
        """Load trained parameters and data weights from .npy files."""
        self.params = np.load(path)
        dw_path = path.replace('.npy', '_data_weights.npy')
        if os.path.exists(dw_path):
            self.data_weights = np.load(dw_path)
