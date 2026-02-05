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
                                  params: List[sympy.Symbol]) -> cirq.Circuit:
    """Apply Ry(θ) to each qubit."""
    circuit = cirq.Circuit()
    for qubit, param in zip(qubits, params):
        circuit.append(cirq.ry(param)(qubit))
    return circuit


def _entangling_layer(qubits: List[cirq.GridQubit]) -> cirq.Circuit:
    """Linear chain of CNOT gates for nearest-neighbor entanglement."""
    circuit = cirq.Circuit()
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
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
        layer_params = sympy.symbols(
            f"{prefix}_L{layer}_0:{n_qubits}"
        )
        symbols.extend(layer_params)
        circuit += _single_qubit_rotation_layer(qubits, layer_params)
        circuit += _entangling_layer(qubits)

    # Final rotation layer for added expressibility
    final_params = sympy.symbols(f"{prefix}_final_0:{n_qubits}")
    symbols.extend(final_params)
    circuit += _single_qubit_rotation_layer(qubits, final_params)

    return circuit, symbols


def create_quantum_autoencoder_circuit(
    depth: int = 3,
) -> Tuple[cirq.Circuit, List[sympy.Symbol], List[cirq.ops.PauliString]]:
    """
    Construct the full Quantum Autoencoder circuit.

    The circuit consists of:
        1. A variational ansatz acting on all qubits
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
        Trainable parameters.
    readouts : list of cirq.PauliString
        Z-observables on each trash qubit.
    """
    ansatz, symbols = create_variational_ansatz(ALL_QUBITS, depth=depth)

    # Observables: measure ⟨Z⟩ on each trash qubit
    readouts = [cirq.Z(q) for q in TRASH_QUBITS]

    return ansatz, symbols, readouts


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
        self.ansatz, self.symbols, self.readouts = \
            create_quantum_autoencoder_circuit(depth=depth)
        self.n_params = len(self.symbols)
        self.params = np.random.uniform(0, 2 * np.pi, size=self.n_params)
        self.history: Dict[str, list] = {"loss": []}

    def _resolve_circuit(self, prep_circuit: cirq.Circuit,
                         param_values: np.ndarray) -> cirq.Circuit:
        """Combine state-preparation circuit with resolved ansatz."""
        resolver = cirq.ParamResolver(
            {s: v for s, v in zip(self.symbols, param_values)}
        )
        resolved_ansatz = cirq.resolve_parameters(self.ansatz, resolver)
        return prep_circuit + resolved_ansatz

    def _evaluate_trash_expectation(
        self,
        prep_circuits: List[cirq.Circuit],
        param_values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute ⟨Z⟩ for each trash qubit across a batch of circuits.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_trash), expectation values in [-1, +1].
        """
        resolver = cirq.ParamResolver(
            {s: v for s, v in zip(self.symbols, param_values)}
        )
        resolved_ansatz = cirq.resolve_parameters(self.ansatz, resolver)

        results = np.zeros((len(prep_circuits), NUM_TRASH_QUBITS))
        for i, prep in enumerate(prep_circuits):
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
    ) -> float:
        """
        Fidelity-based loss: L = 1 - mean(⟨Z⟩_trash).

        Minimized when all trash qubits are in |0⟩ for normal data.
        """
        expectations = self._evaluate_trash_expectation(prep_circuits, param_values)
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

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                batch_idx = idx[start:start + batch_size]
                batch = [prep_circuits[i] for i in batch_idx]

                # Parameter-shift gradient
                grads = self._parameter_shift_gradient(batch)
                self.params -= learning_rate * grads

                batch_loss = self.fidelity_loss(self.params, batch)
                epoch_losses.append(batch_loss)

            epoch_loss = np.mean(epoch_losses)
            self.history["loss"].append(epoch_loss)

            if verbose:
                print(f"  Epoch {epoch + 1:3d}/{epochs} — loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_params = self.params.copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self.params = best_params
        return self.history

    def _parameter_shift_gradient(
        self,
        prep_circuits: List[cirq.Circuit],
        shift: float = np.pi / 2,
    ) -> np.ndarray:
        """
        Compute gradients via the parameter-shift rule.

        ∂L/∂θ_k = [L(θ_k + π/2) - L(θ_k - π/2)] / 2
        """
        grads = np.zeros(self.n_params)
        for k in range(self.n_params):
            params_plus = self.params.copy()
            params_minus = self.params.copy()
            params_plus[k] += shift
            params_minus[k] -= shift

            loss_plus = self.fidelity_loss(params_plus, prep_circuits)
            loss_minus = self.fidelity_loss(params_minus, prep_circuits)
            grads[k] = (loss_plus - loss_minus) / 2.0
        return grads

    def predict(self, prep_circuits: List[cirq.Circuit]) -> np.ndarray:
        """
        Compute ⟨Z⟩ on trash qubits for each input circuit.

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_trash).
        """
        return self._evaluate_trash_expectation(prep_circuits, self.params)

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
        """Save trained parameters to a .npy file."""
        np.save(path, self.params)

    def load_params(self, path: str):
        """Load trained parameters from a .npy file."""
        self.params = np.load(path)
