"""
Quantum Anomaly Detector — Main Entry Point
=============================================
Train a Quantum Autoencoder on synthetic financial data and evaluate
its ability to detect fraudulent transactions via reconstruction fidelity.

Usage:
    python main.py --mode train --epochs 50 --batch-size 32
    python main.py --mode evaluate --checkpoint results/model_params.npy
    python main.py --mode full     # train + evaluate end-to-end
"""

import argparse
import os
import sys
import numpy as np

from src.train import train
from src.evaluate import evaluate
from src.model.quantum_autoencoder import QuantumAutoencoder, ALL_QUBITS
from src.data.generate_data import create_dataset
from src.data.preprocessing import scale_features, prepare_quantum_data, split_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Anomaly Detector for Financial Fraud"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="Run mode: train, evaluate, or full (train + evaluate).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--depth", type=int, default=3, help="Ansatz circuit depth.")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate.")
    parser.add_argument("--n-normal", type=int, default=5000, help="Number of normal transactions.")
    parser.add_argument("--n-fraud", type=int, default=500, help="Number of fraud transactions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/model_params.npy",
        help="Path to model parameters (for evaluate mode).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["synthetic", "kaggle"],
        default="synthetic",
        help="Data source: 'synthetic' or 'kaggle' (real credit card fraud).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/creditcard.csv",
        help="Path to Kaggle creditcard.csv (only used with --data-source kaggle).",
    )
    return parser.parse_args()


def run_evaluate_only(args):
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        print("Run with --mode train first.")
        sys.exit(1)

    model = QuantumAutoencoder(depth=args.depth)
    model.load_params(args.checkpoint)

    df, labels = create_dataset(
        n_normal=args.n_normal, n_fraud=args.n_fraud, seed=args.seed
    )
    X = df.drop(columns=["label"]).values
    X_train, X_test, y_train, y_test = split_data(X, labels, seed=args.seed)
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.clip(X_test_scaled, 0.0, np.pi)
    q_test = prepare_quantum_data(X_test_scaled, ALL_QUBITS)

    data_bundle = {"q_test": q_test, "y_test": y_test, "history": None}
    results = evaluate(model, data_bundle, results_dir=args.results_dir)
    return results


def main():
    args = parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Anomaly Detector — Financial Fraud Detection  ║")
    print("║   Hybrid Quantum-Classical Autoencoder (Cirq + SciPy)   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    if args.mode == "train":
        model, history, data_bundle = train(
            n_normal=args.n_normal,
            n_fraud=args.n_fraud,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            results_dir=args.results_dir,
            seed=args.seed,
            data_source=args.data_source,
            csv_path=args.csv_path,
        )
        print("\nTraining complete. Run with --mode evaluate to generate metrics.")

    elif args.mode == "evaluate":
        run_evaluate_only(args)

    elif args.mode == "full":
        model, history, data_bundle = train(
            n_normal=args.n_normal,
            n_fraud=args.n_fraud,
            depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            results_dir=args.results_dir,
            seed=args.seed,
            data_source=args.data_source,
            csv_path=args.csv_path,
        )
        data_bundle["history"] = history
        results = evaluate(model, data_bundle, results_dir=args.results_dir)

        print("\n" + "=" * 60)
        print("COMPLETE — Summary")
        print("=" * 60)
        print(f"  AUROC:     {results['auroc']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")
        print(f"  Threshold: {results['threshold']:.4f}")
        print(f"  Results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
