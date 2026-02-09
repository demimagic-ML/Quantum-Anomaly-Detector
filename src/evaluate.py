"""
Evaluation & Visualization
============================
Computes anomaly scores, classification metrics, and generates
publication-quality plots for the Quantum Autoencoder fraud detector.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    classification_report,
    confusion_matrix,
)
from src.model.quantum_autoencoder import QuantumAutoencoder


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """
    Find the anomaly score threshold that maximizes F1 score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0=normal, 1=fraud).
    scores : np.ndarray
        Anomaly scores in [0, 1].

    Returns
    -------
    float
        Optimal threshold.
    """
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 200):
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def evaluate(model: QuantumAutoencoder, data_bundle: dict, results_dir: str = "results"):
    """
    Full evaluation pipeline: scores, metrics, and plots.

    Parameters
    ----------
    model : QuantumAutoencoder
        Trained QAE model.
    data_bundle : dict
        Must contain keys: q_test, y_test.
    results_dir : str
        Directory to save plots and metrics.
    """
    os.makedirs(results_dir, exist_ok=True)

    q_test = data_bundle["q_test"]
    y_test = data_bundle["y_test"]

    print("\n" + "=" * 60)
    print("EVALUATION: Computing anomaly scores")
    print("=" * 60)

    scores = model.compute_anomaly_scores(q_test)

    normal_scores = scores[y_test == 0]
    fraud_scores = scores[y_test == 1]

    print(f"  Normal — mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
    print(f"  Fraud  — mean: {fraud_scores.mean():.4f}, std: {fraud_scores.std():.4f}")

    auroc = roc_auc_score(y_test, scores)
    threshold = find_optimal_threshold(y_test, scores)
    predictions = (scores >= threshold).astype(int)
    f1 = f1_score(y_test, predictions)

    print(f"\n  AUROC:              {auroc:.4f}")
    print(f"  Optimal threshold:  {threshold:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"\n{classification_report(y_test, predictions, target_names=['Normal', 'Fraud'])}")

    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_scores, bins=40, alpha=0.7, label="Normal", color="#2196F3", density=True)
    ax.hist(fraud_scores, bins=40, alpha=0.7, label="Fraud", color="#F44336", density=True)
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Anomaly Score (1 − ⟨Z⟩)")
    ax.set_ylabel("Density")
    ax.set_title("Quantum Autoencoder — Anomaly Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "score_distribution.png"), dpi=150)
    print(f"  Saved: {results_dir}/score_distribution.png")

    fpr, tpr, _ = roc_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, color="#9C27B0", label=f"QAE (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Fraud Detection")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=150)
    print(f"  Saved: {results_dir}/roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, linewidth=2, color="#FF9800")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Fraud Detection")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "precision_recall.png"), dpi=150)
    print(f"  Saved: {results_dir}/precision_recall.png")

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=150)
    print(f"  Saved: {results_dir}/confusion_matrix.png")

    if "history" in data_bundle and data_bundle["history"] is not None:
        loss_data = data_bundle["history"]
        if isinstance(loss_data, dict):
            loss_values = loss_data.get("loss", [])
        else:
            loss_values = loss_data.history.get("loss", [])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_values, linewidth=2, color="#4CAF50")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fidelity Loss")
        ax.set_title("Training Loss — Quantum Autoencoder")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "training_loss.png"), dpi=150)
        print(f"  Saved: {results_dir}/training_loss.png")

    plt.close("all")

    print("\n" + "=" * 60)
    print("RECONSTRUCTION FIDELITY TEST")
    print("=" * 60)
    print("  Protocol: encode → discard trash → reset |0⟩ → apply U† → compare")

    recon_fidelities = model.compute_reconstruction_fidelity(q_test)

    normal_fid = recon_fidelities[y_test == 0]
    fraud_fid = recon_fidelities[y_test == 1]

    print(f"\n  Normal  — mean F: {normal_fid.mean():.4f}, std: {normal_fid.std():.4f}")
    print(f"  Fraud   — mean F: {fraud_fid.mean():.4f}, std: {fraud_fid.std():.4f}")
    print(f"  Fidelity gap: {normal_fid.mean() - fraud_fid.mean():.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_fid, bins=40, alpha=0.7, label="Normal", color="#2196F3", density=True)
    ax.hist(fraud_fid, bins=40, alpha=0.7, label="Fraud", color="#F44336", density=True)
    ax.set_xlabel("Reconstruction Fidelity F(ψ, ψ')")
    ax.set_ylabel("Density")
    ax.set_title("Quantum Reconstruction Fidelity — Acid Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "reconstruction_fidelity.png"), dpi=150)
    print(f"  Saved: {results_dir}/reconstruction_fidelity.png")
    plt.close("all")

    import pandas as pd
    metrics_df = pd.DataFrame([{
        "auroc": auroc,
        "f1_score": f1,
        "threshold": threshold,
        "mean_score_normal": normal_scores.mean(),
        "mean_score_fraud": fraud_scores.mean(),
        "mean_recon_fidelity_normal": float(normal_fid.mean()),
        "mean_recon_fidelity_fraud": float(fraud_fid.mean()),
        "fidelity_gap": float(normal_fid.mean() - fraud_fid.mean()),
        "n_test_normal": int((y_test == 0).sum()),
        "n_test_fraud": int((y_test == 1).sum()),
    }])
    metrics_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved: {metrics_path}")

    return {
        "auroc": auroc,
        "f1": f1,
        "threshold": threshold,
        "scores": scores,
        "predictions": predictions,
        "recon_fidelity_normal": float(normal_fid.mean()),
        "recon_fidelity_fraud": float(fraud_fid.mean()),
    }
