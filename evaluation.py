"""
evaluation.py - Comparative Model Evaluation & Exports (Chapter 8)
UCI Bank Marketing Dataset (bank-full.csv)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)


def evaluate_and_export(
    fitted_models: dict,
    results: dict,
    X_test_processed,
    y_test,
    y_pred_ens,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Export comparative metrics table, ROC curve comparison, and confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("CHAPTER 8: Comparative Model Evaluation & Exports")
    print("=" * 65)

    results_df = pd.DataFrame(results).T
    print("\nComparative Results Table:")
    print(results_df.round(3))

    # Export comparative results CSV
    csv_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_df.round(4).to_csv(csv_path)
    print(f"\nSaved comparative metrics table to: {csv_path}")

    print("\nClassification Report - Soft Voting Ensemble:")
    print(classification_report(y_test, y_pred_ens, target_names=["No", "Yes"]))

    # Save Combined ROC Curve Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fitted_model in fitted_models.items():
        RocCurveDisplay.from_estimator(fitted_model, X_test_processed, y_test, ax=ax, name=name)

    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    ax.set_title("ROC Curve Comparison Across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_comparison.png"), dpi=300)
    plt.close()

    # Save Confusion Matrices for All 4 Models
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, fitted_model) in zip(axes.flat, fitted_models.items()):
        ConfusionMatrixDisplay.from_estimator(
            fitted_model,
            X_test_processed,
            y_test,
            display_labels=["No", "Yes"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    print(f"Saved ROC curves and confusion matrices to '{output_dir}/'.")
    return results_df
