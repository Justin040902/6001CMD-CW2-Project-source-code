"""
eda.py - Exploratory Data Analysis (Chapter 3)
UCI Bank Marketing Dataset (bank-full.csv)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(df: pd.DataFrame, output_dir: str = "outputs") -> None:
    """Perform EDA checks and export summary visualisations."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("CHAPTER 3: Exploratory Data Analysis (EDA)")
    print("=" * 65)

    # 3.1 Descriptive Statistics & Target Distribution
    print("\n--- 3.1 Descriptive Statistics (Numerical) ---")
    print(df.describe().round(2))

    target_counts = df["y"].value_counts()
    target_props = df["y"].value_counts(normalize=True).round(4) * 100
    print("\n--- Target Class Distribution ---")
    for cls in target_counts.index:
        print(f"Class '{cls}': {target_counts[cls]} ({target_props[cls]:.2f}%)")

    # 3.2 Missing Value & 'unknown' Placeholder Analysis
    print("\n--- 3.2 Missing Values (NaN) ---")
    print(df.isnull().sum())

    print("\n--- Placeholder 'unknown' Counts in Categorical Columns ---")
    cat_cols_raw = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols_raw:
        unknown_count = (df[col] == "unknown").sum()
        if unknown_count > 0:
            print(f"{col}: {unknown_count} ({unknown_count / len(df) * 100:.2f}%)")

    # 3.3 Correlation & Feature Relationships
    df_corr = df.copy()
    df_corr["y_num"] = (df_corr["y"] == "yes").astype(int)
    num_cols_raw = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    print("\n--- 3.3 Numerical Feature Correlation with Target ---")
    print(df_corr[num_cols_raw + ["y_num"]].corr()["y_num"].sort_values(ascending=False).round(4))

    print("\n--- Subscription Rate by Previous Campaign Outcome (poutcome) ---")
    print(pd.crosstab(df["poutcome"], df["y"], normalize="index").round(4) * 100)

    print("\n--- Subscription Rate by Month ---")
    print(pd.crosstab(df["month"], df["y"], normalize="index").round(4) * 100)

    # 3.4 Skewness and Outlier Summary (IQR Method)
    print("\n--- 3.4 Skewness of Numerical Features ---")
    print(df[num_cols_raw].skew().round(4))

    print("\n--- Outlier Counts (1.5 * IQR Rule) ---")
    for col in num_cols_raw:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        print(f"{col:12s}: {outliers:5d} ({outliers / len(df) * 100:.2f}%)")

    # Save EDA Figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(data=df, x="y", ax=axes[0], palette="Set2")
    axes[0].set_title("Target Class Distribution")

    sns.histplot(df["age"], bins=30, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Age Distribution")

    sns.boxplot(x=df["balance"], ax=axes[2], color="coral")
    axes[2].set_title("Account Balance Distribution (Outliers)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distributions.png"), dpi=300)
    plt.close()

    # Correlation Heatmap (Numerical Features)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr[num_cols_raw + ["y_num"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Numerical Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=300)
    plt.close()
    print(f"\nSaved EDA plots to '{output_dir}/'.")
