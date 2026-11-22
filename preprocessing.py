"""
preprocessing.py - Data Preparation & Preprocessing Pipeline (Chapter 4)
UCI Bank Marketing Dataset (bank-full.csv)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def prepare_data(df: pd.DataFrame):
    """Remove leaky feature, perform stratified split, and build ColumnTransformer."""
    print("\n" + "=" * 65)
    print("CHAPTER 4: Data Preparation & Preprocessing Pipeline")
    print("=" * 65)

    # 4.1.1 Feature Removal: Drop 'duration' to avoid data leakage
    df_clean = df.drop(columns=["duration"]).copy()

    # Binary target encoding
    y = (df_clean["y"] == "yes").astype(int)
    X = df_clean.drop(columns=["y"])

    # Identify categorical and numerical features (16 predictors total)
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")

    # 4.1.2 - 4.1.4 Preprocessing Pipeline
    # Note: No NaNs exist; 'unknown' is preserved as a legitimate categorical level.
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # 4.1.6 Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Fit preprocessor strictly on training fold to prevent data leakage
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed feature dimension after One-Hot Encoding: {X_train_processed.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor,
    )
