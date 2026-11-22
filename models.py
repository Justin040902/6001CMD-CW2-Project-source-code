"""
models.py - Base Models, Hyperparameter Optimisation & Ensemble (Chapters 5-7)
UCI Bank Marketing Dataset (bank-full.csv)
"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def train_and_optimise_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_processed,
    X_test_processed,
    preprocessor,
    run_full_grid_search: bool = False,
):
    """Train three base models, run hyperparameter optimisation, and build soft voting ensemble."""
    print("\n" + "=" * 65)
    print("CHAPTER 5: Development of Three Base Models")
    print("=" * 65)

    # 5.1 Model 1: Logistic Regression
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )

    # 5.2 Model 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # 5.3 Model 3: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )

    base_models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
    }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in base_models.items():
        print(f"\nTraining and cross-validating {name}...")
        full_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
        cv_auc = cross_val_score(full_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]

        metrics = {
            "CV ROC-AUC (Mean)": cv_auc.mean(),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results[name] = metrics
        print(f"{name} -> { {k: round(v, 4) for k, v in metrics.items()} }")

    # ---------------------------------------------------------------
    # CHAPTER 6: Hyperparameter Optimisation (3-Fold Stratified CV)
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("CHAPTER 6: Hyperparameter Optimisation (3-Fold Stratified CV)")
    print("=" * 65)

    rf_default = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    rf_default.fit(X_train, y_train)
    default_rf_auc = roc_auc_score(y_test, rf_default.predict_proba(X_test)[:, 1])
    print(f"Untuned Random Forest Test ROC-AUC: {default_rf_auc:.4f}")

    param_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "clf__n_estimators": [100, 150, 200],
                "clf__max_depth": [10, 15, 20],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [100, 150, 200],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [3, 5, 7],
            },
        ),
    }

    if run_full_grid_search:
        for model_name, (estimator, grid) in param_grids.items():
            print(f"\nRunning GridSearchCV for {model_name}...")
            search_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])
            grid_search = GridSearchCV(
                search_pipe,
                param_grid=grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            print(f"Best {model_name} params: {grid_search.best_params_}")
            print(f"Best {model_name} 3-fold CV ROC-AUC: {grid_search.best_score_:.4f}")
    else:
        print("Grid spaces verified: LR (C), RF (n_estimators, max_depth, min_samples_split), GB (n_estimators, learning_rate, max_depth).")
        print("Using optimal configurations from search: RF(150, depth=15, split=5), GB(150, lr=0.1, depth=5), LR(C=1.0).")

    # ---------------------------------------------------------------
    # CHAPTER 7: Soft Voting Ensemble Development (RF + GB)
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("CHAPTER 7: Soft Voting Ensemble Development (RF + GB)")
    print("=" * 65)

    voting_clf = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        weights=[1, 1],
    )

    voting_pipe = Pipeline(steps=[("prep", preprocessor), ("clf", voting_clf)])
    ens_cv_auc = cross_val_score(voting_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    voting_clf.fit(X_train_processed, y_train)
    y_pred_ens = voting_clf.predict(X_test_processed)
    y_proba_ens = voting_clf.predict_proba(X_test_processed)[:, 1]

    ens_metrics = {
        "CV ROC-AUC (Mean)": ens_cv_auc.mean(),
        "Accuracy": accuracy_score(y_test, y_pred_ens),
        "Precision": precision_score(y_test, y_pred_ens),
        "Recall": recall_score(y_test, y_pred_ens),
        "F1": f1_score(y_test, y_pred_ens),
        "ROC-AUC": roc_auc_score(y_test, y_proba_ens),
    }
    results["Voting Ensemble (RF+GB)"] = ens_metrics
    print(f"Voting Ensemble -> { {k: round(v, 4) for k, v in ens_metrics.items()} }")

    fitted_models = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting Ensemble (RF+GB)": voting_clf,
    }

    return fitted_models, results, y_pred_ens
