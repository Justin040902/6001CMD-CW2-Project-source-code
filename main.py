"""
main.py - Entry Point for End-to-End Machine Learning Pipeline
6001CMD Machine Learning and Related Applications (Solo Submission)
Dataset: UCI Bank Marketing (bank-full.csv)

Author: Justin Wong Yew Ming
Date: November 2026
"""

import warnings
import pandas as pd

from eda import run_eda
from preprocessing import prepare_data
from models import train_and_optimise_models
from evaluation import evaluate_and_export

warnings.filterwarnings("ignore")

DATA_PATH = "bank-full.csv"
OUTPUT_DIR = "outputs"
RUN_FULL_GRID_SEARCH = False  # Set to True to run full GridSearchCV across all 3 models


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Loaded '{DATA_PATH}' with shape: {df.shape}")

    # Chapter 3: Exploratory Data Analysis
    run_eda(df, output_dir=OUTPUT_DIR)

    # Chapter 4: Data Preparation & Preprocessing
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_processed,
        X_test_processed,
        preprocessor,
    ) = prepare_data(df)

    # Chapters 5-7: Base Models, Hyperparameter Optimisation & Ensemble
    fitted_models, results, y_pred_ens = train_and_optimise_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        preprocessor=preprocessor,
        run_full_grid_search=RUN_FULL_GRID_SEARCH,
    )

    # Chapter 8: Evaluation & Exports
    evaluate_and_export(
        fitted_models=fitted_models,
        results=results,
        X_test_processed=X_test_processed,
        y_test=y_test,
        y_pred_ens=y_pred_ens,
        output_dir=OUTPUT_DIR,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
