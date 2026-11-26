"""
train_model.py
==============
Trains and compares three regression models that predict crop Yield.

  1. Linear Regression
  2. Random Forest Regressor
  3. Support Vector Regression (SVR)

Key design decisions (good points to explain in your viva):
  * No data leakage  : Production is NEVER used as an input (Yield = Production / Area).
  * Chronological split: models train on earlier years and are tested on the latest
                         years, like real forecasting. A random split would be too optimistic.
  * One pipeline per model: encoding + scaling + model are saved together, so the
                         Streamlit app applies exactly the same steps as training.
  * Log-transformed target: yield is very skewed, so models learn log(1 + yield) and
                         predictions are converted back. All metrics are reported on the
                         ORIGINAL yield scale.
  * SVR is slow on big data: it is trained on a random sample (see --svr-sample).

Outputs (in models/):
  best_model.pkl, model_comparison.csv, model_metadata.json, test_predictions.csv

Run:  python train_model.py
"""

import argparse
import json
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR

import data_preprocessing as dp

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
FEATURES = dp.FEATURE_COLUMNS          # State, District, Crop, Season, Year, Area
TARGET = dp.TARGET_COLUMN              # Yield
ENCODED_CATEGORIES = ["State", "Crop", "Season", "Location"]  # Location = State + District
TEST_YEAR_FRACTION = 0.2               # last ~20% of the years are used for testing
RANDOM_STATE = 42

MODEL_PATH = dp.MODELS_DIR / "best_model.pkl"
COMPARISON_PATH = dp.MODELS_DIR / "model_comparison.csv"
METADATA_PATH = dp.MODELS_DIR / "model_metadata.json"
TEST_PREDICTIONS_PATH = dp.MODELS_DIR / "test_predictions.csv"


# ==========================================================================
# Data preparation
# ==========================================================================
def prepare_data():
    """Load the cleaned data; create it first if it does not exist yet."""
    if not dp.CLEAN_CSV.exists():
        print("Cleaned data not found - running preprocessing first ...")
        raw = dp.load_raw_data()
        clean, report = dp.clean_data(raw)
        dp.print_report(report)
        dp.save_clean_data(clean)
    return dp.load_clean_data()


def split_by_year(df):
    """
    Chronological split: oldest years -> train, newest years -> test.
    Falls back to a random split only if the data has fewer than 3 distinct years.
    """
    years = sorted(df["Year"].unique())
    if len(years) >= 3:
        n_test_years = max(1, int(round(len(years) * TEST_YEAR_FRACTION)))
        test_years = [int(y) for y in years[-n_test_years:]]
        test_mask = df["Year"].isin(test_years)
        train_df, test_df = df[~test_mask], df[test_mask]
        strategy = "chronological (train on earlier years, test on the latest years)"
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
        test_years = sorted(int(y) for y in test_df["Year"].unique())
        strategy = "random 80/20 (fewer than 3 distinct years in the data)"

    info = {
        "strategy": strategy,
        "train_year_min": int(train_df["Year"].min()),
        "train_year_max": int(train_df["Year"].max()),
        "test_years": test_years,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return train_df, test_df, info


# ==========================================================================
# Models
# ==========================================================================
def make_preprocessor():
    """One-hot encode categories, scale Year, log+scale Area."""
    area_steps = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(transformers=[
        ("categories", OneHotEncoder(handle_unknown="ignore"), ENCODED_CATEGORIES),
        ("year", StandardScaler(), ["Year"]),
        ("area", area_steps, ["Area"]),
    ])


def build_pipeline(regressor):
    """Full pipeline: add Location -> preprocess -> model, predicting log(1 + Yield)."""
    pipeline = Pipeline([
        ("location", FunctionTransformer(dp.add_location_feature)),
        ("preprocess", make_preprocessor()),
        ("model", regressor),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def build_models(rf_trees):
    return {
        "Linear Regression": build_pipeline(LinearRegression()),
        "Random Forest": build_pipeline(RandomForestRegressor(
            n_estimators=rf_trees,
            max_depth=20,
            min_samples_leaf=10,   # keeps the saved model file reasonably small
            max_features=0.5,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
        "SVR": build_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
    }


# ==========================================================================
# Evaluation
# ==========================================================================
def predict_clipped(model, X):
    """Predict and make sure yields are never negative."""
    return np.clip(model.predict(X), 0, None)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare crop-yield models.")
    parser.add_argument("--svr-sample", type=int, default=10000,
                        help="rows used to train SVR (it is slow on large data). Default 10000")
    parser.add_argument("--rf-trees", type=int, default=60,
                        help="number of trees in the Random Forest. Default 60")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        df = prepare_data()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    train_df, test_df, split_info = split_by_year(df)
    if train_df.empty or test_df.empty:
        print("ERROR: train or test set is empty - check your dataset.")
        sys.exit(1)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print("\n=== DATA SPLIT ===")
    print(f"Strategy   : {split_info['strategy']}")
    print(f"Train rows : {split_info['n_train']:,} "
          f"(years {split_info['train_year_min']}-{split_info['train_year_max']})")
    print(f"Test rows  : {split_info['n_test']:,} (years {split_info['test_years']})")
    print(f"Input features: {FEATURES}   |   Target: {TARGET}")

    models = build_models(args.rf_trees)
    fitted, rows = {}, []

    print("\n=== TRAINING ===")
    for name, model in models.items():
        X_fit, y_fit = X_train, y_train
        if name == "SVR" and len(X_train) > args.svr_sample:
            sampled = train_df.sample(n=args.svr_sample, random_state=RANDOM_STATE)
            X_fit, y_fit = sampled[FEATURES], sampled[TARGET]
            print(f"{name}: using a random sample of {len(X_fit):,} training rows (SVR is slow on big data)")

        print(f"Training {name} ...", flush=True)
        start = time.time()
        model.fit(X_fit, y_fit)
        seconds = time.time() - start

        scores = evaluate(y_test, predict_clipped(model, X_test))
        fitted[name] = model
        rows.append({"Model": name, **scores,
                     "Train_Rows": int(len(X_fit)), "Train_Time_Sec": round(seconds, 1)})
        print(f"  done in {seconds:.1f}s | MAE={scores['MAE']:.3f} RMSE={scores['RMSE']:.3f} R2={scores['R2']:.4f}")

    # ---- compare and choose the best (lowest RMSE on the unseen test years) ----
    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_name = comparison.loc[0, "Model"]
    best_model = fitted[best_name]

    dp.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(6).to_csv(COMPARISON_PATH, index=False)
    joblib.dump(best_model, MODEL_PATH, compress=3)

    # Test-set predictions of the best model (used by the app's "actual vs predicted" chart)
    best_pred = predict_clipped(best_model, X_test)
    out = test_df[["State", "District", "Crop", "Year"]].copy()
    out["Actual_Yield"] = y_test.values
    out["Predicted_Yield"] = best_pred
    out.sample(n=min(5000, len(out)), random_state=RANDOM_STATE).to_csv(TEST_PREDICTIONS_PATH, index=False)

    metadata = {
        "best_model": best_name,
        "selection_rule": "lowest RMSE on the held-out test years",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "target": TARGET,
        "target_transform": "log1p during training, expm1 on prediction; metrics use the original scale",
        "split": split_info,
        "svr_sample_size": int(min(args.svr_sample, len(X_train))),
        "random_forest_trees": int(args.rf_trees),
        "notes": "No weather, soil, rainfall or fertilizer data is available in the dataset.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== MODEL COMPARISON (test years, original yield scale) ===")
    print(comparison.round(4).to_string(index=False))
    print(f"\nBest model: {best_name} (lowest RMSE)")
    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved comparison -> {COMPARISON_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")
    print("\nNext step:  streamlit run app.py")


if __name__ == "__main__":
    main()
