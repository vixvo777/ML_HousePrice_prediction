import hashlib
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports" / "figures"
DATA_PATH = DATA_DIR / "house_prices.csv"
EXPERIMENT_NAME = "HousePricePrediction"

def ensure_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        X, y = make_regression(
            n_samples=600, n_features=5, n_informative=5,
            noise=18.0, random_state=42
        )
        columns = ["area_sqft", "bedrooms", "age_years", "distance_km", "quality_score"]
        # Use a synthetic, reproducible house-price-like target.
        df = pd.DataFrame(X, columns=columns)
        df["price"] = (y * 2500 + 250000).round(2)
        df.to_csv(DATA_PATH, index=False)
        print(f"Created reproducible synthetic dataset: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "not-available"
def get_dataset_hash():
    data_path = ROOT / "data" / "house_prices.csv"
    return hashlib.sha256(data_path.read_bytes()).hexdigest()
def main():
    df = ensure_dataset()
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = [
        ("LinearRegression", LinearRegression(), {"model": "LinearRegression"}),
        ("Ridge_alpha_1", make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
         {"model": "Ridge", "alpha": 1.0}),
        ("RandomForest_100", RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        ), {"model": "RandomForestRegressor", "n_estimators": 100, "max_depth": 8}),
    ]

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=False)
    summary = []
    for run_name, model, params in candidates:
        with mlflow.start_run(run_name=run_name) as run:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = r2_score(y_test, preds)

            mlflow.log_params(params)
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            mlflow.set_tag("git_commit", get_git_commit())
            mlflow.set_tag("dataset_path", "data/house_prices.csv")
            mlflow.set_tag("dataset_sha256", get_dataset_hash())
            mlflow.set_tag("dataset_rows", len(df))
            mlflow.set_tag("dataset_version", "v2-601-rows")

            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            fig_path = REPORT_DIR / f"{run_name}_actual_vs_predicted.png"
            plt.figure(figsize=(6, 5))
            plt.scatter(y_test, preds, alpha=0.7)
            plt.xlabel("Actual price")
            plt.ylabel("Predicted price")
            plt.title(f"{run_name}: Actual vs Predicted")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(str(fig_path), artifact_path="figures")

            report_path = ROOT / "reports" / f"{run_name}_metrics.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({
                "run_name": run_name, "mae": mae, "rmse": rmse, "r2": r2
            }, indent=2))
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            signature = infer_signature(X_train, model.predict(X_train.head(5)))
            mlflow.sklearn.log_model(
             sk_model=model,
             artifact_path="model",
             signature=signature,
             input_example=X_train.head(3),
             registered_model_name=None,
             skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
            summary.append({
                "run_name": run_name, "run_id": run.info.run_id,
                "mae": mae, "rmse": rmse, "r2": r2
            })
            print(f"{run_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}, run_id={run.info.run_id}")

    summary_df = pd.DataFrame(summary).sort_values("rmse")
    summary_path = ROOT / "reports" / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\nModel comparison (sorted by RMSE; lower is better):")
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison table: {summary_path}")
    print("\nOpen the MLflow UI with: mlflow ui --host 127.0.0.1 --port 5000")

if __name__ == "__main__":
    main()
