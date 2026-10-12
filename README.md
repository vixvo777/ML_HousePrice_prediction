# MLOps Lab Assignment 4–5
## Advanced MLflow and Data Versioning with DVC

This project demonstrates a reproducible house-price regression workflow using Git, DVC, MLflow, and scikit-learn.

### Project objectives

- Track and compare regression experiments using MLflow.
- Record parameters, metrics, model signatures, input examples, and artifacts.
- Register model versions and document lifecycle decisions.
- Version datasets using DVC and Git.
- Associate experiment runs with source-code revisions and dataset hashes.
- Verify recovery of earlier dataset versions.

### Environment

- OS: Windows
- Python: 3.13.4
- MLflow: 3.17.0
- DVC: 3.67.1
- Dataset: House-price regression CSV

### Setup

Clone the repository and open its directory in VS Code.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project uses an existing Git repository and DVC configuration. Do not initialize them again when working with the existing checkout.

### MLflow experiment tracking

Start the MLflow UI:

```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in your browser.

Run the training script in another terminal:

```powershell
python src/train.py
```

The workflow compares Linear Regression, Ridge Regression, and Random Forest models using MAE, RMSE, and R². Experiment artifacts include metric reports and actual-versus-predicted plots.

### Model registry

The registered model is `HousePriceModel`. The captured registry evidence shows three model versions, with Version 2 marked for production using an alias and lifecycle metadata. Versions 1 and 3 are marked retired.

The production designation documents the registry decision; it does not, by itself, prove deployment to a live serving endpoint.

### DVC dataset versioning

The dataset is tracked through `data/house_prices.csv.dvc`. Git stores the metadata and its history, while DVC stores the dataset contents in its cache and configured remote.

The demonstrated dataset versions are:

| Version | Rows | Columns |
|---|---:|---:|
| Baseline | 600 | 6 |
| Version 2 | 601 | 6 |

The baseline was restored successfully in a separate recovery worktree using the configured local DVC remote. The latest dataset remains in the main project.

Check the current dataset and DVC state:

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/house_prices.csv'); print('Rows:',len(df)); print('Columns:',len(df.columns))"
dvc status
```

The local DVC remote is intended for this lab environment. It is not shared cloud storage and is not automatically available to another engineer cloning the repository.

### Reports and evidence

The `reports/` directory contains model metric JSON files and the model comparison CSV. The `reports/figures/` directory contains visual evaluation artifacts.

The accompanying lab report should include screenshots of MLflow runs and artifacts, registered model versions and lifecycle metadata, Git/DVC dataset history, and the successful recovery test.

### Repository

GitHub: https://github.com/vixvo777/ML_HousePrice_prediction

Submission branch: `mlops-lab-submission`

### Tools and references

- MLflow: https://mlflow.org/docs/latest/
- DVC: https://dvc.org/doc
- scikit-learn: https://scikit-learn.org/stable/
