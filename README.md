# MLOps Lab Assignment 4–5

This project demonstrates MLflow experiment tracking and DVC dataset versioning for a reproducible house-price regression example.

## Windows / VS Code setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
git init
python src/train.py
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## MLflow

In one terminal:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000. In another terminal, run `python src/train.py`.

## DVC

After the dataset has been generated:
```powershell
dvc init
dvc add data/house_prices.csv
git add .dvc .dvcignore data/house_prices.csv.dvc .gitignore
git commit -m "Initialize DVC and track baseline dataset"
```

Create a local DVC remote outside the repository (PowerShell):
```powershell
New-Item -ItemType Directory -Force ..\dvc-storage
dvc remote add -d localremote ..\dvc-storage
git add .dvc/config
git commit -m "Configure local DVC remote"
dvc push
```

A local remote is for practice on this machine; it is not shared cloud storage.

## Notes

- The CSV is generated deterministically if it does not exist.
- The three regression candidates are compared using MAE, RMSE, and R².
- MLflow logs metrics, parameters, model signatures, input examples, plots, and JSON reports.
- Dataset and model lifecycle version registration/promotion should be completed using the installed MLflow version and documented in the lab report.
