
import mlflow
from mlflow import MlflowClient
from mlflow.tracking import MlflowClient as Client

EXPERIMENT_NAME = "HousePricePrediction"
REGISTERED_MODEL_NAME = "HousePriceModel"

client = MlflowClient()

# Find the experiment containing our training runs
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError(
        "Experiment not found. Run python src/train.py first."
    )

# Fetch completed runs that have logged a model artifact
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.rmse ASC"],
    max_results=100,
)

# Keep only runs with a logged model
valid_runs = []
for run in runs:
    try:
        client.get_run(run.info.run_id)
        client.download_artifacts(run.info.run_id, "model")
        valid_runs.append(run)
    except Exception:
        continue

if len(valid_runs) < 3:
    raise RuntimeError(
        f"Only {len(valid_runs)} usable runs found. "
        "Run python src/train.py successfully first."
    )

# Create the registered model if it doesn't exist
try:
    client.create_registered_model(REGISTERED_MODEL_NAME)
    print(f"Created registered model: {REGISTERED_MODEL_NAME}")
except Exception as exc:
    if "already exists" in str(exc).lower():
        print(f"Using existing registered model: {REGISTERED_MODEL_NAME}")
    else:
        raise

# Register three distinct runs as model versions
for index, run in enumerate(valid_runs[:3], start=1):
    model_uri = f"runs:/{run.info.run_id}/model"

    version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    metrics = run.data.metrics
    description = (
        f"Candidate {index}. Run ID: {run.info.run_id}. "
        f"RMSE: {metrics.get('rmse', 'N/A')}. "
        f"MAE: {metrics.get('mae', 'N/A')}. "
        f"R2: {metrics.get('r2', 'N/A')}."
    )

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        description=description,
    )

    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version,
        "candidate", "true"
    )

    print(
        f"Registered Version {version.version} "
        f"from run {run.info.run_id}"
    )

print("\nRegistration completed.")
print("Open the MLflow UI and inspect the Models section.")
