
import mlflow
from mlflow import MlflowClient

MODEL_NAME = "HousePriceModel"
client = MlflowClient()

# Fetch all registered versions
versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if len(versions) < 3:
    raise RuntimeError(
        "At least 3 registered model versions are required. "
        "Complete the registration step first."
    )

# Select the three most recently registered versions
versions = sorted(
    versions,
    key=lambda v: int(v.version),
    reverse=True
)[:3]

# Find metrics for each version's source run
scored = []
for version in versions:
    run_id = version.run_id

    if not run_id:
        print(f"Skipping version {version.version}: no source run ID")
        continue

    run = client.get_run(run_id)
    metrics = run.data.metrics

    if "rmse" not in metrics:
        print(f"Skipping version {version.version}: RMSE missing")
        continue

    scored.append({
        "version": version,
        "rmse": metrics["rmse"],
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
    })

if len(scored) < 3:
    raise RuntimeError(
        "Could not find metrics for three versions. "
        "Check that the registered versions came from successful training runs."
    )

# Lower RMSE is better for this regression task
scored.sort(key=lambda item: item["rmse"])
candidate = scored[0]
production = candidate

# Step A: mark the selected model as a candidate
client.set_model_version_tag(
    MODEL_NAME, candidate["version"].version,
    "lifecycle", "candidate"
)

print(
    f"Candidate: Version {candidate['version'].version}, "
    f"RMSE={candidate['rmse']:.2f}"
)

# Step B: promote it to production using an MLflow alias
try:
    client.set_registered_model_alias(
        MODEL_NAME,
        "production",
        production["version"].version
    )
    client.set_model_version_tag(
        MODEL_NAME, production["version"].version,
        "lifecycle", "production"
    )
    print(
        f"Production alias now points to Version "
        f"{production['version'].version}"
    )
except Exception as exc:
    print("Production alias unavailable in this MLflow setup.")
    print("Using lifecycle tags as a documented alternative:", exc)
    client.set_model_version_tag(
        MODEL_NAME, production["version"].version,
        "lifecycle", "production"
    )

# Step C: mark the other two versions as retired
for item in scored[1:]:
    version_number = item["version"].version
    client.set_model_version_tag(
        MODEL_NAME, version_number,
        "lifecycle", "retired"
    )
    print(
        f"Version {version_number} marked retired "
        f"(RMSE={item['rmse']:.2f})"
    )

print("\nLifecycle management completed.")
print("Acceptance rule: select the lowest RMSE among the three versions.")
print("Evidence: source-run RMSE, MAE and R2 metrics.")
