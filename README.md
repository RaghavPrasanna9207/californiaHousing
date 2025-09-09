# California Housing - Model Export Scaffold

This repository converts the notebook `housingCalifornia.ipynb` into a small production-ready scaffold to train, evaluate, save, and package a model artifact with metadata, tests, and CI.

## Quickstart

### 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2) Train

```bash
python -m src.models.train --config configs/config.yaml
```
This will create artifacts under `artifacts/`:
- `model_vYYYYMMDD_HHMMSS.joblib`
- `model_vYYYYMMDD_HHMMSS.metadata.json`
- `config_vYYYYMMDD_HHMMSS.yaml`

### 3) Export / Package

```bash
python -m src.models.export --artifacts_dir artifacts
```
- If ONNX conversion is available, it writes `model_v....onnx`.
- Always creates `package_model_v....zip` containing `.joblib` and `.metadata.json` (and `.onnx` if available).

### 4) Inference

```bash
python -m src.inference.predict --model artifacts/model_vYYYYMMDD_HHMMSS.joblib --input_json examples/example_request.json --output_csv predictions.csv
```

## Configuration
See `configs/config.yaml`.
- Set `data.path` to your real CSV (must include target `MedHouseVal` by default). If your target column differs, update the config and code.
- Update `model.params` to adjust RandomForest hyperparameters.

## Testing
```bash
python -m pytest -q
```

## Docker (inference)
A minimal Dockerfile is provided to run inference:

```bash
# Build
docker build -t cal-housing-infer .

# Run (expects artifacts and example on host)
docker run --rm -v %cd%/artifacts:/app/artifacts -v %cd%/examples:/app/examples cal-housing-infer \
  python -m src.inference.predict --model artifacts/<your_model>.joblib --input_json examples/example_request.json --output_csv predictions.csv
```

## CI
GitHub Actions workflow in `.github/workflows/ci.yml` runs tests on push.

## Notes & TODOs
- TODO: Confirm real dataset path and schema. Update `configs/config.yaml` and `src/data/preprocessing.py` if columns differ.
- TODO: If your notebook used a different model, adjust `configs/config.yaml:model.type` and augment `src/models/train.py` to support it.
- If ONNX export fails, you still get a ZIP package with `.joblib` and `.metadata.json`.
