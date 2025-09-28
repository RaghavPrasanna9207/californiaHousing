# California Housing - Model Export Scaffold

This repository converts the notebook `housingCalifornia.ipynb` into a production-ready scaffold to train, evaluate, save, and package a model artifact with metadata, tests, CI, and a complete web application for housing price predictions.

## Features

- **Machine Learning Pipeline**: Train and export RandomForest models for California housing price prediction
- **FastAPI Backend**: Production-ready REST API with automatic model loading and health checks
- **Modern Web Frontend**: Clean, responsive UI with real-time validation and loading states
- **Model Artifacts**: Automated versioning, metadata tracking, and ONNX export support
- **Testing & CI**: Comprehensive test suite with GitHub Actions workflow
- **Docker Support**: Containerized inference for easy deployment

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

### 3) Run the Web Application

**Start the Backend API:**
```bash
python -m app.main
```
The API will be available at `http://localhost:8000` with automatic documentation at `/docs`.

**Open the Frontend:**
Open `frontend/index.html` in your web browser. The frontend will connect to the backend API automatically.

### 4) Export / Package (Optional)

```bash
python -m src.models.export --artifacts_dir artifacts
```
- If ONNX conversion is available, it writes `model_v....onnx`.
- Always creates `package_model_v....zip` containing `.joblib` and `.metadata.json` (and `.onnx` if available).

### 5) Command Line Inference (Optional)

```bash
python -m src.inference.predict --model artifacts/model_vYYYYMMDD_HHMMSS.joblib --input_json examples/example_request.json --output_csv predictions.csv
```

## Web Application

### Backend API
The FastAPI backend provides:
- **POST /predict**: Make housing price predictions
- **GET /health**: Health check endpoint
- **GET /docs**: Interactive API documentation
- **Automatic model loading**: Finds and loads the latest trained model
- **Input validation**: Comprehensive request validation with helpful error messages
- **CORS support**: Configured for frontend integration

### Frontend Features
- **Modern UI**: Clean, responsive design with purple gradient theme
- **Real-time validation**: Input validation with contextual error messages
- **Loading states**: Visual feedback during API requests
- **Reset functionality**: One-click form reset
- **Responsive design**: Works on desktop, tablet, and mobile
- **Accessibility**: WCAG compliant with proper focus management

## Configuration
See `configs/config.yaml`.
- Set `data.path` to your real CSV (must include target `MedHouseVal` by default). If your target column differs, update the config and code.
- Update `model.params` to adjust RandomForest hyperparameters.

## API Usage

### Making Predictions
Send a POST request to `/predict` with housing data:

```json
{
  "raw": {
    "MedInc": 5.0,
    "HouseAge": 20,
    "AveRooms": 6.0,
    "AveBedrms": 2.2,
    "Population": 1500,
    "AveOccup": 3.0,
    "Latitude": 37.5,
    "Longitude": -121.9
  }
}
```

Response:
```json
{
  "predictions": [2.408],
  "model_version": "20250909_130932",
  "metadata": {
    "train_metrics": {"rmse": 49401.57, "r2": 0.8239},
    "feature_names": ["MedInc", "HouseAge", ...]
  }
}
```

## Testing
```bash
python -m pytest -q
```

Tests include:
- Model training and artifact generation
- API endpoint functionality
- Input validation and error handling
- Preprocessing pipeline validation

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

## Project Structure

```
├── app/                    # FastAPI backend application
│   ├── main.py            # Main FastAPI app with prediction endpoints
│   ├── schemas.py         # Pydantic models for request/response validation
│   └── predict.py         # Prediction utilities and CLI interface
├── frontend/              # Web frontend
│   ├── index.html         # Main HTML page with form and results
│   └── styles.css         # Modern CSS with responsive design
├── src/                   # Core ML pipeline
│   ├── models/            # Model training and export
│   ├── inference/         # Prediction utilities
│   ├── utils/             # Shared utilities (model loading, preprocessing)
│   └── load_data.py       # Data loading with synthetic fallback
├── tests/                 # Test suite
├── configs/               # Configuration files
├── examples/              # Example request data
└── artifacts/             # Generated model artifacts (created after training)
```

## Notes & TODOs
- TODO: Confirm real dataset path and schema. Update `configs/config.yaml` and preprocessing if columns differ.
- TODO: If your notebook used a different model, adjust `configs/config.yaml:model.type` and augment `src/models/train.py` to support it.
- If ONNX export fails, you still get a ZIP package with `.joblib` and `.metadata.json`.
- The web application uses synthetic data by default. For production, configure a real dataset path in the config file.
