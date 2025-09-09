import argparse
import json
import os
import zipfile
from typing import Optional

import joblib

# Optional imports for ONNX conversion
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False



def export_to_onnx(joblib_path: str, metadata_path: str, onnx_output_path: str) -> bool:
    try:
        model = joblib.load(joblib_path)
        # Build a dummy input DataFrame with correct column names to satisfy ColumnTransformer
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])
        if not feature_names:
            raise RuntimeError("No feature_names found in metadata; cannot create ONNX input schema.")

        # Create a single-row float32 DataFrame with the right column names
        dummy = pd.DataFrame({name: np.array([0.0], dtype=np.float32) for name in feature_names})

        # Determine whether the final estimator is a classifier; only apply zipmap for classifiers.
        try:
            from sklearn.base import is_classifier
            final_est = model
            if hasattr(model, "named_steps"):
                # pipeline -> last step is the estimator
                final_est = list(model.named_steps.values())[-1]
            options = {final_est.__class__.__name__: {"zipmap": False}} if is_classifier(final_est) else None
        except Exception:
            # If anything goes wrong while detecting estimator type, fall back to no options.
            options = None

        # Call to_onnx with options only if relevant
        if options:
            onnx_model = to_onnx(model, dummy, options=options)
        else:
            onnx_model = to_onnx(model, dummy)

        with open(onnx_output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return True
    except Exception as exc:
        print(f"ONNX export failed: {exc}")
        return False




def package_artifacts(joblib_path: str, metadata_path: str, output_zip_path: str, onnx_path: Optional[str] = None) -> str:
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(joblib_path, arcname=os.path.basename(joblib_path))
        zf.write(metadata_path, arcname=os.path.basename(metadata_path))
        if onnx_path and os.path.exists(onnx_path):
            zf.write(onnx_path, arcname=os.path.basename(onnx_path))
    return output_zip_path



def main() -> int:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX and/or package it")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts", help="Artifacts directory")
    parser.add_argument("--model_basename", type=str, required=False, help="Model base name like model_vYYYYMMDD_HHMMSS")
    args = parser.parse_args()

    artifacts = sorted([f for f in os.listdir(args.artifacts_dir) if f.startswith("model_v")])
    if not artifacts:
        print("No artifacts found. Please run training first.")
        return 1

    if args.model_basename:
        model_base = args.model_basename
    else:
        # Pick the latest by name sort which works due to timestamp naming
        latest_joblib = [f for f in artifacts if f.endswith('.joblib')]
        if not latest_joblib:
            print("No .joblib model artifacts found.")
            return 1
        model_base = latest_joblib[-1].split(".joblib")[0]

    joblib_path = os.path.join(args.artifacts_dir, f"{model_base}.joblib")
    metadata_path = os.path.join(args.artifacts_dir, f"{model_base}.metadata.json")
    if not os.path.exists(joblib_path) or not os.path.exists(metadata_path):
        print("Model or metadata not found for the specified basename.")
        return 1

    onnx_path = os.path.join(args.artifacts_dir, f"{model_base}.onnx")
    onnx_success = False
    if ONNX_AVAILABLE:
        onnx_success = export_to_onnx(joblib_path, metadata_path, onnx_path)
    else:
        print("ONNX export not available. TODO: Install onnx and skl2onnx if needed.")

    package_zip_path = os.path.join(args.artifacts_dir, f"package_{model_base}.zip")
    package_artifacts(joblib_path, metadata_path, package_zip_path, onnx_path if onnx_success else None)
    print(f"Packaged artifacts into: {package_zip_path}")
    if onnx_success:
        print(f"ONNX model saved to: {onnx_path}")
    return 0


if __name__ == "__main__":
    exit(main())
