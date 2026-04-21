"""
api.py
======
FastAPI backend exposing bias audit capabilities as REST endpoints.

Endpoints:
  GET  /                     — health check + available datasets
  GET  /datasets             — list all available datasets
  POST /audit/data/{name}    — run data audit on a named dataset
  POST /audit/model/{name}   — run model audit on a named dataset
  POST /audit/full/{name}    — run full pipeline (data + model + mitigation + report)
  GET  /report/{name}/json   — download JSON report for a dataset
  GET  /report/{name}/pdf    — download PDF report for a dataset
  GET  /results              — get cached results for all audited datasets
  POST /audit/upload         — upload custom CSV and run audit

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/home/claude/bias_audit/scripts")

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse as FR2
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("/home/claude/bias_audit/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory cache ──
_cache = {
    "datasets":     {},
    "data_audits":  {},
    "model_audits": {},
    "mitigations":  {},
    "reports":      {},
}

app = FastAPI(
    title="FairLens API",
    description="FairLens — AI Bias Detection & Mitigation Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files if directory exists
import os as _os
_frontend_dir = _os.path.join(_os.path.dirname(__file__), "frontend")
if _os.path.isdir(_frontend_dir):
    app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

@app.get("/ui", include_in_schema=False)
def serve_ui():
    import os as _os2
    fp = _os2.path.join(_os2.path.dirname(__file__), "frontend", "index.html")
    if _os2.path.exists(fp):
        return FR2(fp, media_type="text/html")
    return {"error": "Frontend not found — place frontend/index.html next to api.py"}

AVAILABLE_DATASETS = {
    "adult":       "UCI Adult (Census Income) — income prediction, gender + race bias",
    "compas":      "COMPAS Recidivism — criminal justice, racial bias",
    "german":      "German Credit — loan approval, age + gender bias",
    "utrecht":     "Utrecht Fairness Recruitment — hiring, gender + age bias",
    "diabetes130": "Diabetes 130-US Hospitals — medical readmission, age/race/gender bias",
}


# ─────────────────────────────────────────────────────────────
# Helper: lazy load datasets
# ─────────────────────────────────────────────────────────────

def _get_dataset(name: str):
    if name not in AVAILABLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found. "
                            f"Available: {list(AVAILABLE_DATASETS.keys())}")
    if name not in _cache["datasets"]:
        from data_loader import (
            load_adult, load_compas, load_german_credit,
            load_utrecht, load_diabetes_130
        )
        loaders = {
            "adult":       load_adult,
            "compas":      load_compas,
            "german":      load_german_credit,
            "utrecht":     load_utrecht,
            "diabetes130": load_diabetes_130,
        }
        _cache["datasets"][name] = loaders[name]()
    return _cache["datasets"][name]


def _serialise(obj):
    """Recursively make an object JSON-safe."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Skip unpicklable objects like trained models
    if hasattr(obj, "predict"):
        return "<trained_model>"
    if hasattr(obj, "transform") and not isinstance(obj, dict):
        return "<sklearn_transformer>"
    return obj


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "FairLens API v1.0",
        "ui": "/ui",
        "available_datasets": AVAILABLE_DATASETS,
        "endpoints": [
            "GET  /datasets",
            "POST /audit/data/{name}",
            "POST /audit/model/{name}",
            "POST /audit/full/{name}",
            "GET  /report/{name}/json",
            "GET  /report/{name}/pdf",
            "GET  /results",
            "POST /audit/upload",
        ],
    }


@app.get("/datasets")
def list_datasets():
    return {
        "datasets": [
            {
                "name": name,
                "description": desc,
                "loaded": name in _cache["datasets"],
                "audited": name in _cache["data_audits"],
            }
            for name, desc in AVAILABLE_DATASETS.items()
        ]
    }


@app.post("/audit/data/{name}")
def run_data_audit(name: str):
    """Run statistical data audit on the named dataset."""
    t0 = time.time()
    dataset = _get_dataset(name)

    from data_auditor import audit_dataset
    result = audit_dataset(dataset, verbose=False)
    _cache["data_audits"][name] = result

    return {
        "status": "ok",
        "elapsed_seconds": round(time.time() - t0, 2),
        "result": _serialise(result),
    }


@app.post("/audit/model/{name}")
def run_model_audit(name: str, model_type: str = "logistic"):
    """Train a model and audit it for bias. model_type: logistic | random_forest"""
    if model_type not in ["logistic", "random_forest"]:
        raise HTTPException(400, "model_type must be 'logistic' or 'random_forest'")

    t0 = time.time()
    dataset = _get_dataset(name)

    from model_auditor import audit_model
    result = audit_model(dataset, model_type=model_type, verbose=False)
    _cache["model_audits"][name] = result

    return {
        "status": "ok",
        "elapsed_seconds": round(time.time() - t0, 2),
        "result": _serialise(result),
    }


@app.post("/audit/full/{name}")
def run_full_audit(name: str, model_type: str = "logistic"):
    """
    Run the complete pipeline:
    data audit → model audit → 3x mitigation → PDF + JSON report
    """
    t0 = time.time()
    dataset = _get_dataset(name)

    try:
        from data_auditor import audit_dataset
        from model_auditor import audit_model
        from mitigation import apply_all_mitigations
        from report_generator import generate_reports

        data_result  = audit_dataset(dataset, verbose=False)
        model_result = audit_model(dataset, model_type=model_type, verbose=False)
        mit_result   = apply_all_mitigations(dataset, model_result, verbose=False)
        report_paths = generate_reports(name, data_result, model_result, mit_result)

        _cache["data_audits"][name]  = data_result
        _cache["model_audits"][name] = model_result
        _cache["mitigations"][name]  = mit_result
        _cache["reports"][name]      = report_paths

        return {
            "status": "ok",
            "elapsed_seconds": round(time.time() - t0, 2),
            "data_bias_score":  data_result.get("overall_bias_score"),
            "data_severity":    data_result.get("overall_severity"),
            "model_bias_score": model_result.get("bias_score"),
            "model_severity":   model_result.get("severity"),
            "mitigation_improvements": {
                s: r.get("improvement", 0)
                for s, r in mit_result.items()
            },
            "report_json": f"/report/{name}/json",
            "report_pdf":  f"/report/{name}/pdf",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Full audit failed: {str(e)}\n{traceback.format_exc()}")


@app.get("/report/{name}/json")
def get_json_report(name: str):
    """Return JSON audit report for a dataset (runs full audit if needed)."""
    json_path = OUTPUT_DIR / f"{name}_bias_report.json"
    if not json_path.exists():
        # Run full audit first
        run_full_audit(name)
    if not json_path.exists():
        raise HTTPException(404, "Report not found — run /audit/full/{name} first")
    return FileResponse(str(json_path), media_type="application/json",
                        filename=f"{name}_bias_report.json")


@app.get("/report/{name}/pdf")
def get_pdf_report(name: str):
    """Return PDF audit report for a dataset (runs full audit if needed)."""
    pdf_path = OUTPUT_DIR / f"{name}_bias_report.pdf"
    if not pdf_path.exists():
        run_full_audit(name)
    if not pdf_path.exists():
        raise HTTPException(404, "PDF report not found — reportlab may not be installed")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        filename=f"{name}_bias_report.pdf")


@app.get("/results")
def get_all_results():
    """Return cached audit results for all previously audited datasets."""
    summary = {}
    for name in AVAILABLE_DATASETS:
        entry = {"name": name, "loaded": name in _cache["datasets"]}
        if name in _cache["data_audits"]:
            da = _cache["data_audits"][name]
            entry["data_bias_score"] = da.get("overall_bias_score")
            entry["data_severity"]   = da.get("overall_severity")
        if name in _cache["model_audits"]:
            ma = _cache["model_audits"][name]
            entry["model_bias_score"] = ma.get("bias_score")
            entry["model_severity"]   = ma.get("severity")
            entry["model_accuracy"]   = ma.get("overall_metrics", {}).get("accuracy")
        summary[name] = entry
    return {"results": summary}


class AuditConfig(BaseModel):
    target_column:    str
    protected_column: str
    positive_label:   Optional[str] = "1"
    dataset_label:    Optional[str] = "Custom dataset"


@app.post("/audit/upload")
async def audit_uploaded_csv(
    file: UploadFile = File(...),
    target_column: str = "target",
    protected_column: str = "protected",
    positive_label: str = "1",
    dataset_label: str = "Uploaded dataset",
):
    """
    Upload a CSV and run a full bias audit.
    The CSV must have a target column and a binary protected attribute column.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    t0 = time.time()
    content = await file.read()

    try:
        import io
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found. "
                            f"Columns: {list(df.columns)}")
    if protected_column not in df.columns:
        raise HTTPException(400, f"Protected column '{protected_column}' not found. "
                            f"Columns: {list(df.columns)}")

    # Binarise target
    unique_vals = df[target_column].unique()
    if len(unique_vals) > 2:
        raise HTTPException(400, f"Target must be binary. Got {len(unique_vals)} unique values")

    df["_target_bin"] = (df[target_column].astype(str) == str(positive_label)).astype(int)
    df["_prot_bin"]   = pd.to_numeric(df[protected_column], errors="coerce").fillna(0).astype(int)

    dataset_dict = {
        "df": df,
        "target": "_target_bin",
        "protected": [protected_column],
        "binary_protected": "_prot_bin",
        "label": dataset_label,
        "task": f"Predict {target_column}",
        "positive_label": 1,
    }

    from data_auditor import audit_dataset
    from model_auditor import audit_model
    from mitigation import apply_all_mitigations

    data_result  = audit_dataset(dataset_dict, verbose=False)
    model_result = audit_model(dataset_dict, verbose=False)
    mit_result   = apply_all_mitigations(dataset_dict, model_result, verbose=False)

    name = f"custom_{int(time.time())}"
    _cache["datasets"][name]     = dataset_dict
    _cache["data_audits"][name]  = data_result
    _cache["model_audits"][name] = model_result
    _cache["mitigations"][name]  = mit_result

    return {
        "status": "ok",
        "elapsed_seconds": round(time.time() - t0, 2),
        "n_rows": len(df),
        "data_bias_score":  data_result.get("overall_bias_score"),
        "data_severity":    data_result.get("overall_severity"),
        "model_bias_score": model_result.get("bias_score"),
        "model_severity":   model_result.get("severity"),
        "tpr_gap":          model_result.get("tpr_gap"),
        "fpr_gap":          model_result.get("fpr_gap"),
        "mitigation_improvements": {
            s: r.get("improvement", 0)
            for s, r in mit_result.items()
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)