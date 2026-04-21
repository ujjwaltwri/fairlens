"""
api.py  —  FairLens
FastAPI backend. Serves the REST API and the frontend dashboard.

Run from the backend/ directory:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Or from the repo root:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional

# ── Add backend/ to sys.path so sibling imports always work ──
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# ── Output directory: always relative to backend/ ──
OUTPUT_DIR = Path(_HERE) / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Frontend: backend/../frontend/ ──
_FRONTEND_DIR = os.path.join(os.path.dirname(_HERE), "frontend")

_cache = {
    "datasets": {}, "data_audits": {},
    "model_audits": {}, "mitigations": {}, "reports": {},
}

app = FastAPI(
    title="FairLens API",
    description="FairLens — AI Bias Detection & Mitigation Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Serve frontend at /ui ──
@app.get("/ui", include_in_schema=False)
def serve_ui():
    fp = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(fp):
        return FileResponse(fp, media_type="text/html")
    return JSONResponse({"error": "Frontend not found. Place frontend/index.html at repo root."})

if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

AVAILABLE_DATASETS = {
    "adult":       "UCI Adult (Census Income) — income prediction, gender + race bias",
    "compas":      "COMPAS Recidivism — criminal justice, racial bias",
    "german":      "German Credit — loan approval, age + gender bias",
    "utrecht":     "Utrecht Fairness Recruitment — hiring, gender + age bias",
    "diabetes130": "Diabetes 130-US Hospitals — medical readmission, age/race/gender bias",
}


def _get_dataset(name: str):
    if name not in AVAILABLE_DATASETS:
        raise HTTPException(404, f"Dataset '{name}' not found. Available: {list(AVAILABLE_DATASETS)}")
    if name not in _cache["datasets"]:
        from data_loader import (load_adult, load_compas, load_german_credit,
                                  load_utrecht, load_diabetes_130)
        loaders = {"adult": load_adult, "compas": load_compas, "german": load_german_credit,
                   "utrecht": load_utrecht, "diabetes130": load_diabetes_130}
        _cache["datasets"][name] = loaders[name]()
    return _cache["datasets"][name]


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(i) for i in obj]
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, np.bool_):       return bool(obj)
    if hasattr(obj, "predict"):         return "<trained_model>"
    if hasattr(obj, "transform") and not isinstance(obj, dict): return "<transformer>"
    return obj


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "FairLens API v1.0",
        "dashboard": "/ui",
        "api_docs": "/docs",
        "available_datasets": list(AVAILABLE_DATASETS.keys()),
    }

@app.get("/datasets")
def list_datasets():
    return {"datasets": [
        {"name": n, "description": d,
         "loaded": n in _cache["datasets"],
         "audited": n in _cache["data_audits"]}
        for n, d in AVAILABLE_DATASETS.items()
    ]}

@app.post("/audit/data/{name}")
def run_data_audit(name: str):
    t0 = time.time()
    from data_auditor import audit_dataset
    result = audit_dataset(_get_dataset(name), verbose=False)
    _cache["data_audits"][name] = result
    return {"status": "ok", "elapsed_seconds": round(time.time()-t0, 2),
            "result": _clean(result)}

@app.post("/audit/model/{name}")
def run_model_audit(name: str, model_type: str = "logistic"):
    if model_type not in ["logistic", "random_forest"]:
        raise HTTPException(400, "model_type must be 'logistic' or 'random_forest'")
    t0 = time.time()
    from model_auditor import audit_model
    result = audit_model(_get_dataset(name), model_type=model_type, verbose=False)
    _cache["model_audits"][name] = result
    return {"status": "ok", "elapsed_seconds": round(time.time()-t0, 2),
            "result": _clean(result)}

@app.post("/audit/full/{name}")
def run_full_audit(name: str, model_type: str = "logistic"):
    t0 = time.time()
    dataset = _get_dataset(name)
    try:
        from data_auditor import audit_dataset
        from model_auditor import audit_model
        from mitigation import apply_all_mitigations
        from report_generator import generate_reports

        da = audit_dataset(dataset, verbose=False)
        ma = audit_model(dataset, model_type=model_type, verbose=False)
        mt = apply_all_mitigations(dataset, ma, verbose=False)
        rp = generate_reports(name, da, ma, mt)

        _cache["data_audits"][name]  = da
        _cache["model_audits"][name] = ma
        _cache["mitigations"][name]  = mt
        _cache["reports"][name]      = rp

        return {
            "status": "ok",
            "elapsed_seconds": round(time.time()-t0, 2),
            "data_bias_score":  da.get("overall_bias_score"),
            "data_severity":    da.get("overall_severity"),
            "model_bias_score": ma.get("bias_score"),
            "model_severity":   ma.get("severity"),
            "mitigation_improvements": {s: r.get("improvement", 0) for s, r in mt.items()},
            "report_json": f"/report/{name}/json",
            "report_pdf":  f"/report/{name}/pdf",
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Full audit failed: {e}\n{traceback.format_exc()}")

@app.get("/report/{name}/json")
def get_json_report(name: str):
    p = OUTPUT_DIR / f"{name}_bias_report.json"
    if not p.exists():
        run_full_audit(name)
    if not p.exists():
        raise HTTPException(404, "Report not found — run /audit/full/{name} first")
    return FileResponse(str(p), media_type="application/json",
                        filename=f"{name}_bias_report.json")

@app.get("/report/{name}/pdf")
def get_pdf_report(name: str):
    p = OUTPUT_DIR / f"{name}_bias_report.pdf"
    if not p.exists():
        run_full_audit(name)
    if not p.exists():
        raise HTTPException(404, "PDF not found — reportlab may not be installed")
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{name}_bias_report.pdf")

@app.get("/results")
def get_all_results():
    summary = {}
    for name in AVAILABLE_DATASETS:
        entry = {"name": name, "loaded": name in _cache["datasets"]}
        if name in _cache["data_audits"]:
            entry["data_bias_score"] = _cache["data_audits"][name].get("overall_bias_score")
            entry["data_severity"]   = _cache["data_audits"][name].get("overall_severity")
        if name in _cache["model_audits"]:
            entry["model_bias_score"] = _cache["model_audits"][name].get("bias_score")
            entry["model_severity"]   = _cache["model_audits"][name].get("severity")
            entry["accuracy"] = _cache["model_audits"][name].get("overall_metrics", {}).get("accuracy")
        summary[name] = entry
    return {"results": summary}

@app.post("/audit/upload")
async def audit_uploaded_csv(
    file: UploadFile = File(...),
    target_column: str = "target",
    protected_column: str = "protected",
    positive_label: str = "1",
    dataset_label: str = "Uploaded dataset",
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    t0 = time.time()
    import io
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    for col in [target_column, protected_column]:
        if col not in df.columns:
            raise HTTPException(400, f"Column '{col}' not found. Columns: {list(df.columns)}")

    if df[target_column].nunique() > 2:
        raise HTTPException(400, f"Target must be binary. Got {df[target_column].nunique()} unique values")

    df["_target_bin"] = (df[target_column].astype(str) == str(positive_label)).astype(int)
    df["_prot_bin"]   = pd.to_numeric(df[protected_column], errors="coerce").fillna(0).astype(int)

    dataset_dict = {
        "df": df, "target": "_target_bin",
        "protected": [protected_column], "binary_protected": "_prot_bin",
        "label": dataset_label, "task": f"Predict {target_column}", "positive_label": 1,
    }

    from data_auditor import audit_dataset
    from model_auditor import audit_model
    from mitigation import apply_all_mitigations

    da = audit_dataset(dataset_dict, verbose=False)
    ma = audit_model(dataset_dict, verbose=False)
    mt = apply_all_mitigations(dataset_dict, ma, verbose=False)

    name = f"custom_{int(time.time())}"
    _cache["datasets"][name]     = dataset_dict
    _cache["data_audits"][name]  = da
    _cache["model_audits"][name] = ma
    _cache["mitigations"][name]  = mt

    return {
        "status": "ok",
        "elapsed_seconds": round(time.time()-t0, 2),
        "n_rows": len(df),
        "data_bias_score":  da.get("overall_bias_score"),
        "data_severity":    da.get("overall_severity"),
        "model_bias_score": ma.get("bias_score"),
        "model_severity":   ma.get("severity"),
        "tpr_gap":          ma.get("tpr_gap"),
        "fpr_gap":          ma.get("fpr_gap"),
        "mitigation_improvements": {s: r.get("improvement", 0) for s, r in mt.items()},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)