"""
api.py
======
FastAPI backend — now with:
  - Supabase persistence (uploads, jobs, audit results)
  - Supabase Storage (CSV files + PDF/JSON reports)
  - Async background jobs (no more blocking 60s requests)
  - Gemini chat endpoint
  - Job status polling
  - Rate limiting via slowapi (10 req/hour/IP on heavy endpoints)

Endpoints:
  GET  /                        — health check
  GET  /datasets                — list available datasets
  GET  /jobs                    — list recent audit jobs
  GET  /jobs/{job_id}           — get job status
  GET  /results                 — list recent audit results (summary)
  GET  /results/{job_id}        — full audit result for a job

  POST /audit/data/{name}       — run data audit (sync, fast)
  POST /audit/model/{name}      — run model audit (sync)
  POST /audit/full/{name}       — full pipeline async → returns job_id immediately
  POST /audit/upload            — upload CSV → full pipeline async → job_id

  GET  /report/{name}/json      — redirect to Supabase Storage JSON URL
  GET  /report/{name}/pdf       — redirect to Supabase Storage PDF URL

  POST /chat/{job_id}           — Gemini Q&A on a completed audit
  POST /chat/dataset/{name}     — Gemini Q&A using latest audit for a named dataset

  GET  /uploads                 — list all uploaded CSVs

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

import sys
import os
import json
import time
import asyncio
import traceback
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

# ── Rate limiting ──────────────────────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

# ── Internal modules ───────────────────────────────────────────────────────────
import database as db_ops
import storage as storage_ops
from gemini import generate_narrative, generate_recommendation, chat as gemini_chat
from data_validator import validate_csv_file, validate_dataframe, validate_named_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FairLens API",
    description="FairLens — AI Bias Detection & Mitigation Platform",
    version="2.0.0",
)

# Register rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_DATASETS = {
    "adult":       "UCI Adult (Census Income) — income prediction, gender + race bias",
    "compas":      "COMPAS Recidivism — criminal justice, racial bias",
    "german":      "German Credit — loan approval, age + gender bias",
    "utrecht":     "Utrecht Fairness Recruitment — hiring, gender + age bias",
    "diabetes130": "Diabetes 130-US Hospitals — medical readmission, age/race/gender bias",
}

# Heavy-endpoint rate limit — 10 full pipeline runs per hour per IP
HEAVY_LIMIT = "10/hour"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _serialise(obj):
    """Recursively make an object JSON-safe for API responses."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, np.bool_):        return bool(obj)
    if hasattr(obj, "predict") or (hasattr(obj, "transform") and not isinstance(obj, dict)):
        return None
    return obj


def _load_dataset(name: str) -> dict:
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
    if name not in loaders:
        raise HTTPException(404, f"Dataset '{name}' not found. Available: {list(loaders)}")
    return loaders[name]()


# ─────────────────────────────────────────────────────────────
# Background pipeline
# ─────────────────────────────────────────────────────────────

async def _run_pipeline_bg(
    job_id: str,
    dataset_name: str,
    dataset_dict: dict,
    model_type: str = "logistic",
):
    """
    Full pipeline run as a background task.
    Updates job status in Supabase at each step.
    Saves result + uploads reports to Supabase Storage.
    """
    try:
        db_ops.update_job_status(job_id, "running")

        from data_auditor import audit_dataset
        from model_auditor import audit_model
        from mitigation import apply_all_mitigations
        from report_generator import generate_reports

        logger.info(f"[Job {job_id}] Running data audit...")
        data_result = await asyncio.to_thread(audit_dataset, dataset_dict, False)

        logger.info(f"[Job {job_id}] Running model audit...")
        model_result = await asyncio.to_thread(
            audit_model, dataset_dict, model_type, False
        )

        logger.info(f"[Job {job_id}] Running mitigation strategies...")
        mit_result = await asyncio.to_thread(
            apply_all_mitigations, dataset_dict, model_result, False
        )

        # Gemini — non-fatal if API key is missing or call fails
        narrative      = None
        recommendation = None
        if os.environ.get("GEMINI_API_KEY"):
            try:
                logger.info(f"[Job {job_id}] Generating Gemini narrative...")
                narrative = await asyncio.to_thread(
                    generate_narrative,
                    dataset_dict["label"],
                    data_result,
                    model_result,
                )
                recommendation = await asyncio.to_thread(
                    generate_recommendation,
                    dataset_dict["label"],
                    model_result,
                    mit_result,
                )
            except Exception as ge:
                logger.warning(f"[Job {job_id}] Gemini failed (non-fatal): {ge}")
        else:
            logger.info(f"[Job {job_id}] GEMINI_API_KEY not set — skipping AI narrative")

        logger.info(f"[Job {job_id}] Generating reports...")
        report_paths = await asyncio.to_thread(
            generate_reports,
            dataset_name,
            data_result,
            model_result,
            mit_result,
            narrative,
            recommendation,
        )

        # Upload reports to Supabase Storage (non-fatal)
        pdf_url = json_url = None
        try:
            pdf_url, json_url = await asyncio.to_thread(
                storage_ops.upload_both_reports, report_paths, dataset_name
            )
        except Exception as se:
            logger.warning(f"[Job {job_id}] Storage upload failed (non-fatal): {se}")

        # Persist to Supabase DB
        db_ops.save_audit_result(
            job_id=job_id,
            dataset_name=dataset_name,
            data_audit=data_result,
            model_audit=model_result,
            mitigation=mit_result,
            gemini_narrative=narrative,
            gemini_recommendation=recommendation,
            report_json_url=json_url,
            report_pdf_url=pdf_url,
        )

        db_ops.update_job_status(job_id, "done", completed=True)
        logger.info(f"[Job {job_id}] Pipeline complete.")

    except Exception as e:
        logger.error(f"[Job {job_id}] Pipeline failed: {e}\n{traceback.format_exc()}")
        db_ops.update_job_status(job_id, "failed", completed=True)


# ─────────────────────────────────────────────────────────────
# Routes — health + info
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    db_ok = db_ops.health_check()
    return {
        "status":             "ok",
        "service":            "FairLens API v2.0",
        "database":           "connected" if db_ok else "error",
        "gemini_enabled":     bool(os.environ.get("GEMINI_API_KEY")),
        "available_datasets": AVAILABLE_DATASETS,
    }


@app.get("/datasets")
def list_datasets():
    return {"datasets": [
        {"name": n, "description": d}
        for n, d in AVAILABLE_DATASETS.items()
    ]}


# ─────────────────────────────────────────────────────────────
# Routes — jobs
# ─────────────────────────────────────────────────────────────

@app.get("/jobs")
def list_jobs():
    return {"jobs": db_ops.list_jobs()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = db_ops.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    if job["status"] == "done":
        result = db_ops.get_audit_result(job_id)
        if result:
            job["data_bias_score"]  = result.get("data_bias_score")
            job["data_severity"]    = result.get("data_severity")
            job["model_bias_score"] = result.get("model_bias_score")
            job["model_severity"]   = result.get("model_severity")
            job["report_pdf_url"]   = result.get("report_pdf_url")
            job["report_json_url"]  = result.get("report_json_url")
    return job


# ─────────────────────────────────────────────────────────────
# Routes — results
# ─────────────────────────────────────────────────────────────

@app.get("/results")
def list_results():
    return {"results": db_ops.list_results()}


@app.get("/results/{job_id}")
def get_result(job_id: str):
    result = db_ops.get_audit_result(job_id)
    if not result:
        raise HTTPException(404, f"No result found for job '{job_id}'")
    return {
        "job_id":                 result["job_id"],
        "dataset_name":           result["dataset_name"],
        "data_bias_score":        result["data_bias_score"],
        "data_severity":          result["data_severity"],
        "model_bias_score":       result["model_bias_score"],
        "model_severity":         result["model_severity"],
        "data_audit":             result.get("data_audit_json"),
        "model_audit":            result.get("model_audit_json"),
        "mitigation":             result.get("mitigation_json"),
        "gemini_narrative":       result.get("gemini_narrative"),
        "gemini_recommendation":  result.get("gemini_recommendation"),
        "report_pdf_url":         result.get("report_pdf_url"),
        "report_json_url":        result.get("report_json_url"),
        "created_at":             result.get("created_at"),
    }


@app.get("/results/dataset/{name}")
def get_result_by_dataset(name: str):
    """Fetch the full audit result using the dataset name for the frontend explorer."""
    result = db_ops.get_latest_result_for_dataset(name)
    if not result:
        raise HTTPException(404, f"No result found for '{name}'")
    return get_result(result["job_id"])


# ─────────────────────────────────────────────────────────────
# Routes — audit (sync, lightweight)
# ─────────────────────────────────────────────────────────────

@app.post("/audit/data/{name}")
def run_data_audit(name: str):
    """Sync data audit — fast, no model training."""
    if name not in AVAILABLE_DATASETS:
        raise HTTPException(404, f"Dataset '{name}' not found")
    t0 = time.time()
    dataset = _load_dataset(name)
    from data_auditor import audit_dataset
    result = audit_dataset(dataset, verbose=False)
    return {
        "status":          "ok",
        "elapsed_seconds": round(time.time() - t0, 2),
        "result":          _serialise(result),
    }


@app.post("/audit/model/{name}")
def run_model_audit(name: str, model_type: str = "logistic"):
    """Sync model audit — trains + evaluates a model."""
    if name not in AVAILABLE_DATASETS:
        raise HTTPException(404, f"Dataset '{name}' not found")
    if model_type not in ["logistic", "random_forest"]:
        raise HTTPException(400, "model_type must be 'logistic' or 'random_forest'")
    t0 = time.time()
    dataset = _load_dataset(name)
    from model_auditor import audit_model
    result = audit_model(dataset, model_type=model_type, verbose=False)
    return {
        "status":          "ok",
        "elapsed_seconds": round(time.time() - t0, 2),
        "result":          _serialise(result),
    }


# ─────────────────────────────────────────────────────────────
# Routes — full pipeline (async + rate-limited)
# ─────────────────────────────────────────────────────────────

@app.post("/audit/full/{name}")
@limiter.limit(HEAVY_LIMIT)
async def run_full_audit(
    request: Request,
    name: str,
    background_tasks: BackgroundTasks,
    model_type: str = "logistic",
):
    """
    Start the full audit pipeline as a background job.
    Rate-limited: 10 requests per hour per IP.
    Returns job_id immediately — poll GET /jobs/{job_id} for status.
    """
    if name not in AVAILABLE_DATASETS:
        raise HTTPException(404, f"Dataset '{name}' not found")
    if model_type not in ["logistic", "random_forest"]:
        raise HTTPException(400, "model_type must be 'logistic' or 'random_forest'")

    job = db_ops.create_job(dataset_name=name, model_type=model_type)
    job_id = job["id"]

    try:
        dataset = _load_dataset(name)
    except Exception as e:
        db_ops.update_job_status(job_id, "failed", completed=True)
        raise HTTPException(500, f"Failed to load dataset: {e}")

    val = validate_named_dataset(dataset)
    if not val["valid"]:
        db_ops.update_job_status(job_id, "failed", completed=True)
        raise HTTPException(422, detail={
            "message": "Dataset validation failed",
            "errors":  val["errors"],
        })

    background_tasks.add_task(
        _run_pipeline_bg, job_id, name, dataset, model_type
    )

    return {
        "status":   "accepted",
        "job_id":   job_id,
        "poll_url": f"/jobs/{job_id}",
        "message":  "Pipeline started. Poll /jobs/{job_id} for status.",
    }


@app.post("/audit/upload")
@limiter.limit(HEAVY_LIMIT)
async def audit_uploaded_csv(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    protected_column: str = Form(...),
    positive_label: str = Form("1"),
    dataset_label: str = Form("Uploaded dataset"),
    model_type: str = Form("logistic"),
):
    """
    Upload a CSV → save to Supabase Storage → run full audit pipeline async.
    Rate-limited: 10 requests per hour per IP.
    Returns job_id immediately.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    content = await file.read()

    # File-level validation (size, encoding, extension)
    file_val = validate_csv_file(content, file.filename)
    if not file_val["valid"]:
        raise HTTPException(422, detail={
            "message":     "File validation failed",
            "errors":      file_val["errors"],
            "suggestions": file_val["suggestions"],
        })

    try:
        import io
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    # Surface available columns in every 422 so the UI can show them
    available_columns = list(df.columns)

    # Quick column-existence check before running full validation —
    # gives a clear, actionable error instead of a cryptic validator message
    missing = []
    if target_column not in df.columns:
        missing.append(f"Target column '{target_column}' not found in CSV")
    if protected_column not in df.columns:
        missing.append(f"Protected column '{protected_column}' not found in CSV")

    if missing:
        raise HTTPException(422, detail={
            "message":           "Column names do not match your CSV",
            "errors":            missing,
            "available_columns": available_columns,
            "suggestions":       [
                f"Your CSV has these columns: {available_columns}",
                "Set target_column and protected_column to match exactly (case-sensitive).",
            ],
        })

    # DataFrame-level validation (columns, balance, group sizes, etc.)
    df_val = validate_dataframe(
        df,
        target_column=target_column,
        protected_column=protected_column,
        positive_label=positive_label,
        dataset_label=dataset_label,
    )
    if not df_val["valid"]:
        raise HTTPException(422, detail={
            "message":           "Data validation failed",
            "errors":            df_val["errors"],
            "warnings":          df_val["warnings"],
            "suggestions":       df_val["suggestions"],
            "stats":             df_val["stats"],
            "available_columns": available_columns,
        })

    validation_warnings = df_val["warnings"]

    # Upload CSV to Supabase Storage (non-fatal)
    storage_path = public_url = ""
    try:
        storage_path, public_url = storage_ops.upload_csv(content, file.filename)
    except Exception as e:
        logger.warning(f"Storage upload failed (continuing without): {e}")

    # Save upload record to DB (non-fatal)
    upload_id = None
    try:
        upload_record = db_ops.save_upload(
            filename=file.filename,
            storage_path=storage_path,
            public_url=public_url,
            row_count=len(df),
            target_col=target_column,
            protected_col=protected_column,
        )
        upload_id = upload_record["id"]
    except Exception as e:
        logger.warning(f"DB upload record failed (continuing): {e}")

    # Build dataset dict
    df["_target_bin"] = (df[target_column].astype(str) == str(positive_label)).astype(int)

    # Binarise protected column — handle both numeric and string (e.g. male/female)
    prot_series = df[protected_column]
    prot_numeric = pd.to_numeric(prot_series, errors="coerce")
    if prot_numeric.isna().mean() > 0.5:
        # String-valued protected attribute (e.g. "male"/"female", "White"/"Black")
        # Encode as 0/1 using sorted unique values — last alphabetically = 1
        prot_vals = sorted(prot_series.dropna().unique().tolist())
        df["_prot_bin"] = (prot_series == prot_vals[-1]).astype(int)
        logger.info(
            f"String protected column '{protected_column}': "
            f"encoded {prot_vals[0]}=0, {prot_vals[-1]}=1"
        )
    else:
        df["_prot_bin"] = prot_numeric.fillna(0).astype(int)

    dataset_dict = {
        "df":               df,
        "target":           "_target_bin",
        "protected":        [protected_column],
        "binary_protected": "_prot_bin",
        "label":            dataset_label,
        "task":             f"Predict {target_column}",
        "positive_label":   1,
    }

    dataset_name = f"upload_{int(time.time())}"
    job = db_ops.create_job(
        dataset_name=dataset_name,
        model_type=model_type,
        upload_id=upload_id,
    )
    job_id = job["id"]

    background_tasks.add_task(
        _run_pipeline_bg, job_id, dataset_name, dataset_dict, model_type
    )

    return {
        "status":               "accepted",
        "job_id":               job_id,
        "upload_id":            upload_id,
        "n_rows":               len(df),
        "validation_warnings":  validation_warnings,
        "poll_url":             f"/jobs/{job_id}",
        "message":              "CSV uploaded and pipeline started. Poll /jobs/{job_id} for status.",
    }


# ─────────────────────────────────────────────────────────────
# Routes — reports (redirect to Supabase Storage URLs)
# ─────────────────────────────────────────────────────────────

@app.get("/report/{name}/json")
def get_json_report(name: str):
    result = db_ops.get_latest_result_for_dataset(name)
    if not result or not result.get("report_json_url"):
        raise HTTPException(
            404, f"No JSON report found for '{name}'. Run /audit/full/{name} first."
        )
    return RedirectResponse(url=result["report_json_url"])


@app.get("/report/{name}/pdf")
def get_pdf_report(name: str):
    result = db_ops.get_latest_result_for_dataset(name)
    if not result or not result.get("report_pdf_url"):
        raise HTTPException(
            404, f"No PDF report found for '{name}'. Run /audit/full/{name} first."
        )
    return RedirectResponse(url=result["report_pdf_url"])


# ─────────────────────────────────────────────────────────────
# Routes — Gemini chat
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[list] = None


@app.post("/chat/{job_id}")
def chat_with_audit(job_id: str, body: ChatRequest):
    """
    Ask Gemini a question about a completed audit.
    Fetches full context from Supabase.

    Body: { "question": "Why is the bias score so high?", "conversation_history": [...] }
    """
    result = db_ops.get_audit_result(job_id)
    if not result:
        raise HTTPException(404, f"No audit result for job '{job_id}'")

    job = db_ops.get_job(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(400, "Audit job is not yet complete")

    data_audit  = result.get("data_audit_json",  {}) or {}
    model_audit = result.get("model_audit_json", {}) or {}
    mit_results = result.get("mitigation_json",  {}) or {}

    answer = gemini_chat(
        question=body.question,
        dataset_label=result.get("dataset_name", "Unknown dataset"),
        data_audit=data_audit,
        model_audit=model_audit,
        mitigation_results=mit_results,
        gemini_narrative=result.get("gemini_narrative"),
        conversation_history=body.conversation_history,
    )

    return {"job_id": job_id, "question": body.question, "answer": answer}


@app.post("/chat/dataset/{name}")
def chat_with_dataset(name: str, body: ChatRequest):
    """
    Same as /chat/{job_id} but looks up the latest audit for a named dataset.
    Convenience endpoint for the dashboard.
    """
    result = db_ops.get_latest_result_for_dataset(name)
    if not result:
        raise HTTPException(404, f"No audit result found for dataset '{name}'")

    data_audit  = result.get("data_audit_json",  {}) or {}
    model_audit = result.get("model_audit_json", {}) or {}
    mit_results = result.get("mitigation_json",  {}) or {}

    answer = gemini_chat(
        question=body.question,
        dataset_label=result.get("dataset_name", name),
        data_audit=data_audit,
        model_audit=model_audit,
        mitigation_results=mit_results,
        gemini_narrative=result.get("gemini_narrative"),
        conversation_history=body.conversation_history,
    )

    return {"dataset_name": name, "question": body.question, "answer": answer}


# ─────────────────────────────────────────────────────────────
# Routes — uploads list
# ─────────────────────────────────────────────────────────────

@app.get("/uploads")
def list_uploads():
    return {"uploads": db_ops.list_uploads()}


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)