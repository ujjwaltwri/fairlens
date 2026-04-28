"""
database.py
===========
Supabase client and all database operations for FairLens.

Tables:
  uploads      — uploaded CSVs (metadata + storage path)
  audit_jobs   — job status tracking (pending / running / done / failed)
  audit_results — full audit output stored as JSONB

Setup:
  Set these env vars (or put in .env):
    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-service-role-key   ← use service role, not anon
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
import math
import numpy as np
import pandas as pd  # <-- Added pandas import
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────

def get_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set as environment variables."
        )
    return create_client(url, key)


# Singleton — reused across requests
_client: Optional[Client] = None

def db() -> Client:
    global _client
    if _client is None:
        _client = get_client()
    return _client


# ─────────────────────────────────────────────────────────────
# uploads table
# ─────────────────────────────────────────────────────────────

def save_upload(
    filename: str,
    storage_path: str,
    public_url: str,
    row_count: int,
    target_col: str,
    protected_col: str,
) -> dict:
    """
    Insert a new upload record. Returns the created row.
    """
    payload = {
        "filename":     filename,
        "storage_path": storage_path,
        "public_url":   public_url,
        "row_count":    row_count,
        "target_col":   target_col,
        "protected_col": protected_col,
    }
    res = db().table("uploads").insert(payload).execute()
    if not res.data:
        raise RuntimeError(f"Failed to save upload: {res}")
    logger.info(f"[DB] Upload saved: {filename} ({row_count} rows)")
    return res.data[0]


def get_upload(upload_id: str) -> Optional[dict]:
    """Fetch a single upload record by UUID."""
    res = db().table("uploads").select("*").eq("id", upload_id).execute()
    return res.data[0] if res.data else None


def list_uploads(limit: int = 50) -> list:
    """List recent uploads, newest first."""
    res = (
        db().table("uploads")
        .select("*")
        .order("uploaded_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


# ─────────────────────────────────────────────────────────────
# audit_jobs table
# ─────────────────────────────────────────────────────────────

def create_job(
    dataset_name: str,
    model_type: str = "logistic",
    upload_id: Optional[str] = None,
) -> dict:
    """
    Create a new audit job in 'pending' state. Returns the job row.
    """
    payload = {
        "dataset_name": dataset_name,
        "model_type":   model_type,
        "status":       "pending",
        "upload_id":    upload_id,
    }
    res = db().table("audit_jobs").insert(payload).execute()
    if not res.data:
        raise RuntimeError(f"Failed to create job: {res}")
    job = res.data[0]
    logger.info(f"[DB] Job created: {job['id']} for dataset '{dataset_name}'")
    return job


def update_job_status(
    job_id: str,
    status: str,
    completed: bool = False,
) -> dict:
    """
    Update job status. status: 'pending' | 'running' | 'done' | 'failed'
    """
    payload: dict = {"status": status}
    if completed:
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    res = (
        db().table("audit_jobs")
        .update(payload)
        .eq("id", job_id)
        .execute()
    )
    if not res.data:
        raise RuntimeError(f"Failed to update job {job_id}: {res}")
    return res.data[0]


def get_job(job_id: str) -> Optional[dict]:
    """Fetch a single job by UUID."""
    res = db().table("audit_jobs").select("*").eq("id", job_id).execute()
    return res.data[0] if res.data else None


def list_jobs(limit: int = 50) -> list:
    """List recent audit jobs, newest first."""
    res = (
        db().table("audit_jobs")
        .select("*, audit_results(data_bias_score, model_bias_score, data_severity, model_severity)")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


# ─────────────────────────────────────────────────────────────
# audit_results table
# ─────────────────────────────────────────────────────────────

def _serialise_for_pg(obj):
    """Recursively make an object safe for Postgres JSONB."""
    if isinstance(obj, dict):
        return {k: _serialise_for_pg(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_serialise_for_pg(i) for i in obj]
        
    # --- Pandas handling added here ---
    if isinstance(obj, pd.Series):
        return _serialise_for_pg(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _serialise_for_pg(obj.to_dict(orient="records"))
    # ----------------------------------

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v  # nan/inf → NULL
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj  # catch plain Python nan/inf too
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # Skip sklearn objects
    if hasattr(obj, "predict") or (hasattr(obj, "transform") and not isinstance(obj, dict)):
        return None
    return obj


def save_audit_result(
    job_id: str,
    dataset_name: str,
    data_audit: dict,
    model_audit: dict,
    mitigation: dict,
    gemini_narrative: Optional[str] = None,
    gemini_recommendation: Optional[str] = None,
    report_json_url: Optional[str] = None,
    report_pdf_url: Optional[str] = None,
) -> dict:
    """
    Persist the full audit result for a job.
    Large nested dicts are stored as JSONB columns.
    """
    payload = {
        "job_id":          job_id,
        "dataset_name":    dataset_name,
        "data_bias_score": data_audit.get("overall_bias_score"),
        "data_severity":   data_audit.get("overall_severity"),
        "model_bias_score": model_audit.get("bias_score"),
        "model_severity":   model_audit.get("severity"),
        "data_audit_json":  _serialise_for_pg(data_audit),
        "model_audit_json": _serialise_for_pg(model_audit),
        "mitigation_json":  _serialise_for_pg(mitigation),
        "gemini_narrative":      gemini_narrative,
        "gemini_recommendation": gemini_recommendation,
        "report_json_url": report_json_url,
        "report_pdf_url":  report_pdf_url,
    }
    res = db().table("audit_results").insert(payload).execute()
    if not res.data:
        raise RuntimeError(f"Failed to save audit result for job {job_id}: {res}")
    logger.info(
        f"[DB] Audit result saved: job={job_id} "
        f"data={payload['data_bias_score']} model={payload['model_bias_score']}"
    )
    return res.data[0]


def get_audit_result(job_id: str) -> Optional[dict]:
    """Fetch audit result by job UUID."""
    res = (
        db().table("audit_results")
        .select("*")
        .eq("job_id", job_id)
        .execute()
    )
    return res.data[0] if res.data else None


def get_latest_result_for_dataset(dataset_name: str) -> Optional[dict]:
    """
    Get the most recent audit result for a named dataset (e.g. 'adult').
    Used by the /chat endpoint to fetch context without a job_id.
    """
    res = (
        db().table("audit_results")
        .select("*")
        .eq("dataset_name", dataset_name)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def list_results(limit: int = 50) -> list:
    """List recent audit results with summary fields only."""
    res = (
        db().table("audit_results")
        .select(
            "id, job_id, dataset_name, data_bias_score, data_severity, "
            "model_bias_score, model_severity, report_json_url, report_pdf_url, created_at"
        )
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

def health_check() -> bool:
    """Returns True if Supabase connection is alive."""
    try:
        db().table("audit_jobs").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"[DB] Health check failed: {e}")
        return False