"""
storage.py
==========
Supabase Storage operations for FairLens.

Buckets (create these in Supabase dashboard → Storage):
  fairlens-uploads   — uploaded CSVs from /audit/upload
  fairlens-reports   — generated PDF and JSON reports

All buckets should be set to PUBLIC so the frontend can
display/download reports without auth headers.

Env vars required (same as database.py):
  SUPABASE_URL
  SUPABASE_KEY
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional

from database import db

logger = logging.getLogger(__name__)

UPLOADS_BUCKET = "fairlens-uploads"
REPORTS_BUCKET = "fairlens-reports"


# ─────────────────────────────────────────────────────────────
# CSV uploads
# ─────────────────────────────────────────────────────────────

def upload_csv(
    file_bytes: bytes,
    filename: str,
    content_type: str = "text/csv",
) -> tuple[str, str]:
    """
    Upload a CSV file to Supabase Storage.

    Returns:
        (storage_path, public_url)

    storage_path is what you store in the DB.
    public_url is what the frontend uses to download.
    """
    import time
    # Prefix with timestamp to avoid collisions
    storage_path = f"uploads/{int(time.time())}_{filename}"

    db().storage.from_(UPLOADS_BUCKET).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": content_type},
    )

    public_url = _get_public_url(UPLOADS_BUCKET, storage_path)
    logger.info(f"[Storage] CSV uploaded: {storage_path}")
    return storage_path, public_url


def download_csv_as_bytes(storage_path: str) -> bytes:
    """
    Download a CSV from Supabase Storage and return raw bytes.
    Used by data_loader.load_from_supabase().
    """
    data = db().storage.from_(UPLOADS_BUCKET).download(storage_path)
    logger.info(f"[Storage] CSV downloaded: {storage_path}")
    return data


# ─────────────────────────────────────────────────────────────
# Report uploads (PDF + JSON)
# ─────────────────────────────────────────────────────────────

def upload_report_pdf(local_path: str, dataset_name: str) -> Optional[str]:
    """
    Upload a locally-generated PDF report to Supabase Storage.
    Returns public URL or None if file doesn't exist.
    """
    p = Path(local_path)
    if not p.exists():
        logger.warning(f"[Storage] PDF not found at {local_path} — skipping upload")
        return None

    storage_path = f"reports/{dataset_name}_bias_report.pdf"
    with open(p, "rb") as f:
        file_bytes = f.read()

    # Upsert — overwrite if already exists
    try:
        db().storage.from_(REPORTS_BUCKET).remove([storage_path])
    except Exception:
        pass  # doesn't exist yet, that's fine

    db().storage.from_(REPORTS_BUCKET).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/pdf"},
    )

    url = _get_public_url(REPORTS_BUCKET, storage_path)
    logger.info(f"[Storage] PDF report uploaded: {storage_path}")
    return url


def upload_report_json(local_path: str, dataset_name: str) -> Optional[str]:
    """
    Upload a locally-generated JSON report to Supabase Storage.
    Returns public URL or None if file doesn't exist.
    """
    p = Path(local_path)
    if not p.exists():
        logger.warning(f"[Storage] JSON not found at {local_path} — skipping upload")
        return None

    storage_path = f"reports/{dataset_name}_bias_report.json"
    with open(p, "rb") as f:
        file_bytes = f.read()

    try:
        db().storage.from_(REPORTS_BUCKET).remove([storage_path])
    except Exception:
        pass

    db().storage.from_(REPORTS_BUCKET).upload(
        path=storage_path,
        file=file_bytes,
        file_options={"content-type": "application/json"},
    )

    url = _get_public_url(REPORTS_BUCKET, storage_path)
    logger.info(f"[Storage] JSON report uploaded: {storage_path}")
    return url


def upload_both_reports(
    report_paths: dict,
    dataset_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Convenience wrapper — uploads both PDF and JSON from report_generator output.

    Args:
        report_paths: dict with keys 'pdf' and 'json' (local file paths)
        dataset_name: used to name the storage object

    Returns:
        (pdf_url, json_url)
    """
    pdf_url  = upload_report_pdf(report_paths.get("pdf", ""), dataset_name)
    json_url = upload_report_json(report_paths.get("json", ""), dataset_name)
    return pdf_url, json_url


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _get_public_url(bucket: str, storage_path: str) -> str:
    """
    Build the public URL for a Supabase Storage object.
    Works for public buckets — no signed URL needed.
    """
    res = db().storage.from_(bucket).get_public_url(storage_path)
    return res