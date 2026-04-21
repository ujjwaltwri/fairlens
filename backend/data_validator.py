"""
data_validator.py
=================
Validates uploaded CSVs and named datasets before the audit pipeline runs.

Checks performed:
  1.  File size & row count — too small = unreliable statistics
  2.  Target column exists and is binary
  3.  Protected column exists and is binary or near-binary
  4.  Class balance — target must not be >95% one class
  5.  Group size — each protected group must have enough samples
  6.  Missing value rate — columns with >40% nulls are flagged
  7.  Protected attribute coverage — not >95% one group
  8.  Feature variance — constant columns are flagged
  9.  Duplicate rows — high duplication rate is flagged
  10. Data type sanity — target and protected must be numeric-castable

Returns a ValidationResult dict with:
  valid        : bool — whether pipeline should proceed
  errors       : list of blocking issues (pipeline will not run)
  warnings     : list of non-blocking issues (pipeline runs but results may be unreliable)
  stats        : basic dataset statistics
  suggestions  : actionable fixes for each error/warning
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

MIN_ROWS               = 100     # below this, statistics are unreliable
MIN_ROWS_PER_GROUP     = 30      # minimum samples per protected group
MAX_MISSING_RATE       = 0.40    # 40% nulls in a column = warning
MAX_CLASS_IMBALANCE    = 0.95    # 95% one class = error
MAX_GROUP_IMBALANCE    = 0.95    # 95% one protected group = error
MAX_DUPLICATE_RATE     = 0.30    # 30% duplicate rows = warning
MIN_FEATURE_VARIANCE   = 1e-10   # below this = constant column warning
MAX_FILE_SIZE_MB       = 100     # upper limit for upload


# ─────────────────────────────────────────────────────────────
# Main validator
# ─────────────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    target_column: str,
    protected_column: str,
    positive_label: Optional[str] = None,
    dataset_label: str = "Uploaded dataset",
) -> dict:
    """
    Validate a DataFrame before running the audit pipeline.

    Args:
        df               : the loaded DataFrame
        target_column    : name of the target/outcome column
        protected_column : name of the protected attribute column
        positive_label   : value of target that counts as positive (optional)
        dataset_label    : human-readable name for error messages

    Returns:
        ValidationResult dict — see module docstring
    """
    errors   = []
    warnings = []
    suggestions = []
    stats    = {}

    n_rows, n_cols = df.shape
    stats["n_rows"]    = n_rows
    stats["n_cols"]    = n_cols
    stats["columns"]   = list(df.columns)

    # ── Check 1: Minimum row count ──────────────────────────
    if n_rows < MIN_ROWS:
        errors.append(
            f"Dataset has only {n_rows} rows — minimum {MIN_ROWS} required for reliable statistics."
        )
        suggestions.append(
            f"Collect more data. At least {MIN_ROWS} rows are needed; "
            f"{MIN_ROWS_PER_GROUP * 2} or more is recommended."
        )
    elif n_rows < 500:
        warnings.append(
            f"Dataset has only {n_rows} rows. Results may be statistically unreliable. "
            f"500+ rows recommended."
        )

    # ── Check 2: Target column exists ───────────────────────
    if target_column not in df.columns:
        errors.append(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        suggestions.append(
            f"Check the column name spelling. If the target column has a different name, "
            f"pass it as the 'target_column' parameter."
        )
        # Can't continue without target
        return _build_result(False, errors, warnings, suggestions, stats)

    # ── Check 3: Target is binary ────────────────────────────
    target_series = df[target_column].dropna()
    unique_target = target_series.unique()
    stats["target_unique_values"] = [str(v) for v in unique_target[:10].tolist()]
    stats["target_missing_rate"]  = round(df[target_column].isna().mean(), 4)

    if len(unique_target) < 2:
        errors.append(
            f"Target column '{target_column}' has only one unique value ({unique_target[0]}). "
            f"Cannot train a classifier."
        )
        suggestions.append("Target must have exactly 2 unique values (binary classification).")

    elif len(unique_target) > 2:
        errors.append(
            f"Target column '{target_column}' has {len(unique_target)} unique values "
            f"({unique_target[:5].tolist()}...). Only binary targets are supported."
        )
        suggestions.append(
            f"Binarise the target before uploading. "
            f"For example, if target is 0/1/2, combine classes into 0 vs 1."
        )

    else:
        # Binary — check castability
        try:
            if positive_label:
                binary_target = (target_series.astype(str) == str(positive_label)).astype(int)
            else:
                binary_target = pd.to_numeric(target_series, errors="coerce")
                if binary_target.isna().any():
                    # Try treating as categorical
                    vals = sorted(target_series.unique())
                    binary_target = (target_series == vals[1]).astype(int)

            pos_rate = binary_target.mean()
            stats["target_positive_rate"] = round(float(pos_rate), 4)

            # Check 4: Class balance
            if pos_rate > MAX_CLASS_IMBALANCE:
                errors.append(
                    f"Target is {pos_rate*100:.1f}% positive — severely imbalanced. "
                    f"Max allowed: {MAX_CLASS_IMBALANCE*100:.0f}%."
                )
                suggestions.append(
                    "Try oversampling the minority class (SMOTE) or collecting more "
                    "negative examples before uploading."
                )
            elif pos_rate < (1 - MAX_CLASS_IMBALANCE):
                errors.append(
                    f"Target is only {pos_rate*100:.1f}% positive — severely imbalanced. "
                    f"Min allowed: {(1-MAX_CLASS_IMBALANCE)*100:.0f}%."
                )
                suggestions.append(
                    "Try oversampling the minority class or collecting more positive examples."
                )
            elif pos_rate > 0.80 or pos_rate < 0.20:
                warnings.append(
                    f"Target is {pos_rate*100:.1f}% positive — moderately imbalanced. "
                    f"Fairness metrics may be less reliable."
                )

        except Exception as e:
            errors.append(f"Could not parse target column '{target_column}': {e}")
            suggestions.append(
                "Ensure the target column contains only numeric values (0/1) or "
                "two consistent string labels."
            )

    # ── Check 5: Protected column exists ────────────────────
    if protected_column not in df.columns:
        errors.append(
            f"Protected attribute column '{protected_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        suggestions.append(
            f"Specify the correct column name. Common protected attributes: "
            f"gender, sex, race, age, ethnicity."
        )
        return _build_result(False, errors, warnings, suggestions, stats)

    # ── Check 6: Protected column coverage ──────────────────
    prot_series = df[protected_column].dropna()
    unique_prot = prot_series.unique()
    stats["protected_unique_values"] = [str(v) for v in unique_prot[:10].tolist()]
    stats["protected_missing_rate"]  = round(df[protected_column].isna().mean(), 4)
    stats["n_protected_groups"]      = len(unique_prot)

    if len(unique_prot) < 2:
        errors.append(
            f"Protected column '{protected_column}' has only one unique value. "
            f"Cannot measure group disparities."
        )
        suggestions.append(
            "The protected attribute must have at least 2 groups (e.g. male/female, 0/1)."
        )

    elif len(unique_prot) > 10:
        warnings.append(
            f"Protected column '{protected_column}' has {len(unique_prot)} unique values. "
            f"It will be binarised — this may lose nuance."
        )
        suggestions.append(
            "Consider binarising the protected attribute yourself before uploading "
            "(e.g. age → under_40: 0/1) for more meaningful results."
        )

    else:
        # Check group imbalance
        try:
            prot_numeric = pd.to_numeric(prot_series, errors="coerce")
            if prot_numeric.isna().mean() < 0.1:
                prot_bin = (prot_numeric > prot_numeric.median()).astype(int)
            else:
                vals = sorted(prot_series.unique())
                prot_bin = (prot_series == vals[-1]).astype(int)

            group_rate = prot_bin.mean()
            stats["protected_group1_rate"] = round(float(group_rate), 4)

            if group_rate > MAX_GROUP_IMBALANCE or group_rate < (1 - MAX_GROUP_IMBALANCE):
                errors.append(
                    f"Protected attribute is {group_rate*100:.1f}% one group — "
                    f"too imbalanced to measure disparities reliably."
                )
                suggestions.append(
                    "Ensure both protected groups have sufficient representation. "
                    f"Each group needs at least {MIN_ROWS_PER_GROUP} samples."
                )

            # Check per-group size
            n_group1 = int(prot_bin.sum())
            n_group0 = int(len(prot_bin) - n_group1)
            stats["n_group0"] = n_group0
            stats["n_group1"] = n_group1

            if n_group0 < MIN_ROWS_PER_GROUP:
                errors.append(
                    f"Protected group 0 has only {n_group0} samples — "
                    f"minimum {MIN_ROWS_PER_GROUP} required per group."
                )
                suggestions.append(
                    f"Collect at least {MIN_ROWS_PER_GROUP} samples from each protected group."
                )
            elif n_group0 < 100:
                warnings.append(
                    f"Protected group 0 has only {n_group0} samples. "
                    f"Results may be unreliable — 100+ per group recommended."
                )

            if n_group1 < MIN_ROWS_PER_GROUP:
                errors.append(
                    f"Protected group 1 has only {n_group1} samples — "
                    f"minimum {MIN_ROWS_PER_GROUP} required per group."
                )
            elif n_group1 < 100:
                warnings.append(
                    f"Protected group 1 has only {n_group1} samples. "
                    f"Results may be unreliable — 100+ per group recommended."
                )

        except Exception as e:
            warnings.append(f"Could not analyse protected group distribution: {e}")

    # ── Check 7: Missing value rates ─────────────────────────
    missing_rates = df.isnull().mean()
    high_missing  = missing_rates[missing_rates > MAX_MISSING_RATE]
    stats["missing_rates"] = {
        col: round(float(rate), 4)
        for col, rate in missing_rates[missing_rates > 0].items()
    }

    if target_column in high_missing.index:
        errors.append(
            f"Target column '{target_column}' has "
            f"{high_missing[target_column]*100:.1f}% missing values."
        )
        suggestions.append("Remove rows where the target is missing before uploading.")

    if protected_column in high_missing.index:
        errors.append(
            f"Protected column '{protected_column}' has "
            f"{high_missing[protected_column]*100:.1f}% missing values."
        )
        suggestions.append(
            "Remove rows where the protected attribute is missing, or impute carefully."
        )

    other_high_missing = [
        c for c in high_missing.index
        if c not in [target_column, protected_column]
    ]
    if other_high_missing:
        warnings.append(
            f"Columns with >40% missing values: {other_high_missing}. "
            f"These will be dropped or imputed automatically."
        )

    # ── Check 8: Constant / zero-variance columns ───────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [
        c for c in numeric_cols
        if df[c].nunique() <= 1
    ]
    if constant_cols:
        warnings.append(
            f"Constant columns (zero variance) detected: {constant_cols}. "
            f"These will be dropped — they carry no information."
        )
        stats["constant_columns"] = constant_cols

    # ── Check 9: Duplicate rows ──────────────────────────────
    n_dupes = df.duplicated().sum()
    dupe_rate = n_dupes / max(n_rows, 1)
    stats["n_duplicate_rows"] = int(n_dupes)
    stats["duplicate_rate"]   = round(float(dupe_rate), 4)

    if dupe_rate > MAX_DUPLICATE_RATE:
        warnings.append(
            f"{n_dupes} duplicate rows detected ({dupe_rate*100:.1f}% of data). "
            f"This may inflate bias metrics."
        )
        suggestions.append("Consider deduplicating the dataset before uploading.")

    # ── Check 10: Enough numeric features to train on ────────
    feature_cols = [
        c for c in df.columns
        if c not in [target_column, protected_column]
    ]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    stats["n_numeric_features"]     = len(numeric_features)
    stats["n_categorical_features"] = len(categorical_features)
    stats["feature_columns"]        = feature_cols[:20]  # cap for readability

    if len(numeric_features) + len(categorical_features) < 2:
        errors.append(
            "Dataset has fewer than 2 usable feature columns (excluding target and protected). "
            "Cannot train a meaningful model."
        )
        suggestions.append(
            "Add more columns. The model needs at least 2 features to learn from."
        )

    # High-cardinality categoricals warning
    high_card = [
        c for c in categorical_features
        if df[c].nunique() > 50
    ]
    if high_card:
        warnings.append(
            f"High-cardinality categorical columns will be dropped during training: {high_card}. "
            f"Consider encoding them before uploading."
        )

    # ── Summary stats ────────────────────────────────────────
    stats["n_features_total"]  = len(feature_cols)
    stats["dataset_label"]     = dataset_label

    valid = len(errors) == 0
    return _build_result(valid, errors, warnings, suggestions, stats)


# ─────────────────────────────────────────────────────────────
# Validate named dataset (built-in)
# ─────────────────────────────────────────────────────────────

def validate_named_dataset(dataset_dict: dict) -> dict:
    """
    Lightweight validation for the 5 built-in datasets.
    These are pre-validated but we still check the loaded DataFrame
    hasn't been corrupted (e.g. empty load due to network failure).
    """
    df              = dataset_dict.get("df")
    target          = dataset_dict.get("target")
    binary_protected = dataset_dict.get("binary_protected")
    label           = dataset_dict.get("label", "Unknown")

    errors   = []
    warnings = []

    if df is None or len(df) == 0:
        errors.append(f"Dataset '{label}' loaded empty — check network or local fallback files.")
        return _build_result(False, errors, warnings, [], {"n_rows": 0})

    if target not in df.columns:
        errors.append(f"Target column '{target}' missing from loaded DataFrame.")
        return _build_result(False, errors, warnings, [], {"n_rows": len(df)})

    if binary_protected and binary_protected not in df.columns:
        errors.append(f"Binary protected column '{binary_protected}' missing from DataFrame.")
        return _build_result(False, errors, warnings, [], {"n_rows": len(df)})

    # Check for suspiciously small load
    if len(df) < 100:
        warnings.append(
            f"Dataset loaded with only {len(df)} rows — "
            f"network fallback may have partially failed."
        )

    stats = {
        "n_rows":   len(df),
        "n_cols":   len(df.columns),
        "label":    label,
        "target":   target,
        "protected": binary_protected,
    }

    return _build_result(True, errors, warnings, [], stats)


# ─────────────────────────────────────────────────────────────
# File-level checks (before parsing)
# ─────────────────────────────────────────────────────────────

def validate_csv_file(
    file_bytes: bytes,
    filename: str,
) -> dict:
    """
    Quick checks on the raw file before attempting to parse it.
    Call this before pd.read_csv().

    Returns ValidationResult — if not valid, don't attempt parsing.
    """
    errors   = []
    warnings = []
    stats    = {}

    # Check extension
    if not filename.lower().endswith(".csv"):
        errors.append(f"File '{filename}' is not a CSV. Only .csv files are supported.")
        return _build_result(False, errors, warnings, [], stats)

    # Check file size
    size_mb = len(file_bytes) / (1024 * 1024)
    stats["file_size_mb"] = round(size_mb, 2)

    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(
            f"File size {size_mb:.1f} MB exceeds maximum {MAX_FILE_SIZE_MB} MB. "
            f"Consider sampling your dataset."
        )
        return _build_result(False, errors, warnings, [], stats)

    if size_mb > 50:
        warnings.append(
            f"File size {size_mb:.1f} MB is large. "
            f"The audit may take several minutes."
        )

    # Check it's not empty
    if len(file_bytes) < 50:
        errors.append("File appears to be empty or too small to be a valid CSV.")
        return _build_result(False, errors, warnings, [], stats)

    # Try to detect encoding issues
    try:
        file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        warnings.append(
            "File may have non-UTF-8 encoding. Will attempt to parse anyway — "
            "if it fails, re-save the CSV as UTF-8."
        )

    return _build_result(True, errors, warnings, [], stats)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _build_result(
    valid: bool,
    errors: list,
    warnings: list,
    suggestions: list,
    stats: dict,
) -> dict:
    return {
        "valid":       valid,
        "errors":      errors,
        "warnings":    warnings,
        "suggestions": suggestions,
        "stats":       stats,
        "summary":     (
            "Validation passed." if valid and not warnings
            else f"Validation passed with {len(warnings)} warning(s)." if valid
            else f"Validation failed: {len(errors)} error(s), {len(warnings)} warning(s)."
        ),
    }


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Testing validator with synthetic data...\n")

    # Good dataset
    import numpy as np
    np.random.seed(42)
    n = 1000
    good_df = pd.DataFrame({
        "age":       np.random.randint(18, 65, n),
        "income":    np.random.normal(50000, 15000, n),
        "gender":    np.random.choice([0, 1], n, p=[0.48, 0.52]),
        "education": np.random.choice(["high_school", "college", "graduate"], n),
        "hired":     np.random.choice([0, 1], n, p=[0.55, 0.45]),
    })
    result = validate_dataframe(good_df, "hired", "gender", dataset_label="Good test dataset")
    print("Good dataset:")
    print(json.dumps(result, indent=2))

    # Bad dataset — too small, bad target
    bad_df = pd.DataFrame({
        "feature": range(50),
        "target":  [1] * 50,       # only one class
        "prot":    [0, 1] * 25,
    })
    result2 = validate_dataframe(bad_df, "target", "prot", dataset_label="Bad test dataset")
    print("\nBad dataset:")
    print(json.dumps(result2, indent=2))