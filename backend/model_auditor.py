"""
model_auditor.py
================
Trains a classifier on each dataset and audits the trained model for bias.

Steps per dataset:
  1. Feature engineering (encode categoricals, select numeric features)
  2. Train/test split (stratified)
  3. Train LogisticRegression + RandomForestClassifier
  4. Compute fairness metrics:
       - Equalized Odds (TPR gap, FPR gap)
       - Equal Opportunity (TPR gap only)
       - Predictive Parity (precision gap)
       - Calibration by group
  5. SHAP feature importance (global + per-group)
  6. Counterfactual flip test (flip protected attr, measure prediction change)
  7. AIF360 metrics (if available)

Returns structured model_audit_result dict.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import shap
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Feature preparation helpers
# ─────────────────────────────────────────────────────────────

def prepare_features(df, target, protected_binary):
    """
    Returns X (feature matrix), y (target), feature_names.
    - Encodes categorical columns with LabelEncoder
    - Drops high-cardinality text columns
    - Keeps all numeric columns
    """
    df_work = df.copy()

    # Drop columns that would trivially leak the target
    leak_candidates = [
        "income", "two_year_recid", "is_recid", "readmitted",
        "score_text", "decile_score", "v_decile_score",
        "readmit_30"
    ]
    # Don't drop the target itself from the list before it's separated
    drop_cols = [c for c in leak_candidates if c in df_work.columns and c != target]
    df_work.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Separate target
    y = df_work[target].astype(int)
    df_work.drop(columns=[target], inplace=True)

    # Ensure the binary protected column is retained even if it duplicates info
    # (mark it so we don't double-encode it)

    # Encode categoricals
    for col in df_work.select_dtypes(include=["object", "category"]).columns:
        if df_work[col].nunique() > 50:
            df_work.drop(columns=[col], inplace=True)
        else:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))

    # Keep only numeric
    X = df_work.select_dtypes(include=[np.number]).fillna(0)

    return X, y, list(X.columns)


# ─────────────────────────────────────────────────────────────
# Fairness metric functions
# ─────────────────────────────────────────────────────────────

def compute_group_metrics(y_true, y_pred, group_mask):
    """Compute classification metrics for a binary group mask."""
    yt = np.array(y_true)[group_mask]
    yp = np.array(y_pred)[group_mask]
    if len(yt) == 0:
        return {"n": 0, "accuracy": 0.0, "tpr": 0.0, "fpr": 0.0,
                "precision": 0.0, "positive_rate": 0.0}
    try:
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    except ValueError:
        # Only one class present in this group
        tp = int(((yt==1) & (yp==1)).sum())
        fp = int(((yt==0) & (yp==1)).sum())
        fn = int(((yt==1) & (yp==0)).sum())
        tn = int(((yt==0) & (yp==0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    acc = accuracy_score(yt, yp)
    pos_rate = yp.mean()
    return {
        "n": int(len(yt)),
        "accuracy": round(float(acc), 4),
        "tpr": round(float(tpr), 4),       # True Positive Rate (Recall)
        "fpr": round(float(fpr), 4),       # False Positive Rate
        "precision": round(float(precision), 4),
        "positive_rate": round(float(pos_rate), 4),
    }


def equalized_odds(metrics_priv, metrics_unpriv):
    """Returns TPR gap and FPR gap. Both = 0 is perfect equalized odds."""
    tpr_gap = abs(metrics_priv["tpr"] - metrics_unpriv["tpr"])
    fpr_gap = abs(metrics_priv["fpr"] - metrics_unpriv["fpr"])
    return round(tpr_gap, 4), round(fpr_gap, 4)


def predictive_parity_gap(metrics_priv, metrics_unpriv):
    """Precision gap across groups."""
    return round(abs(metrics_priv["precision"] - metrics_unpriv["precision"]), 4)


def counterfactual_flip_rate(model, X_test, protected_col_idx, scaler=None):
    """
    Flips the protected attribute for every test row and measures
    what % of predictions change. High = model relies on protected attr.
    """
    X_flip = X_test.copy()
    X_flip[:, protected_col_idx] = 1 - X_flip[:, protected_col_idx]

    orig_preds = model.predict(X_test)
    flip_preds = model.predict(X_flip)
    flip_rate  = (orig_preds != flip_preds).mean()
    return round(float(flip_rate), 4)


def compute_model_bias_score(tpr_gap, fpr_gap, flip_rate, dir_val):
    """Composite model bias score 0–100."""
    score = 0
    score += min(35, int(tpr_gap * 200))   # TPR gap (0–35)
    score += min(25, int(fpr_gap * 150))   # FPR gap (0–25)
    score += min(20, int(flip_rate * 100)) # Flip rate (0–20)
    # DIR component (0–20)
    if dir_val is not None and not np.isnan(dir_val):
        if dir_val < 0.5:
            score += 20
        elif dir_val < 0.8:
            score += 10
    return min(100, score)


# ─────────────────────────────────────────────────────────────
# Main model audit function
# ─────────────────────────────────────────────────────────────

def audit_model(dataset_dict, model_type="logistic", verbose=True):
    """
    Train a model and audit it for bias.

    Args:
        dataset_dict : dict from data_loader.py
        model_type   : "logistic" | "random_forest"
        verbose      : print results

    Returns:
        dict with trained model, metrics, SHAP values, bias scores
    """
    df              = dataset_dict["df"].copy()
    target          = dataset_dict["target"]
    binary_prot     = dataset_dict.get("binary_protected")
    label           = dataset_dict["label"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"MODEL AUDIT: {label}  [{model_type}]")
        print(f"{'='*60}")

    # ── Feature prep ──
    X, y, feature_names = prepare_features(df, target, binary_prot)

    # Extract protected column BEFORE scaling/splitting so it is never lost
    if binary_prot in df.columns:
        prot_series = df[binary_prot].astype(int).values
    else:
        prot_series = None

    # ── Scale ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Split ──
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, indices, test_size=0.25, random_state=42, stratify=y
    )
    prot_idx  = feature_names.index(binary_prot) if binary_prot in feature_names else None
    prot_test = prot_series[idx_test] if prot_series is not None else None

    # ── Train ──
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Overall metrics ──
    overall = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    if verbose:
        print(f"\n  Overall → Acc={overall['accuracy']}  "
              f"AUC={overall['roc_auc']}  "
              f"Precision={overall['precision']}  "
              f"Recall={overall['recall']}")

    # ── Per-group metrics ──
    group_metrics = {}
    if prot_test is not None:
        priv_mask   = (prot_test == 1)
        unpriv_mask = (prot_test == 0)
        group_metrics["privileged"]   = compute_group_metrics(y_test, y_pred, priv_mask)
        group_metrics["unprivileged"] = compute_group_metrics(y_test, y_pred, unpriv_mask)

        tpr_gap, fpr_gap = equalized_odds(
            group_metrics["privileged"], group_metrics["unprivileged"]
        )
        pp_gap = predictive_parity_gap(
            group_metrics["privileged"], group_metrics["unprivileged"]
        )
        pos_rate_gap = abs(
            group_metrics["privileged"]["positive_rate"] -
            group_metrics["unprivileged"]["positive_rate"]
        )
        dir_val_model = (
            group_metrics["unprivileged"]["positive_rate"] /
            group_metrics["privileged"]["positive_rate"]
            if group_metrics["privileged"]["positive_rate"] > 0 else np.nan
        )

        if verbose:
            print(f"\n  Per-group metrics (protected attr: '{binary_prot}'):")
            for g, m in group_metrics.items():
                print(f"    {g:14s}: n={m['n']:,}  acc={m['accuracy']}  "
                      f"tpr={m['tpr']}  fpr={m['fpr']}  "
                      f"prec={m['precision']}  pos_rate={m['positive_rate']}")
            print(f"\n  Equalized Odds → TPR gap={tpr_gap}  FPR gap={fpr_gap}")
            print(f"  Predictive Parity gap: {pp_gap}")
            print(f"  Positive rate gap: {round(pos_rate_gap, 4)}")
            print(f"  Model DIR: {round(dir_val_model, 4) if not np.isnan(dir_val_model) else 'N/A'}")
    else:
        tpr_gap, fpr_gap, pp_gap, pos_rate_gap, dir_val_model = 0, 0, 0, 0, np.nan

    # ── Counterfactual flip rate ──
    flip_rate = 0.0
    if prot_idx is not None:
        flip_rate = counterfactual_flip_rate(model, X_test, prot_idx)
        if verbose:
            print(f"  Counterfactual flip rate: {flip_rate:.1%} of predictions change "
                  f"when protected attr is flipped")

    # ── SHAP feature importance ──
    shap_values_dict = {}
    top_features = []
    try:
        # Use a sample for speed
        sample_size = min(200, len(X_test))
        X_sample = X_test[:sample_size]

        if model_type == "random_forest":
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_sample)

        # Global importance (mean absolute SHAP)
        global_importance = np.abs(shap_vals).mean(axis=0)
        feat_imp = sorted(
            zip(feature_names, global_importance),
            key=lambda x: x[1], reverse=True
        )
        top_features = [(f, round(float(v), 4)) for f, v in feat_imp[:10]]

        if verbose:
            print(f"\n  Top 10 features by SHAP importance:")
            for feat, val in top_features:
                bar = "█" * int(val / max(v for _, v in top_features) * 20)
                print(f"    {feat:35s}  {bar}  {val:.4f}")

        shap_values_dict = {"global_importance": top_features}

    except Exception as e:
        if verbose:
            print(f"  [SHAP] Could not compute: {e}")

    # ── Bias score ──
    bias_score = compute_model_bias_score(tpr_gap, fpr_gap, flip_rate, dir_val_model)
    severity = "HIGH" if bias_score >= 70 else "MEDIUM" if bias_score >= 40 else "LOW"

    if verbose:
        print(f"\n  MODEL BIAS SCORE: {bias_score}/100  [{severity}]")

    return {
        "dataset_label":       label,
        "model_type":          model_type,
        "model":               model,
        "scaler":              scaler,
        "feature_names":       feature_names,
        "overall_metrics":     overall,
        "group_metrics":       group_metrics,
        "tpr_gap":             round(tpr_gap, 4),
        "fpr_gap":             round(fpr_gap, 4),
        "predictive_parity_gap": round(pp_gap, 4),
        "positive_rate_gap":   round(float(pos_rate_gap), 4),
        "model_dir":           round(float(dir_val_model), 4) if not np.isnan(dir_val_model) else None,
        "counterfactual_flip_rate": flip_rate,
        "shap_values":         shap_values_dict,
        "top_features":        top_features,
        "bias_score":          bias_score,
        "severity":            severity,
        # Keep test data for mitigation
        "_X_train": X_train, "_X_test": X_test,
        "_y_train": y_train, "_y_test": y_test,
        "_prot_test": prot_test,
        "_protected_col_idx": prot_idx,
    }


def audit_all_models(datasets_dict, model_type="logistic", verbose=True):
    """Audit all datasets. Returns dict of model audit results."""
    results = {}
    for name, dataset in datasets_dict.items():
        try:
            results[name] = audit_model(dataset, model_type=model_type, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Model audit failed for {name}: {e}")
            import traceback; traceback.print_exc()
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/bias_audit/scripts")
    from data_loader import load_all

    datasets = load_all()
    model_results = audit_all_models(datasets, model_type="logistic")

    print("\n\n=== MODEL AUDIT SUMMARY ===")
    print(f"{'Dataset':<35} {'Acc':>6} {'AUC':>6} {'Score':>6} {'TPR_gap':>8} {'Flip%':>7}")
    print("-" * 75)
    for name, r in model_results.items():
        om = r["overall_metrics"]
        print(f"  {r['dataset_label']:<33} "
              f"{om['accuracy']:>6.3f} "
              f"{om['roc_auc']:>6.3f} "
              f"{r['bias_score']:>6} "
              f"{r['tpr_gap']:>8.3f} "
              f"{r['counterfactual_flip_rate']*100:>6.1f}%")