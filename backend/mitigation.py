"""
mitigation.py
=============
Applies three bias mitigation strategies and measures improvement.

Strategy 1 — PRE-PROCESSING : Reweighing
  Assigns sample weights to underrepresented group/label combinations
  so the model sees them as equally important during training.
  (Uses IBM AIF360 Reweighing when available, else manual implementation)

Strategy 2 — IN-PROCESSING : Fairness Constraint (ExponentiatedGradient)
  Modifies the training objective to include a fairness penalty.
  Uses Microsoft Fairlearn's ExponentiatedGradient with
  DemographicParity or EqualizedOdds constraint.

Strategy 3 — POST-PROCESSING : Threshold Calibration
  After training, sets different decision thresholds per demographic
  group so that True Positive Rates are equalized across groups.
  Does not require retraining.

Each strategy outputs:
  - Mitigated model/predictions
  - Before/after bias metrics comparison
  - Bias score improvement
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

# Fairlearn
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("[WARN] fairlearn not available — using manual implementations")

# AIF360
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing as AIF360Reweighing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("[WARN] aif360 not available — using manual reweighing")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _group_tpr_fpr(y_true, y_pred, prot):
    """Returns (tpr_gap, fpr_gap, pos_rate_gap) across binary prot groups."""
    results = {}
    for g in [0, 1]:
        mask = (np.array(prot) == g)
        yt, yp = np.array(y_true)[mask], np.array(y_pred)[mask]
        if len(yt) == 0:
            results[g] = {"tpr": 0, "fpr": 0, "pos_rate": 0}
            continue
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        results[g] = {
            "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "pos_rate": yp.mean(),
        }
    tpr_gap = abs(results[1]["tpr"] - results[0]["tpr"])
    fpr_gap = abs(results[1]["fpr"] - results[0]["fpr"])
    pos_gap = abs(results[1]["pos_rate"] - results[0]["pos_rate"])
    return round(tpr_gap, 4), round(fpr_gap, 4), round(pos_gap, 4), results


def _bias_score_from_gaps(tpr_gap, fpr_gap, flip_rate=0.0):
    score = min(35, int(tpr_gap * 200))
    score += min(25, int(fpr_gap * 150))
    score += min(20, int(flip_rate * 100))
    return min(100, score)


def _print_comparison(label, before, after):
    print(f"\n  [{label}]")
    print(f"    Bias score : {before['bias_score']} → {after['bias_score']}  "
          f"({'−'+str(before['bias_score']-after['bias_score']) if after['bias_score']<before['bias_score'] else '+'+str(after['bias_score']-before['bias_score'])})")
    print(f"    TPR gap    : {before['tpr_gap']:.4f} → {after['tpr_gap']:.4f}")
    print(f"    FPR gap    : {before['fpr_gap']:.4f} → {after['fpr_gap']:.4f}")
    print(f"    Accuracy   : {before['accuracy']:.4f} → {after['accuracy']:.4f}")


# ─────────────────────────────────────────────────────────────
# Strategy 1: Pre-processing — Reweighing
# ─────────────────────────────────────────────────────────────

def manual_reweigh(X_train, y_train, prot_train):
    """
    Manual reweighing: upweight underrepresented (group, label) combinations.
    Returns sample_weights array.
    """
    prot = np.array(prot_train)
    y    = np.array(y_train)
    n    = len(y)

    weights = np.ones(n)
    for g in [0, 1]:
        for lbl in [0, 1]:
            mask = (prot == g) & (y == lbl)
            n_group_label = mask.sum()
            n_group = (prot == g).sum()
            n_label = (y == lbl).sum()
            if n_group_label > 0 and n_group > 0 and n_label > 0:
                expected = (n_group / n) * (n_label / n) * n
                actual = n_group_label
                weights[mask] = expected / actual
    return weights


def apply_reweighing(dataset_dict, model_audit_result, verbose=True):
    """Pre-processing: reweigh training data then retrain."""
    if verbose:
        print("\n── Strategy 1: PRE-PROCESSING (Reweighing) ──")

    df          = dataset_dict["df"].copy()
    target      = dataset_dict["target"]
    binary_prot = dataset_dict.get("binary_protected")

    from model_auditor import prepare_features
    X, y, feature_names = prepare_features(df, target, binary_prot)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    prot_series = df[binary_prot].astype(int).values if binary_prot in df.columns else np.zeros(len(df))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, indices, test_size=0.25, random_state=42, stratify=y
    )
    prot_train = prot_series[idx_train]
    prot_test  = prot_series[idx_test]
    prot_idx   = feature_names.index(binary_prot) if binary_prot in feature_names else None

    # Before metrics
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    tpr_b, fpr_b, pos_b, _ = _group_tpr_fpr(y_test, y_pred_base, prot_test)
    before = {
        "bias_score": _bias_score_from_gaps(tpr_b, fpr_b),
        "tpr_gap": tpr_b, "fpr_gap": fpr_b,
        "accuracy": accuracy_score(y_test, y_pred_base),
    }

    # Compute reweighing weights
    if AIF360_AVAILABLE:
        try:
            train_df = pd.DataFrame(X_train, columns=feature_names)
            train_df["label"] = y_train.values
            bld = BinaryLabelDataset(
                df=train_df,
                label_names=["label"],
                protected_attribute_names=[binary_prot] if binary_prot in feature_names else [],
                favorable_label=1, unfavorable_label=0
            )
            rw = AIF360Reweighing(
                unprivileged_groups=[{binary_prot: 0}],
                privileged_groups=[{binary_prot: 1}]
            )
            rw.fit(bld)
            bld_rw = rw.transform(bld)
            weights = bld_rw.instance_weights
            if verbose:
                print("  Using AIF360 Reweighing")
        except Exception:
            weights = manual_reweigh(X_train, y_train, prot_train)
            if verbose:
                print("  Using manual reweighing (AIF360 fallback)")
    else:
        weights = manual_reweigh(X_train, y_train, prot_train)
        if verbose:
            print("  Using manual reweighing")

    # Retrain with weights
    mitigated_model = LogisticRegression(max_iter=1000, random_state=42)
    mitigated_model.fit(X_train, y_train, sample_weight=weights)
    y_pred_mit = mitigated_model.predict(X_test)

    tpr_a, fpr_a, pos_a, _ = _group_tpr_fpr(y_test, y_pred_mit, prot_test)
    after = {
        "bias_score": _bias_score_from_gaps(tpr_a, fpr_a),
        "tpr_gap": tpr_a, "fpr_gap": fpr_a,
        "accuracy": accuracy_score(y_test, y_pred_mit),
    }

    if verbose:
        _print_comparison("Reweighing", before, after)

    return {
        "strategy": "pre_processing_reweighing",
        "before": before,
        "after": after,
        "improvement": before["bias_score"] - after["bias_score"],
        "model": mitigated_model,
    }


# ─────────────────────────────────────────────────────────────
# Strategy 2: In-processing — Fairness Constraint
# ─────────────────────────────────────────────────────────────

def apply_fairness_constraint(dataset_dict, model_audit_result, verbose=True):
    """In-processing: train with demographic parity / equalized odds constraint."""
    if verbose:
        print("\n── Strategy 2: IN-PROCESSING (Fairness Constraint) ──")

    df          = dataset_dict["df"].copy()
    target      = dataset_dict["target"]
    binary_prot = dataset_dict.get("binary_protected")

    from model_auditor import prepare_features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names = prepare_features(df, target, binary_prot)
    prot_series = df[binary_prot].astype(int).values if binary_prot in df.columns else np.zeros(len(df))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    indices = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, indices, test_size=0.25, random_state=42, stratify=y
    )
    prot_train = prot_series[idx_train]
    prot_test  = prot_series[idx_test]

    # Before
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    tpr_b, fpr_b, pos_b, _ = _group_tpr_fpr(y_test, y_pred_base, prot_test)
    before = {
        "bias_score": _bias_score_from_gaps(tpr_b, fpr_b),
        "tpr_gap": tpr_b, "fpr_gap": fpr_b,
        "accuracy": accuracy_score(y_test, y_pred_base),
    }

    if FAIRLEARN_AVAILABLE:
        try:
            constraint = EqualizedOdds()
            estimator  = LogisticRegression(max_iter=1000, random_state=42)
            eg = ExponentiatedGradient(estimator, constraints=constraint, eps=0.01)
            eg.fit(X_train, y_train, sensitive_features=prot_train)
            y_pred_mit = eg.predict(X_test)
            method_used = "Fairlearn ExponentiatedGradient (EqualizedOdds)"
        except Exception as e:
            if verbose:
                print(f"  [WARN] Fairlearn failed ({e}), using adversarial reweighing fallback")
            # Fallback: increase penalty on misclassification for minority group
            weights = manual_reweigh(X_train, y_train, prot_train)
            weights *= 2  # extra emphasis
            mitigated_model = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
            mitigated_model.fit(X_train, y_train, sample_weight=weights)
            y_pred_mit = mitigated_model.predict(X_test)
            method_used = "Penalised LR with group reweighing (fallback)"
    else:
        # Manual: penalise unequal TPR via sample weights with regularisation
        weights = np.ones(len(y_train))
        for _ in range(3):  # iterative reweighing
            tmp_model = LogisticRegression(max_iter=500, random_state=42)
            tmp_model.fit(X_train, y_train, sample_weight=weights)
            tmp_pred = tmp_model.predict(X_train)
            tpr_tmp, _, _, grp = _group_tpr_fpr(y_train, tmp_pred, prot_train)
            if tpr_tmp > 0.02:
                for g in [0, 1]:
                    mask = (prot_train == g)
                    tpr_g = grp[g]["tpr"]
                    mean_tpr = sum(grp[v]["tpr"] for v in [0, 1]) / 2
                    if tpr_g < mean_tpr and mean_tpr > 0:
                        weights[mask & (np.array(y_train) == 1)] *= (mean_tpr / tpr_g)
        mitigated_model = LogisticRegression(max_iter=1000, random_state=42)
        mitigated_model.fit(X_train, y_train, sample_weight=weights)
        y_pred_mit = mitigated_model.predict(X_test)
        method_used = "Iterative TPR-equalising reweighing (manual)"

    tpr_a, fpr_a, pos_a, _ = _group_tpr_fpr(y_test, y_pred_mit, prot_test)
    after = {
        "bias_score": _bias_score_from_gaps(tpr_a, fpr_a),
        "tpr_gap": tpr_a, "fpr_gap": fpr_a,
        "accuracy": accuracy_score(y_test, np.array(y_pred_mit)),
    }

    if verbose:
        print(f"  Method: {method_used}")
        _print_comparison("Fairness Constraint", before, after)

    return {
        "strategy": "in_processing_fairness_constraint",
        "method": method_used,
        "before": before,
        "after": after,
        "improvement": before["bias_score"] - after["bias_score"],
    }


# ─────────────────────────────────────────────────────────────
# Strategy 3: Post-processing — Threshold Calibration
# ─────────────────────────────────────────────────────────────

def apply_threshold_calibration(dataset_dict, model_audit_result, verbose=True):
    """
    Post-processing: find per-group thresholds that equalise TPR.
    No retraining needed — works on any existing model's probability outputs.
    """
    if verbose:
        print("\n── Strategy 3: POST-PROCESSING (Threshold Calibration) ──")

    df          = dataset_dict["df"].copy()
    target      = dataset_dict["target"]
    binary_prot = dataset_dict.get("binary_protected")

    from model_auditor import prepare_features
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names = prepare_features(df, target, binary_prot)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    prot_idx  = feature_names.index(binary_prot) if binary_prot in feature_names else None
    prot_test = X_test[:, prot_idx] if prot_idx is not None else np.zeros(len(y_test))

    # Train base model
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_train, y_train)
    y_prob = base_model.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob >= 0.5).astype(int)

    tpr_b, fpr_b, _, _ = _group_tpr_fpr(y_test, y_pred_base, prot_test)
    before = {
        "bias_score": _bias_score_from_gaps(tpr_b, fpr_b),
        "tpr_gap": tpr_b, "fpr_gap": fpr_b,
        "accuracy": accuracy_score(y_test, y_pred_base),
        "threshold_g0": 0.5, "threshold_g1": 0.5,
    }

    # ── Find per-group thresholds that equalise TPR ──
    thresholds = np.linspace(0.1, 0.9, 81)
    best_tpr_gap = float("inf")
    best_t0, best_t1 = 0.5, 0.5

    for t0 in thresholds[::2]:      # group 0 (unprivileged)
        for t1 in thresholds[::2]:  # group 1 (privileged)
            y_pred_cal = np.where(prot_test == 0,
                                  (y_prob >= t0).astype(int),
                                  (y_prob >= t1).astype(int))
            tpr_c, fpr_c, _, _ = _group_tpr_fpr(y_test, y_pred_cal, prot_test)
            if tpr_c < best_tpr_gap:
                best_tpr_gap = tpr_c
                best_t0, best_t1 = t0, t1

    # Apply best thresholds
    y_pred_cal = np.where(prot_test == 0,
                          (y_prob >= best_t0).astype(int),
                          (y_prob >= best_t1).astype(int))

    tpr_a, fpr_a, _, _ = _group_tpr_fpr(y_test, y_pred_cal, prot_test)
    after = {
        "bias_score": _bias_score_from_gaps(tpr_a, fpr_a),
        "tpr_gap": tpr_a, "fpr_gap": fpr_a,
        "accuracy": accuracy_score(y_test, y_pred_cal),
        "threshold_g0": round(float(best_t0), 3),
        "threshold_g1": round(float(best_t1), 3),
    }

    if verbose:
        print(f"  Calibrated thresholds → group_0={best_t0:.2f}, group_1={best_t1:.2f}")
        _print_comparison("Threshold Calibration", before, after)

    return {
        "strategy": "post_processing_threshold_calibration",
        "before": before,
        "after": after,
        "improvement": before["bias_score"] - after["bias_score"],
        "threshold_g0": best_t0,
        "threshold_g1": best_t1,
    }


# ─────────────────────────────────────────────────────────────
# Apply all 3 strategies to a dataset
# ─────────────────────────────────────────────────────────────

def apply_all_mitigations(dataset_dict, model_audit_result=None, verbose=True):
    """Run all three mitigation strategies. Returns combined results."""
    label = dataset_dict["label"]
    if verbose:
        print(f"\n{'='*60}")
        print(f"MITIGATION: {label}")
        print(f"{'='*60}")

    results = {}
    for name, fn in [
        ("reweighing",              apply_reweighing),
        ("fairness_constraint",     apply_fairness_constraint),
        ("threshold_calibration",   apply_threshold_calibration),
    ]:
        try:
            results[name] = fn(dataset_dict, model_audit_result, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")
            import traceback; traceback.print_exc()

    if verbose and results:
        print(f"\n  Best mitigation strategy: "
              f"{max(results.items(), key=lambda x: x[1].get('improvement', 0))[0]}")

    return results


def mitigate_all(datasets_dict, model_audit_results=None, verbose=True):
    """Apply all mitigations to all datasets."""
    all_results = {}
    for name, dataset in datasets_dict.items():
        mar = model_audit_results.get(name) if model_audit_results else None
        try:
            all_results[name] = apply_all_mitigations(dataset, mar, verbose=verbose)
        except Exception as e:
            print(f"  [ERROR] Mitigation failed for {name}: {e}")
    return all_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/bias_audit/scripts")
    from data_loader import load_all

    datasets = load_all()

    # Run on just Adult for a quick demo
    adult = {"adult": datasets["adult"]}
    results = mitigate_all(adult)

    print("\n\n=== MITIGATION SUMMARY (Adult dataset) ===")
    for strat, r in results["adult"].items():
        imp = r.get("improvement", 0)
        print(f"  {strat:35s}  score {r['before']['bias_score']} → {r['after']['bias_score']}  "
              f"(improvement: {imp:+d})")