"""
data_auditor.py
===============
Audits raw datasets for statistical bias BEFORE any model is trained.

Checks performed:
  1. Disparate Impact Ratio (four-fifths / 80% rule)
  2. Demographic Parity Gap
  3. Class Imbalance by group
  4. Proxy Feature Detection (correlation with protected attrs)
  5. Statistical significance (chi-square test)

Returns a structured audit_result dict and prints a summary report.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────

def disparate_impact_ratio(df, target, protected_binary):
    """
    DIR = P(positive | unprivileged) / P(positive | privileged)
    Below 0.8 = legally flagged (four-fifths rule, US EEOC).
    """
    priv   = df[df[protected_binary] == 1][target].mean()
    unpriv = df[df[protected_binary] == 0][target].mean()
    if priv == 0:
        return np.nan
    return round(unpriv / priv, 4)


def demographic_parity_gap(df, target, protected_binary):
    """
    DPG = |P(pos | group=1) - P(pos | group=0)|
    Ideal = 0.0. Above 0.1 is generally considered concerning.
    """
    p1 = df[df[protected_binary] == 1][target].mean()
    p0 = df[df[protected_binary] == 0][target].mean()
    return round(abs(p1 - p0), 4), round(p1, 4), round(p0, 4)


def class_imbalance_by_group(df, target, protected_binary):
    """Returns representation of each protected group per label."""
    result = {}
    for g_val in [0, 1]:
        group = df[df[protected_binary] == g_val]
        label = "privileged" if g_val == 1 else "unprivileged"
        result[label] = {
            "n": len(group),
            "pct_of_total": round(len(group) / len(df) * 100, 2),
            "positive_rate": round(group[target].mean() * 100, 2),
            "negative_rate": round((1 - group[target].mean()) * 100, 2),
        }
    return result


def chi_square_test(df, target, protected_binary):
    """
    Chi-square test of independence between protected attribute and target.
    Returns p-value. p < 0.05 = statistically significant association.
    """
    ct = pd.crosstab(df[protected_binary], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return round(float(chi2), 4), round(float(p), 6), dof


def proxy_feature_detection(df, target, protected_binary, top_n=5):
    """
    Detects features that are highly correlated with the protected attribute.
    These are potential proxy discriminators.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target, protected_binary]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    correlations = []
    for col in feature_cols:
        if df[col].nunique() > 1:
            try:
                corr = abs(df[col].corr(df[protected_binary]))
                if not np.isnan(corr):
                    correlations.append((col, round(corr, 4)))
            except Exception:
                pass

    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations[:top_n]


def compute_bias_score(dir_val, dpg_val, chi2_pval):
    """
    Composite bias score 0–100.
    Higher = more biased.
    """
    score = 0

    # Disparate impact (0–40 points)
    if dir_val is not None and not np.isnan(dir_val):
        if dir_val < 0.5:
            score += 40
        elif dir_val < 0.6:
            score += 32
        elif dir_val < 0.7:
            score += 24
        elif dir_val < 0.8:
            score += 16
        elif dir_val < 0.9:
            score += 8
        else:
            score += 0

    # Demographic parity gap (0–40 points)
    if dpg_val is not None and not np.isnan(dpg_val):
        score += min(40, int(dpg_val * 200))

    # Statistical significance (0–20 points)
    if chi2_pval is not None:
        if chi2_pval < 0.001:
            score += 20
        elif chi2_pval < 0.01:
            score += 14
        elif chi2_pval < 0.05:
            score += 8

    return min(100, score)


def severity_label(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────────────────────
# Main audit function
# ─────────────────────────────────────────────────────────────

def audit_dataset(dataset_dict, verbose=True):
    """
    Run full data audit on a dataset dict from data_loader.py.

    Returns:
        audit_result : dict with all metrics and scores
    """
    df              = dataset_dict["df"]
    target          = dataset_dict["target"]
    protected_attrs = dataset_dict["protected"]
    binary_protected= dataset_dict.get("binary_protected", protected_attrs[0])
    label           = dataset_dict["label"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"DATA AUDIT: {label}")
        print(f"{'='*60}")
        print(f"Rows: {len(df):,}  |  Target: '{target}'  |  Protected: {protected_attrs}")

    results = {}

    for prot_attr in [binary_protected]:
        if prot_attr not in df.columns:
            if verbose:
                print(f"  [SKIP] Column '{prot_attr}' not found")
            continue

        # Ensure no NaNs in key columns
        sub = df[[prot_attr, target]].dropna()
        sub[target] = sub[target].astype(int)
        sub[prot_attr] = sub[prot_attr].astype(int)

        dir_val = disparate_impact_ratio(sub, target, prot_attr)
        dpg_val, p1, p0 = demographic_parity_gap(sub, target, prot_attr)
        group_stats = class_imbalance_by_group(sub, target, prot_attr)
        chi2, pval, dof = chi_square_test(sub, target, prot_attr)
        proxies = proxy_feature_detection(df, target, prot_attr)
        bias_score = compute_bias_score(dir_val, dpg_val, pval)
        severity = severity_label(bias_score)

        result = {
            "protected_attr":       prot_attr,
            "disparate_impact_ratio": dir_val,
            "dir_pass":             dir_val >= 0.8 if dir_val is not None else None,
            "demographic_parity_gap": dpg_val,
            "positive_rate_privileged":   p1,
            "positive_rate_unprivileged": p0,
            "group_stats":          group_stats,
            "chi2_statistic":       chi2,
            "chi2_pvalue":          pval,
            "chi2_significant":     pval < 0.05,
            "top_proxy_features":   proxies,
            "bias_score":           bias_score,
            "severity":             severity,
        }
        results[prot_attr] = result

        if verbose:
            _print_attr_results(result)

    overall_score = int(np.mean([r["bias_score"] for r in results.values()])) if results else 0

    audit_result = {
        "dataset_label":  label,
        "n_rows":         len(df),
        "target":         target,
        "attribute_results": results,
        "overall_bias_score": overall_score,
        "overall_severity":   severity_label(overall_score),
    }

    if verbose:
        print(f"\n  OVERALL BIAS SCORE: {overall_score}/100  [{severity_label(overall_score)}]")

    return audit_result


def _print_attr_results(r):
    print(f"\n  Protected attribute: '{r['protected_attr']}'")
    print(f"  ┌─ Disparate Impact Ratio : {r['disparate_impact_ratio']} "
          f"({'PASS ✓' if r['dir_pass'] else 'FAIL ✗ (< 0.80)'})")
    print(f"  ├─ Demographic Parity Gap : {r['demographic_parity_gap']}")
    print(f"  │    Privileged   pos rate : {r['positive_rate_privileged']*100:.1f}%")
    print(f"  │    Unprivileged pos rate : {r['positive_rate_unprivileged']*100:.1f}%")
    print(f"  ├─ Chi-square p-value      : {r['chi2_pvalue']} "
          f"({'significant ✗' if r['chi2_significant'] else 'not significant ✓'})")
    print(f"  ├─ Group stats:")
    for g, s in r["group_stats"].items():
        print(f"  │    {g:14s}: n={s['n']:,}  "
              f"({s['pct_of_total']:.1f}% of data)  "
              f"pos_rate={s['positive_rate']:.1f}%")
    if r["top_proxy_features"]:
        print(f"  ├─ Top proxy features (corr w/ protected attr):")
        for feat, corr in r["top_proxy_features"]:
            bar = "█" * int(corr * 20)
            print(f"  │    {feat:30s}  {bar}  {corr:.3f}")
    print(f"  └─ Bias Score: {r['bias_score']}/100  [{r['severity']}]")


# ─────────────────────────────────────────────────────────────
# Audit all datasets
# ─────────────────────────────────────────────────────────────

def audit_all(datasets_dict):
    """Run data audit on all loaded datasets. Returns dict of results."""
    all_results = {}
    for name, dataset in datasets_dict.items():
        try:
            result = audit_dataset(dataset)
            all_results[name] = result
        except Exception as e:
            print(f"  [ERROR] Could not audit {name}: {e}")
    return all_results


if __name__ == "__main__":
    import sys
    from data_loader import load_all

    print("Loading datasets...")
    datasets = load_all()

    print("\nRunning data audits...")
    results = audit_all(datasets)

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Dataset':<35} {'Score':>6} {'Severity':>8} {'DIR':>7} {'DPG':>7}")
    print("-" * 70)
    for name, r in results.items():
        for attr, ar in r["attribute_results"].items():
            dir_str = f"{ar['disparate_impact_ratio']:.3f}" if ar['disparate_impact_ratio'] else "N/A"
            dpg_str = f"{ar['demographic_parity_gap']:.3f}"
            print(f"  {r['dataset_label']:<33} {ar['bias_score']:>6} "
                  f"{ar['severity']:>8} {dir_str:>7} {dpg_str:>7}")