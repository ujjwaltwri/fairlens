"""
report_generator.py
===================
Generates audit reports in two formats:
  1. JSON — machine-readable, suitable for API responses
  2. PDF  — human-readable report for compliance officers

Each report contains:
  - Dataset overview
  - Data audit results (DIR, DPG, proxy features)
  - Model audit results (equalized odds, SHAP top features)
  - Mitigation comparison (before/after per strategy)
  - Severity ratings and recommended actions
"""

import json
import os
import datetime
import numpy as np
from pathlib import Path

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("[WARN] reportlab not available — PDF generation disabled")


OUTPUT_DIR = Path("/home/claude/bias_audit/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# JSON Report
# ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def build_json_report(dataset_name, data_audit, model_audit, mitigation_results):
    """Build a structured JSON report dict."""
    timestamp = datetime.datetime.now().isoformat()

    # Simplify model audit (remove large objects)
    model_summary = {
        k: v for k, v in model_audit.items()
        if not k.startswith("_") and k not in ["model", "scaler", "shap_values"]
    }

    # Summarise mitigation
    mit_summary = {}
    for strat, r in mitigation_results.items():
        mit_summary[strat] = {
            "before_score": r["before"]["bias_score"],
            "after_score":  r["after"]["bias_score"],
            "improvement":  r.get("improvement", 0),
            "before_tpr_gap": r["before"]["tpr_gap"],
            "after_tpr_gap":  r["after"]["tpr_gap"],
            "before_accuracy": r["before"]["accuracy"],
            "after_accuracy":  r["after"]["accuracy"],
        }

    best_strat = max(mit_summary.items(), key=lambda x: x[1]["improvement"])[0] \
                 if mit_summary else "N/A"

    report = {
        "report_metadata": {
            "generated_at": timestamp,
            "dataset_name":  dataset_name,
            "dataset_label": data_audit.get("dataset_label", dataset_name),
            "report_version": "1.0",
        },
        "data_audit": {
            "n_rows": data_audit.get("n_rows"),
            "target": data_audit.get("target"),
            "overall_bias_score": data_audit.get("overall_bias_score"),
            "overall_severity": data_audit.get("overall_severity"),
            "attribute_results": {
                attr: {
                    "disparate_impact_ratio": r.get("disparate_impact_ratio"),
                    "dir_pass": r.get("dir_pass"),
                    "demographic_parity_gap": r.get("demographic_parity_gap"),
                    "positive_rate_privileged": r.get("positive_rate_privileged"),
                    "positive_rate_unprivileged": r.get("positive_rate_unprivileged"),
                    "chi2_pvalue": r.get("chi2_pvalue"),
                    "chi2_significant": r.get("chi2_significant"),
                    "bias_score": r.get("bias_score"),
                    "severity": r.get("severity"),
                    "top_proxy_features": r.get("top_proxy_features", [])[:5],
                }
                for attr, r in data_audit.get("attribute_results", {}).items()
            },
        },
        "model_audit": {
            "model_type": model_audit.get("model_type"),
            "overall_metrics": model_audit.get("overall_metrics"),
            "group_metrics": model_audit.get("group_metrics"),
            "tpr_gap": model_audit.get("tpr_gap"),
            "fpr_gap": model_audit.get("fpr_gap"),
            "predictive_parity_gap": model_audit.get("predictive_parity_gap"),
            "counterfactual_flip_rate": model_audit.get("counterfactual_flip_rate"),
            "model_dir": model_audit.get("model_dir"),
            "bias_score": model_audit.get("bias_score"),
            "severity": model_audit.get("severity"),
            "top_shap_features": model_audit.get("top_features", [])[:10],
        },
        "mitigation": {
            "strategies": mit_summary,
            "best_strategy": best_strat,
            "recommended_actions": _get_recommendations(data_audit, model_audit),
        },
    }
    return report


def save_json_report(report, dataset_name):
    path = OUTPUT_DIR / f"{dataset_name}_bias_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"  JSON report saved: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────
# PDF Report
# ─────────────────────────────────────────────────────────────

def _severity_color(severity):
    return {
        "HIGH":   colors.HexColor("#FCEBEB"),
        "MEDIUM": colors.HexColor("#FAEEDA"),
        "LOW":    colors.HexColor("#EAF3DE"),
    }.get(severity, colors.white)


def _severity_text_color(severity):
    return {
        "HIGH":   colors.HexColor("#A32D2D"),
        "MEDIUM": colors.HexColor("#633806"),
        "LOW":    colors.HexColor("#27500A"),
    }.get(severity, colors.black)


def _get_recommendations(data_audit, model_audit):
    recs = []
    score = model_audit.get("bias_score", 0)
    tpr_gap = model_audit.get("tpr_gap", 0)
    flip_rate = model_audit.get("counterfactual_flip_rate", 0)

    for attr, r in data_audit.get("attribute_results", {}).items():
        dir_val = r.get("disparate_impact_ratio")
        if dir_val is not None and dir_val < 0.8:
            recs.append(f"Apply reweighing to fix disparate impact on '{attr}' (DIR={dir_val:.2f})")
        if r.get("chi2_significant"):
            recs.append(f"Protected attribute '{attr}' is statistically associated with outcome — investigate proxy features")

    if tpr_gap > 0.1:
        recs.append(f"True positive rate gap of {tpr_gap:.3f} detected — apply equalized odds post-processing")
    if flip_rate > 0.1:
        recs.append(f"Counterfactual flip rate {flip_rate:.1%} — model is directly using protected attribute")
    if score >= 70:
        recs.append("HIGH RISK: Do not deploy this model without bias mitigation")
    elif score >= 40:
        recs.append("MEDIUM RISK: Apply at least one mitigation strategy before deployment")
    else:
        recs.append("LOW RISK: Monitor for drift, re-audit quarterly")

    return recs if recs else ["No critical issues detected — continue monitoring"]


def generate_pdf_report(report_data, dataset_name):
    if not REPORTLAB_AVAILABLE:
        print("  [SKIP] reportlab not available — skipping PDF")
        return None

    path = OUTPUT_DIR / f"{dataset_name}_bias_report.pdf"
    doc  = SimpleDocTemplate(str(path), pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title2", parent=styles["Title"],
                                  fontSize=20, spaceAfter=6, alignment=TA_LEFT)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=14, spaceBefore=14, spaceAfter=4,
                               textColor=colors.HexColor("#1a1a1a"))
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"],
                               fontSize=11, spaceBefore=8, spaceAfter=2,
                               textColor=colors.HexColor("#444444"))
    body_style = ParagraphStyle("Body2", parent=styles["Normal"],
                                 fontSize=10, spaceAfter=4, leading=14)
    small_style = ParagraphStyle("Small", parent=styles["Normal"],
                                  fontSize=8.5, textColor=colors.HexColor("#666666"))
    code_style = ParagraphStyle("Code", parent=styles["Code"],
                                 fontSize=8.5, leading=12,
                                 backColor=colors.HexColor("#F5F5F0"))

    meta = report_data["report_metadata"]
    da   = report_data["data_audit"]
    ma   = report_data["model_audit"]
    mit  = report_data["mitigation"]

    severity = da.get("overall_severity", "MEDIUM")
    overall_score = da.get("overall_bias_score", 0)
    model_score   = ma.get("bias_score", 0)

    story = []

    # ── Header ──
    story.append(Paragraph("Bias Audit Report", title_style))
    story.append(Paragraph(f"{meta['dataset_label']}", h2_style))
    story.append(Paragraph(
        f"Generated: {meta['generated_at'][:19].replace('T', ' ')}  |  "
        f"Model: {ma.get('model_type', 'N/A')}  |  "
        f"Version: {meta['report_version']}",
        small_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#DDDDDD")))
    story.append(Spacer(1, 0.3*cm))

    # ── Overall scores ──
    story.append(Paragraph("Overall risk assessment", h2_style))
    scores_data = [
        ["Metric", "Score", "Severity"],
        ["Data bias score",  str(overall_score),  severity],
        ["Model bias score", str(model_score),
         ma.get("severity", "MEDIUM")],
    ]
    t = Table(scores_data, colWidths=[8*cm, 3*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C2C2A")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [_severity_color(severity), _severity_color(ma.get("severity", "MEDIUM"))]),
        ("TEXTCOLOR", (2, 1), (2, 1), _severity_text_color(severity)),
        ("TEXTCOLOR", (2, 2), (2, 2), _severity_text_color(ma.get("severity", "MEDIUM"))),
        ("FONTNAME",  (2, 1), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Data Audit ──
    story.append(Paragraph("Data audit results", h2_style))
    story.append(Paragraph(
        f"Dataset: {da['n_rows']:,} rows | Target: '{da['target']}'", body_style
    ))

    for attr, r in da.get("attribute_results", {}).items():
        story.append(Paragraph(f"Protected attribute: {attr}", h3_style))
        dir_val = r.get("disparate_impact_ratio")
        dir_str = f"{dir_val:.4f}" if dir_val else "N/A"
        dir_status = "PASS ✓" if r.get("dir_pass") else "FAIL ✗"
        rows = [
            ["Metric", "Value", "Status"],
            ["Disparate Impact Ratio (80% rule)", dir_str, dir_status],
            ["Demographic Parity Gap",
             f"{r.get('demographic_parity_gap', 0):.4f}",
             "WARN" if r.get("demographic_parity_gap", 0) > 0.1 else "OK"],
            ["Positive rate — privileged",
             f"{r.get('positive_rate_privileged', 0)*100:.1f}%", ""],
            ["Positive rate — unprivileged",
             f"{r.get('positive_rate_unprivileged', 0)*100:.1f}%", ""],
            ["Chi-square p-value",
             f"{r.get('chi2_pvalue', 1):.6f}",
             "Significant ✗" if r.get("chi2_significant") else "Not significant ✓"],
            ["Bias Score", f"{r.get('bias_score', 0)}/100", r.get("severity", "")],
        ]
        t = Table(rows, colWidths=[9*cm, 4*cm, 4*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#534AB7")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F6")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*cm))

        proxies = r.get("top_proxy_features", [])
        if proxies:
            proxy_text = "Top proxy features: " + ", ".join(
                f"{f} ({v:.3f})" for f, v in proxies[:5]
            )
            story.append(Paragraph(proxy_text, small_style))
        story.append(Spacer(1, 0.3*cm))

    # ── Model Audit ──
    story.append(Paragraph("Model audit results", h2_style))
    om = ma.get("overall_metrics", {})
    model_rows = [
        ["Metric", "Value"],
        ["Accuracy",    f"{om.get('accuracy', 0):.4f}"],
        ["ROC-AUC",     f"{om.get('roc_auc', 0):.4f}"],
        ["Precision",   f"{om.get('precision', 0):.4f}"],
        ["Recall",      f"{om.get('recall', 0):.4f}"],
        ["TPR gap (Equalized Odds)",    f"{ma.get('tpr_gap', 0):.4f}"],
        ["FPR gap (Equalized Odds)",    f"{ma.get('fpr_gap', 0):.4f}"],
        ["Predictive Parity gap",       f"{ma.get('predictive_parity_gap', 0):.4f}"],
        ["Counterfactual flip rate",    f"{ma.get('counterfactual_flip_rate', 0)*100:.1f}%"],
        ["Model DIR",   str(ma.get('model_dir', 'N/A'))],
        ["Model bias score", f"{ma.get('bias_score', 0)}/100"],
    ]
    t = Table(model_rows, colWidths=[10*cm, 7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F6E56")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FAF5")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # Top SHAP features
    top_feats = ma.get("top_shap_features", [])
    if top_feats:
        story.append(Paragraph("Top features by SHAP importance:", h3_style))
        feat_rows = [["Feature", "Mean |SHAP|"]] + [
            [f, f"{v:.4f}"] for f, v in top_feats[:8]
        ]
        t = Table(feat_rows, colWidths=[12*cm, 5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#BA7517")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFFBF0")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))

    # ── Mitigation ──
    story.append(Paragraph("Mitigation strategies", h2_style))
    mit_rows = [["Strategy", "Before", "After", "Improvement", "Acc (after)"]]
    for strat, r in mit.get("strategies", {}).items():
        imp = r.get("improvement", 0)
        imp_str = f"+{imp}" if imp > 0 else str(imp)
        mit_rows.append([
            strat.replace("_", " ").title(),
            str(r.get("before_score")),
            str(r.get("after_score")),
            imp_str,
            f"{r.get('after_accuracy', 0):.3f}",
        ])
    t = Table(mit_rows, colWidths=[6*cm, 2.5*cm, 2.5*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D9E75")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FAF5")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"Best strategy: {mit.get('best_strategy', 'N/A').replace('_', ' ').title()}",
        body_style
    ))

    # ── Recommendations ──
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Recommended actions", h2_style))
    for i, rec in enumerate(mit.get("recommended_actions", []), 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))

    # ── Footer ──
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC")))
    story.append(Paragraph(
        "This report was generated by the Bias Audit System. "
        "Results should be reviewed by qualified data scientists and legal/compliance teams "
        "before making deployment decisions.",
        small_style
    ))

    doc.build(story)
    print(f"  PDF report saved: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────
# Main: generate both formats
# ─────────────────────────────────────────────────────────────

def generate_reports(dataset_name, data_audit, model_audit, mitigation_results):
    """Generate JSON + PDF reports for one dataset."""
    print(f"\n  Generating reports for: {dataset_name}")

    report = build_json_report(dataset_name, data_audit, model_audit, mitigation_results)
    json_path = save_json_report(report, dataset_name)
    pdf_path  = generate_pdf_report(report, dataset_name)

    return {"json": json_path, "pdf": pdf_path, "report_data": report}


def generate_all_reports(datasets_dict, data_audits, model_audits, mitigations):
    """Generate reports for all datasets."""
    all_paths = {}
    for name in datasets_dict:
        if name in data_audits and name in model_audits and name in mitigations:
            paths = generate_reports(name, data_audits[name],
                                     model_audits[name], mitigations[name])
            all_paths[name] = paths
        else:
            print(f"  [SKIP] {name} — missing audit or mitigation results")
    return all_paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/bias_audit/scripts")
    from data_loader import load_all
    from data_auditor import audit_all
    from model_auditor import audit_all_models
    from mitigation import mitigate_all

    print("Loading datasets...")
    datasets = {"adult": load_all()["adult"]}

    print("Running data audits...")
    data_audits = audit_all(datasets)

    print("Running model audits...")
    model_audits = audit_all_models(datasets, verbose=False)

    print("Running mitigations...")
    mitigations = mitigate_all(datasets, model_audits, verbose=False)

    print("Generating reports...")
    paths = generate_all_reports(datasets, data_audits, model_audits, mitigations)
    print(f"\nReport files: {paths}")