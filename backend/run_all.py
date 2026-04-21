"""
run_all.py
==========
Single entry point: runs the complete bias audit pipeline on all 5 datasets.

Pipeline per dataset:
  1. Load data
  2. Data audit (statistical bias analysis)
  3. Model audit (train + SHAP + fairness metrics)
  4. Apply 3 mitigation strategies
  5. Generate JSON + PDF report

Usage:
  python run_all.py                   # Run all 5 datasets
  python run_all.py --dataset adult   # Run only one dataset
  python run_all.py --model rf        # Use RandomForest instead of LogisticRegression
  python run_all.py --skip-report     # Skip PDF generation
  python run_all.py --quick           # Run only Adult dataset (fastest demo)
"""

import sys
import argparse
import time
import json
from pathlib import Path

sys.path.insert(0, "/home/claude/bias_audit/scripts")

OUTPUT_DIR = Path("/home/claude/bias_audit/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           BIAS AUDIT SYSTEM — Full Pipeline Runner           ║
║         Unbiased AI Decision — Hack2Skill Hackathon          ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_pipeline(dataset_names, model_type="logistic", skip_report=False, verbose=True):
    """Run the full audit pipeline on the given dataset names."""

    results_summary = {}

    # ── Step 1: Load datasets ──
    print_section("Step 1 / 5 — Loading datasets")
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

    datasets = {}
    for name in dataset_names:
        if name not in loaders:
            print(f"  [SKIP] Unknown dataset: {name}")
            continue
        try:
            t0 = time.time()
            datasets[name] = loaders[name]()
            print(f"  ✓ {name} ({len(datasets[name]['df']):,} rows) "
                  f"in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    if not datasets:
        print("No datasets loaded. Exiting.")
        return {}

    # ── Step 2: Data audits ──
    print_section("Step 2 / 5 — Data audits (statistical bias analysis)")
    from data_auditor import audit_dataset
    data_audits = {}
    for name, ds in datasets.items():
        t0 = time.time()
        try:
            data_audits[name] = audit_dataset(ds, verbose=verbose)
            score = data_audits[name]["overall_bias_score"]
            sev   = data_audits[name]["overall_severity"]
            print(f"\n  [{name}] Data bias score: {score}/100  [{sev}]")
        except Exception as e:
            print(f"  ✗ {name} data audit failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 3: Model audits ──
    print_section(f"Step 3 / 5 — Model audits (train + SHAP + fairness metrics) [{model_type}]")
    from model_auditor import audit_model
    model_audits = {}
    for name, ds in datasets.items():
        t0 = time.time()
        try:
            model_audits[name] = audit_model(ds, model_type=model_type, verbose=verbose)
            score = model_audits[name]["bias_score"]
            sev   = model_audits[name]["severity"]
            acc   = model_audits[name]["overall_metrics"]["accuracy"]
            auc   = model_audits[name]["overall_metrics"]["roc_auc"]
            tpr_g = model_audits[name]["tpr_gap"]
            flip  = model_audits[name]["counterfactual_flip_rate"]
            print(f"\n  [{name}] Model bias score: {score}/100 [{sev}] | "
                  f"Acc={acc:.3f} AUC={auc:.3f} TPR_gap={tpr_g:.3f} Flip={flip:.1%}")
        except Exception as e:
            print(f"  ✗ {name} model audit failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 4: Mitigations ──
    print_section("Step 4 / 5 — Applying bias mitigation strategies")
    from mitigation import apply_all_mitigations
    mitigations = {}
    for name, ds in datasets.items():
        mar = model_audits.get(name)
        t0 = time.time()
        try:
            mitigations[name] = apply_all_mitigations(ds, mar, verbose=verbose)
            best = max(
                mitigations[name].items(),
                key=lambda x: x[1].get("improvement", 0)
            )
            print(f"\n  [{name}] Best: {best[0]} "
                  f"(improvement: {best[1].get('improvement', 0):+d} points)")
        except Exception as e:
            print(f"  ✗ {name} mitigation failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 5: Reports ──
    if not skip_report:
        print_section("Step 5 / 5 — Generating reports (JSON + PDF)")
        from report_generator import generate_reports
        report_paths = {}
        for name in datasets:
            if name in data_audits and name in model_audits and name in mitigations:
                try:
                    paths = generate_reports(
                        name, data_audits[name], model_audits[name], mitigations[name]
                    )
                    report_paths[name] = paths
                except Exception as e:
                    print(f"  ✗ {name} report failed: {e}")
    else:
        print_section("Step 5 / 5 — Reports (skipped)")
        report_paths = {}

    # ── Final summary ──
    print_section("COMPLETE — Summary")
    print(f"\n  {'Dataset':<20} {'Data':>6} {'Model':>6} {'Best mit.':>10} {'Improv.':>8}")
    print(f"  {'─'*55}")

    for name in datasets:
        da_score = data_audits.get(name, {}).get("overall_bias_score", "–")
        ma_score = model_audits.get(name, {}).get("bias_score", "–")
        if name in mitigations and mitigations[name]:
            best_name, best_r = max(
                mitigations[name].items(),
                key=lambda x: x[1].get("improvement", 0)
            )
            best_short = best_name.split("_")[0]
            improvement = best_r.get("improvement", 0)
        else:
            best_short, improvement = "–", 0

        print(f"  {name:<20} {str(da_score):>6} {str(ma_score):>6} "
              f"{best_short:>10} {improvement:>+7}")

    print()
    if report_paths:
        print("  Output files:")
        for name, paths in report_paths.items():
            for fmt, path in paths.items():
                if path:
                    print(f"    {name} [{fmt}]: {path}")

    # Build results dict
    for name in datasets:
        results_summary[name] = {
            "data_bias_score":  data_audits.get(name, {}).get("overall_bias_score"),
            "data_severity":    data_audits.get(name, {}).get("overall_severity"),
            "model_bias_score": model_audits.get(name, {}).get("bias_score"),
            "model_severity":   model_audits.get(name, {}).get("severity"),
            "model_accuracy":   model_audits.get(name, {}).get("overall_metrics", {}).get("accuracy"),
            "model_auc":        model_audits.get(name, {}).get("overall_metrics", {}).get("roc_auc"),
            "tpr_gap":          model_audits.get(name, {}).get("tpr_gap"),
            "fpr_gap":          model_audits.get(name, {}).get("fpr_gap"),
            "counterfactual_flip_rate": model_audits.get(name, {}).get("counterfactual_flip_rate"),
            "mitigation_improvements": {
                s: r.get("improvement", 0)
                for s, r in mitigations.get(name, {}).items()
            },
            "reports": report_paths.get(name, {}),
        }

    # Save summary JSON
    summary_path = OUTPUT_DIR / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        # Convert numpy types
        def _clean(obj):
            if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_clean(i) for i in obj]
            if hasattr(obj, 'item'):  return obj.item()
            return obj
        json.dump(_clean(results_summary), f, indent=2)
    print(f"\n  Pipeline summary: {summary_path}")

    return results_summary


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bias Audit System — Full Pipeline Runner"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Run only one dataset (adult|compas|german|utrecht|diabetes130)"
    )
    parser.add_argument(
        "--model", type=str, default="logistic", choices=["logistic", "rf"],
        help="Model type: logistic (default) or rf (random forest)"
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Skip PDF/JSON report generation"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the Adult dataset (fastest, good for demo)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose per-step output"
    )
    args = parser.parse_args()

    print_banner()

    all_datasets = ["adult", "compas", "german", "utrecht", "diabetes130"]

    if args.quick:
        chosen = ["adult"]
        print("  Quick mode: running Adult dataset only")
    elif args.dataset:
        if args.dataset not in all_datasets:
            print(f"  Unknown dataset '{args.dataset}'. Choose from: {all_datasets}")
            sys.exit(1)
        chosen = [args.dataset]
    else:
        chosen = all_datasets

    model_type = "random_forest" if args.model == "rf" else "logistic"

    t_total = time.time()
    run_pipeline(
        dataset_names=chosen,
        model_type=model_type,
        skip_report=args.skip_report,
        verbose=not args.quiet,
    )
    print(f"\n  Total elapsed: {time.time()-t_total:.1f}s")
    print("  Done.\n")


if __name__ == "__main__":
    main()