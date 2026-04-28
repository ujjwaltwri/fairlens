"""
gemini.py
=========
Gemini API integration for FairLens.

Three capabilities:
  1. generate_narrative()      — plain-English explanation of audit numbers
  2. generate_recommendation() — which mitigation strategy to deploy and why
  3. chat()                    — conversational Q&A with full audit context

Env vars required:
  GEMINI_API_KEY   — from Google AI Studio (aistudio.google.com)

Model used: gemini-2.5-flash  (fast, cheap, good enough for structured summaries)
"""

import os
import logging
from typing import Optional

from google import genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-flash"


# ─────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def _generate(prompt: str) -> str:
    """Single-turn text generation."""
    client = _get_client()
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────
# 1. Narrative — plain-English audit explanation
# ─────────────────────────────────────────────────────────────

def generate_narrative(
    dataset_label: str,
    data_audit: dict,
    model_audit: dict,
) -> str:
    data_score   = data_audit.get("overall_bias_score", "N/A")
    data_sev     = data_audit.get("overall_severity", "N/A")
    model_score  = model_audit.get("bias_score", "N/A")
    model_sev    = model_audit.get("severity", "N/A")
    tpr_gap      = model_audit.get("tpr_gap", "N/A")
    fpr_gap      = model_audit.get("fpr_gap", "N/A")
    flip_rate    = model_audit.get("counterfactual_flip_rate", "N/A")
    accuracy     = model_audit.get("overall_metrics", {}).get("accuracy", "N/A")
    top_features = model_audit.get("top_features", [])[:5]

    attr_summaries = []
    for attr, r in data_audit.get("attribute_results", {}).items():
        attr_summaries.append(
            f"  - Protected attribute '{attr}': "
            f"Disparate Impact Ratio = {r.get('disparate_impact_ratio')}, "
            f"Demographic Parity Gap = {r.get('demographic_parity_gap')}, "
            f"Chi-square p-value = {r.get('chi2_pvalue')} "
            f"({'significant' if r.get('chi2_significant') else 'not significant'}), "
            f"Bias score = {r.get('bias_score')}/100 [{r.get('severity')}]"
        )

    attr_block    = "\n".join(attr_summaries) if attr_summaries else "  No attributes analysed."
    feature_block = ", ".join(f"{f} ({v:.3f})" for f, v in top_features) if top_features else "N/A"

    prompt = f"""You are a senior AI ethics and compliance expert writing an audit report for non-technical stakeholders.

You have just run a bias audit on the following dataset and trained model. Based on the numbers below, write a clear, professional explanation in 3 paragraphs with NO markdown formatting and NO bullet points — plain prose only.

Dataset: {dataset_label}

DATA AUDIT RESULTS:
Overall data bias score: {data_score}/100 [{data_sev}]
Per-attribute findings:
{attr_block}

MODEL AUDIT RESULTS:
Overall model bias score: {model_score}/100 [{model_sev}]
Model accuracy: {accuracy}
TPR gap (Equalized Odds): {tpr_gap}
FPR gap (Equalized Odds): {fpr_gap}
Counterfactual flip rate: {flip_rate}
Top SHAP features driving predictions: {feature_block}

Write exactly 3 paragraphs:
1. What bias was found in the raw data and why it matters in plain terms.
2. What the trained model is doing — whether it absorbed and amplified those biases, citing the TPR gap, flip rate, and key features.
3. What this means for real people affected by this model's decisions.

Do not use any bullet points, headers, or markdown. Write in formal but accessible English."""

    try:
        narrative = _generate(prompt)
        logger.info(f"[Gemini] Narrative generated for '{dataset_label}'")
        return narrative
    except Exception as e:
        logger.error(f"[Gemini] Narrative generation failed: {e}")
        return (
            f"Automated narrative unavailable ({e}). "
            f"Data bias score: {data_score}/100 [{data_sev}]. "
            f"Model bias score: {model_score}/100 [{model_sev}]."
        )


# ─────────────────────────────────────────────────────────────
# 2. Recommendation — which mitigation strategy to deploy
# ─────────────────────────────────────────────────────────────

def generate_recommendation(
    dataset_label: str,
    model_audit: dict,
    mitigation_results: dict,
) -> str:
    strategies = []
    for name, r in mitigation_results.items():
        before = r.get("before", {})
        after  = r.get("after", {})
        strategies.append(
            f"  - {name}: "
            f"bias score {before.get('bias_score')} → {after.get('bias_score')} "
            f"(improvement: {r.get('improvement', 0):+d} pts), "
            f"TPR gap {before.get('tpr_gap')} → {after.get('tpr_gap')}, "
            f"accuracy {before.get('accuracy', 'N/A')} → {after.get('accuracy', 'N/A')}"
        )

    strat_block   = "\n".join(strategies)
    base_accuracy = model_audit.get("overall_metrics", {}).get("accuracy", "N/A")
    base_score    = model_audit.get("bias_score", "N/A")

    prompt = f"""You are a senior AI ethics expert advising a company on which bias mitigation strategy to deploy.

Dataset: {dataset_label}
Baseline model — bias score: {base_score}/100, accuracy: {base_accuracy}

Three mitigation strategies were tested:
{strat_block}

Write exactly 2 paragraphs (plain prose, no markdown, no bullet points):
1. Which strategy you recommend deploying and specifically why — weigh bias reduction against accuracy cost.
2. What the organisation should do next: monitoring cadence, re-audit triggers, and any legal or compliance implications.

Be concrete and direct. If one strategy clearly dominates, say so plainly."""

    try:
        recommendation = _generate(prompt)
        logger.info(f"[Gemini] Recommendation generated for '{dataset_label}'")
        return recommendation
    except Exception as e:
        logger.error(f"[Gemini] Recommendation generation failed: {e}")
        if mitigation_results:
            best = max(mitigation_results.items(), key=lambda x: x[1].get("improvement", 0))
            return (
                f"Automated recommendation unavailable ({e}). "
                f"Based on raw results, '{best[0]}' showed the highest improvement "
                f"({best[1].get('improvement', 0):+d} points)."
            )
        return f"Automated recommendation unavailable: {e}"


# ─────────────────────────────────────────────────────────────
# 3. Chat — conversational Q&A with full audit context
# ─────────────────────────────────────────────────────────────

def chat(
    question: str,
    dataset_label: str,
    data_audit: dict,
    model_audit: dict,
    mitigation_results: dict,
    gemini_narrative: Optional[str] = None,
    conversation_history: Optional[list] = None,
) -> str:
    data_score  = data_audit.get("overall_bias_score")
    data_sev    = data_audit.get("overall_severity")
    model_score = model_audit.get("bias_score")
    model_sev   = model_audit.get("severity")
    tpr_gap     = model_audit.get("tpr_gap")
    fpr_gap     = model_audit.get("fpr_gap")
    flip_rate   = model_audit.get("counterfactual_flip_rate")
    accuracy    = model_audit.get("overall_metrics", {}).get("accuracy")
    top_feats   = model_audit.get("top_features", [])[:5]

    attr_lines = []
    for attr, r in data_audit.get("attribute_results", {}).items():
        attr_lines.append(
            f"{attr}: DIR={r.get('disparate_impact_ratio')}, "
            f"DPG={r.get('demographic_parity_gap')}, "
            f"score={r.get('bias_score')}/100"
        )

    mit_lines = []
    for name, r in mitigation_results.items():
        mit_lines.append(
            f"{name}: {r.get('before', {}).get('bias_score')} → "
            f"{r.get('after', {}).get('bias_score')} "
            f"(+{r.get('improvement', 0)} pts)"
        )

    system_context = f"""You are FairLens, an AI bias auditing assistant. Answer the user's questions about the audit results clearly and concisely.

AUDIT CONTEXT:
Dataset: {dataset_label}
Data bias score: {data_score}/100 [{data_sev}]
Model bias score: {model_score}/100 [{model_sev}]
Accuracy: {accuracy}
TPR gap: {tpr_gap} | FPR gap: {fpr_gap} | Counterfactual flip rate: {flip_rate}
Top features: {', '.join(f"{f}({v:.3f})" for f, v in top_feats)}
Protected attribute results: {' | '.join(attr_lines)}
Mitigation results: {' | '.join(mit_lines)}
{f'Prior narrative: {gemini_narrative[:500]}...' if gemini_narrative else ''}

Answer in plain English. Be specific and cite numbers from the audit. If the question is outside the scope of this audit, say so clearly."""

    try:
        client = _get_client()

        if conversation_history:
            # Build contents list from history + new question
            contents = []
            for turn in conversation_history:
                contents.append({
                    "role": turn["role"],
                    "parts": [{"text": p if isinstance(p, str) else p.get("text", "")}
                               for p in turn["parts"]]
                })
            # Prepend system context to first user message
            full_question = f"{system_context}\n\nUser question: {question}"
            contents.append({"role": "user", "parts": [{"text": full_question}]})
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=contents,
            )
        else:
            full_prompt = f"{system_context}\n\nUser question: {question}"
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=full_prompt,
            )

        answer = response.text.strip()
        logger.info(f"[Gemini] Chat answered for '{dataset_label}': {question[:60]}...")
        return answer

    except Exception as e:
        logger.error(f"[Gemini] Chat failed: {e}")
        return f"I couldn't process your question right now ({e}). Please try again."


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

def gemini_available() -> bool:
    """Returns True if Gemini API key is set and reachable."""
    try:
        _generate("ping")
        return True
    except Exception:
        return False