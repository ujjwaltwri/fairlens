# FairLens 🔍

> **AI Bias Detection & Mitigation Platform**  
> Detect, measure, and fix hidden discrimination in machine learning models — before they impact real people.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hack2Skill](https://img.shields.io/badge/Hack2Skill-Unbiased%20AI%20Decision-6C47FF?style=flat)](https://hack2skill.com)

---

## 🧠 What is FairLens?

FairLens is an end-to-end bias auditing system that helps organizations identify and fix discriminatory patterns in their AI models and training datasets. It provides:

- **Statistical data auditing** — catch bias before training
- **Model-level fairness metrics** — measure discrimination in deployed models
- **SHAP explainability** — understand *which* features drive biased decisions
- **Three mitigation strategies** — automatically fix the bias found
- **PDF + JSON audit reports** — compliance-ready documentation
- **REST API** — integrate FairLens into any existing ML pipeline

Built for the **[Hack2Skill] Unbiased AI Decision** challenge.

---

## 📊 Validated Datasets

FairLens has been validated across 5 real-world fairness benchmarks:

| Dataset | Domain | Protected Attribute | Bias Found |
|---|---|---|---|
| [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) | Income prediction | Gender, Race | DIR = 0.50 — HIGH |
| [COMPAS Recidivism](https://github.com/propublica/compas-analysis) | Criminal justice | Race | Chi-sq significant — MEDIUM |
| [German Credit (Statlog)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | Loan approval | Age | TPR gap = 0.15 — MEDIUM |
| [Utrecht Fairness Recruitment](https://www.kaggle.com/datasets/ictinstitute/utrecht-fairness-recruitment-dataset) | Hiring | Gender | Parity gap — MEDIUM |
| [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | Medical readmission | Race | DIR = 0.43 — HIGH |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FairLens Platform                     │
├──────────────┬──────────────────────┬───────────────────┤
│  Data Layer  │    AI/Audit Core     │   Output Layer    │
│              │                      │                   │
│ data_loader  │  data_auditor        │  report_generator │
│              │  ├─ DIR ratio        │  ├─ PDF report    │
│ 5 datasets   │  ├─ Parity gap       │  ├─ JSON report   │
│ + CSV upload │  ├─ Chi-square       │  └─ Dashboard     │
│              │  └─ Proxy features   │                   │
│              │                      │   api.py          │
│              │  model_auditor       │  ├─ /audit/full   │
│              │  ├─ SHAP             │  ├─ /report/pdf   │
│              │  ├─ Equalized Odds   │  └─ /audit/upload │
│              │  └─ Counterfactual   │                   │
│              │                      │                   │
│              │  mitigation.py       │                   │
│              │  ├─ Reweighing       │                   │
│              │  ├─ EqualizedOdds    │                   │
│              │  └─ Threshold Cal.   │                   │
└──────────────┴──────────────────────┴───────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/fairlens.git
cd fairlens

pip install -r requirements.txt
```

### 2. Run Full Pipeline (All 5 Datasets)

```bash
python run_all.py
```

**Demo mode** (Adult dataset only, ~10 seconds):
```bash
python run_all.py --quick
```

**Random Forest** instead of Logistic Regression:
```bash
python run_all.py --model rf
```

### 3. Start the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://localhost:8000](http://localhost:8000)  
Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit: [http://localhost:5173](http://localhost:5173)

---

## 📁 Project Structure

```
fairlens/
├── README.md
├── requirements.txt
│
├── data_loader.py        # Load all 5 datasets (+ CSV upload)
├── data_auditor.py       # Statistical bias analysis on raw data
├── model_auditor.py      # Train models + SHAP + fairness metrics
├── mitigation.py         # 3 bias mitigation strategies
├── report_generator.py   # PDF + JSON audit report generation
├── api.py                # FastAPI REST backend
├── run_all.py            # Single entry point — full pipeline
│
├── data/                 # Synthetic dataset files (fallback)
│   ├── adult_synthetic.csv
│   ├── compas_synthetic.csv
│   ├── german_synthetic.csv
│   ├── utrecht_synthetic.csv
│   └── diabetes130_synthetic.csv
│
├── outputs/              # Generated reports (gitignored)
│   ├── adult_bias_report.pdf
│   ├── adult_bias_report.json
│   └── pipeline_summary.json
│
└── frontend/             # React dashboard
    ├── index.html
    ├── package.json
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── components/
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check + available datasets |
| `GET` | `/datasets` | List all available datasets |
| `POST` | `/audit/data/{name}` | Run statistical data audit |
| `POST` | `/audit/model/{name}` | Train model + audit for bias |
| `POST` | `/audit/full/{name}` | Full pipeline: audit + mitigate + report |
| `GET` | `/report/{name}/json` | Download JSON audit report |
| `GET` | `/report/{name}/pdf` | Download PDF audit report |
| `GET` | `/results` | Get all cached audit results |
| `POST` | `/audit/upload` | Upload custom CSV and audit it |

**Example:**
```bash
# Run full audit on the Adult dataset
curl -X POST http://localhost:8000/audit/full/adult

# Upload your own CSV
curl -X POST http://localhost:8000/audit/upload \
  -F "file=@mydata.csv" \
  -F "target_column=outcome" \
  -F "protected_column=gender"
```

---

## 📐 Fairness Metrics Explained

### Data Audit
| Metric | Formula | Threshold |
|---|---|---|
| **Disparate Impact Ratio** | P(pos\|unprivileged) / P(pos\|privileged) | < 0.80 = FAIL (US EEOC four-fifths rule) |
| **Demographic Parity Gap** | \|P(pos\|g=1) − P(pos\|g=0)\| | > 0.10 = concerning |
| **Chi-square p-value** | Statistical test of independence | < 0.05 = significant association |

### Model Audit
| Metric | Description |
|---|---|
| **Equalized Odds** | Equal TPR + FPR across groups |
| **Equal Opportunity** | Equal TPR (recall) across groups |
| **Predictive Parity** | Equal precision across groups |
| **Counterfactual Flip Rate** | % predictions that change when protected attr is flipped |
| **SHAP Feature Importance** | Which features drive biased predictions |

### Mitigation Strategies
| Strategy | When applied | Method |
|---|---|---|
| **Pre-processing (Reweighing)** | Before training | Upweight underrepresented group/label combinations |
| **In-processing (Fairness Constraint)** | During training | ExponentiatedGradient with EqualizedOdds constraint |
| **Post-processing (Threshold Calibration)** | After training | Per-group decision thresholds to equalize TPR |

---

## 📦 Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
aif360>=0.6
shap>=0.44
fairlearn>=0.10
ucimlrepo>=0.0.3
fastapi>=0.110
uvicorn>=0.27
python-multipart>=0.0.9
reportlab>=4.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.13
requests>=2.31
```

Install all:
```bash
pip install -r requirements.txt
```

---

## ☁️ Deployment

### Google Cloud Run (Recommended)

```bash
# Build and deploy
gcloud run deploy fairlens-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi
```

### Render.com (Free tier)

1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Docker

```bash
docker build -t fairlens .
docker run -p 8000:8000 fairlens
```

---

## 🧪 Run Tests

```bash
# Quick smoke test — Adult dataset only (~10s)
python run_all.py --quick

# Full pipeline — all datasets (~60s)
python run_all.py

# API only
uvicorn api:app --port 8000
curl http://localhost:8000/
```

---

## 📊 Results

From our validation run across all 5 datasets:

| Dataset | Data Bias | Model Bias | Best Mitigation | Improvement |
|---|---|---|---|---|
| UCI Adult | 92/100 HIGH | 60/100 MEDIUM | Fairness Constraint | −8 pts |
| COMPAS | 55/100 MEDIUM | — | — | — |
| German Credit | 60/100 MEDIUM | 50/100 MEDIUM | Reweighing | — |
| Utrecht Recruitment | 51/100 MEDIUM | — | — | — |
| **Diabetes 130** | **97/100 HIGH** | **83/100 HIGH** | **Fairness Constraint** | **−48 pts** |

---

## 👥 Team

Built for **Hack2Skill — Unbiased AI Decision** challenge.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [IBM AIF360](https://github.com/Trusted-AI/AIF360) — fairness metrics and mitigation algorithms
- [Microsoft Fairlearn](https://fairlearn.org) — ExponentiatedGradient constraint optimizer
- [SHAP](https://shap.readthedocs.io) — model explainability
- [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis) — recidivism dataset
- [UCI ML Repository](https://archive.ics.uci.edu) — Adult and German Credit datasets