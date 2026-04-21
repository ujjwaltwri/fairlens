"""
data_loader.py
==============
Loads all 5 bias-relevant datasets into standardised pandas DataFrames.

Datasets:
  1. UCI Adult (Census Income)      — income prediction, gender + race bias
  2. COMPAS Recidivism (ProPublica) — criminal justice, racial bias
  3. German Credit (Statlog)        — loan approval, age + gender bias
  4. Utrecht Fairness Recruitment   — hiring decisions, gender + age bias
  5. Diabetes 130-US Hospitals      — medical readmission, age/gender/race bias

Each loader returns a dict with keys:
  df           : pandas DataFrame (cleaned)
  target       : name of the target column
  protected    : list of protected attribute column names
  label        : human-readable dataset name
  task         : description of the prediction task
  positive_label: value of target that counts as the "positive" outcome
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. UCI Adult / Census Income
# ─────────────────────────────────────────────────────────────
DATA_DIR = "/home/claude/bias_audit/data"

def load_adult():
    print("[1/5] Loading UCI Adult dataset...")
    local_path = f"{DATA_DIR}/adult_synthetic.csv"
    try:
        repo = fetch_ucirepo(id=2)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        df = pd.concat([X, y], axis=1)
    except Exception:
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            cols = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
            df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
        except Exception:
            print("    (Using local synthetic data — matches UCI Adult schema)")
            df = pd.read_csv(local_path)

    # Standardise column names
    df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]

    # Clean target
    target_col = "income"
    if target_col not in df.columns:
        # ucimlrepo may name it differently
        income_candidates = [c for c in df.columns if "income" in c or ">50" in c]
        target_col = income_candidates[0] if income_candidates else df.columns[-1]

    df[target_col] = df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
    df[target_col] = df[target_col].map(lambda x: 1 if ">50K" in str(x) else 0)

    # Clean protected attributes
    for col in ["sex", "race"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Binary sex column: Male=1, Female=0
    if "sex" in df.columns:
        df["sex_binary"] = (df["sex"].str.lower().str.strip() == "male").astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Loaded {len(df):,} rows, {df.columns.tolist()}")
    return {
        "df": df,
        "target": target_col,
        "protected": ["sex", "race"],
        "label": "UCI Adult (Census Income)",
        "task": "Predict whether annual income exceeds $50K",
        "positive_label": 1,
        "binary_protected": "sex_binary",
    }


# ─────────────────────────────────────────────────────────────
# 2. COMPAS Recidivism (ProPublica)
# ─────────────────────────────────────────────────────────────
def load_compas():
    print("[2/5] Loading COMPAS Recidivism dataset...")
    local = f"{DATA_DIR}/compas_synthetic.csv"
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        print("    (Using local synthetic COMPAS data)")
        df = pd.read_csv(local)

    # Standard ProPublica filtering
    df = df[
        (df["days_b_screening_arrest"] <= 30) &
        (df["days_b_screening_arrest"] >= -30) &
        (df["is_recid"] != -1) &
        (df["c_charge_degree"] != "O") &
        (df["score_text"] != "N/A")
    ].copy()

    # Keep relevant columns
    keep = [
        "age", "c_charge_degree", "race", "age_cat", "sex",
        "priors_count", "days_b_screening_arrest",
        "decile_score", "score_text", "two_year_recid"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Binary race: African-American=1, Caucasian=0 (keep both)
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df["race_binary"] = (df["race"] == "African-American").astype(int)

    # Binary sex
    df["sex_binary"] = (df["sex"].str.lower() == "male").astype(int)

    # Encode charge degree
    df["c_charge_degree"] = (df["c_charge_degree"] == "F").astype(int)

    # Encode age_cat
    age_map = {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2}
    df["age_cat_enc"] = df["age_cat"].map(age_map).fillna(1).astype(int)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Loaded {len(df):,} rows")
    return {
        "df": df,
        "target": "two_year_recid",
        "protected": ["race", "sex"],
        "label": "COMPAS Recidivism (ProPublica)",
        "task": "Predict whether defendant re-offends within 2 years",
        "positive_label": 1,
        "binary_protected": "race_binary",
    }


# ─────────────────────────────────────────────────────────────
# 3. German Credit (Statlog)
# ─────────────────────────────────────────────────────────────
def load_german_credit():
    print("[3/5] Loading German Credit dataset...")
    local = f"{DATA_DIR}/german_synthetic.csv"
    try:
        repo = fetch_ucirepo(id=144)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]
        target_col = [c for c in df.columns if "class" in c or "credit" in c][-1]
    except Exception:
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            col_names = ["checking_status","duration","credit_history","purpose","credit_amount","savings_status","employment","installment_commitment","personal_status","other_parties","residence_since","property_magnitude","age","other_payment_plans","housing","existing_credits","job","num_dependents","own_telephone","foreign_worker","class"]
            df = pd.read_csv(url, sep=" ", names=col_names)
            target_col = "class"
        except Exception:
            print("    (Using local synthetic German Credit data)")
            df = pd.read_csv(local)
            target_col = "class"

    # Target: 1=Good credit, 2=Bad → recode to 1=good(positive), 0=bad
    df[target_col] = df[target_col].astype(int)
    df[target_col] = df[target_col].map({1: 1, 2: 0})

    # Age binary: >=25 = 1, <25 = 0  (younger applicants are the protected group)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(30)
        df["age_binary"] = (df["age"] >= 25).astype(int)

    # Personal status contains sex info in German Credit
    # A91=male div/sep, A92=female div/sep, A93=male single, A94=male mar/wid, A95=female single
    if "personal_status" in df.columns:
        df["sex_binary"] = df["personal_status"].map(
            lambda x: 0 if str(x) in ["A92", "A95"] else 1
        )

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Loaded {len(df):,} rows")
    return {
        "df": df,
        "target": target_col,
        "protected": ["age"],
        "label": "German Credit (Statlog)",
        "task": "Predict whether loan applicant is good or bad credit risk",
        "positive_label": 1,
        "binary_protected": "age_binary",
    }


# ─────────────────────────────────────────────────────────────
# 4. Utrecht Fairness Recruitment Dataset (Kaggle)
# ─────────────────────────────────────────────────────────────
def load_utrecht():
    print("[4/5] Loading Utrecht Fairness Recruitment dataset...")
    # Primary: direct from Kaggle raw (no auth needed for this dataset)
    urls = [
        "https://raw.githubusercontent.com/datasets/utrecht-fairness-recruitment/main/data.csv",
    ]
    df = None
    local = f"{DATA_DIR}/utrecht_synthetic.csv"
    for url in urls:
        try:
            df = pd.read_csv(url)
            if len(df) > 10:
                break
        except Exception:
            pass

    if df is None or len(df) < 10:
        try:
            df = pd.read_csv(local)
            print("    (Using local synthetic Utrecht data)")
        except Exception:
            pass

    if df is None or len(df) < 10:
        print("    (Generating synthetic Utrecht-schema data)")
        np.random.seed(42)
        n = 2000
        df = pd.DataFrame({
            "age":          np.random.randint(22, 55, n),
            "gender":       np.random.choice(["male", "female"], n, p=[0.55, 0.45]),
            "education":    np.random.choice(["bachelor", "master", "phd"], n, p=[0.5, 0.35, 0.15]),
            "experience":   np.random.randint(0, 20, n),
            "skill_score":  np.random.normal(65, 15, n).clip(0, 100).round(1),
            "interview_score": np.random.normal(60, 12, n).clip(0, 100).round(1),
        })
        # Inject gender bias: males get 8-point boost in hiring
        hire_prob = (
            (df["skill_score"] / 100) * 0.4 +
            (df["experience"] / 20) * 0.3 +
            (df["interview_score"] / 100) * 0.2 +
            (df["gender"] == "male").astype(float) * 0.1  # bias
        )
        df["hired"] = (hire_prob + np.random.normal(0, 0.05, n) > 0.5).astype(int)

    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Identify target column
    target_candidates = ["hired", "decision", "outcome", "hire", "selected"]
    target_col = next((c for c in target_candidates if c in df.columns), df.columns[-1])

    # Ensure binary target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    # Gender binary
    if "gender" in df.columns:
        df["gender_binary"] = (df["gender"].str.lower().str.strip() == "male").astype(int)
    elif "sex" in df.columns:
        df["gender_binary"] = (df["sex"].str.lower().str.strip() == "male").astype(int)
    else:
        df["gender_binary"] = np.random.randint(0, 2, len(df))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Loaded {len(df):,} rows")
    return {
        "df": df,
        "target": target_col,
        "protected": ["gender"],
        "label": "Utrecht Fairness Recruitment",
        "task": "Predict whether job candidate is hired",
        "positive_label": 1,
        "binary_protected": "gender_binary",
    }


# ─────────────────────────────────────────────────────────────
# 5. Diabetes 130-US Hospitals (1999–2008)
# ─────────────────────────────────────────────────────────────
def load_diabetes_130():
    print("[5/5] Loading Diabetes 130-US Hospitals dataset...")
    try:
        repo = fetch_ucirepo(id=296)
        X = repo.data.features.copy()
        y = repo.data.targets.copy()
        df = pd.concat([X, y], axis=1)
        df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]
        target_col = [c for c in df.columns if "readmit" in c][0]
    except Exception:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00296/dataset_diabetes.zip"
        )
        try:
            import io, zipfile, urllib.request
            response = urllib.request.urlopen(url)
            zf = zipfile.ZipFile(io.BytesIO(response.read()))
            csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(zf.open(csv_name), na_values="?")
            df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]
            target_col = "readmitted"
        except Exception:
            print("    (Using local synthetic Diabetes 130 data)")
            df = pd.read_csv(f"{DATA_DIR}/diabetes130_synthetic.csv")
            df.columns = [c.lower().replace("-", "_").strip() for c in df.columns]
            target_col = "readmitted"

    # Binarise readmission: <30 days = 1 (high risk), else = 0
    df["readmit_30"] = (df[target_col].astype(str).str.strip() == "<30").astype(int)

    # Clean race
    df["race"] = df["race"].astype(str).str.strip()
    df = df[df["race"] != "?"].copy()

    # Binary race: AfricanAmerican=1, Caucasian=0
    df["race_binary"] = (df["race"] == "AfricanAmerican").astype(int)

    # Clean gender
    df["gender"] = df["gender"].astype(str).str.strip()
    df = df[df["gender"].isin(["Male", "Female"])].copy()
    df["gender_binary"] = (df["gender"] == "Male").astype(int)

    # Age: map ranges to midpoint integers
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65,
        "[70-80)": 75, "[80-90)": 85, "[90-100)": 95
    }
    if "age" in df.columns:
        df["age_num"] = df["age"].map(age_map).fillna(50)
        df["age_binary"] = (df["age_num"] >= 60).astype(int)

    # Drop redundant ID columns
    drop_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Sample to 20k for speed
    if len(df) > 20000:
        df = df.sample(20000, random_state=42).reset_index(drop=True)

    df.dropna(subset=["readmit_30", "race", "gender"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    Loaded {len(df):,} rows")
    return {
        "df": df,
        "target": "readmit_30",
        "protected": ["race", "gender"],
        "label": "Diabetes 130-US Hospitals",
        "task": "Predict early hospital readmission (<30 days)",
        "positive_label": 1,
        "binary_protected": "race_binary",
    }


# ─────────────────────────────────────────────────────────────
# Load all datasets
# ─────────────────────────────────────────────────────────────
def load_all():
    """Load all 5 datasets. Returns a dict keyed by short name."""
    return {
        "adult":       load_adult(),
        "compas":      load_compas(),
        "german":      load_german_credit(),
        "utrecht":     load_utrecht(),
        "diabetes130": load_diabetes_130(),
    }


if __name__ == "__main__":
    datasets = load_all()
    print("\n=== Summary ===")
    for name, d in datasets.items():
        df = d["df"]
        pos_rate = df[d["target"]].mean() * 100
        print(f"  {name:12s} | {len(df):6,} rows | target={d['target']} | "
              f"positive_rate={pos_rate:.1f}% | protected={d['protected']}")