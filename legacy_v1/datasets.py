"""
datasets.py – load, clean and return (X, y, name) tuples for each benchmark dataset.
All preprocessing is done OUTSIDE the cross-validation loop here (only label encoding
and column selection). StandardScaler + PCA are applied INSIDE the CV loop to prevent
leakage (see evaluate.py).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
import urllib.request, io, os

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────
def _download(url: str, filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  Downloading {filename} …")
        urllib.request.urlretrieve(url, path)
    return path


# ── dataset loaders ────────────────────────────────────────────────────────────
def load_breast_cancer_data():
    """Wisconsin Breast Cancer – sklearn built-in."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(int)            # 0=malignant, 1=benign
    return X, y, "Breast Cancer Wisconsin"


def load_heart_disease_data():
    """Heart Disease Cleveland – UCI (binary: 0=no disease, 1=disease)."""
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    try:
        path = _download(url, "heart_cleveland.csv")
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(path, header=None, names=cols, na_values="?")
    except Exception:
        # Fallback: generate a reproducible proxy from breast-cancer PCA
        rng = np.random.default_rng(2024)
        X = rng.standard_normal((303, 13)).astype(np.float32)
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        return X, y, "Heart Disease UCI (proxy)"
    df = df.dropna()
    df["target"] = (df["target"] > 0).astype(int)
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(int)
    return X, y, "Heart Disease UCI"


def load_diabetes_data():
    """Pima Indians Diabetes – Kaggle/UCI mirror."""
    url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
           "pima-indians-diabetes.data.csv")
    try:
        path = _download(url, "pima_diabetes.csv")
        cols = ["preg","glucose","bp","skin","insulin","bmi","dpf","age","target"]
        df = pd.read_csv(path, header=None, names=cols)
    except Exception:
        rng = np.random.default_rng(2025)
        X = rng.standard_normal((768, 8)).astype(np.float32)
        y = rng.integers(0, 2, 768)
        return X, y, "Pima Diabetes (proxy)"
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(int)
    return X, y, "Pima Diabetes"


def load_heart_failure_data():
    """
    Heart Failure Clinical Records (UCI, Chicco & Jurman 2020).
    Task: predict all-cause mortality (DEATH_EVENT) at follow-up.
    Features: age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
              high_blood_pressure, platelets, serum_creatinine, serum_sodium,
              sex, smoking, time (follow-up days).
    n=299, binary (96 deaths / 203 survivors).
    Source: UCI ML Repository — direct CSV, no authentication needed.
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00519/heart_failure_clinical_records_dataset.csv")
    try:
        path = _download(url, "heart_failure_records.csv")
        df = pd.read_csv(path)
        df = df.dropna()
        target_col = "DEATH_EVENT"
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols].values.astype(np.float32)
        y = df[target_col].values.astype(int)
        return X, y, "Heart Failure (UCI)"
    except Exception as e:
        print(f"  Heart failure download failed ({e}); using proxy dataset.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=299, n_features=12, n_informative=6,
                                   n_redundant=2, weights=[0.68, 0.32], random_state=42)
        return X.astype(np.float32), y.astype(int), "Heart Failure (proxy)"


def load_tcga_brca_data():
    """
    TCGA-BRCA mRNA expression (UCSC Xena) + ER status (binary classification).
    Task: ER+ vs ER- (determines endocrine therapy eligibility — direct clinical impact).
    Features: top-200 variance genes (PCA to 4 applied inside CV loop).
    Source: UCSC Xena public S3 bucket (no credentials required).
    """
    expr_url = ("https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
                "TCGA.BRCA.sampleMap%2FHiSeqV2_percentile.gz")
    clin_url = ("https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
                "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix")
    try:
        import gzip
        clin_path = _download(clin_url, "tcga_brca_clinical.tsv")
        expr_path = _download(expr_url, "tcga_brca_expr.gz")

        clin_df = pd.read_csv(clin_path, sep="\t", index_col=0, low_memory=False)
        clin_df.index = clin_df.index.str.strip()

        er_col = next((c for c in ["ER_Status_nature2012", "er_status_by_ihc",
                                   "ER_status", "er_status_ihc"] if c in clin_df.columns), None)
        if er_col is None:
            raise ValueError(f"ER column not found; available: {list(clin_df.columns[:10])}")

        # Binary ER label
        er_raw = clin_df[er_col].astype(str).str.upper().str.strip()
        pos_vals = {"POSITIVE", "POS", "+"}
        neg_vals = {"NEGATIVE", "NEG", "-"}
        er_mask = er_raw.isin(pos_vals | neg_vals)
        er_labeled = er_raw[er_mask]
        labeled_samples = er_labeled.index

        # Load expression (genes × samples) — transpose to (samples × genes)
        with gzip.open(expr_path, "rt") as fh:
            expr_df = pd.read_csv(fh, sep="\t", index_col=0)
        expr_df.columns = expr_df.columns.str.strip()
        expr_df = expr_df.T  # samples × genes

        common = expr_df.index.intersection(labeled_samples)
        if len(common) < 50:
            raise ValueError(f"Too few matched samples: {len(common)}")

        expr_df = expr_df.loc[common]
        y_series = er_labeled.loc[common]
        y = y_series.isin(pos_vals).astype(int).values

        # Keep top-200 variance genes (PCA to 4 happens inside CV)
        var = expr_df.var(axis=0)
        top_genes = var.nlargest(200).index
        X = expr_df[top_genes].values.astype(np.float32)

        nan_mask = ~np.isnan(X).any(axis=1)
        X, y = X[nan_mask], y[nan_mask]

        # Subsample to ≤1500 (stratified)
        if len(y) > 1500:
            rng = np.random.default_rng(42)
            classes, counts = np.unique(y, return_counts=True)
            idx = []
            for cls, cnt in zip(classes, counts):
                k = max(1, int(round(1500 * cnt / len(y))))
                k = min(k, cnt)
                idx.extend(rng.choice(np.where(y == cls)[0], k, replace=False).tolist())
            X, y = X[np.array(idx)], y[np.array(idx)]

        return X, y, "TCGA-BRCA (ER status)"

    except Exception as e:
        print(f"  TCGA-BRCA download failed ({e}); using proxy.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=200, n_informative=15,
                                   n_redundant=10, weights=[0.27, 0.73],
                                   random_state=2024)
        return X.astype(np.float32), y.astype(int), "TCGA-BRCA (proxy)"


def load_nhanes_diabetes_data():
    """
    NHANES 2017-2018 diabetes prevalence dataset.
    Task: Type 2 diabetes (HbA1c >= 6.5 % or doctor-diagnosed) vs. no diabetes.
    Features: age, sex, BMI, waist circumference, systolic BP, diastolic BP, HbA1c.
    Source: CDC public XPT files (no registration required).
    n ~ 2000 after stratified subsample; class imbalance ~15 % positive (real-world).
    """
    # CDC reorganised NHANES hosting in 2024; correct base URL uses DataFiles path
    BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    files = {
        "demo": (BASE + "DEMO_J.xpt", "nhanes_demo2.xpt"),
        "ghb":  (BASE + "GHB_J.xpt",  "nhanes_ghb.xpt"),
        "diq":  (BASE + "DIQ_J.xpt",  "nhanes_diq.xpt"),
        "bmx":  (BASE + "BMX_J.xpt",  "nhanes_bmx.xpt"),
        "bpx":  (BASE + "BPX_J.xpt",  "nhanes_bpx.xpt"),
    }
    try:
        dfs = {}
        for key, (url, fname) in files.items():
            path = _download(url, fname)
            # Try pyreadstat first (handles both XPORT v5 and v8), fall back to pandas
            try:
                import pyreadstat
                df_tmp, _ = pyreadstat.read_xport(str(path))
                dfs[key] = df_tmp
            except Exception:
                dfs[key] = pd.read_sas(str(path), encoding="latin-1")

        df = dfs["demo"][["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
        for key, cols in [("ghb", ["SEQN", "LBXGH"]),
                          ("diq", ["SEQN", "DIQ010"]),
                          ("bmx", ["SEQN", "BMXBMI", "BMXWAIST"]),
                          ("bpx", ["SEQN", "BPXSY1", "BPXDI1"])]:
            avail = [c for c in cols if c in dfs[key].columns]
            df = df.merge(dfs[key][avail], on="SEQN", how="left")

        df = df[df["RIDAGEYR"] >= 18].copy()

        # Diabetes label: HbA1c >= 6.5 OR diagnosed (DIQ010 == 1)
        has_hba1c = "LBXGH" in df.columns
        has_diag  = "DIQ010" in df.columns
        if has_hba1c and has_diag:
            df["diabetes"] = ((df["LBXGH"] >= 6.5) | (df["DIQ010"] == 1)).astype(int)
        elif has_hba1c:
            df["diabetes"] = (df["LBXGH"] >= 6.5).astype(int)
        else:
            df["diabetes"] = (df["DIQ010"] == 1).astype(int)

        feat_cols = [c for c in ["RIDAGEYR", "RIAGENDR", "BMXBMI", "BMXWAIST",
                                  "BPXSY1", "BPXDI1", "LBXGH"] if c in df.columns]
        df = df[feat_cols + ["diabetes"]].dropna()

        X = df[feat_cols].values.astype(np.float32)
        y = df["diabetes"].values.astype(int)

        # Stratified subsample to ≤2000
        if len(y) > 2000:
            rng = np.random.default_rng(42)
            classes, counts = np.unique(y, return_counts=True)
            idx = []
            for cls, cnt in zip(classes, counts):
                k = max(1, int(round(2000 * cnt / len(y))))
                k = min(k, cnt)
                idx.extend(rng.choice(np.where(y == cls)[0], k, replace=False).tolist())
            X, y = X[np.array(idx)], y[np.array(idx)]

        return X, y, "NHANES Diabetes 2017-18"

    except Exception as e:
        print(f"  NHANES download failed ({e}); using proxy.")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=7, n_informative=5,
                                   n_redundant=1, weights=[0.85, 0.15], random_state=2025)
        return X.astype(np.float32), y.astype(int), "NHANES Diabetes (proxy)"


# ── main entry point ────────────────────────────────────────────────────────────
LOADERS = {
    "sklearn_bc":     load_breast_cancer_data,
    "uci_heart":      load_heart_disease_data,
    "pima_diabetes":  load_diabetes_data,
    "heart_failure":  load_heart_failure_data,
    "tcga_brca":      load_tcga_brca_data,
    "nhanes_diabetes":load_nhanes_diabetes_data,
}

def get_dataset(key: str):
    return LOADERS[key]()
