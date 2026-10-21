"""
QML-Health-READY Benchmark v3 — single-file Colab version
==========================================================
Paper:   Parity Without Advantage: A Matched, Equivalence-Tested Benchmark of
         Quantum Machine Learning on Six Clinical Datasets and the
         QML-Health-READY Reporting Framework
Authors: Mohammed Siraj B, Ruban S
Code:    github.com/sirajbb/qml-health-ready

Changes from v2 (responding to internal Q1-style review):
  A4  TCGA-BRCA: primary-tumour samples only (barcode suffix -01), one per patient.
  A5  Heart failure: follow-up `time` removed from predictors (post-baseline leakage).
      Pima: physiologically impossible zeros set to missing; median imputation
      fitted inside each training fold.
  B4  Finite-shot arm: noisy QSVM (p = 0.01) with kernel entries estimated from
      1,024 shots (binomial model), plus a shot-count sweep.
  C1  Classical baselines on ALL original features (suffix _full), and a
      qubit-scaling experiment (4 / 8 / 12 qubits) on TCGA-BRCA and breast cancer.
  C2  Bandwidth grid extended down to 0.005.
  C3  VQC reads out all qubits (linear readout), so the no-entanglement ablation
      isolates entanglement instead of also dropping three features.
  C4  Implementation-sensitivity ("fault ladder") experiment: QSVM with
      (i) bandwidth fixed at 1, (ii) + unscaled PCA inputs, (iii) + the CZ-RZ-CZ
      map, which cancels to single-qubit rotations (no entanglement).
  A2  Geometric difference reported at several regularisation values, plus
      kernel-kernel alignment between the quantum kernel and the best RBF kernel.
  B5  Balanced accuracy added to the per-split metrics.

HOW TO USE IN GOOGLE COLAB
  Upload this file, then in a cell:   !python benchmark_colab_v3.py
  Results are saved after every dataset to qml_results_v3/ (safe to resume:
  re-run the same cell after a disconnect). Download
  qml_results_v3/benchmark_results.json when finished.
  Runtime (Colab CPU): FAST_MODE ~15 min; full ~5-6 h (NHANES alone ~1.5-2 h).
  Keep the Colab tab open; if the session disconnects, re-run the cell to resume.
"""

# ===== CELL 1 — Install dependencies =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "pennylane>=0.45", "pyreadstat", "-q"], check=False)

# ===== CELL 2 — Imports & configuration =====
import os, json, math, time, gzip, warnings, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             matthews_corrcoef, balanced_accuracy_score)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

FAST_MODE = os.environ.get("QML_FAST", "0") == "1"   # or set True here for a quick test
ONLY      = os.environ.get("QML_DATASETS")            # e.g. "breast_cancer"

N_QUBITS    = 4
ZZ_REPS     = 2
N_JOBS      = -1
NOISE_EVAL  = 0.01
SHOTS_EVAL  = 1024
MARGINS     = [0.01, 0.02, 0.05]
VQC_LAYERS, VQC_LR = 3, 0.05
QUBIT_SCALING_DATASETS = ["tcga_brca", "breast_cancer"]

if FAST_MODE:
    N_REPEAT_SEED, CV_OUTER, CV_INNER = 1, 3, 2
    BANDWIDTHS  = [0.01, 0.1]
    QSVM_C      = [1.0]
    VQC_EPOCHS  = 15
    NOISE_CURVE = [0.0, 0.05]
    SHOT_CURVE  = [128, 1024]
    QUBIT_SCALING = [4, 8]
    PARAM_GRIDS = {
        "LR":       {"C": [1]},
        "SVM_RBF":  {"C": [1], "gamma": ["scale"]},
        "SVM_POLY": {"C": [1], "degree": [2], "gamma": ["scale"], "coef0": [1]},
        "RF":       {"n_estimators": [100], "max_depth": [5]},
        "GB":       {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
    }
    print("[FAST MODE] for testing only, not for the paper")
else:
    N_REPEAT_SEED, CV_OUTER, CV_INNER = 5, 5, 3
    BANDWIDTHS  = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    QSVM_C      = [0.1, 1, 10]
    VQC_EPOCHS  = 100
    NOISE_CURVE = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    SHOT_CURVE  = [64, 256, 1024, 8192]
    QUBIT_SCALING = [4, 8, 12]
    PARAM_GRIDS = {
        "LR":       {"C": [0.01, 0.1, 1, 10]},
        "SVM_RBF":  {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
        "SVM_POLY": {"C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"], "coef0": [0, 1]},
        "RF":       {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
                     "min_samples_leaf": [1, 3]},
        "GB":       {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2],
                     "max_depth": [3, 5]},
    }
    print("[FULL MODE] 5 seeds x 5-fold nested CV")

SEEDS = [42 + 7 * i for i in range(N_REPEAT_SEED)]
DATA_DIR = Path("qml_data");       DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("qml_results_v3"); OUT_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("breast_cancer",   "Breast Cancer Wisconsin"),
    ("heart_disease",   "Heart Disease Cleveland"),
    ("diabetes",        "Pima Indians Diabetes"),
    ("heart_failure",   "Heart Failure Clinical Records"),
    ("tcga_brca",       "TCGA-BRCA ER status"),
    ("nhanes_diabetes", "NHANES Diabetes 2017-18"),
]
if ONLY:
    DATASETS = [d for d in DATASETS if d[0] in ONLY.split(",")]

CLASSICAL = ["LR", "SVM_RBF", "SVM_POLY", "RF", "GB"]

# ===== CELL 3 — Dataset loaders (fail loudly: no synthetic fallbacks) =====

def _download(url, filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, path)
    return path

def _stratified_cap(X, y, n_max, seed=42):
    if len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        k = min(max(1, int(round(n_max * len(cls_idx) / len(y)))), len(cls_idx))
        idx.extend(rng.choice(cls_idx, k, replace=False).tolist())
    idx = np.sort(np.array(idx))
    return X[idx], y[idx]

DATA_NOTES = {}

def load_breast_cancer_data():
    d = load_breast_cancer()
    return d.data.astype(np.float64), d.target.astype(int)

def load_heart_disease_data():
    path = _download("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "heart-disease/processed.cleveland.data", "heart_cleveland.csv")
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(path, header=None, names=cols, na_values="?").dropna()
    y = (df.pop("target") > 0).astype(int).values
    return df.values.astype(np.float64), y

def load_diabetes_data():
    path = _download("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
                     "pima-indians-diabetes.data.csv", "pima_diabetes.csv")
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :8].values.astype(np.float64)
    # glucose, blood pressure, skin thickness, insulin, BMI: zero means missing
    for j in [1, 2, 3, 4, 5]:
        X[X[:, j] == 0, j] = np.nan
    DATA_NOTES["diabetes"] = {"missing_fraction_by_column":
                              np.isnan(X).mean(0).round(3).tolist()}
    return X, df.iloc[:, 8].values.astype(int)

def load_heart_failure_data():
    path = _download("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "00519/heart_failure_clinical_records_dataset.csv",
                     "heart_failure_records.csv")
    df = pd.read_csv(path).dropna()
    y = df.pop("DEATH_EVENT").astype(int).values
    df = df.drop(columns=["time"])          # follow-up time is post-baseline information
    DATA_NOTES["heart_failure"] = {"predictors": list(df.columns)}
    return df.values.astype(np.float64), y

def load_tcga_brca_data():
    base = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
    clin = pd.read_csv(_download(base + "TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix",
                                 "tcga_brca_clinical.tsv"),
                       sep="\t", index_col=0, low_memory=False)
    clin.index = clin.index.str.strip()
    er_col = next(c for c in ["ER_Status_nature2012", "er_status_by_ihc",
                              "ER_status", "er_status_ihc"] if c in clin.columns)
    er = clin[er_col].astype(str).str.upper().str.strip()
    pos, neg = {"POSITIVE", "POS", "+"}, {"NEGATIVE", "NEG", "-"}
    er = er[er.isin(pos | neg)]
    with gzip.open(_download(base + "TCGA.BRCA.sampleMap%2FHiSeqV2_percentile.gz",
                             "tcga_brca_expr.gz"), "rt") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0).T
    expr.index = expr.index.str.strip()
    common = expr.index.intersection(er.index)
    n_before = len(common)
    types = Counter(s[13:15] for s in common)
    # Primary solid tumour only (barcode sample type 01), one sample per patient
    primary = [s for s in common if s[13:15] == "01"]
    keep, seen = [], set()
    for s in sorted(primary):
        if s[:12] not in seen:
            seen.add(s[:12]); keep.append(s)
    DATA_NOTES["tcga_brca"] = {"samples_with_er_label": n_before,
                               "sample_type_counts": dict(types),
                               "primary_tumour_one_per_patient": len(keep),
                               "er_column": er_col}
    expr = expr.loc[keep]
    y = er.loc[keep].isin(pos).astype(int).values
    # Top-200 variance genes: unsupervised, label-free selection on the cohort
    X = expr[expr.var(axis=0).nlargest(200).index].values.astype(np.float64)
    ok = ~np.isnan(X).any(axis=1)
    return _stratified_cap(X[ok], y[ok], 1500)

def load_nhanes_diabetes_data():
    base = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
    import pyreadstat
    tabs = {}
    for key in ["DEMO_J", "GHB_J", "DIQ_J", "BMX_J", "BPX_J"]:
        tabs[key], _ = pyreadstat.read_xport(str(_download(base + key + ".xpt",
                                                           f"nhanes_{key}.xpt")))
    df = tabs["DEMO_J"][["SEQN", "RIDAGEYR", "RIAGENDR"]]
    for key, cols in [("GHB_J", ["LBXGH"]), ("DIQ_J", ["DIQ010"]),
                      ("BMX_J", ["BMXBMI", "BMXWAIST"]), ("BPX_J", ["BPXSY1", "BPXDI1"])]:
        df = df.merge(tabs[key][["SEQN"] + cols], on="SEQN", how="left")
    df = df[df["RIDAGEYR"] >= 18]
    df = df.dropna(subset=["RIDAGEYR", "RIAGENDR", "BMXBMI", "BMXWAIST",
                           "BPXSY1", "BPXDI1", "LBXGH", "DIQ010"])
    DATA_NOTES["nhanes_diabetes"] = {"eligible_adults_complete": int(len(df))}
    y = ((df["LBXGH"] >= 6.5) | (df["DIQ010"] == 1)).astype(int).values
    # HbA1c defines the label, so it is excluded from the predictors (no target leakage)
    X = df[["RIDAGEYR", "RIAGENDR", "BMXBMI", "BMXWAIST",
            "BPXSY1", "BPXDI1"]].values.astype(np.float64)
    return _stratified_cap(X, y, 2000)

LOADERS = {
    "breast_cancer":   load_breast_cancer_data,
    "heart_disease":   load_heart_disease_data,
    "diabetes":        load_diabetes_data,
    "heart_failure":   load_heart_failure_data,
    "tcga_brca":       load_tcga_brca_data,
    "nhanes_diabetes": load_nhanes_diabetes_data,
}

# ===== CELL 4 — Quantum feature maps, kernels, shots and VQC =====

def _depolarise(p, wires):
    if p > 0:
        for w in wires:
            qml.DepolarizingChannel(p, wires=w)

def zz_feature_map(x, n_q, entangle=True, p=0.0, cz_bug=False):
    """Havlicek et al. (2019) ZZ feature map, linear chain, depolarising channel after
    every gate. cz_bug=True reproduces the faulty CZ-RZ-CZ block (equal to a plain RZ)."""
    for _ in range(ZZ_REPS):
        for i in range(n_q):
            qml.Hadamard(wires=i);              _depolarise(p, [i])
            qml.RZ(2.0 * x[..., i], wires=i);   _depolarise(p, [i])
        if entangle:
            two_q = qml.CZ if cz_bug else qml.CNOT
            for i in range(n_q - 1):
                phi = 2.0 * (np.pi - x[..., i]) * (np.pi - x[..., i + 1])
                two_q(wires=[i, i + 1]);        _depolarise(p, [i, i + 1])
                qml.RZ(phi, wires=i + 1);       _depolarise(p, [i + 1])
                two_q(wires=[i, i + 1]);        _depolarise(p, [i, i + 1])

_QNODES = {}

def _state_qnode(n_q, entangle, p, cz_bug):
    key = (n_q, entangle, p, cz_bug)
    if key not in _QNODES:
        wires = list(range(n_q))
        if p > 0:
            dev = qml.device("default.mixed", wires=n_q)
            @qml.qnode(dev)
            def circuit(x):
                zz_feature_map(x, n_q, entangle, p, cz_bug)
                return qml.density_matrix(wires=wires)
        else:
            dev = qml.device("default.qubit", wires=n_q)
            @qml.qnode(dev)
            def circuit(x):
                zz_feature_map(x, n_q, entangle, 0.0, cz_bug)
                return qml.state()
        _QNODES[key] = circuit
    return _QNODES[key]

def embed(Z, bandwidth, entangle=True, p=0.0, cz_bug=False, chunk=256):
    n_q = Z.shape[1]
    f = _state_qnode(n_q, entangle, p, cz_bug)
    X = bandwidth * np.asarray(Z, dtype=np.float64)
    return np.concatenate([np.asarray(f(X[i:i + chunk])) for i in range(0, len(X), chunk)])

def quantum_gram(A, B):
    """|<a|b>|^2 for pure states; Tr(rho_a rho_b) for mixed states."""
    if A.ndim == 2:
        return np.abs(A.conj() @ B.T) ** 2
    d = A.shape[1]
    return np.real(A.reshape(len(A), d * d) @ np.transpose(B, (0, 2, 1)).reshape(len(B), d * d).T)

def shot_sample(K, shots, rng, symmetric):
    """Binomial shot model: each entry estimated as the success frequency of `shots`
    Bernoulli trials with success probability equal to the exact kernel value."""
    P = np.clip(K, 0.0, 1.0)
    if symmetric:
        iu = np.triu_indices(len(P))
        S = np.zeros_like(P)
        S[iu] = rng.binomial(shots, P[iu]) / shots
        S = S + S.T - np.diag(np.diag(S))
        w, V = np.linalg.eigh(S)                      # project onto the PSD cone
        return (V * np.clip(w, 0, None)) @ V.T
    return rng.binomial(shots, P) / shots

def _vqc_qnode(entangle, n_layers):
    wires = list(range(N_QUBITS))
    dev = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=wires, rotation="X")
        if entangle:
            qml.StronglyEntanglingLayers(w, wires=wires)
        else:
            for l in range(n_layers):
                for q in wires:
                    qml.Rot(w[l, q, 0], w[l, q, 1], w[l, q, 2], wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in wires]
    return circuit

def vqc_fit_predict(Z_tr, y_tr, Z_te, entangle, seed, layers=VQC_LAYERS, lr=VQC_LR):
    """AngleEmbedding + StronglyEntanglingLayers; logit = a . <Z_q> + b over all qubits;
    full-batch Adam on binary cross-entropy."""
    circuit = _vqc_qnode(entangle, layers)
    rng = np.random.default_rng(seed)
    Xtr = pnp.array(np.clip(Z_tr, -np.pi, np.pi), requires_grad=False)
    Xte = np.clip(Z_te, -np.pi, np.pi)
    yt = pnp.array(y_tr.astype(float), requires_grad=False)
    w = pnp.array(rng.normal(0, 0.5, (layers, N_QUBITS, 3)), requires_grad=True)
    a = pnp.array(np.full(N_QUBITS, 0.5), requires_grad=True)
    b = pnp.array(0.0, requires_grad=True)

    def logits(X, w, a, b):
        return pnp.dot(a, pnp.stack(circuit(X, w))) + b

    def loss(w, a, b):
        z = logits(Xtr, w, a, b)
        return pnp.mean(pnp.logaddexp(0.0, z) - yt * z)

    opt = qml.AdamOptimizer(lr)
    for _ in range(VQC_EPOCHS):
        w, a, b = opt.step(loss, w, a, b)
    return np.asarray(logits(Xte, w, a, b))

# ===== CELL 5 — Evaluation =====

def _metrics(y_true, y_pred, score):
    return {"auc": roc_auc_score(y_true, score),
            "acc": accuracy_score(y_true, y_pred),
            "bacc": balanced_accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)}

def _aggregate(scores):
    out = {}
    for m in scores[0]:
        v = [float(s[m]) for s in scores]
        se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        t = stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
        out[m] = {"mean": float(np.mean(v)), "ci": float(t * se), "n": len(v), "values": v}
    return out

def _prep(X_tr, X_te, n_comp, seed, restandardise=True):
    imp = SimpleImputer(strategy="median").fit(X_tr)
    s1 = StandardScaler().fit(imp.transform(X_tr))
    F_tr, F_te = s1.transform(imp.transform(X_tr)), s1.transform(imp.transform(X_te))
    pca = PCA(n_comp, random_state=seed).fit(F_tr)
    P_tr, P_te = pca.transform(F_tr), pca.transform(F_te)
    if restandardise:
        s2 = StandardScaler().fit(P_tr)
        P_tr, P_te = s2.transform(P_tr), s2.transform(P_te)
    return F_tr, F_te, P_tr, P_te

def make_folds(X, y):
    folds = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(CV_OUTER, shuffle=True, random_state=seed).split(X, y):
            F_tr, F_te, Z_tr, Z_te = _prep(X[tr], X[te], N_QUBITS, seed)
            _, _, U_tr, U_te = _prep(X[tr], X[te], N_QUBITS, seed, restandardise=False)
            mm = MinMaxScaler((0.0, np.pi)).fit(Z_tr)
            folds.append({"seed": seed, "tr": tr, "te": te, "y_tr": y[tr], "y_te": y[te],
                          "Z_tr": Z_tr, "Z_te": Z_te,          # 4 standardised PCs
                          "U_tr": U_tr, "U_te": U_te,          # 4 raw PCs (fault ladder)
                          "M_tr": mm.transform(Z_tr),          # 4 PCs min-max scaled to [0, pi]
                          "M_te": mm.transform(Z_te),
                          "F_tr": F_tr, "F_te": F_te})         # all features, standardised
    return folds

def _estimator(name):
    return {"LR": LogisticRegression(max_iter=3000),
            "SVM_RBF": SVC(kernel="rbf", probability=True),
            "SVM_POLY": SVC(kernel="poly", probability=True),
            "RF": RandomForestClassifier(random_state=0),
            "GB": GradientBoostingClassifier(random_state=0)}[name]

def run_classical(name, folds, full=False, decision=False):
    """decision=True scores an SVM by its decision function (no Platt scaling), as for the QSVM."""
    scores = []
    for f in folds:
        Xtr, Xte = (f["F_tr"], f["F_te"]) if full else (f["Z_tr"], f["Z_te"])
        est = SVC(kernel="rbf") if decision else _estimator(name)
        gs = GridSearchCV(est, PARAM_GRIDS[name], scoring="roc_auc", n_jobs=N_JOBS,
                          cv=StratifiedKFold(CV_INNER, shuffle=True, random_state=f["seed"]))
        gs.fit(Xtr, f["y_tr"])
        score = gs.decision_function(Xte) if decision else gs.predict_proba(Xte)[:, 1]
        scores.append(_metrics(f["y_te"], gs.predict(Xte), score))
    return scores

def _kernels(f, bw, entangle, p, inputs, cz_bug, shots, rng, cache):
    key = (bw, entangle, p, inputs, cz_bug)
    if cache is not None and key in cache:
        K, K_te = cache[key]
    else:
        S_tr = embed(f[inputs + "_tr"], bw, entangle, p, cz_bug)
        S_te = embed(f[inputs + "_te"], bw, entangle, p, cz_bug)
        K, K_te = quantum_gram(S_tr, S_tr), quantum_gram(S_te, S_tr)
        if cache is not None:
            cache[key] = (K, K_te)
    if shots:
        K, K_te = shot_sample(K, shots, rng, True), shot_sample(K_te, shots, rng, False)
    return K, K_te

def qsvm_fold(f, entangle=True, p=0.0, bandwidths=None, inputs="Z", cz_bug=False,
              shots=None, fixed_C=None, cache=None):
    """Inner CV over (bandwidth, C) on the training kernel; refit; score the test fold."""
    bandwidths = bandwidths or BANDWIDTHS
    Cs = [fixed_C] if fixed_C is not None else QSVM_C
    y = f["y_tr"]
    inner = list(StratifiedKFold(CV_INNER, shuffle=True,
                                 random_state=f["seed"]).split(f[inputs + "_tr"], y))
    rng = np.random.default_rng(f["seed"] * 1000 + len(f["tr"]))
    best = (-np.inf, None, None, None)
    for bw in bandwidths:
        K, K_te = _kernels(f, bw, entangle, p, inputs, cz_bug, shots, rng, cache)
        for C in Cs:
            aucs = []
            for a, v in inner:
                svm = SVC(kernel="precomputed", C=C).fit(K[np.ix_(a, a)], y[a])
                aucs.append(roc_auc_score(y[v], svm.decision_function(K[np.ix_(v, a)])))
            if np.mean(aucs) > best[0]:
                best = (np.mean(aucs), bw, C, (K, K_te))
    _, bw, C, (K, K_te) = best
    svm = SVC(kernel="precomputed", C=C).fit(K, y)
    return _metrics(f["y_te"], svm.predict(K_te), svm.decision_function(K_te)), bw, C

def run_qsvm(folds, **kw):
    scores, chosen = [], []
    for f in folds:
        m, bw, C = qsvm_fold(f, **kw)
        scores.append(m); chosen.append((bw, C))
    return scores, chosen

def run_vqc(folds, entangle=True):
    scores = []
    for f in folds:
        z = vqc_fit_predict(f["Z_tr"], f["y_tr"], f["Z_te"], entangle, f["seed"])
        scores.append(_metrics(f["y_te"], (z > 0).astype(int), z))
    return scores

VQC_GRID = [(1, 0.05), (3, 0.05), (5, 0.05), (3, 0.15)]   # (layers, Adam learning rate)

def run_vqc_tuned(folds):
    """VQC with (layers, learning rate) chosen by 3-fold inner CV on AUROC, as for the QSVM."""
    scores, chosen = [], []
    for f in folds:
        y = f["y_tr"]
        inner = list(StratifiedKFold(CV_INNER, shuffle=True,
                                     random_state=f["seed"]).split(f["Z_tr"], y))
        cv = {}
        for layers, lr in VQC_GRID:
            cv[(layers, lr)] = np.mean([roc_auc_score(y[v], vqc_fit_predict(
                f["Z_tr"][a], y[a], f["Z_tr"][v], True, f["seed"], layers, lr))
                for a, v in inner])
        layers, lr = max(cv, key=cv.get)
        z = vqc_fit_predict(f["Z_tr"], y, f["Z_te"], True, f["seed"], layers, lr)
        scores.append(_metrics(f["y_te"], (z > 0).astype(int), z))
        chosen.append((layers, lr))
    return scores, chosen

def paired_stats(q, c, ratio):
    """Quantum minus comparator, paired over identical outer folds
    (Nadeau-Bengio / Bouckaert-Frank corrected variance)."""
    q, c = np.asarray(q), np.asarray(c)
    d = q - c
    J, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se = sd * math.sqrt(1.0 / J + ratio)
    out = {"mean_diff": mean, "sd": sd, "se_corrected": se, "df": J - 1,
           "p_corrected_t": float(2 * stats.t.sf(abs(mean / se), J - 1)) if se > 0 else float("nan"),
           "cohens_dz": mean / sd if sd > 0 else float("nan")}
    for M in MARGINS:
        out[f"p_tost_{M}"] = float(max(stats.t.sf((mean + M) / se, J - 1),
                                       stats.t.cdf((mean - M) / se, J - 1))) if se > 0 else float("nan")
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(q, c, zero_method="zsplit").pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    return out

# ===== CELL 6 — Data complexity, kernel similarity and qubit scaling =====

def centred_alignment(K1, K2):
    n = len(K1)
    H = np.eye(n) - np.ones((n, n)) / n
    A, B = H @ K1 @ H, H @ K2 @ H
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))

def geometric_difference(K_c, K_q, lam):
    """g(K_C || K_Q) of Huang et al. (2021); kernels trace-normalised to n."""
    n = len(K_c)
    K_c = K_c * n / np.trace(K_c)
    K_q = K_q * n / np.trace(K_q)
    w, V = np.linalg.eigh(K_q)
    sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    return float(np.sqrt(np.linalg.norm(sq @ np.linalg.inv(K_c + lam * np.eye(n)) @ sq, 2)))

def complexity(X, y, bandwidth, n_max=300):
    Xs, ys = _stratified_cap(X, y, n_max, seed=0)
    _, _, Z, _ = _prep(Xs, Xs[:1], N_QUBITS, 0)
    K_q = quantum_gram(*(2 * [embed(Z, bandwidth, True)]))
    K_p = quantum_gram(*(2 * [embed(Z, bandwidth, False)]))
    K_c = rbf_kernel(Z, gamma=1.0 / N_QUBITS)
    gammas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    sims = {g: centred_alignment(K_q, rbf_kernel(Z, gamma=g)) for g in gammas}
    g_best = max(sims, key=sims.get)
    off = ~np.eye(len(Z), dtype=bool)
    yy = 2.0 * ys - 1.0
    c0, c1 = Z[ys == 0], Z[ys == 1]
    _, nn = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    return {"n_used": len(Z), "bandwidth": bandwidth,
            "F1_fisher_max": float(np.max((c0.mean(0) - c1.mean(0)) ** 2
                                          / (c0.var(0) + c1.var(0) + 1e-12))),
            "N2_nn_purity": float((ys[nn[:, 1]] == ys).mean()),
            "KTA_quantum": centred_alignment(K_q, np.outer(yy, yy)),
            "KTA_product": centred_alignment(K_p, np.outer(yy, yy)),
            "KTA_rbf": centred_alignment(K_c, np.outer(yy, yy)),
            "geom_diff_CQ": {str(l): geometric_difference(K_c, K_q, l)
                             for l in [1e-4, 1e-3, 1e-2, 1e-1]},
            "sqrt_n": float(math.sqrt(len(Z))),
            "kq_vs_rbf_alignment_best": sims[g_best], "kq_vs_rbf_best_gamma": g_best,
            "kq_vs_product_alignment": centred_alignment(K_q, K_p),
            "kq_offdiag_mean": float(K_q[off].mean()), "kq_offdiag_var": float(K_q[off].var())}

def qubit_scaling(X, y):
    """First-seed folds: QSVM (tuned bandwidth) vs SVM-RBF on q principal components,
    plus kernel concentration at the default bandwidth of 1."""
    out = {}
    for q in QUBIT_SCALING:
        if q > X.shape[1]:
            continue
        q_auc, c_auc, bws, var_b1 = [], [], [], []
        for tr, te in StratifiedKFold(CV_OUTER, shuffle=True, random_state=SEEDS[0]).split(X, y):
            _, _, Z_tr, Z_te = _prep(X[tr], X[te], q, SEEDS[0])
            f = {"seed": SEEDS[0], "tr": tr, "y_tr": y[tr], "y_te": y[te],
                 "Z_tr": Z_tr, "Z_te": Z_te}
            m, bw, _ = qsvm_fold(f)
            q_auc.append(m["auc"]); bws.append(bw)
            gs = GridSearchCV(_estimator("SVM_RBF"), PARAM_GRIDS["SVM_RBF"], scoring="roc_auc",
                              cv=StratifiedKFold(CV_INNER, shuffle=True, random_state=SEEDS[0]),
                              n_jobs=N_JOBS).fit(Z_tr, y[tr])
            c_auc.append(roc_auc_score(y[te], gs.predict_proba(Z_te)[:, 1]))
            S = embed(Z_tr[:200], 1.0, True)
            K = quantum_gram(S, S)
            var_b1.append(float(K[~np.eye(len(K), dtype=bool)].var()))
        out[str(q)] = {"qsvm_auc": float(np.mean(q_auc)), "svm_rbf_auc": float(np.mean(c_auc)),
                       "chosen_bandwidths": bws, "offdiag_var_b1": float(np.mean(var_b1))}
        print(f"    qubits={q:<3} QSVM {np.mean(q_auc):.3f}  SVM-RBF {np.mean(c_auc):.3f}  "
              f"bw {Counter(bws).most_common(1)[0][0]}  offdiag var(b=1) {np.mean(var_b1):.2e}",
              flush=True)
    return out

# ===== CELL 7 — Benchmark one dataset =====

def _report(name, res, times):
    print(f"  {name:<20} AUC {res[name]['auc']['mean']:.3f} ± {res[name]['auc']['ci']:.3f}"
          f"   ({times[name]:.0f}s)", flush=True)

def benchmark_dataset(label, X, y):
    folds = make_folds(X, y)
    ratio = float(np.mean([len(f["te"]) / len(f["tr"]) for f in folds]))
    res, times, meta = {}, {}, {}

    for name in CLASSICAL:
        t0 = time.time(); res[name] = _aggregate(run_classical(name, folds))
        times[name] = time.time() - t0; _report(name, res, times)
    for name in CLASSICAL:
        full = name + "_full"
        t0 = time.time(); res[full] = _aggregate(run_classical(name, folds, full=True))
        times[full] = time.time() - t0; _report(full, res, times)

    # Noisy exact and noisy finite-shot arms share the simulated density matrices per fold
    t0 = time.time()
    sc_n, sc_s, ch_n, ch_s = [], [], [], []
    for f in folds:
        cache = {}
        m, bw, C = qsvm_fold(f, True, NOISE_EVAL, cache=cache)
        sc_n.append(m); ch_n.append((bw, C))
        m, bw, C = qsvm_fold(f, True, NOISE_EVAL, shots=SHOTS_EVAL, cache=cache)
        sc_s.append(m); ch_s.append((bw, C))
    res["QSVM_noisy"], res["QSVM_noisy_shots"] = _aggregate(sc_n), _aggregate(sc_s)
    meta["QSVM_noisy_chosen_bandwidth_C"], meta["QSVM_noisy_shots_chosen_bandwidth_C"] = ch_n, ch_s
    times["QSVM_noisy"] = times["QSVM_noisy_shots"] = time.time() - t0
    _report("QSVM_noisy", res, times); _report("QSVM_noisy_shots", res, times)

    quantum_runs = [
        ("QSVM",                dict(entangle=True, p=0.0)),
        ("QSVM_noent",          dict(entangle=False, p=0.0)),
        ("QSVM_fault_b1",       dict(entangle=True, bandwidths=[1.0])),
        ("QSVM_fault_unscaled", dict(entangle=True, bandwidths=[1.0], inputs="U")),
        ("QSVM_fault_v1map",    dict(entangle=True, bandwidths=[1.0], inputs="U", cz_bug=True)),
    ]
    for name, kw in quantum_runs:
        t0 = time.time()
        sc, chosen = run_qsvm(folds, **kw)
        res[name] = _aggregate(sc); times[name] = time.time() - t0
        meta[f"{name}_chosen_bandwidth_C"] = chosen
        _report(name, res, times)

    for name, ent in [("VQC", True), ("VQC_noent", False)]:
        t0 = time.time(); res[name] = _aggregate(run_vqc(folds, ent))
        times[name] = time.time() - t0; _report(name, res, times)

    best = max(CLASSICAL, key=lambda k: res[k]["auc"]["mean"])
    best_full = max([c + "_full" for c in CLASSICAL], key=lambda k: res[k]["auc"]["mean"])
    v = lambda k: res[k]["auc"]["values"]
    comparisons = {
        "QSVM_vs_best": paired_stats(v("QSVM"), v(best), ratio),
        "QSVM_vs_SVM_RBF": paired_stats(v("QSVM"), v("SVM_RBF"), ratio),
        "QSVM_noent_vs_best": paired_stats(v("QSVM_noent"), v(best), ratio),
        "QSVM_noisy_shots_vs_best": paired_stats(v("QSVM_noisy_shots"), v(best), ratio),
        "VQC_vs_best": paired_stats(v("VQC"), v(best), ratio),
        "QSVM_vs_QSVM_noent": paired_stats(v("QSVM"), v("QSVM_noent"), ratio),
        "QSVM_vs_QSVM_noisy": paired_stats(v("QSVM"), v("QSVM_noisy"), ratio),
        "QSVM_vs_QSVM_noisy_shots": paired_stats(v("QSVM"), v("QSVM_noisy_shots"), ratio),
        "VQC_vs_VQC_noent": paired_stats(v("VQC"), v("VQC_noent"), ratio),
        "best_full_vs_best_4pc": paired_stats(v(best_full), v(best), ratio),
    }
    for fault in ["QSVM_fault_b1", "QSVM_fault_unscaled", "QSVM_fault_v1map"]:
        comparisons[f"{fault}_vs_best"] = paired_stats(v(fault), v(best), ratio)

    # Noise and shot sweeps on first-seed folds, at the (bandwidth, C) chosen by ideal QSVM
    t0 = time.time()
    first = [(i, f) for i, f in enumerate(folds) if f["seed"] == SEEDS[0]]
    picks = meta["QSVM_chosen_bandwidth_C"]
    noise_curve = {str(p): float(np.mean([
        qsvm_fold(f, True, p, bandwidths=[picks[i][0]], fixed_C=picks[i][1])[0]["auc"]
        for i, f in first])) for p in NOISE_CURVE}
    shot_curve = {str(s): float(np.mean([
        qsvm_fold(f, True, 0.0, bandwidths=[picks[i][0]], fixed_C=picks[i][1], shots=s)[0]["auc"]
        for i, f in first])) for s in SHOT_CURVE}
    times["sweeps"] = time.time() - t0
    print(f"  noise curve {noise_curve}\n  shot curve  {shot_curve}   ({times['sweeps']:.0f}s)",
          flush=True)

    modal_bw = Counter(bw for bw, _ in picks).most_common(1)[0][0]
    comp = complexity(X, y, modal_bw)
    print(f"  complexity  g(lam=1e-3)={comp['geom_diff_CQ']['0.001']:.2f}  "
          f"KQ~RBF alignment={comp['kq_vs_rbf_alignment_best']:.3f}", flush=True)
    scaling = None
    if label in QUBIT_SCALING_DATASETS:
        t0 = time.time()
        print("  qubit scaling:", flush=True)
        scaling = qubit_scaling(X, y)
        times["qubit_scaling"] = time.time() - t0

    res["__meta__"] = {
        "dataset": label, "n": int(len(y)), "n_features": int(X.shape[1]),
        "pos_rate": float(y.mean()), "n_observations": len(folds),
        "test_train_ratio": ratio, "best_classical": best, "best_classical_full": best_full,
        "n_train_mean": float(np.mean([len(f["tr"]) for f in folds])),
        "n_test_mean": float(np.mean([len(f["te"]) for f in folds])),
        "comparisons": comparisons, "time_sec": times,
        "noise_curve": noise_curve, "shot_curve": shot_curve,
        "complexity": comp, "qubit_scaling": scaling,
        "data_notes": DATA_NOTES.get(label, {}), **meta,
        "config": {"version": "v3", "n_qubits": N_QUBITS, "zz_reps": ZZ_REPS, "seeds": SEEDS,
                   "cv_outer": CV_OUTER, "cv_inner": CV_INNER, "bandwidths": BANDWIDTHS,
                   "qsvm_C": QSVM_C, "noise_eval": NOISE_EVAL, "shots_eval": SHOTS_EVAL,
                   "vqc_layers": VQC_LAYERS, "vqc_epochs": VQC_EPOCHS, "vqc_lr": VQC_LR,
                   "vqc_readout": "linear over all qubits", "margins": MARGINS,
                   "fast_mode": FAST_MODE, "pennylane": qml.__version__},
    }
    return res

# ===== CELL 8 — Run all datasets (resumable) =====

class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

BENCH_PATH = OUT_DIR / ("benchmark_results_fast.json" if FAST_MODE else "benchmark_results.json")

# ----- VQC tuning mode (v3c): tuned VQC on identical folds, one output file per dataset so
# datasets can run in parallel. Run with:  QML_MODE=vqc_tune QML_DATASETS=<one dataset> python ...
if os.environ.get("QML_MODE") == "vqc_tune":
    base = json.loads(BENCH_PATH.read_text())
    for label, description in DATASETS:
        out = OUT_DIR / f"vqc_tuned_{label}.json"
        if out.exists():
            print(f"Skipping {label} (already tuned)")
            continue
        print(f"\n{description}", flush=True)
        X, y = LOADERS[label]()
        folds = make_folds(X, y)
        lr_chk = run_classical("LR", folds)
        diff = max(abs(a - b["auc"]) for a, b in zip(base[label]["LR"]["auc"]["values"], lr_chk))
        if diff > 1e-9:
            sys.exit(f"  Fold check FAILED for {label} (LR max diff {diff:.2e})")
        t0 = time.time()
        sc, chosen = run_vqc_tuned(folds)
        res = {"VQC_tuned": _aggregate(sc), "chosen_layers_lr": chosen,
               "grid": VQC_GRID, "time_sec": time.time() - t0, "fold_check_lr_max_diff": diff}
        out.write_text(json.dumps(res, cls=_Enc, indent=1))
        print(f"  VQC tuned AUC {res['VQC_tuned']['auc']['mean']:.3f} "
              f"[untuned {base[label]['VQC']['auc']['mean']:.3f}]  chosen "
              f"{Counter(map(tuple, chosen)).most_common(3)}  ({res['time_sec']/60:.0f} min)",
              flush=True)
    sys.exit(0)

# ----- Patch mode (v3b): add decision-function RBF SVM and the min-max [0, pi] QSVM to an
# existing v3 results file, on identical folds. Run with:  QML_MODE=patch python benchmark_colab_v3.py
if os.environ.get("QML_MODE") == "patch":
    all_labels = ["breast_cancer", "heart_disease", "diabetes", "heart_failure",
                  "tcga_brca", "nhanes_diabetes"]
    complete = lambda p: p.exists() and all(l in json.loads(p.read_text()) for l in all_labels)
    # Resume from a complete patched file if one exists; otherwise start from the upload
    src = next((p for p in [BENCH_PATH, Path("benchmark_results_v3.json")] if complete(p)), None)
    if src is None:
        sys.exit("Upload the complete benchmark_results_v3.json next to this script first.")
    all_results = json.loads(src.read_text())
    missing = [l for l, _ in DATASETS if l not in all_results]
    if missing:
        sys.exit(f"{src} is incomplete (missing {missing}). Upload the complete "
                 f"benchmark_results_v3.json next to this script.")
    print(f"Patching {src}")
    for label, description in DATASETS:
        r = all_results[label]
        if "SVM_RBF_df" in r and "QSVM_fault_minmax" in r:
            print(f"Skipping {label} (already patched)")
            continue
        print(f"\n{description}", flush=True)
        X, y = LOADERS[label]()
        folds = make_folds(X, y)
        # Fold identity: logistic regression is deterministic and well conditioned, so its
        # per-split AUROC must reproduce exactly; the near-flat bandwidth-1 QSVM kernel is
        # reported for information (it can differ slightly across platforms).
        lr = run_classical("LR", folds)
        stored = r["LR"]["auc"]["values"]
        diff = max(abs(a - b["auc"]) for a, b in zip(stored, lr))
        if len(lr) != len(stored) or diff > 1e-9:
            sys.exit(f"  Fold check FAILED for {label} (LR max diff {diff:.2e}); not patching.")
        qchk, _ = run_qsvm(folds, entangle=True, bandwidths=[1.0])
        qdiff = max(abs(a - b["auc"]) for a, b in zip(r["QSVM_fault_b1"]["auc"]["values"], qchk))
        print(f"  Fold check passed (LR max diff {diff:.1e}; bandwidth-1 QSVM max diff "
              f"{qdiff:.1e})", flush=True)
        r["__meta__"]["patch_fold_check"] = {"lr_max_diff": diff, "qsvm_b1_max_diff": qdiff}
        t0 = time.time()
        r["SVM_RBF_df"] = _aggregate(run_classical("SVM_RBF", folds, decision=True))
        t1 = time.time()
        sc, chosen = run_qsvm(folds, entangle=True, bandwidths=[1.0], inputs="M")
        r["QSVM_fault_minmax"] = _aggregate(sc)
        r["__meta__"]["time_sec"]["SVM_RBF_df"] = t1 - t0
        r["__meta__"]["time_sec"]["QSVM_fault_minmax"] = time.time() - t1
        r["__meta__"]["QSVM_fault_minmax_chosen_bandwidth_C"] = chosen
        r["__meta__"]["patch"] = "v3b"
        print(f"  SVM_RBF (decision function) AUC {r['SVM_RBF_df']['auc']['mean']:.3f}  "
              f"[Platt: {r['SVM_RBF']['auc']['mean']:.3f}]   "
              f"QSVM min-max [0,pi] AUC {r['QSVM_fault_minmax']['auc']['mean']:.3f}", flush=True)
        BENCH_PATH.write_text(json.dumps(all_results, cls=_Enc, indent=1))
        print(f"  Saved -> {BENCH_PATH}")
    print(f"\nPatch complete. Download {BENCH_PATH.resolve()}")
    sys.exit(0)

all_results = json.loads(BENCH_PATH.read_text()) if BENCH_PATH.exists() else {}
if all_results:
    print(f"Resuming — already done: {sorted(all_results)}")

for label, description in DATASETS:
    if label in all_results:
        print(f"Skipping {label} (already done)")
        continue
    print(f"\n{'=' * 64}\n{description}\n{'=' * 64}")
    X, y = LOADERS[label]()
    print(f"  shape={X.shape}, class counts={np.bincount(y).tolist()}  "
          f"notes={DATA_NOTES.get(label, {})}", flush=True)
    t0 = time.time()
    all_results[label] = benchmark_dataset(label, X, y)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min")
    BENCH_PATH.write_text(json.dumps(all_results, cls=_Enc, indent=1))
    print(f"  Saved -> {BENCH_PATH}")

# ===== CELL 9 — Summary =====

labels = [l for l, _ in DATASETS if l in all_results]
ORDER = (CLASSICAL + [c + "_full" for c in CLASSICAL] +
         ["QSVM", "QSVM_noent", "QSVM_noisy", "QSVM_noisy_shots",
          "QSVM_fault_b1", "QSVM_fault_unscaled", "QSVM_fault_v1map", "VQC", "VQC_noent"])
print("\n" + "=" * 110 + "\nAUROC (mean ± 95% CI)\n" + "=" * 110)
print(f"{'Model':<20}" + "".join(f"{l[:15]:>18}" for l in labels))
for m in ORDER:
    print(f"{m:<20}" + "".join(
        f"{all_results[l][m]['auc']['mean']:>11.3f}±{all_results[l][m]['auc']['ci']:.3f}"
        for l in labels))

print("\nKey paired comparisons (difference, corrected-t p, TOST p at ±0.02)")
for l in labels:
    c = all_results[l]["__meta__"]["comparisons"]
    for k in ["QSVM_vs_best", "QSVM_vs_SVM_RBF", "VQC_vs_best", "QSVM_noisy_shots_vs_best",
              "best_full_vs_best_4pc"]:
        s = c[k]
        print(f"  {l:<16} {k:<26} d={s['mean_diff']:+.3f}  p={s['p_corrected_t']:.4f}"
              f"  TOST(.02)={s['p_tost_0.02']:.4f}")

# Optional: copy to Google Drive
# from google.colab import drive; drive.mount('/content/drive')
# import shutil; shutil.copy(str(BENCH_PATH), "/content/drive/MyDrive/qml_benchmark_results_v3.json")

print(f"\nResults saved to: {BENCH_PATH.resolve()}")
