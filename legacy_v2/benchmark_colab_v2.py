"""
QML-Health-READY Benchmark v2 — single-file Colab version
==========================================================
Paper: Quantum Machine Learning in Medicine: An Evidence-Graded Critical
       Review and the QML-Health-READY Framework
Authors: Mohammed Siraj B, Ruban S
Code:    github.com/sirajbb/qml-health-ready

Changes from v1 (v1 results must not be used):
  1. ZZ feature map now entangles (CNOT-RZ-CNOT; v1 used CZ-RZ-CZ, which cancels
     to a plain RZ). Havlicek et al. 2019 structure, reps = 2, linear entanglement.
  2. Depolarising noise is actually simulated (default.mixed + DepolarizingChannel
     after every gate). v1 silently ignored the noise argument.
  3. Quantum kernel bandwidth is a tuned hyperparameter (Shaydulin & Wild 2022).
  4. Each embedded state is simulated once per sample, so quantum models train on
     the FULL outer-fold training set (v1 subsampled to n = 50 / 25).
  5. All models share identical outer folds and test sets -> genuinely paired tests:
     two-sided Wilcoxon, Nadeau-Bengio corrected resampled t-test, TOST equivalence,
     paired Cohen's dz.
  6. VQC: full training set, Adam + binary cross-entropy with trainable readout.
  7. Dataset download failures now stop the run instead of silently using
     synthetic proxy data.
  8. Data-complexity metrics (centred KTA, Huang geometric difference, kernel
     concentration) and a QSVM noise-sweep curve are computed and saved.

HOW TO USE IN GOOGLE COLAB
  Upload this file, then in a cell:   !python benchmark_colab.py
  Results are saved after every dataset to qml_results_v2/ (safe to resume).
  Runtime (Colab CPU): FAST_MODE=True ~10-15 min; FAST_MODE=False ~2-4 h.
"""

# ===== CELL 1 — Install dependencies =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "pennylane>=0.45", "pyreadstat", "-q"], check=False)

# ===== CELL 2 — Imports & configuration =====
import os, io, json, math, time, gzip, warnings, urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

FAST_MODE = os.environ.get("QML_FAST", "0") == "1"   # or set True here for a quick test
ONLY      = os.environ.get("QML_DATASETS")            # e.g. "breast_cancer" for a single dataset

N_QUBITS  = 4
ZZ_REPS   = 2
N_JOBS    = -1
NOISE_EVAL = 0.01
TOST_MARGIN = 0.05            # AUROC equivalence margin

if FAST_MODE:
    N_REPEAT_SEED, CV_OUTER, CV_INNER = 1, 3, 2
    BANDWIDTHS   = [0.1, 0.5]
    QSVM_C       = [1.0]
    VQC_EPOCHS   = 20
    NOISE_CURVE  = [0.0, 0.05]
    PARAM_GRIDS = {
        "LR":       {"C": [1]},
        "SVM_RBF":  {"C": [1], "gamma": ["scale"]},
        "SVM_POLY": {"C": [1], "degree": [2], "gamma": ["scale"], "coef0": [1]},
        "RF":       {"n_estimators": [100], "max_depth": [5]},
        "GB":       {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [3]},
    }
    print("[FAST MODE] 1 seed x 3-fold — for testing only, not for the paper")
else:
    N_REPEAT_SEED, CV_OUTER, CV_INNER = 5, 5, 3
    BANDWIDTHS   = [0.05, 0.1, 0.2, 0.5, 1.0]
    QSVM_C       = [0.1, 1, 10]
    VQC_EPOCHS   = 100
    NOISE_CURVE  = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
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

VQC_LAYERS, VQC_LR = 3, 0.05
SEEDS = [42 + 7 * i for i in range(N_REPEAT_SEED)]

DATA_DIR = Path("qml_data");       DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path("qml_results_v2"); OUT_DIR.mkdir(exist_ok=True)

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
QUANTUM   = ["QSVM", "QSVM_noisy", "QSVM_noent", "VQC", "VQC_noent"]

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
    return df.iloc[:, :8].values.astype(np.float64), df.iloc[:, 8].values.astype(int)

def load_heart_failure_data():
    path = _download("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "00519/heart_failure_clinical_records_dataset.csv",
                     "heart_failure_records.csv")
    df = pd.read_csv(path).dropna()
    y = df.pop("DEATH_EVENT").astype(int).values
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
    expr = expr.loc[common]
    y = er.loc[common].isin(pos).astype(int).values
    # Top-200 variance genes: unsupervised, label-free selection on the full cohort
    X = expr[expr.var(axis=0).nlargest(200).index].values.astype(np.float64)
    keep = ~np.isnan(X).any(axis=1)
    return _stratified_cap(X[keep], y[keep], 1500)

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

# ===== CELL 4 — Quantum feature map, kernels and VQC =====

WIRES = list(range(N_QUBITS))

def _depolarise(p, wires):
    if p > 0:
        for w in wires:
            qml.DepolarizingChannel(p, wires=w)

def zz_feature_map(x, entangle=True, p=0.0):
    """Havlicek et al. (2019) ZZ feature map; depolarising channel after every gate."""
    for _ in range(ZZ_REPS):
        for i in WIRES:
            qml.Hadamard(wires=i);              _depolarise(p, [i])
            qml.RZ(2.0 * x[..., i], wires=i);   _depolarise(p, [i])
        if entangle:
            for i in range(N_QUBITS - 1):
                phi = 2.0 * (np.pi - x[..., i]) * (np.pi - x[..., i + 1])
                qml.CNOT(wires=[i, i + 1]);     _depolarise(p, [i, i + 1])
                qml.RZ(phi, wires=i + 1);       _depolarise(p, [i + 1])
                qml.CNOT(wires=[i, i + 1]);     _depolarise(p, [i, i + 1])

_QNODES = {}

def _state_qnode(entangle, p):
    key = (entangle, p)
    if key not in _QNODES:
        if p > 0:
            dev = qml.device("default.mixed", wires=N_QUBITS)
            @qml.qnode(dev)
            def circuit(x):
                zz_feature_map(x, entangle, p)
                return qml.density_matrix(wires=WIRES)
        else:
            dev = qml.device("default.qubit", wires=N_QUBITS)
            @qml.qnode(dev)
            def circuit(x):
                zz_feature_map(x, entangle, 0.0)
                return qml.state()
        _QNODES[key] = circuit
    return _QNODES[key]

def embed(Z, bandwidth, entangle=True, p=0.0, chunk=512):
    """Simulate the embedded state of every row of Z (parameter broadcasting).
    Returns state vectors (n, 2^q) if p == 0, density matrices (n, 2^q, 2^q) otherwise."""
    f = _state_qnode(entangle, p)
    X = bandwidth * np.asarray(Z, dtype=np.float64)
    return np.concatenate([np.asarray(f(X[i:i + chunk])) for i in range(0, len(X), chunk)])

def quantum_gram(A, B):
    """Fidelity kernel: |<a|b>|^2 for pure states, Tr(rho_a rho_b) for mixed states
    (the quantity estimated by a swap test on the noisy embedded states)."""
    if A.ndim == 2:
        return np.abs(A.conj() @ B.T) ** 2
    d = A.shape[1]
    return np.real(A.reshape(len(A), d * d) @ np.transpose(B, (0, 2, 1)).reshape(len(B), d * d).T)

def _vqc_qnode(entangle):
    dev = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=WIRES, rotation="X")
        if entangle:
            qml.StronglyEntanglingLayers(w, wires=WIRES)
        else:
            for l in range(VQC_LAYERS):
                for q in WIRES:
                    qml.Rot(w[l, q, 0], w[l, q, 1], w[l, q, 2], wires=q)
        return qml.expval(qml.PauliZ(0))
    return circuit

def vqc_fit_predict(Z_tr, y_tr, Z_te, entangle, seed):
    """AngleEmbedding + StronglyEntanglingLayers; logit = a*<Z0> + b; Adam on BCE."""
    circuit = _vqc_qnode(entangle)
    rng = np.random.default_rng(seed)
    Xtr = pnp.array(np.clip(Z_tr, -np.pi, np.pi), requires_grad=False)
    Xte = np.clip(Z_te, -np.pi, np.pi)
    yt  = pnp.array(y_tr.astype(float), requires_grad=False)
    w = pnp.array(rng.normal(0, 0.5, (VQC_LAYERS, N_QUBITS, 3)), requires_grad=True)
    a = pnp.array(2.0, requires_grad=True)
    b = pnp.array(0.0, requires_grad=True)

    def loss(w, a, b):
        z = a * circuit(Xtr, w) + b
        return pnp.mean(pnp.logaddexp(0.0, z) - yt * z)

    opt = qml.AdamOptimizer(VQC_LR)
    for _ in range(VQC_EPOCHS):
        w, a, b = opt.step(loss, w, a, b)
    return float(a) * np.asarray(circuit(Xte, w)) + float(b)

# ===== CELL 5 — Evaluation =====

def _metrics(y_true, y_pred, score):
    return {"auc": roc_auc_score(y_true, score),
            "acc": accuracy_score(y_true, y_pred),
            "f1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred)}

def _aggregate(scores):
    out = {}
    for m in scores[0]:
        v = [float(s[m]) for s in scores]
        se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
        t = stats.t.ppf(0.975, len(v) - 1) if len(v) > 1 else 0.0
        out[m] = {"mean": float(np.mean(v)), "ci": float(t * se), "n": len(v), "values": v}
    return out

def make_folds(X, y):
    """Identical outer folds for every model; preprocessing fitted on training part only.
    StandardScaler -> PCA(N_QUBITS) -> StandardScaler (unit-variance components)."""
    folds = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(CV_OUTER, shuffle=True, random_state=seed).split(X, y):
            s1 = StandardScaler().fit(X[tr])
            pca = PCA(N_QUBITS, random_state=seed).fit(s1.transform(X[tr]))
            s2 = StandardScaler().fit(pca.transform(s1.transform(X[tr])))
            f = lambda A: s2.transform(pca.transform(s1.transform(A)))
            folds.append({"seed": seed, "tr": tr, "te": te,
                          "Z_tr": f(X[tr]), "Z_te": f(X[te]),
                          "y_tr": y[tr], "y_te": y[te]})
    return folds

def run_classical(name, folds):
    est = {"LR": LogisticRegression(max_iter=2000),
           "SVM_RBF": SVC(kernel="rbf", probability=True),
           "SVM_POLY": SVC(kernel="poly", probability=True),
           "RF": RandomForestClassifier(random_state=0),
           "GB": GradientBoostingClassifier(random_state=0)}[name]
    scores = []
    for f in folds:
        gs = GridSearchCV(est, PARAM_GRIDS[name], scoring="roc_auc", n_jobs=N_JOBS,
                          cv=StratifiedKFold(CV_INNER, shuffle=True, random_state=f["seed"]))
        gs.fit(f["Z_tr"], f["y_tr"])
        scores.append(_metrics(f["y_te"], gs.predict(f["Z_te"]),
                               gs.predict_proba(f["Z_te"])[:, 1]))
    return scores

def qsvm_fold(f, entangle=True, p=0.0, bandwidths=None):
    """Inner CV over (bandwidth, C) on precomputed train kernel; refit; score test fold."""
    bandwidths = bandwidths or BANDWIDTHS
    inner = list(StratifiedKFold(CV_INNER, shuffle=True,
                                 random_state=f["seed"]).split(f["Z_tr"], f["y_tr"]))
    y = f["y_tr"]
    best = (-np.inf, None, None, None)
    for bw in bandwidths:
        S_tr = embed(f["Z_tr"], bw, entangle, p)
        K = quantum_gram(S_tr, S_tr)
        for C in QSVM_C:
            aucs = []
            for a, v in inner:
                svm = SVC(kernel="precomputed", C=C).fit(K[np.ix_(a, a)], y[a])
                aucs.append(roc_auc_score(y[v], svm.decision_function(K[np.ix_(v, a)])))
            if np.mean(aucs) > best[0]:
                best = (np.mean(aucs), bw, C, (S_tr, K))
    _, bw, C, (S_tr, K) = best
    svm = SVC(kernel="precomputed", C=C).fit(K, y)
    K_te = quantum_gram(embed(f["Z_te"], bw, entangle, p), S_tr)
    m = _metrics(f["y_te"], svm.predict(K_te), svm.decision_function(K_te))
    return m, bw, C

def run_qsvm(folds, entangle=True, p=0.0):
    scores, chosen = [], []
    for f in folds:
        m, bw, C = qsvm_fold(f, entangle, p)
        scores.append(m); chosen.append((bw, C))
    return scores, chosen

def run_vqc(folds, entangle=True):
    scores = []
    for f in folds:
        z = vqc_fit_predict(f["Z_tr"], f["y_tr"], f["Z_te"], entangle, f["seed"])
        scores.append(_metrics(f["y_te"], (z > 0).astype(int), z))
    return scores

def paired_stats(q, c, test_train_ratio):
    """Quantum minus classical, paired over identical outer folds."""
    q, c = np.asarray(q), np.asarray(c)
    d = q - c
    J, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se = sd * math.sqrt(1.0 / J + test_train_ratio)        # Nadeau & Bengio (2003)
    tc = stats.t.ppf(0.975, J - 1)
    if se > 0:
        p_nb = float(2 * stats.t.sf(abs(mean / se), J - 1))
        p_tost = float(max(stats.t.sf((mean + TOST_MARGIN) / se, J - 1),
                           stats.t.cdf((mean - TOST_MARGIN) / se, J - 1)))
    else:
        p_nb, p_tost = float("nan"), float("nan")
    try:
        p_w = float(stats.wilcoxon(q, c, alternative="two-sided", zero_method="zsplit").pvalue)
    except ValueError:
        p_w = float("nan")
    return {"mean_diff": mean, "ci_corrected": [mean - tc * se, mean + tc * se],
            "p_corrected_t": p_nb, "p_wilcoxon": p_w, "p_tost": p_tost,
            "tost_margin": TOST_MARGIN, "cohens_dz": mean / sd if sd > 0 else float("nan")}

# ===== CELL 6 — Data-complexity analysis =====

def centred_kta(K, y):
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    ypm = 2.0 * y - 1.0
    Y = np.outer(ypm, ypm)
    return float(np.sum(Kc * Y) / (np.linalg.norm(Kc) * np.linalg.norm(Y) + 1e-12))

def geometric_difference(K_c, K_q, lam=1e-3):
    """g(K_C || K_Q) of Huang et al. (2021); both kernels trace-normalised to n."""
    n = len(K_c)
    K_c = K_c * n / np.trace(K_c)
    K_q = K_q * n / np.trace(K_q)
    w, V = np.linalg.eigh(K_q)
    sq = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    M = sq @ np.linalg.inv(K_c + lam * np.eye(n)) @ sq
    return float(np.sqrt(np.linalg.norm(M, 2)))

def complexity(X, y, bandwidth, n_max=300):
    Xs, ys = _stratified_cap(X, y, n_max, seed=0)
    s1 = StandardScaler().fit(Xs)
    Z = StandardScaler().fit_transform(PCA(N_QUBITS, random_state=0).fit_transform(s1.transform(Xs)))
    K_q = quantum_gram(*(2 * [embed(Z, bandwidth, True, 0.0)]))
    K_c = rbf_kernel(Z, gamma=1.0 / N_QUBITS)
    off = ~np.eye(len(Z), dtype=bool)
    c0, c1 = Z[ys == 0], Z[ys == 1]
    fisher = np.max((c0.mean(0) - c1.mean(0)) ** 2 / (c0.var(0) + c1.var(0) + 1e-12))
    _, nn = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
    return {"n_used": len(Z), "bandwidth": bandwidth,
            "F1_fisher_max": float(fisher),
            "N2_nn_purity": float((ys[nn[:, 1]] == ys).mean()),
            "KTA_quantum": centred_kta(K_q, ys), "KTA_rbf": centred_kta(K_c, ys),
            "geom_diff_CQ": geometric_difference(K_c, K_q),
            "kq_offdiag_mean": float(K_q[off].mean()), "kq_offdiag_var": float(K_q[off].var())}

# ===== CELL 7 — Benchmark one dataset =====

def benchmark_dataset(label, X, y):
    folds = make_folds(X, y)
    ratio = float(np.mean([len(f["te"]) / len(f["tr"]) for f in folds]))
    res, times, meta = {}, {}, {}

    for name in CLASSICAL:
        t0 = time.time()
        res[name] = _aggregate(run_classical(name, folds))
        times[name] = time.time() - t0
        print(f"  {name:<11} AUC {res[name]['auc']['mean']:.3f} ± {res[name]['auc']['ci']:.3f}"
              f"   ({times[name]:.0f}s)", flush=True)

    for name, kw in [("QSVM", dict(entangle=True, p=0.0)),
                     ("QSVM_noisy", dict(entangle=True, p=NOISE_EVAL)),
                     ("QSVM_noent", dict(entangle=False, p=0.0))]:
        t0 = time.time()
        sc, chosen = run_qsvm(folds, **kw)
        res[name] = _aggregate(sc)
        times[name] = time.time() - t0
        meta[f"{name}_chosen_bandwidth_C"] = chosen
        print(f"  {name:<11} AUC {res[name]['auc']['mean']:.3f} ± {res[name]['auc']['ci']:.3f}"
              f"   ({times[name]:.0f}s)", flush=True)

    for name, ent in [("VQC", True), ("VQC_noent", False)]:
        t0 = time.time()
        res[name] = _aggregate(run_vqc(folds, ent))
        times[name] = time.time() - t0
        print(f"  {name:<11} AUC {res[name]['auc']['mean']:.3f} ± {res[name]['auc']['ci']:.3f}"
              f"   ({times[name]:.0f}s)", flush=True)

    best = max(CLASSICAL, key=lambda k: res[k]["auc"]["mean"])
    cl = res[best]["auc"]["values"]
    comparisons = {q: paired_stats(res[q]["auc"]["values"], cl, ratio) for q in QUANTUM}
    comparisons["QSVM_vs_QSVM_noent"] = paired_stats(
        res["QSVM"]["auc"]["values"], res["QSVM_noent"]["auc"]["values"], ratio)
    comparisons["QSVM_vs_QSVM_noisy"] = paired_stats(
        res["QSVM"]["auc"]["values"], res["QSVM_noisy"]["auc"]["values"], ratio)
    comparisons["VQC_vs_VQC_noent"] = paired_stats(
        res["VQC"]["auc"]["values"], res["VQC_noent"]["auc"]["values"], ratio)

    # Noise sweep: QSVM on the first seed's folds, hyperparameters re-tuned at each level
    t0 = time.time()
    first = [f for f in folds if f["seed"] == SEEDS[0]]
    noise_curve = {}
    for p in NOISE_CURVE:
        noise_curve[str(p)] = float(np.mean([qsvm_fold(f, True, p)[0]["auc"] for f in first]))
    print(f"  noise curve {noise_curve}   ({time.time() - t0:.0f}s)", flush=True)

    modal_bw = Counter(bw for bw, _ in meta["QSVM_chosen_bandwidth_C"]).most_common(1)[0][0]
    comp = complexity(X, y, modal_bw)
    print(f"  complexity  {comp}", flush=True)

    res["__meta__"] = {
        "dataset": label, "n": int(len(y)), "n_features": int(X.shape[1]),
        "pos_rate": float(y.mean()), "n_observations": len(folds),
        "best_classical": best,
        "wilcoxon_qsvm_vs_best_classical_p": comparisons["QSVM"]["p_wilcoxon"],
        "comparisons": comparisons, "time_sec": times,
        "noise_curve": noise_curve, "complexity": comp, **meta,
        "config": {"n_qubits": N_QUBITS, "zz_reps": ZZ_REPS, "seeds": SEEDS,
                   "cv_outer": CV_OUTER, "cv_inner": CV_INNER, "bandwidths": BANDWIDTHS,
                   "qsvm_C": QSVM_C, "noise_eval": NOISE_EVAL, "vqc_layers": VQC_LAYERS,
                   "vqc_epochs": VQC_EPOCHS, "vqc_lr": VQC_LR, "fast_mode": FAST_MODE,
                   "pennylane": qml.__version__},
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
all_results = json.loads(BENCH_PATH.read_text()) if BENCH_PATH.exists() else {}
if all_results:
    print(f"Resuming — already done: {sorted(all_results)}")

for label, description in DATASETS:
    if label in all_results:
        print(f"Skipping {label} (already done)")
        continue
    print(f"\n{'=' * 64}\n{description}\n{'=' * 64}")
    X, y = LOADERS[label]()
    print(f"  shape={X.shape}, class counts={np.bincount(y).tolist()}", flush=True)
    t0 = time.time()
    all_results[label] = benchmark_dataset(label, X, y)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min")
    BENCH_PATH.write_text(json.dumps(all_results, cls=_Enc, indent=1))
    print(f"  Saved -> {BENCH_PATH}")

# ===== CELL 9 — Summary =====

ORDER = CLASSICAL + QUANTUM
labels = [l for l, _ in DATASETS if l in all_results]
print("\n" + "=" * 100 + "\nAUROC (mean ± 95% CI)\n" + "=" * 100)
print(f"{'Model':<12}" + "".join(f"{l[:15]:>18}" for l in labels))
for m in ORDER:
    print(f"{m:<12}" + "".join(
        f"{all_results[l][m]['auc']['mean']:>11.3f}±{all_results[l][m]['auc']['ci']:.3f}"
        for l in labels))

print("\nQuantum minus best classical (paired, identical folds)")
for l in labels:
    meta = all_results[l]["__meta__"]
    for q in ["QSVM", "VQC"]:
        s = meta["comparisons"][q]
        print(f"  {l:<16} {q:<5} vs {meta['best_classical']:<8} "
              f"dAUC={s['mean_diff']:+.3f} [{s['ci_corrected'][0]:+.3f},{s['ci_corrected'][1]:+.3f}]"
              f"  p_t={s['p_corrected_t']:.4f}  p_W={s['p_wilcoxon']:.4f}"
              f"  p_TOST(±{TOST_MARGIN})={s['p_tost']:.4f}  dz={s['cohens_dz']:+.2f}")

# Optional: copy to Google Drive
# from google.colab import drive; drive.mount('/content/drive')
# import shutil; shutil.copy(str(BENCH_PATH), "/content/drive/MyDrive/qml_benchmark_results_v2.json")

print(f"\nResults saved to: {BENCH_PATH.resolve()}")
