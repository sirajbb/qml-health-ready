"""
evaluate.py – nested cross-validation with confidence intervals and statistical tests.

Design (documented for the paper):
  - Outer loop : 5-fold StratifiedKFold, repeated N_REPEAT_SEED times with different seeds
  - Inner loop : 3-fold StratifiedKFold GridSearchCV (classical) or manual scan (QML)
  - Pre-processing inside CV: StandardScaler → PCA(n_components=N_QUBITS) per fold
  - Metrics     : AUROC, accuracy, F1 (macro), Matthews CC
  - Uncertainty : 95 % CI via t-distribution (mean ± t * SE across outer folds × seeds)
  - Statistics  : Wilcoxon signed-rank test (quantum vs best classical) per dataset

Fast-mode QSVM optimisation (FAST_MODE=True):
  - Precomputes the full n×n kernel matrix ONCE per QSVM variant (exploiting symmetry:
    only n(n+1)/2 circuit evaluations instead of n² × n_folds)
  - CV is performed by row/column indexing into the precomputed matrix
  - This yields a ~30× reduction in total quantum circuit evaluations
  - Kernel is computed after standard preprocessing fitted on the subsampled batch;
    slight preprocessing leakage is documented as a limitation (labels never used)
"""

import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from scipy import stats

import config
from config import (N_QUBITS, N_JOBS, N_LAYERS_VQC, LR_VQC, NOISE_EVAL, PARAM_GRIDS)
# Mutable fast-mode variables accessed via config.* to pick up run-time overrides
from quantum_models import QSVM, VQC

# Maximum training samples for quantum models (kernel matrix is O(n²) circuits).
MAX_QUANTUM_TRAIN    = 50   # used in full-mode nested CV
MAX_VQC_TRAIN        = 25   # VQC training limit
MAX_QUANTUM_TEST     = 25
MAX_PRECOMPUTED_SAMP = 50   # fast-mode: subsample for precomputed kernel CV

warnings.filterwarnings("ignore")


# ── classical model factories ──────────────────────────────────────────────────
def _classical_estimators():
    return {
        "LR":       LogisticRegression(solver="lbfgs", max_iter=1000),
        "SVM_RBF":  SVC(kernel="rbf", probability=True),
        "SVM_POLY": SVC(kernel="poly", probability=True),
        "RF":       RandomForestClassifier(random_state=0),
        "GB":       GradientBoostingClassifier(random_state=0),
    }


# ── per-fold scoring ───────────────────────────────────────────────────────────
def _score(y_true, y_pred, y_proba):
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        auc = float("nan")
    return {
        "auc":  auc,
        "acc":  accuracy_score(y_true, y_pred),
        "f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc":  matthews_corrcoef(y_true, y_pred),
    }


# ── stratified subsampler ──────────────────────────────────────────────────────
def _stratified_subsample(y, n, rng):
    """Return indices for a stratified subsample of size n from array y."""
    classes, counts = np.unique(y, return_counts=True)
    fracs = counts / counts.sum()
    idx = []
    for cls, frac in zip(classes, fracs):
        cls_idx = np.where(y == cls)[0]
        k = max(1, int(round(n * frac)))
        k = min(k, len(cls_idx))
        chosen = rng.choice(cls_idx, k, replace=False)
        idx.extend(chosen.tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return idx[:n]


# ── nested CV for one estimator (full mode) ────────────────────────────────────
def _nested_cv(estimator, param_grid, X, y, seed, is_quantum=False):
    """
    Returns a list of score dicts, one per outer fold.
    Pre-processing (scaler + PCA) is applied inside each outer fold.
    """
    outer_cv = StratifiedKFold(n_splits=config.CV_OUTER, shuffle=True, random_state=seed)
    inner_cv = StratifiedKFold(n_splits=config.CV_INNER, shuffle=True, random_state=seed)
    fold_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx],  y[test_idx]

        # pre-process INSIDE the fold (no leakage)
        scaler = StandardScaler()
        pca    = PCA(n_components=N_QUBITS, random_state=seed)
        X_tr_t = pca.fit_transform(scaler.fit_transform(X_tr))
        X_te_t = pca.transform(scaler.transform(X_te))

        # Quantum models subsample for tractability (reported as limitation)
        if is_quantum:
            rng_sub = np.random.default_rng(seed)
            is_vqc  = isinstance(estimator, VQC)
            max_tr  = MAX_VQC_TRAIN if is_vqc else MAX_QUANTUM_TRAIN
            if len(X_tr_t) > max_tr:
                tr_sub = _stratified_subsample(y_tr, max_tr, rng_sub)
                X_tr_t, y_tr = X_tr_t[tr_sub], y_tr[tr_sub]
            if len(X_te_t) > MAX_QUANTUM_TEST:
                te_sub = _stratified_subsample(y_te, MAX_QUANTUM_TEST, rng_sub)
                X_te_t, y_te = X_te_t[te_sub], y_te[te_sub]

        if is_quantum:
            # manual hyper-param search (small grid)
            best_score, best_est = -np.inf, None
            best_C = param_grid.get("C", [1.0])[0]
            for C_val in param_grid.get("C", [1.0]):
                est = _clone_quantum(estimator, C=C_val, seed=seed)
                inner_cv2 = StratifiedKFold(n_splits=config.CV_INNER, shuffle=True,
                                            random_state=seed)
                cv_aucs = []
                for tr2, va2 in inner_cv2.split(X_tr_t, y_tr):
                    try:
                        est.fit(X_tr_t[tr2], y_tr[tr2])
                        p = est.predict_proba(X_tr_t[va2])
                        cv_aucs.append(roc_auc_score(y_tr[va2], p[:, 1]))
                    except Exception:
                        cv_aucs.append(0.5)
                mean_auc = np.mean(cv_aucs)
                if mean_auc > best_score:
                    best_score, best_C = mean_auc, C_val
            est = _clone_quantum(estimator, C=best_C, seed=seed)
        else:
            gs = GridSearchCV(estimator, param_grid, cv=config.CV_INNER,
                              scoring="roc_auc", n_jobs=N_JOBS, refit=True)
            est = gs

        try:
            est.fit(X_tr_t, y_tr)
            y_pred  = est.predict(X_te_t)
            y_proba = est.predict_proba(X_te_t)
            fold_scores.append(_score(y_te, y_pred, y_proba))
        except Exception as e:
            fold_scores.append({"auc": np.nan, "acc": np.nan,
                                 "f1": np.nan, "mcc": np.nan})

    return fold_scores


# ── FAST-MODE: precomputed kernel CV for QSVM variants ─────────────────────────
def _build_kernel_matrix(X, kernel_fn):
    """
    Build the full n×n symmetric kernel matrix with only n(n+1)/2 circuit calls.
    For n=50 this is 1,275 evaluations vs 2,500 for the naive double loop.
    """
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = float(kernel_fn(X[i], X[j]))
            K[i, j] = v
            K[j, i] = v
    return K


def _qsvm_precomputed_cv(qsvm_model, X_pre, y_sub, seed):
    """
    Fast-mode QSVM evaluation using a precomputed kernel matrix.

    X_pre : already preprocessed (scaled+PCA) subsampled array of shape (n_sub, n_qubits)
    Builds K once, then does StratifiedKFold CV by row/col indexing.
    The C hyperparameter is fixed (no inner tuning — grid is single-value in fast mode).
    """
    kfn = qsvm_model._build_kernel_fn()
    K   = _build_kernel_matrix(X_pre, kfn)

    cv = StratifiedKFold(n_splits=config.CV_OUTER, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, te_idx in cv.split(X_pre, y_sub):
        K_tr = K[np.ix_(tr_idx, tr_idx)]
        K_te = K[np.ix_(te_idx, tr_idx)]
        svm  = CalibratedClassifierCV(
            SVC(kernel="precomputed", C=qsvm_model.C), ensemble=False)
        try:
            svm.fit(K_tr, y_sub[tr_idx])
            y_pred  = svm.predict(K_te)
            y_proba = svm.predict_proba(K_te)
            scores.append(_score(y_sub[te_idx], y_pred, y_proba))
        except Exception:
            scores.append({"auc": np.nan, "acc": np.nan, "f1": np.nan, "mcc": np.nan})
    return scores


def _eval_qsvm_fast(qsvm_variants, X, y, seeds):
    """
    Fast-mode evaluation of all QSVM variants.

    For each seed:
      1. Stratified subsample MAX_PRECOMPUTED_SAMP points
      2. Fit StandardScaler+PCA on the subsample
      3. For each QSVM variant: build K (with symmetry), do 3-fold CV by indexing

    Returns dict: variant_name -> list of fold score dicts
    """
    all_scores = {name: [] for name in qsvm_variants}

    for seed in seeds:
        rng = np.random.default_rng(seed)
        sub_idx = _stratified_subsample(y, MAX_PRECOMPUTED_SAMP, rng)
        X_sub = X[sub_idx]
        y_sub = y[sub_idx]

        # Preprocessing fitted on the subsample (slight leakage vs full outer fold;
        # documented as a limitation — labels not used, no SVM fitted on test kernel)
        scaler = StandardScaler()
        pca    = PCA(n_components=N_QUBITS, random_state=seed)
        X_pre  = pca.fit_transform(scaler.fit_transform(X_sub))

        for name, qsvm_model in qsvm_variants.items():
            fold_s = _qsvm_precomputed_cv(qsvm_model, X_pre, y_sub, seed)
            all_scores[name].extend(fold_s)

    return all_scores


# ── VQC nested CV (fast mode: no inner tuning, 3-fold outer) ──────────────────
def _eval_vqc_fast(vqc_variants, X, y, seeds):
    """Fast-mode VQC evaluation: small subsample, outer CV only (no inner tuning)."""
    all_scores = {name: [] for name in vqc_variants}

    for seed in seeds:
        rng     = np.random.default_rng(seed)
        outer_cv = StratifiedKFold(n_splits=config.CV_OUTER, shuffle=True,
                                   random_state=seed)
        for train_idx, test_idx in outer_cv.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx],  y[test_idx]

            scaler = StandardScaler()
            pca    = PCA(n_components=N_QUBITS, random_state=seed)
            X_tr_t = pca.fit_transform(scaler.fit_transform(X_tr))
            X_te_t = pca.transform(scaler.transform(X_te))

            # subsample training for VQC
            vqc_limit = config.MAX_VQC_TRAIN_FAST if config.FAST_MODE else MAX_VQC_TRAIN
            if len(X_tr_t) > vqc_limit:
                sub = _stratified_subsample(y_tr, vqc_limit, rng)
                X_tr_t, y_tr_s = X_tr_t[sub], y_tr[sub]
            else:
                y_tr_s = y_tr

            if len(X_te_t) > MAX_QUANTUM_TEST:
                sub = _stratified_subsample(y_te, MAX_QUANTUM_TEST, rng)
                X_te_t, y_te = X_te_t[sub], y_te[sub]

            for name, vqc_model in vqc_variants.items():
                vqc = _clone_quantum(vqc_model, seed=seed)
                try:
                    vqc.fit(X_tr_t, y_tr_s)
                    y_pred  = vqc.predict(X_te_t)
                    y_proba = vqc.predict_proba(X_te_t)
                    all_scores[name].append(_score(y_te, y_pred, y_proba))
                except Exception:
                    all_scores[name].append({"auc": np.nan, "acc": np.nan,
                                              "f1": np.nan, "mcc": np.nan})
    return all_scores


def _clone_quantum(model, C=1.0, seed=42):
    if isinstance(model, QSVM):
        return QSVM(n_qubits=model.n_qubits, C=C,
                    noise_prob=model.noise_prob,
                    entanglement=model.entanglement)
    else:
        return VQC(n_qubits=model.n_qubits, n_layers=model.n_layers,
                   n_epochs=model.n_epochs, lr=model.lr,
                   noise_prob=model.noise_prob,
                   entanglement=model.entanglement, seed=seed)


# ── aggregate across seeds ─────────────────────────────────────────────────────
def _aggregate(all_scores):
    """all_scores: list of fold score dicts.  Returns mean ± 95% CI dict."""
    metrics = list(all_scores[0].keys())
    result  = {}
    for m in metrics:
        vals = [s[m] for s in all_scores if not np.isnan(s[m])]
        if len(vals) == 0:
            result[m] = {"mean": np.nan, "ci": np.nan}
            continue
        mn  = np.mean(vals)
        se  = np.std(vals, ddof=1) / np.sqrt(len(vals))
        t95 = stats.t.ppf(0.975, df=len(vals) - 1) if len(vals) > 1 else 0
        result[m] = {"mean": float(mn), "ci": float(t95 * se),
                     "n": len(vals), "values": vals}
    return result


# ── full benchmark for one dataset ────────────────────────────────────────────
def benchmark_dataset(X, y, dataset_name: str, verbose=True):
    """
    Run all models on one dataset.
    Returns a dict: model_name -> aggregated metric dict.
    """
    seeds = [42 + i * 7 for i in range(config.N_REPEAT_SEED)]

    # ── classical models ──────────────────────────────────────────────────────
    results = {}
    for name, est in _classical_estimators().items():
        if verbose:
            print(f"  [{dataset_name}] {name} … ", end="", flush=True)
        all_scores = []
        for seed in seeds:
            fold_s = _nested_cv(est, PARAM_GRIDS[name], X, y, seed,
                                 is_quantum=False)
            all_scores.extend(fold_s)
        results[name] = _aggregate(all_scores)
        if verbose:
            auc = results[name]["auc"]
            print(f"AUC {auc['mean']:.3f} ± {auc['ci']:.3f}")

    # ── quantum models ────────────────────────────────────────────────────────
    qsvm_variants = {
        "QSVM":       QSVM(n_qubits=N_QUBITS, C=1.0, noise_prob=0.0),
        "QSVM_noisy": QSVM(n_qubits=N_QUBITS, C=1.0, noise_prob=NOISE_EVAL),
        "QSVM_noent": QSVM(n_qubits=N_QUBITS, C=1.0, noise_prob=0.0,
                           entanglement=False),
    }
    vqc_variants = {
        "VQC":      VQC(n_qubits=N_QUBITS, n_layers=N_LAYERS_VQC,
                        n_epochs=config.N_EPOCHS_VQC, lr=LR_VQC, noise_prob=0.0),
        "VQC_noent": VQC(n_qubits=N_QUBITS, n_layers=N_LAYERS_VQC,
                         n_epochs=config.N_EPOCHS_VQC, lr=LR_VQC, noise_prob=0.0,
                         entanglement=False),
    }

    if config.FAST_MODE:
        # Precomputed-kernel fast path (see module docstring)
        if verbose:
            print(f"  [{dataset_name}] QSVM variants (precomputed kernel) … ",
                  end="", flush=True)
        qsvm_scores = _eval_qsvm_fast(qsvm_variants, X, y, seeds)
        for name, scores in qsvm_scores.items():
            results[name] = _aggregate(scores)
        if verbose:
            auc = results["QSVM"]["auc"]
            print(f"QSVM AUC {auc['mean']:.3f} ± {auc['ci']:.3f}  "
                  f"| noisy {results['QSVM_noisy']['auc']['mean']:.3f}  "
                  f"| noent {results['QSVM_noent']['auc']['mean']:.3f}")

        if verbose:
            print(f"  [{dataset_name}] VQC variants … ", end="", flush=True)
        vqc_scores = _eval_vqc_fast(vqc_variants, X, y, seeds)
        for name, scores in vqc_scores.items():
            results[name] = _aggregate(scores)
        if verbose:
            auc = results["VQC"]["auc"]
            print(f"VQC AUC {auc['mean']:.3f} ± {auc['ci']:.3f}  "
                  f"| noent {results['VQC_noent']['auc']['mean']:.3f}")
    else:
        # Full nested-CV path
        all_quantum = {**qsvm_variants, **vqc_variants}
        for name, model in all_quantum.items():
            if verbose:
                print(f"  [{dataset_name}] {name} … ", end="", flush=True)
            all_scores = []
            for seed in seeds:
                fold_s = _nested_cv(model, PARAM_GRIDS.get("QSVM", {}),
                                    X, y, seed, is_quantum=True)
                all_scores.extend(fold_s)
            results[name] = _aggregate(all_scores)
            if verbose:
                auc = results[name]["auc"]
                print(f"AUC {auc['mean']:.3f} ± {auc['ci']:.3f}")

    # Wilcoxon test: best classical vs QSVM
    best_cl = max(
        ["LR","SVM_RBF","SVM_POLY","RF","GB"],
        key=lambda k: results[k]["auc"]["mean"])
    qsvm_vals = results["QSVM"]["auc"].get("values", [])
    cl_vals   = results[best_cl]["auc"].get("values", [])
    n = min(len(qsvm_vals), len(cl_vals))
    if n >= 5:
        try:
            stat, pval = stats.wilcoxon(qsvm_vals[:n], cl_vals[:n],
                                        alternative="greater",
                                        zero_method="zsplit")
        except Exception:
            pval = float("nan")
    else:
        pval = float("nan")
    results["__meta__"] = {
        "dataset": dataset_name,
        "best_classical": best_cl,
        "wilcoxon_qsvm_vs_best_classical_p": pval,
    }
    return results
