"""
config.py – global constants for the QML-Health-READY benchmark.
All random states, qubit counts, CV settings and tuning grids live here.
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED   = 42
N_REPEAT_SEED = 5          # independent random seeds per CV fold
FAST_MODE         = False      # set True by run_benchmark.py --fast
MAX_VQC_TRAIN_FAST = 10        # fast-mode VQC training limit (vs 25 full)
CV_OUTER      = 5          # outer StratifiedKFold folds (nested CV)
CV_INNER      = 3          # inner (tuning) folds
N_JOBS        = 1          # parallelism (set >1 if you have cores to spare)

# ── Quantum circuit settings ──────────────────────────────────────────────────
N_QUBITS      = 4          # all circuits use 4 qubits; PCA reduces features to N_QUBITS
N_LAYERS_VQC  = 3          # strongly-entangling layers for VQC
N_EPOCHS_VQC  = 80         # gradient-descent epochs per VQC fit
LR_VQC        = 0.05       # Adam learning rate

# Noise model parameters (depolarizing channel, applied to each gate)
NOISE_LEVELS  = [0.0, 0.005, 0.01, 0.02]   # 0 = ideal; rest = noisy
NOISE_EVAL    = 0.01        # level used in main results table

# ── Classical baseline grids ──────────────────────────────────────────────────
PARAM_GRIDS = {
    "LR": {"C": [0.01, 0.1, 1, 10], "max_iter": [1000]},
    "SVM_RBF": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
    "SVM_POLY": {"C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]},
    "RF": {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
           "min_samples_leaf": [1, 3]},
    "GB": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2],
           "max_depth": [3, 5]},
    "QSVM": {"C": [0.1, 1, 10]},          # QSVM – C only (kernel is fixed quantum)
}

# ── Dataset definitions ───────────────────────────────────────────────────────
# Each entry: (label, loader_key, description, domain)
DATASETS = [
    ("breast_cancer",  "sklearn_bc",      "Breast Cancer Wisconsin (n=569, 30 feat.)",            "Oncology"),
    ("heart_disease",  "uci_heart",       "Heart Disease UCI (n=303, 13 feat.)",                  "Cardiology"),
    ("diabetes",       "pima_diabetes",   "Pima Indians Diabetes (n=768, 8 feat.)",               "Metabolic"),
    ("heart_failure",  "heart_failure",   "Heart Failure Clinical Records (n=299, 12 feat.)",        "Cardiology"),
    ("tcga_brca",      "tcga_brca",       "TCGA-BRCA ER Status (n~1500, 200 feat.)",              "Oncology"),
    ("nhanes_diabetes","nhanes_diabetes", "NHANES Diabetes 2017-18 (n~2000, 7 feat.)",            "Metabolic"),
]
