"""
complexity.py – data complexity metrics following Huang et al. (2021) and Ho & Basu (2002).
Used for the use-case triage section and to compute the geometric difference between
quantum and classical (RBF) kernels.

Metrics implemented:
  - Fisher Discriminant Ratio (F1)  : class separability (high = easier)
  - Geometry of Overlap (N2)        : nearest-neighbour class purity
  - Kernel alignment (KA)           : alignment between quantum and RBF kernel matrices
  - Geometric difference s(K_Q, K_C): Huang et al. (2021) Eq.2 normalised trace ratio
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel


def fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Mean Fisher DR over all features.  Higher = more linearly separable."""
    classes = np.unique(y)
    if len(classes) != 2:
        return float("nan")
    c0, c1 = X[y == classes[0]], X[y == classes[1]]
    mu0, mu1 = c0.mean(0), c1.mean(0)
    var0, var1 = c0.var(0) + 1e-12, c1.var(0) + 1e-12
    ratio = (mu1 - mu0) ** 2 / (var0 + var1)
    return float(np.mean(ratio))


def nn_class_purity(X: np.ndarray, y: np.ndarray) -> float:
    """
    N2 metric: fraction of samples whose nearest neighbour has the SAME label.
    Range [0,1]; 1 = perfectly separable.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X)
    _, idx = nn.kneighbors(X)
    same = (y[idx[:, 1]] == y).mean()
    return float(same)


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Kernel-target alignment (Cortes et al. 2012).
    A(K, yy^T) = <K, yy^T>_F / (||K||_F * ||yy^T||_F)
    Range [-1, 1]; higher = kernel is well matched to the task.
    """
    y_pm  = 2.0 * y.astype(float) - 1.0
    Y     = np.outer(y_pm, y_pm)
    num   = np.sum(K * Y)
    denom = np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro") + 1e-12
    return float(num / denom)


def quantum_rbf_geometric_diff(K_q: np.ndarray, K_c: np.ndarray,
                                n_samples: int = None) -> float:
    """
    Geometric difference s(K_Q, K_C) – Huang et al. (2021), Eq. 2.
    s = sqrt( ||K_C^{-1/2} K_Q K_C^{-1/2}||_F / n )
    Approximated via eigendecomposition to avoid full inversion.
    High s → quantum kernel is 'far' from the RBF kernel → potential advantage.
    """
    n = K_q.shape[0]
    K_c_reg = K_c + 1e-6 * np.eye(n)   # regularise
    # K_c^{-1/2} via eigendecomp
    evals, evecs = np.linalg.eigh(K_c_reg)
    evals = np.maximum(evals, 1e-10)
    K_c_invsqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    M     = K_c_invsqrt @ K_q @ K_c_invsqrt
    s     = float(np.sqrt(np.linalg.norm(M, "fro") / n))
    return s


def compute_complexity(X: np.ndarray, y: np.ndarray,
                       n_qubits: int = 4) -> dict:
    """
    Run all complexity metrics on a pre-processed (scaled+PCA-reduced) dataset.
    Returns a dict suitable for reporting in Table 5 of the paper.
    """
    # standardise + reduce to n_qubits dims (replicating benchmark pre-processing)
    scaler = StandardScaler()
    pca    = PCA(n_components=n_qubits, random_state=42)
    Xr     = pca.fit_transform(scaler.fit_transform(X))

    # Build RBF kernel (gamma = 'scale')
    gamma_scale = 1.0 / (n_qubits * Xr.var())
    K_rbf = rbf_kernel(Xr, gamma=gamma_scale)

    # Build quantum kernel (import lazily to avoid circular imports)
    from quantum_models import QSVM
    qsvm = QSVM(n_qubits=n_qubits, C=1.0, noise_prob=0.0)
    kfn  = qsvm._build_kernel_fn()
    K_q  = np.zeros((len(Xr), len(Xr)))
    # compute only upper triangle + symmetry for speed (subsample to ≤80)
    n_sub = min(len(Xr), 80)
    rng   = np.random.default_rng(42)
    idx   = rng.choice(len(Xr), n_sub, replace=False)
    Xs    = Xr[idx]; ys = y[idx]
    K_qs  = np.zeros((n_sub, n_sub))
    for i in range(n_sub):
        for j in range(i, n_sub):
            v = kfn(Xs[i], Xs[j])
            K_qs[i, j] = v; K_qs[j, i] = v
    K_rbf_s = rbf_kernel(Xs, gamma=gamma_scale)

    return {
        "F1_fisher":        fisher_discriminant_ratio(Xr, y),
        "N2_nn_purity":     nn_class_purity(Xr, y),
        "KA_quantum":       kernel_target_alignment(K_qs, ys),
        "KA_rbf":           kernel_target_alignment(K_rbf_s, ys),
        "geom_diff_sq":     quantum_rbf_geometric_diff(K_qs, K_rbf_s),
        "n_samples":        len(X),
        "n_features_orig":  X.shape[1],
    }
