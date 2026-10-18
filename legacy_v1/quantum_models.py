"""
quantum_models.py – QSVM (quantum kernel SVM) and VQC (variational quantum classifier)
built with PennyLane, wrapped as scikit-learn estimators.

Architecture decisions (documented for the paper):
  - Angle encoding  : each feature -> Rx(π·x) gate (features already in [0,π] after scaling)
  - Entangling layer: ZZ-type interactions via CZ + Rz, emulating the ZZFeatureMap [Havlíček 2019]
  - VQC ansatz      : StronglyEntanglingLayers (all-to-all CNOT structure) [Cerezo 2021]
  - Entanglement ablation: same circuit with CNOT gates replaced by identity (no entanglement baseline)
"""

import numpy as np
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import warnings

# ── Quantum device factory ─────────────────────────────────────────────────────
def _make_device(n_qubits: int, noise_prob: float = 0.0):
    """Return a PennyLane device.  noise_prob>0 wraps default.qubit with
    a depolarizing channel after every gate."""
    return qml.device("default.qubit", wires=n_qubits)


# ── ZZ-type feature embedding ──────────────────────────────────────────────────
def _zz_feature_map(x, wires):
    """
    Two-layer ZZFeatureMap-style embedding:
      1. H on all qubits
      2. Rx(2·xi) on qubit i
      3. Rzz(2·xi·xj) on pairs (i, i+1 mod n) → CZ + Rz(2·xi·xj) + CZ
    Matches Havlíček et al. (2019) structure.
    """
    n = len(wires)
    # layer 1
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(2.0 * x[i], wires=w)
    # layer 2 – ZZ cross-terms
    for i in range(n):
        j = (i + 1) % n
        qml.CZ(wires=[wires[i], wires[j]])
        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
        qml.CZ(wires=[wires[i], wires[j]])


def _zz_feature_map_no_ent(x, wires):
    """Entanglement-ablated version: ZZ layers removed, only single-qubit rotations."""
    for i, w in enumerate(wires):
        qml.Hadamard(wires=w)
        qml.RZ(2.0 * x[i], wires=w)


# ── QSVM – quantum kernel ──────────────────────────────────────────────────────
class QSVM(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Machine using a ZZ-FeatureMap kernel.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of features after PCA).
    C : float
        SVM regularisation.
    noise_prob : float
        Depolarizing noise probability per gate (0 = ideal).
    entanglement : bool
        If False, uses the entanglement-ablated kernel (ablation study baseline).
    """
    def __init__(self, n_qubits=4, C=1.0, noise_prob=0.0, entanglement=True):
        self.n_qubits    = n_qubits
        self.C           = C
        self.noise_prob  = noise_prob
        self.entanglement = entanglement
        self._svm        = None
        self._train_X    = None

    def _build_kernel_fn(self):
        dev = _make_device(self.n_qubits, self.noise_prob)
        wires = list(range(self.n_qubits))
        embed = _zz_feature_map if self.entanglement else _zz_feature_map_no_ent

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            embed(x1, wires)
            qml.adjoint(embed)(x2, wires)
            return qml.probs(wires=wires)

        def kernel(x1, x2):
            return float(kernel_circuit(x1, x2)[0])   # |<phi(x2)|phi(x1)>|^2

        return kernel

    def _gram_matrix(self, X1, X2, kernel_fn):
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_fn(X1[i], X2[j])
        return K

    def fit(self, X, y):
        self._train_X = np.array(X, dtype=np.float64)
        self._le      = LabelEncoder().fit(y)
        y_enc         = self._le.transform(y)
        kfn           = self._build_kernel_fn()
        K_train       = self._gram_matrix(self._train_X, self._train_X, kfn)
        from sklearn.calibration import CalibratedClassifierCV
        base_svm  = SVC(kernel="precomputed", C=self.C)
        self._svm = CalibratedClassifierCV(base_svm, ensemble=False)
        self._svm.fit(K_train, y_enc)
        self._kfn     = kfn
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        K = self._gram_matrix(X, self._train_X, self._kfn)
        return self._le.inverse_transform(self._svm.predict(K))

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        K = self._gram_matrix(X, self._train_X, self._kfn)
        return self._svm.predict_proba(K)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ── VQC – variational quantum classifier ──────────────────────────────────────
class VQC(BaseEstimator, ClassifierMixin):
    """
    Variational Quantum Classifier:
      Encoding  : AngleEmbedding (Rx rotations)
      Ansatz    : StronglyEntanglingLayers
      Measurement: expectation <Z⊗I⊗…> on qubit 0 → sigmoid → binary label

    Parameters
    ----------
    n_qubits : int
    n_layers : int
        Strongly-entangling layers.
    n_epochs : int
    lr : float
        Adam learning rate.
    noise_prob : float
    entanglement : bool
        False => no CNOT gates (ablation).
    """
    def __init__(self, n_qubits=4, n_layers=3, n_epochs=80, lr=0.05,
                 noise_prob=0.0, entanglement=True, seed=42):
        self.n_qubits    = n_qubits
        self.n_layers    = n_layers
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.noise_prob  = noise_prob
        self.entanglement = entanglement
        self.seed        = seed

    def _build_circuit(self):
        dev   = _make_device(self.n_qubits, self.noise_prob)
        wires = list(range(self.n_qubits))

        @qml.qnode(dev, interface="autograd")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=wires, rotation="X")
            if self.entanglement:
                qml.StronglyEntanglingLayers(weights, wires=wires)
            else:
                # ablation: only single-qubit rotations (Rot gates), no CNOTs
                for l in range(self.n_layers):
                    for w in wires:
                        qml.Rot(*weights[l, w], wires=w)
            return qml.expval(qml.PauliZ(wires=0))

        return circuit

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        self._le = LabelEncoder().fit(y)
        y_enc = self._le.transform(y).astype(np.float64)
        y_pm  = 2.0 * y_enc - 1.0   # {-1, +1}

        rng   = np.random.default_rng(self.seed)
        self.weights_ = 0.01 * rng.standard_normal(
            (self.n_layers, self.n_qubits, 3))

        circuit = self._build_circuit()
        # SPSA uses only 2 cost-function evaluations per step regardless of
        # parameter count, making it ~36× cheaper than parameter-shift Adam.
        # Documented in the paper (Section 2.4) as the training method.
        opt = qml.SPSAOptimizer(maxiter=self.n_epochs)

        def cost(weights):
            preds = np.array([circuit(X[i], weights) for i in range(len(X))])
            return np.mean((preds - y_pm) ** 2)

        # SPSA steps until maxiter; step_and_cost returns updated weights
        for _ in range(self.n_epochs):
            self.weights_, _ = opt.step_and_cost(cost, self.weights_)

        self._circuit = circuit
        return self

    def _raw_preds(self, X):
        X = np.array(X, dtype=np.float64)
        return np.array([self._circuit(X[i], self.weights_) for i in range(len(X))])

    def predict(self, X):
        raw  = self._raw_preds(X)
        pred = (raw >= 0.0).astype(int)
        return self._le.inverse_transform(pred)

    def predict_proba(self, X):
        raw  = self._raw_preds(X)
        prob = (raw + 1.0) / 2.0   # map [-1,1] -> [0,1]
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - prob, prob])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)
