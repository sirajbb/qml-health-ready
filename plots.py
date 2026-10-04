"""
plots.py – publication-quality figures for the paper.

Figure 1: Grouped bar chart of AUC (mean ± 95% CI) for all models × datasets
Figure 2: Noise degradation curve (AUC vs noise_prob) for QSVM
Figure 3: QML-Health-READY domain compliance bar chart (from extraction data)
Figure 4: Kernel target alignment: quantum vs RBF across datasets
Figure 5: Cohen's d effect size heatmap (quantum vs best classical)
Figure 6: Use-case triage matrix
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def _t_crit_95(df):
    """Return approximate two-sided 95% t critical value (t_{df, 0.025})."""
    # Lookup for common df values used in this benchmark
    _TABLE = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
              20: 2.086, 25: 2.060, 29: 2.045, 30: 2.042}
    if df in _TABLE:
        return _TABLE[df]
    # For large df, approximate with normal quantile 1.96
    return 1.96 + 5.5 / df

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

PALETTE = {
    "LR":          "#4878CF",
    "SVM_RBF":     "#6ACC65",
    "SVM_POLY":    "#D65F5F",
    "RF":          "#B47CC7",
    "GB":          "#C4AD66",
    "QSVM":        "#77BEDB",
    "QSVM_noisy":  "#3A7BBF",
    "QSVM_noent":  "#AAD4E8",
    "VQC":         "#F0A500",
    "VQC_noent":   "#F5D07A",
}

MODEL_ORDER = ["LR","SVM_RBF","SVM_POLY","RF","GB",
               "QSVM","QSVM_noisy","QSVM_noent","VQC","VQC_noent"]
MODEL_LABELS = {
    "LR":          "Logistic Regression",
    "SVM_RBF":     "SVM (RBF kernel)",
    "SVM_POLY":    "SVM (Poly kernel)",
    "RF":          "Random Forest",
    "GB":          "Gradient Boosting",
    "QSVM":        "QSVM (ideal)",
    "QSVM_noisy":  "QSVM (noisy, p=0.01)",
    "QSVM_noent":  "QSVM (no entanglement)",
    "VQC":         "VQC (ideal)",
    "VQC_noent":   "VQC (no entanglement)",
}


def fig1_grouped_bar(all_results: dict, save=True):
    """
    all_results: {dataset_name: {model_name: aggregated_metric_dict}}
    """
    datasets = list(all_results.keys())
    nd = len(datasets)
    nm = len(MODEL_ORDER)
    x  = np.arange(nd)
    width = 0.08

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, m in enumerate(MODEL_ORDER):
        means = [all_results[d].get(m, {}).get("auc", {}).get("mean", np.nan) for d in datasets]
        cis   = [all_results[d].get(m, {}).get("auc", {}).get("ci",   np.nan) for d in datasets]
        offset = (i - nm / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=MODEL_LABELS[m],
                      color=PALETTE[m], yerr=cis, capsize=2, linewidth=0.5,
                      error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", " ").title() for d in datasets], fontsize=10)
    ax.set_ylabel("AUROC (mean ± 95 % CI)", fontsize=11)
    ax.set_ylim(0.40, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, label="Random (0.50)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.8)
    ax.set_title("Figure 1. AUROC across six clinical datasets — all models", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig1_auc_grouped.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig2_noise_curve(noise_results: dict, save=True):
    """
    noise_results: {noise_prob: {dataset_name: auc_mean}}
    """
    fig, axes = plt.subplots(1, len(noise_results.get("datasets", [])),
                              figsize=(12, 3.5), sharey=True)
    datasets = noise_results.get("datasets", [])
    noise_levels = noise_results.get("levels", [])
    data = noise_results.get("data", {})

    if not datasets:
        plt.close(); return

    for ax, ds in zip(axes, datasets):
        qsvm_means = [data.get((p, ds, "QSVM"), np.nan) for p in noise_levels]
        gb_means   = [data.get((p, ds, "GB"),   np.nan) for p in noise_levels]
        ax.plot(noise_levels, qsvm_means, "o-", color=PALETTE["QSVM"],
                label="QSVM", linewidth=1.5)
        ax.axhline(gb_means[0] if gb_means else 0, color=PALETTE["GB"],
                   linestyle="--", linewidth=1.2, label="GB (classical)")
        ax.set_title(ds.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Depolarizing noise p", fontsize=9)
        ax.set_ylim(0.40, 1.0)

    axes[0].set_ylabel("AUROC", fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 2. QSVM AUROC vs depolarizing noise level", fontsize=11)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig2_noise_curve.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig3_ready_compliance(compliance: dict, save=True):
    """
    compliance: {"domain_label": fraction_meeting_criteria, ...}
    """
    labels = list(compliance.keys())
    values = [compliance[l] * 100 for l in labels]
    colors = ["#4878CF" if v >= 50 else "#D65F5F" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Studies meeting criterion (%)", fontsize=10)
    ax.set_title("Figure 3. QML-Health-READY domain compliance in included studies",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig3_ready_compliance.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig4_kernel_alignment(complexity_data: dict, save=True):
    """
    complexity_data: {dataset_name: complexity_dict}
    """
    datasets  = list(complexity_data.keys())
    ka_q  = [complexity_data[d].get("KA_quantum", np.nan) for d in datasets]
    ka_c  = [complexity_data[d].get("KA_rbf",     np.nan) for d in datasets]
    gd    = [complexity_data[d].get("geom_diff_sq", np.nan) for d in datasets]
    x = np.arange(len(datasets))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.35
    ax1.bar(x - w/2, ka_q, w, label="Quantum kernel KA", color=PALETTE["QSVM"])
    ax1.bar(x + w/2, ka_c, w, label="RBF kernel KA",    color=PALETTE["SVM_RBF"])
    ax1.set_xticks(x); ax1.set_xticklabels([d.replace("_", " ").title() for d in datasets], fontsize=9)
    ax1.set_ylabel("Kernel-target alignment", fontsize=10)
    ax1.set_title("Kernel alignment", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2.bar(x, gd, color=PALETTE["VQC"])
    ax2.set_xticks(x); ax2.set_xticklabels([d.replace("_", " ").title() for d in datasets], fontsize=9)
    ax2.set_ylabel("Geometric difference s(K_Q, K_C)", fontsize=10)
    ax2.set_title("Quantum–classical kernel divergence", fontsize=10)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("Figure 4. Data complexity: kernel alignment and geometric difference", fontsize=11)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig4_kernel_alignment.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig5_cohens_d_heatmap(all_results: dict, save=True):
    """Approximate Cohen's d: quantum models vs best classical per dataset."""
    DATASETS = ["breast_cancer", "heart_disease", "diabetes",
                "heart_failure", "tcga_brca", "nhanes_diabetes"]
    DS_SHORT = {
        "breast_cancer":   "BC\nWisconsin",
        "heart_disease":   "Heart\nDisease",
        "diabetes":        "Pima\nDiabetes",
        "heart_failure":   "Heart\nFailure",
        "tcga_brca":       "TCGA-\nBRCA",
        "nhanes_diabetes": "NHANES\nDiabetes",
    }
    Q_MODELS = ["QSVM", "VQC"]
    Q_LABELS = {"QSVM": "QSVM (ideal)", "VQC": "VQC (ideal)"}
    # n_obs: breast_cancer medium mode (5×3=15), rest fast (3×2=6)
    N_OBS = {"breast_cancer": 15, "heart_disease": 6, "diabetes": 6,
             "heart_failure": 6, "tcga_brca": 6, "nhanes_diabetes": 6}

    matrix = np.full((len(Q_MODELS), len(DATASETS)), np.nan)

    for j, ds in enumerate(DATASETS):
        n = N_OBS[ds]
        t_crit = _t_crit_95(n - 1)
        # best classical (among non-quantum)
        best_c_mean, best_c_ci = -1.0, np.nan
        for m in ["LR", "SVM_RBF", "RF", "GB"]:
            v = all_results.get(ds, {}).get(m, {}).get("auc", {}).get("mean", -1.0)
            if v > best_c_mean:
                best_c_mean = v
                best_c_ci = all_results.get(ds, {}).get(m, {}).get("auc", {}).get("ci", np.nan)
        std_c = (best_c_ci * np.sqrt(n) / t_crit) if not np.isnan(best_c_ci) else np.nan
        for i, qm in enumerate(Q_MODELS):
            q_mean = all_results.get(ds, {}).get(qm, {}).get("auc", {}).get("mean", np.nan)
            q_ci   = all_results.get(ds, {}).get(qm, {}).get("auc", {}).get("ci",   np.nan)
            if np.isnan(q_mean) or np.isnan(std_c):
                continue
            std_q = (q_ci * np.sqrt(n) / t_crit) if not np.isnan(q_ci) else std_c
            pooled = np.sqrt((std_c ** 2 + std_q ** 2) / 2.0)
            if pooled > 0:
                matrix[i, j] = (q_mean - best_c_mean) / pooled

    fig, ax = plt.subplots(figsize=(10, 3.2))
    vmax = max(0.5, np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=9)
    ax.set_yticks(range(len(Q_MODELS)))
    ax.set_yticklabels([Q_LABELS[m] for m in Q_MODELS], fontsize=10)

    for i in range(len(Q_MODELS)):
        for j in range(len(DATASETS)):
            v = matrix[i, j]
            if not np.isnan(v):
                col = "white" if abs(v) > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=col)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cohen's d  (quantum − classical; negative = quantum worse)", fontsize=8)
    ax.set_title("Figure 5. Effect size: quantum vs best classical AUROC (approximate Cohen's d)",
                 fontsize=10)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig5_cohens_d_heatmap.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig6_triage_matrix(save=True):
    """Visual use-case triage matrix for QML in clinical applications."""
    use_cases = [
        "Molecular simulation\n& drug discovery",
        "High-dim omics\n(proteomics, metabolomics)",
        "Combinatorial optimisation\n(scheduling, radiotherapy)",
        "Neuro / mental-health\nsignals (EEG, survey)",
        "Routine tabular risk\n(EHR, structured data)",
        "Medical image\nclassification",
    ]
    dimensions = ["Theoretical\nbasis", "Data-loading\nfeasibility",
                  "NISQ hardware\nreadiness", "Plausibility\nof benefit"]
    # 1 = Low, 2 = Moderate, 3 = High
    scores = np.array([
        [3, 3, 1, 3],   # Molecular sim — strong theory, feasible loading, needs FT HW
        [2, 2, 2, 2],   # Omics — possible theory, manageable at small n
        [2, 3, 1, 2],   # Combinatorial — QUBO formulation, QAOA needs deep circuits
        [2, 2, 2, 1],   # Neuro/mental-health — leakage/overfitting risk high
        [1, 2, 3, 1],   # Routine tabular — GB too strong, QML overhead high
        [1, 1, 2, 1],   # Medical imaging — heavy compression kills advantage
    ])

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").resampled(3)
    im = ax.imshow(scores, cmap=cmap, vmin=0.5, vmax=3.5, aspect="auto")

    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_yticks(range(len(use_cases)))
    ax.set_yticklabels(use_cases, fontsize=9)

    label_map = {1: "Low", 2: "Moderate", 3: "High"}
    for i in range(len(use_cases)):
        for j in range(len(dimensions)):
            v = scores[i, j]
            col = "black"
            ax.text(j, i, label_map[v], ha="center", va="center",
                    fontsize=8.5, color=col, fontweight="bold")

    patches = [
        mpatches.Patch(color=cmap(0), label="Low"),
        mpatches.Patch(color=cmap(1), label="Moderate"),
        mpatches.Patch(color=cmap(2), label="High"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.22, 1.0), title="Rating")
    ax.set_title("Figure 6. QML use-case triage: plausibility of quantum advantage by clinical application",
                 fontsize=10)
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig6_triage_matrix.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig3_ready_compliance_illustrative(save=True):
    """
    Illustrative QML-Health-READY domain compliance chart.
    Estimates based on Gupta et al. 2025 (n=16 included studies).
    """
    compliance = {
        "R – Rationale (data complexity)":         0.12,
        "E – Encoding details reported":            0.56,
        "A – Algorithm (≥4 tuned baselines)":       0.25,
        "D – Dataset quality (real clinical data)": 0.31,
        "Y – Yield (95% CI, multiple seeds)":       0.06,
        "R – Robustness to noise":                  0.19,
        "E – External evaluation / validation":     0.13,
        "Y – Yardstick (compute cost reported)":    0.19,
    }
    labels = list(compliance.keys())
    values = [compliance[l] * 100 for l in labels]
    colors = ["#4878CF" if v >= 50 else "#D65F5F" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, label="50% threshold")
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Studies meeting criterion (%, illustrative)", fontsize=10)
    ax.set_title("Figure 3. QML-Health-READY domain compliance\n"
                 "(illustrative estimates based on Gupta et al. 2025 review, n≈16 studies)",
                 fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    if save:
        path = FIG_DIR / "fig3_ready_compliance.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()
