"""
plots.py - figures for the manuscript. No in-image titles: numbering and
descriptions live in the manuscript captions. Statistics are computed by
build_manuscript.py and passed in.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

DATASETS = ["breast_cancer", "heart_disease", "diabetes",
            "heart_failure", "tcga_brca", "nhanes_diabetes"]
DS_SHORT = {
    "breast_cancer": "Breast Cancer", "heart_disease": "Heart Disease",
    "diabetes": "Pima Diabetes", "heart_failure": "Heart Failure",
    "tcga_brca": "TCGA-BRCA", "nhanes_diabetes": "NHANES Diabetes",
}
CLASSICAL = ["LR", "SVM_RBF", "SVM_POLY", "RF", "GB"]
QUANTUM = ["QSVM", "QSVM_noisy", "QSVM_noent", "VQC", "VQC_noent"]
LABEL = {
    "LR": "Logistic regression", "SVM_RBF": "SVM (RBF)", "SVM_POLY": "SVM (polynomial)",
    "RF": "Random forest", "GB": "Gradient boosting",
    "QSVM": "QSVM", "QSVM_noisy": "QSVM (noisy, p = 0.01)",
    "QSVM_noent": "QSVM (product-state)", "VQC": "VQC", "VQC_noent": "VQC (no entanglement)",
}
C_CLASSICAL, C_QUANTUM, C_FULL = "#4C72B0", "#C44E52", "#555555"
COMP_COLOR = {"QSVM_vs_SVM_RBF": "#2E86AB", "QSVM_vs_best": "#1B4F72", "VQC_vs_best": "#F0A500"}
COMP_LABEL = {"QSVM_vs_SVM_RBF": "QSVM − SVM (RBF), kernel-matched",
              "QSVM_vs_best": "QSVM − best classical",
              "VQC_vs_best": "VQC − best classical"}


def _save(name):
    path = FIG_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close()


def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_auroc_dots(R, best_full=None):
    """Per-dataset dot plot of AUROC ± 95% CI; dashed line = best full-feature classical."""
    models = CLASSICAL + QUANTUM
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, ds in zip(axes.ravel(), DATASETS):
        for yi, m in zip(y, models):
            r = R[ds][m]["auc"]
            col = C_CLASSICAL if m in CLASSICAL else C_QUANTUM
            ax.errorbar(r["mean"], yi, xerr=r["ci"], fmt="o", color=col, ms=4.5,
                        capsize=2, lw=1.2)
        if best_full and ds in best_full:
            ax.axvline(best_full[ds], color=C_FULL, ls="--", lw=1)
        ax.axhline(len(QUANTUM) - 0.5, color="#BBBBBB", lw=0.6)
        ax.set_title(DS_SHORT[ds], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#EEEEEE")
        _clean(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[m] for m in models], fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("AUROC", fontsize=9)
    handles = [mpatches.Patch(color=C_CLASSICAL, label="Classical (4 PCs)"),
               mpatches.Patch(color=C_QUANTUM, label="Quantum (4 qubits)")]
    if best_full:
        handles.append(plt.Line2D([], [], color=C_FULL, ls="--",
                                  label="Best classical, all features"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    _save("fig1_auroc.png")


def fig_forest(S, margin_lines=(0.02, 0.05), pooled=None):
    """S[comp][ds] = dict(mean_diff, ci95=(lo, hi)); pooled[comp] adds random-effects diamonds."""
    comps = [c for c in ["QSVM_vs_SVM_RBF", "QSVM_vs_best", "VQC_vs_best"] if c in S]
    fig, ax = plt.subplots(figsize=(8.5, 7.8 if pooled else 7))
    ax.axvspan(-margin_lines[1], margin_lines[1], color="#E8F3E8")
    ax.axvspan(-margin_lines[0], margin_lines[0], color="#CFE8CF")
    ax.axvline(0, color="black", lw=0.8)
    yt, yl, yi = [], [], 0
    if pooled:
        for c in comps[::-1]:
            p = pooled[c]
            lo, hi, m = p["ci95"][0], p["ci95"][1], p["mean_diff"]
            ax.fill([lo, m, hi, m], [yi, yi + 0.3, yi, yi - 0.3], color=COMP_COLOR[c])
            yi += 1
        yt.append(yi - (len(comps) + 1) / 2)
        yl.append("Pooled (random effects)")
        ax.axhline(yi - 0.1, color="#888888", lw=0.6)
        yi += 0.8
    for ds in DATASETS[::-1]:
        for c in comps[::-1]:
            s = S[c][ds]
            ax.plot(s["ci95"], [yi, yi], color=COMP_COLOR[c], lw=2.2)
            ax.plot(s["mean_diff"], yi, "o", color=COMP_COLOR[c], ms=5.5)
            yi += 1
        yt.append(yi - (len(comps) + 1) / 2)
        yl.append(DS_SHORT[ds])
        yi += 0.8
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=9.5)
    ax.set_xlabel("Paired AUROC difference, quantum − classical (mean, corrected 95% CI)",
                  fontsize=9.5)
    handles = [plt.Line2D([], [], color=COMP_COLOR[c], marker="o", lw=2, label=COMP_LABEL[c])
               for c in comps]
    handles += [mpatches.Patch(color="#CFE8CF", label=f"±{margin_lines[0]} margin"),
                mpatches.Patch(color="#E8F3E8", label=f"±{margin_lines[1]} margin")]
    ax.legend(handles=handles, fontsize=8, loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.45, -0.08))
    _clean(ax)
    plt.tight_layout()
    _save("fig2_paired_differences.png")


def fig_noise(R, shot_curves=None):
    """AUROC change relative to p = 0 on a shared axis; optional shot-count panel."""
    ncols = 2 if shot_curves else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.2), squeeze=False)
    ax = axes[0, 0]
    cmap = plt.get_cmap("tab10")
    for k, ds in enumerate(DATASETS):
        nc = R[ds]["__meta__"]["noise_curve"]
        lv = list(nc.keys())
        ax.plot(range(len(lv)), [nc[p] - nc[lv[0]] for p in lv], "o-", color=cmap(k),
                label=DS_SHORT[ds], lw=1.5, ms=4)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(lv)))
    ax.set_xticklabels(lv, fontsize=8.5)
    ax.set_xlabel("Depolarising probability per gate", fontsize=9.5)
    ax.set_ylabel("AUROC change vs. noiseless QSVM", fontsize=9.5)
    ax.set_title("A", loc="left", fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False)
    _clean(ax)
    if shot_curves:
        ax2 = axes[0, 1]
        for k, ds in enumerate(DATASETS):
            sc = shot_curves[ds]
            sv = list(sc.keys())
            ax2.plot(range(len(sv)), [sc[s] - R[ds]["__meta__"]["noise_curve"]["0.0"] for s in sv],
                     "o-", color=cmap(k), lw=1.5, ms=4)
        ax2.axhline(0, color="black", lw=0.7)
        ax2.set_xticks(range(len(sv)))
        ax2.set_xticklabels(sv, fontsize=8.5)
        ax2.set_xlabel("Shots per kernel entry (noiseless circuit)", fontsize=9.5)
        ax2.set_title("B", loc="left", fontweight="bold")
        _clean(ax2)
    plt.tight_layout()
    _save("fig3_noise.png")


def fig_kernels(comp, g_key):
    x = np.arange(len(DATASETS))
    w = 0.27
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    have_prod = all("KTA_product" in comp[d] for d in DATASETS)
    series = [("KTA_quantum", "Quantum ZZ kernel", "#2E86AB")]
    if have_prod:
        series.append(("KTA_product", "Product-state kernel", "#9CCBE0"))
    series.append(("KTA_rbf", "RBF kernel", "#6ACC65"))
    for i, (k, lab, col) in enumerate(series):
        off = (i - (len(series) - 1) / 2) * w
        ax1.bar(x + off, [comp[d][k] for d in DATASETS], w, label=lab, color=col)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8, rotation=20)
    ax1.set_ylabel("Centred kernel-target alignment", fontsize=9.5)
    ax1.set_title("A", loc="left", fontweight="bold")
    ax1.legend(fontsize=8, frameon=False)
    g = [g_key(comp[d]) for d in DATASETS]
    ax2.bar(x, g, 0.5, color="#F0A500", label="Geometric difference g (λ = 10⁻³)")
    ax2.scatter(x, [np.sqrt(comp[d]["n_used"]) for d in DATASETS], marker="_", s=400,
                color="black", label="Reference scale √n")
    ax2.set_xticks(x)
    ax2.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=8, rotation=20)
    ax2.set_ylabel("g(K_C ‖ K_Q)", fontsize=9.5)
    ax2.set_title("B", loc="left", fontweight="bold")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        _clean(ax)
    plt.tight_layout()
    _save("fig4_kernels.png")


def fig_fault_ladder(R):
    steps = [("QSVM", "Correct pipeline"), ("QSVM_fault_b1", "Bandwidth fixed at 1"),
             ("QSVM_fault_unscaled", "+ unscaled inputs"),
             ("QSVM_fault_v1map", "+ CZ–RZ–CZ map")]
    cols = ["#2E86AB", "#E1A95F", "#D1675B", "#8E3B46"]
    if "QSVM_fault_minmax" in R[DATASETS[0]]:
        steps.append(("QSVM_fault_minmax", "Min–max [0, π], no bandwidth"))
        cols.append("#7F7F7F")
    x = np.arange(len(DATASETS))
    w = 0.8 / len(steps)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, ((m, lab), col) in enumerate(zip(steps, cols)):
        ax.bar(x + (i - (len(steps) - 1) / 2) * w, [R[d][m]["auc"]["mean"] for d in DATASETS], w,
               yerr=[R[d][m]["auc"]["ci"] for d in DATASETS], capsize=2, color=col, label=lab)
    ax.axhline(0.5, color="black", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DS_SHORT[d] for d in DATASETS], fontsize=9)
    ax.set_ylabel("QSVM AUROC (mean ± 95% CI)", fontsize=9.5)
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=8, ncol=len(steps), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    _clean(ax)
    plt.tight_layout()
    _save("fig5_fault_ladder.png")


def fig_qubit_scaling(R, datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 3.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sc = R[ds]["__meta__"]["qubit_scaling"]
        q = list(sc.keys())
        ax.plot(q, [sc[k]["qsvm_auc"] for k in q], "o-", color="#2E86AB", label="QSVM (tuned bandwidth)")
        ax.plot(q, [sc[k]["svm_rbf_auc"] for k in q], "s--", color="#6ACC65", label="SVM (RBF)")
        ax.set_xlabel("Qubits = principal components", fontsize=9)
        ax.set_ylabel("AUROC (first-seed folds)", fontsize=9)
        ax.set_title(DS_SHORT[ds], fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        _clean(ax)
    plt.tight_layout()
    _save("figS2_qubit_scaling.png")


def fig_graphical_abstract(R, matched):
    """Journal graphical abstract (13 x 5 cm, landscape): design, accuracy, measurement cost."""
    fig = plt.figure(figsize=(13 / 2.54 * 2, 5 / 2.54 * 2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.45)

    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    steps = ["6 public clinical datasets", "Identical 4-component inputs",
             "25 identical cross-validation splits",
             "5 tuned classical models  vs  QSVM, VQC",
             "Paired equivalence tests (±0.02, ±0.05)"]
    for i, s in enumerate(steps):
        y = 0.92 - i * 0.205
        ax.text(0.5, y, s, ha="center", va="center", fontsize=8.5, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", fc="#EAF2F8", ec="#2E86AB", lw=0.8))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.135), xytext=(0.5, y - 0.07), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.8))
    ax.set_title("A  Matched design", loc="left", fontsize=9.5, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    ax.axvspan(-0.05, 0.05, color="#E8F3E8")
    ax.axvspan(-0.02, 0.02, color="#CFE8CF")
    ax.axvline(0, color="black", lw=0.7)
    for i, ds in enumerate(DATASETS[::-1]):
        s = matched[ds]
        ax.plot(s["ci95"], [i, i], color="#2E86AB", lw=2)
        ax.plot(s["mean_diff"], i, "o", color="#2E86AB", ms=4.5)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels([DS_SHORT[d] for d in DATASETS[::-1]], fontsize=7.5)
    ax.set_xticks([-0.05, 0, 0.05])
    ax.set_xlabel("AUROC, QSVM − RBF SVM", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("B  Matched, not better", loc="left", fontsize=9.5, fontweight="bold")
    _clean(ax)

    ax = fig.add_subplot(gs[2])
    y = np.arange(len(DATASETS))[::-1]
    a = [R[d]["QSVM"]["auc"]["mean"] for d in DATASETS]
    b = [R[d]["QSVM_noisy_shots"]["auc"]["mean"] for d in DATASETS]
    for yi, ai, bi in zip(y, a, b):
        ax.plot([bi, ai], [yi, yi], color="#BBBBBB", lw=1.5)
    ax.plot(a, y, "o", color="#2E86AB", ms=4.5, label="Exact simulation")
    ax.plot(b, y, "o", color="#C44E52", ms=4.5, label="Noise + 1,024 shots")
    ax.set_yticks(y)
    ax.set_yticklabels([DS_SHORT[d] for d in DATASETS], fontsize=7.5)
    ax.set_xlabel("QSVM AUROC", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("C  Shots are the real cost", loc="left", fontsize=9.5, fontweight="bold")
    _clean(ax)

    for ext in ("png", "pdf"):
        path = FIG_DIR / f"graphical_abstract.{ext}"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close()


def fig_triage():
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
    scores = np.array([[3, 3, 1, 3], [2, 2, 2, 2], [2, 3, 1, 2],
                       [2, 2, 2, 1], [1, 2, 3, 1], [1, 1, 2, 1]])
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = matplotlib.colormaps["RdYlGn"].resampled(3)
    ax.imshow(scores, cmap=cmap, vmin=0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_yticks(range(len(use_cases)))
    ax.set_yticklabels(use_cases, fontsize=9)
    label_map = {1: "Low", 2: "Moderate", 3: "High"}
    for i in range(len(use_cases)):
        for j in range(len(dimensions)):
            ax.text(j, i, label_map[scores[i, j]], ha="center", va="center",
                    fontsize=8.5, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=cmap(k), label=l)
                       for k, l in enumerate(["Low", "Moderate", "High"])],
              loc="upper right", fontsize=8, bbox_to_anchor=(1.22, 1.0), title="Rating")
    plt.tight_layout()
    _save("figS1_triage.png")
