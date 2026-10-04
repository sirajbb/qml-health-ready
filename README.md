# QML-Health-READY Benchmark

**Reproducible benchmark code for:**  
*Quantum Machine Learning in Medicine: An Evidence-Graded Critical Review and the QML-Health-READY Framework for Reporting and Translational Readiness*  
Mohammed Siraj B, Ruban S — St Aloysius (Deemed to be University), Mangalore, India

---

## Overview

This repository contains the complete benchmark pipeline comparing quantum machine learning (QML) models against tuned classical baselines on six clinical datasets spanning four medical domains.

**Key finding:** No statistically significant quantum advantage was observed on any dataset after Bonferroni correction (α* = 0.0083). Depolarising noise at p = 0.01 produced results indistinguishable from ideal simulation on all datasets, consistent with noise-induced barren-plateau saturation.

---

## Datasets

| Dataset | Domain | n | Features | Source |
|---|---|---|---|---|
| Breast Cancer Wisconsin | Oncology | 569 | 30 | scikit-learn / UCI |
| Heart Disease Cleveland | Cardiology | 297 | 13 | UCI ML Repository |
| Pima Indians Diabetes | Metabolic | 768 | 8 | UCI ML Repository |
| Heart Failure Clinical Records | Cardiology | 299 | 12 | UCI ML Repository |
| TCGA-BRCA ER Status | Genomic oncology | 780 | 200 | UCSC Xena S3 |
| NHANES Diabetes 2017–18 | Population epidemiology | 2000 | 7 | CDC public XPT |

All datasets are publicly available with no restrictions on research use. See `datasets.py` for download logic.

---

## Models benchmarked

**Classical baselines:** Logistic Regression (L2), SVM-RBF, SVM-Poly, Random Forest, Gradient Boosting  
**Quantum models:** QSVM (ZZ-FeatureMap kernel), VQC (AngleEmbedding + StronglyEntanglingLayers)  
**Ablations:** Entanglement-removed variants of QSVM and VQC  
**Noise:** Depolarising noise p = 0.01 per gate applied to QSVM

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Usage

### Fast run (< 10 min, for testing)
```bash
python run_benchmark.py --fast
```

### Full run (publishable quality, ~2–4 h)
```bash
python run_benchmark.py
```

### Regenerate manuscript figures and Word document only
```bash
python run_benchmark.py --load   # reloads saved results
python build_manuscript.py
```

Outputs:
- `results/benchmark_results.json` — all AUROC, CI, Wilcoxon results
- `results/complexity_results.json` — KTA, geometric difference, Fisher ratio
- `figures/fig1_auc_grouped.png` — grouped bar chart
- `figures/fig2_noise_curve.png` — noise degradation curves
- `figures/fig3_ready_compliance.png` — QML-Health-READY compliance illustration
- `figures/fig4_kernel_alignment.png` — kernel-target alignment comparison
- `figures/fig5_cohens_d_heatmap.png` — Cohen's d effect size heatmap
- `figures/fig6_triage_matrix.png` — use-case triage matrix

---

## Repository structure

```
benchmark/
├── config.py             # Global constants (seeds, qubit count, CV folds, grids)
├── datasets.py           # Dataset loaders (auto-downloads TCGA-BRCA, NHANES)
├── quantum_models.py     # QSVM and VQC implementations (PennyLane)
├── evaluate.py           # Nested CV, Wilcoxon test, confidence intervals
├── complexity.py         # KTA, geometric difference, Fisher ratio
├── plots.py              # All 6 figure generators
├── run_benchmark.py      # Main runner
├── build_manuscript.py   # Generates Word manuscript from results JSON
└── results/              # JSON output (generated at runtime)
└── figures/              # PNG figures (generated at runtime)
```

---

## QML-Health-READY Framework

The eight-domain reporting checklist proposed in this paper:

| Domain | Name | Description |
|---|---|---|
| R† | Rationale | Theoretical basis + data-complexity analysis (KTA, geometric difference) |
| E† | Encoding | Encoding type, qubit count, PCA fitted inside folds |
| A† | Algorithm & baselines | ≥4 tuned classical baselines + entanglement-ablated quantum model |
| D† | Data & splits | Provenance, class balance, subject-level splitting, leakage checks |
| Y† | Yield (uncertainty) | Nested CV, ≥5 seeds, 95% CI, paired statistical test |
| R† | Robustness to noise | Hardware-calibrated noise model evaluation |
| E | Evaluation in context | Calibration, decision-curve analysis, TRIPOD+AI / CLAIM alignment |
| Y | Yardstick for cost | Wall-clock time, circuit evaluations vs classical baselines |

† = Core item. A study meeting all Core items is **READY-Reportable**.

---

## Citation

> Mohammed Siraj B, Ruban S. Quantum Machine Learning in Medicine: An Evidence-Graded Critical Review and the QML-Health-READY Framework for Reporting and Translational Readiness. *Manuscript under review.* 2026. Code: github.com/sirajbb/qml-health-ready

---

## License

MIT License. See `LICENSE`.

---

## Acknowledgements

Datasets: UCI Machine Learning Repository, UCSC Xena (TCGA-BRCA), CDC NHANES.  
Quantum simulation: PennyLane (Xanadu).
