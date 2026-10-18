# Version 1 (withdrawn) — do not use

These modules are kept only for transparency. Results produced with them were withdrawn
after code review found the following faults (see Supplementary Table S8 of the paper):

- **No entanglement:** the "ZZ" block used CZ–RZ–CZ. All three gates are diagonal, so the
  CZs cancel and the circuit is a product state.
- **Noise never applied:** `_make_device` ignored `noise_prob`, so "noisy" results were
  identical to ideal ones.
- **No feature scaling or bandwidth:** unscaled principal components were fed to the
  feature map, giving near-diagonal kernels and near-chance AUROC.
- **Subsampling:** quantum models were trained on 50 (QSVM) or 25 (VQC) samples while
  classical models used full training folds.
- **Statistics:** one-sided Wilcoxon tests on test sets that differed between models.
- **Target leakage:** HbA1c was used as an NHANES predictor although it defines the label.

The corrected pipeline is `../benchmark_colab_v3.py`.
