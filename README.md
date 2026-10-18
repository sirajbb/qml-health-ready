# QML-Health-READY benchmark

Code and per-split results for:

> Mohammed Siraj B, Ruban S. *Matched but Not Better: An Equivalence-Tested Benchmark of
> Quantum Machine Learning on Six Clinical Datasets and the QML-Health-READY Reporting
> Framework.* Manuscript under review, 2026.

A quantum-kernel SVM (4-qubit ZZ feature map, tuned bandwidth) and a variational quantum
classifier are compared with five tuned classical models on six public clinical datasets,
using identical four-component inputs and 25 identical cross-validation splits, corrected
paired t-tests and equivalence tests. Additional analyses cover entanglement ablation,
depolarising noise, finite-shot kernel estimation, a kernel-matched classical comparator,
classical models on all predictors, implementation faults and qubit scaling (4/8/12 qubits).

## Repository layout

| Path | Contents |
|---|---|
| `benchmark_colab_v3.py` | The full benchmark as one script (runs in Google Colab or locally) |
| `build_manuscript.py`, `plots.py` | Rebuild the paper's tables, figures and statistics from the results file |
| `results/benchmark_results.json` | Per-split metrics for every model, dataset and analysis reported in the paper |
| `results/superseded/` | Results of the superseded version 2 (not reported) |
| `legacy_v1/`, `legacy_v2/` | Earlier code, kept for transparency; see the README in each folder |

## Reproducing the results

```bash
pip install -r requirements.txt
python benchmark_colab_v3.py                      # full run, about 5-6 h on a Colab CPU
QML_MODE=patch python benchmark_colab_v3.py       # adds the v3b analyses to an existing results file
QML_FAST=1 python benchmark_colab_v3.py           # quick smoke test (not for reporting)
```

Datasets are downloaded at run time from the UCI repository, UCSC Xena (TCGA-BRCA) and CDC
NHANES; a failed download stops the run rather than substituting synthetic data. Results are
written to `qml_results_v3/benchmark_results.json` after each dataset, so an interrupted run
resumes where it stopped.

To rebuild the manuscript and supplement from the results file:

```bash
python build_manuscript.py
```

The builder recomputes all statistics (corrected resampled t-test, equivalence tests at
±0.01/0.02/0.05 AUROC, Holm adjustment) from the per-split values and does not require SciPy.

## Version history

| Version | Status | Notes |
|---|---|---|
| 1 | Withdrawn | Implementation faults; see `legacy_v1/README.md` |
| 2 | Superseded | Corrected pipeline; see `legacy_v2/README.md` |
| 3 | Reported | Current code; analysis changes are listed in Supplementary Table S8 |

## Licence

MIT. See `LICENSE`.
