"""
run_benchmark.py – main runner.

Usage:
    python run_benchmark.py               # full run (slow, ~30–60 min)
    python run_benchmark.py --fast        # reduced seeds/epochs (< 10 min)
    python run_benchmark.py --load        # reload saved results, make figures only

Outputs written to:
    results/benchmark_results.json
    results/complexity_results.json
    figures/fig1_auc_grouped.png
    figures/fig2_noise_curve.png
    figures/fig4_kernel_alignment.png
"""

import sys, json, time, argparse, io
import numpy as np
from pathlib import Path

# Force UTF-8 stdout so Unicode chars in dataset names don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# --- path fix so sub-modules find each other ---------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import config
from datasets   import LOADERS
from config     import DATASETS
from evaluate   import benchmark_dataset
from complexity import compute_complexity
from plots      import (fig1_grouped_bar, fig2_noise_curve,
                         fig4_kernel_alignment,
                         fig3_ready_compliance_illustrative,
                         fig5_cohens_d_heatmap,
                         fig6_triage_matrix)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--fast",   action="store_true",
                    help="Reduced run: 2 seeds, 3-fold outer, 10 VQC epochs")
parser.add_argument("--medium", action="store_true",
                    help="Medium run: 3 seeds, 5-fold outer, 50 VQC epochs (~5-8 hrs)")
parser.add_argument("--load",   action="store_true",
                    help="Load saved JSON results and only regenerate figures")
args = parser.parse_args()

if args.fast:
    config.FAST_MODE       = True
    config.N_REPEAT_SEED   = 2
    config.CV_OUTER        = 3
    config.CV_INNER        = 2
    config.N_EPOCHS_VQC    = 10
    config.PARAM_GRIDS["LR"]       = {"C": [0.1, 1]}
    config.PARAM_GRIDS["SVM_RBF"]  = {"C": [0.1, 1], "gamma": ["scale"]}
    config.PARAM_GRIDS["SVM_POLY"] = {"C": [1], "degree": [2], "gamma": ["scale"]}
    config.PARAM_GRIDS["RF"]       = {"n_estimators": [100], "max_depth": [5]}
    config.PARAM_GRIDS["GB"]       = {"n_estimators": [100], "learning_rate": [0.1],
                                       "max_depth": [3]}
    config.PARAM_GRIDS["QSVM"]     = {"C": [1.0]}
    print("[FAST MODE] 2 seeds, 3-fold outer, 2-fold inner, 10 VQC epochs")

elif args.medium:
    config.N_REPEAT_SEED   = 3          # 3 seeds (publishable floor)
    config.CV_OUTER        = 5          # full 5-fold outer
    config.CV_INNER        = 3          # full 3-fold inner
    config.N_EPOCHS_VQC    = 50         # 50 VQC epochs (vs 80 full, vs 10 fast)
    config.PARAM_GRIDS["LR"]       = {"C": [0.01, 0.1, 1, 10]}
    config.PARAM_GRIDS["SVM_RBF"]  = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]}
    config.PARAM_GRIDS["SVM_POLY"] = {"C": [0.1, 1, 10], "degree": [2, 3], "gamma": ["scale"]}
    config.PARAM_GRIDS["RF"]       = {"n_estimators": [100, 200], "max_depth": [5, 10]}
    config.PARAM_GRIDS["GB"]       = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
                                       "max_depth": [3, 5]}
    config.PARAM_GRIDS["QSVM"]     = {"C": [0.1, 1, 10]}
    print("[MEDIUM MODE] 3 seeds, 5-fold outer, 3-fold inner, 50 VQC epochs")


# ── helper: JSON serialise numpy ──────────────────────────────────────────────
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        return super().default(o)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    bench_path = RESULTS_DIR / "benchmark_results.json"
    comp_path  = RESULTS_DIR / "complexity_results.json"

    # ── 1. Run or load benchmark ──────────────────────────────────────────────
    if args.load and bench_path.exists():
        print("Loading saved benchmark results …")
        with open(bench_path) as f:
            all_results = json.load(f)
    else:
        # Resume from partial JSON if it exists (e.g. after a crash)
        if bench_path.exists():
            with open(bench_path) as f:
                all_results = json.load(f)
            done = set(all_results.keys())
            print(f"Resuming — already completed: {sorted(done)}")
        else:
            all_results = {}
            done = set()
        for label, loader_key, description, domain in DATASETS:
            if label in done:
                print(f"  Skipping {label} (already in results)")
                continue
            print(f"\n{'='*60}")
            print(f"Dataset: {description}")
            print(f"{'='*60}")
            X, y, name = LOADERS[loader_key]()
            print(f"  shape={X.shape}, class balance={np.bincount(y)}")

            t0 = time.time()
            res = benchmark_dataset(X, y, label, verbose=True)
            all_results[label] = res
            elapsed = time.time() - t0
            print(f"  Done in {elapsed/60:.1f} min")

            # save incrementally
            with open(bench_path, "w") as f:
                json.dump(all_results, f, cls=_Enc, indent=2)

    # ── 2. Run or load complexity analysis ───────────────────────────────────
    if args.load and comp_path.exists():
        print("\nLoading saved complexity results …")
        with open(comp_path) as f:
            complexity_data = json.load(f)
    else:
        complexity_data = {}
        for label, loader_key, description, domain in DATASETS:
            print(f"\nComplexity: {description}")
            X, y, _ = LOADERS[loader_key]()
            try:
                complexity_data[label] = compute_complexity(X, y, config.N_QUBITS)
                print(f"  {complexity_data[label]}")
            except Exception as e:
                print(f"  FAILED: {e}")
                complexity_data[label] = {}
        with open(comp_path, "w") as f:
            json.dump(complexity_data, f, cls=_Enc, indent=2)

    # ── 3. Print results table ────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("MAIN RESULTS TABLE  –  AUROC (mean ± 95 % CI)")
    print("="*80)
    from plots import MODEL_ORDER, MODEL_LABELS
    header = f"{'Model':<28}" + "".join(f"  {d[:14]:>14}" for d, *_ in DATASETS)
    print(header)
    print("-" * len(header))
    for m in MODEL_ORDER:
        row = f"{MODEL_LABELS[m]:<28}"
        for label, *_ in DATASETS:
            d = all_results.get(label, {})
            r = d.get(m, {}).get("auc", {})
            mn = r.get("mean", float("nan"))
            ci = r.get("ci",   float("nan"))
            row += f"  {mn:.3f}±{ci:.3f}  "
        print(row)

    print("\n\nWILCOXON TESTS  (QSVM ideal vs best classical)")
    for label, *_ in DATASETS:
        meta = all_results.get(label, {}).get("__meta__", {})
        p    = meta.get("wilcoxon_qsvm_vs_best_classical_p", float("nan"))
        bc   = meta.get("best_classical", "?")
        print(f"  {label:20s}  best_classical={bc:8s}  p={p:.4f}"
              + ("  *" if (not np.isnan(p) and p < 0.05) else "  ns"))

    # ── 4. Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig1_grouped_bar(all_results)
    fig4_kernel_alignment(complexity_data)
    fig5_cohens_d_heatmap(all_results)
    fig3_ready_compliance_illustrative()
    fig6_triage_matrix()

    # noise curve – collect QSVM results under different noise levels
    noise_data = {}
    datasets_for_noise = [d for d, *_ in DATASETS]
    for ds in datasets_for_noise:
        r = all_results.get(ds, {})
        noise_data[(0.0,   ds, "QSVM")]   = r.get("QSVM",       {}).get("auc",{}).get("mean", np.nan)
        noise_data[(0.01,  ds, "QSVM")]   = r.get("QSVM_noisy", {}).get("auc",{}).get("mean", np.nan)
        noise_data[(0.0,   ds, "GB")]     = r.get("GB",          {}).get("auc",{}).get("mean", np.nan)

    fig2_noise_curve({"levels": [0.0, 0.01], "datasets": datasets_for_noise,
                      "data": noise_data})

    print("\nAll done.  Results in results/   Figures in figures/")

    # ── 5. Return summary for manuscript builder ──────────────────────────────
    return all_results, complexity_data


if __name__ == "__main__":
    all_results, complexity_data = main()
