"""Merge per-dataset tuned-VQC results (QML_MODE=vqc_tune) into the main results file.

    python merge_vqc_tuned.py [results_dir_with_vqc_tuned_files] [results_json]
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).parent
src_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "qml_results_v3"
target = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results" / "benchmark_results.json"

R = json.loads(target.read_text())
for label in R:
    f = src_dir / f"vqc_tuned_{label}.json"
    if not f.exists():
        sys.exit(f"Missing {f}; not merging anything.")
    t = json.loads(f.read_text())
    assert t["VQC_tuned"]["auc"]["n"] == R[label]["VQC"]["auc"]["n"], label
    R[label]["VQC_tuned"] = t["VQC_tuned"]
    R[label]["__meta__"]["vqc_tuning"] = {"grid": t["grid"], "chosen": t["chosen_layers_lr"],
                                          "fold_check_lr_max_diff": t["fold_check_lr_max_diff"]}
    R[label]["__meta__"]["time_sec"]["VQC_tuned"] = t["time_sec"]
target.write_text(json.dumps(R, indent=1))
print(f"Merged tuned VQC for {len(R)} datasets into {target}")
