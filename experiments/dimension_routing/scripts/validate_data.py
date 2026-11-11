"""Data validation and manifest generation script."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routing_v2.src.data.validation import (
    validate_training_csv,
    validate_eval_csv,
    validate_probe_class_balance,
    check_cross_split_leakage,
    check_train_corpus_overlap,
)


def main():
    parser = argparse.ArgumentParser(description="Validate datasets and generate manifest.")
    parser.add_argument("--training_csv", default="data/final_data.csv", help="Path to pooled training data.")
    parser.add_argument("--manifest_out", default="routing_v2/data/manifests/dataset_manifest.json", help="Manifest output path.")
    args = parser.parse_args()

    manifest = {"training_data": {}, "eval_datasets": {}, "leakage_audit": {}}

    print("=" * 60)
    print("STEP 1: Validating pooled training CSV")
    print("=" * 60)
    train_report = validate_training_csv(args.training_csv)
    manifest["training_data"] = train_report

    print(f"  [OK] Training CSV valid: {train_report['total_rows']:,} rows")
    for src, count in sorted(train_report["source_counts"].items()):
        print(f"       - {src:<18}: {count:,} samples")

    eval_candidates = {
        "TWEET_EVAL_SENT": {
            "type": "single",
            "train": Path("data/tweet_train.csv"),
            "test": Path("data/tweet_test.csv"),
        },
        "BANKING77": {
            "type": "single",
            "train": Path("data/eval/banking77_train.csv"),
            "test": Path("data/eval/banking77_test.csv"),
        },
        "MRPC": {
            "type": "pair",
            "train": Path("data/eval/mrpc_train.csv"),
            "test": Path("data/eval/mrpc_test.csv"),
        },
        "DEEHAN1866/WIC": {
            "type": "pair",
            "train": Path("data/eval/wic_train.csv"),
            "test": Path("data/eval/wic_test.csv"),
        },
        "DGEM_FORMAT": {
            "type": "pair",
            "train": Path("data/eval/dgem_train.csv"),
            "test": Path("data/eval/dgem_test.csv"),
        },
        "SICK": {
            "type": "pair",
            "train": Path("data/eval/sick_train.csv"),
            "test": Path("data/eval/sick_test.csv"),
        },
        "STSB": {
            "type": "pair_regression",
            "train": Path("data/eval/stsb_train.csv"),
            "test": Path("data/eval/stsb_test.csv"),
        },
    }

    print("\n" + "=" * 60)
    print("STEP 2: Checking evaluation CSVs, probe balance, and data leakage")
    print("=" * 60)
    for task_name, paths in eval_candidates.items():
        if paths["train"].exists() and paths["test"].exists():
            tr_rep = validate_eval_csv(paths["train"], paths["type"], f"{task_name}_train")
            te_rep = validate_eval_csv(paths["test"], paths["type"], f"{task_name}_test")

            # Validate probe class balance (>= 3 samples per class) for classification tasks
            class_counts = {}
            if paths["type"] in ("single", "pair"):
                class_counts = validate_probe_class_balance(paths["train"], min_samples_per_class=3)

            # Check cross-split leakage
            leakage = check_cross_split_leakage(paths["train"], paths["test"], task_name, paths["type"])

            # Check corpus overlap
            corpus_overlap = check_train_corpus_overlap(args.training_csv, paths["test"], task_name, paths["type"])

            manifest["eval_datasets"][task_name] = {
                "train": tr_rep,
                "test": te_rep,
                "class_counts": class_counts,
            }
            manifest["leakage_audit"][task_name] = {
                "cross_split_leakage": leakage,
                "corpus_overlap_count": corpus_overlap,
            }

            leak_status = "[LEAK DETECTED]" if leakage["is_leakage"] else "[CLEAN]"
            print(f"  [OK] {task_name}: {tr_rep['num_samples']:,} train, {te_rep['num_samples']:,} test")
            print(f"       Class distribution: {class_counts}")
            print(f"       Split Leakage: {leak_status} (exact: {leakage['exact_duplicates']}, reversed: {leakage['reversed_duplicates']})")
            print(f"       Contrastive Corpus Overlap: {corpus_overlap} test sentences in final_data.csv")

    out_path = Path(args.manifest_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest successfully written to: {out_path}")


if __name__ == "__main__":
    main()
