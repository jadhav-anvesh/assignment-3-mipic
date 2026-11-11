"""Download and prepare standard evaluation datasets for all 7 canonical tasks."""

from pathlib import Path
import sys
import io
import urllib.request
import zipfile
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from routing_v2.src.data.validation import map_scitail_label


def download_and_prepare_eval_datasets(eval_dir: str | Path = "data/eval") -> dict:
    """Download and format eval datasets, returning tasks configuration dict."""
    from datasets import load_dataset
    out_path = Path(eval_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tasks = {}

    # 1. TWEET_EVAL_SENT (Local CSVs already exist)
    tasks["TWEET_EVAL_SENT"] = {
        "type": "single",
        "metric": "macro_f1",
        "train": "data/tweet_train.csv",
        "test": "data/tweet_test.csv",
    }
    print("[✓] TWEET_EVAL_SENT: Using existing local data/tweet_*.csv")

    # 2. BANKING77
    b77_tr = out_path / "banking77_train.csv"
    b77_te = out_path / "banking77_test.csv"
    if not (b77_tr.exists() and b77_te.exists()):
        print("Downloading BANKING77...")
        ds = load_dataset("PolyAI/banking77")
        pd.DataFrame({"text": ds["train"]["text"], "label": ds["train"]["label"]}).to_csv(b77_tr, index=False)
        pd.DataFrame({"text": ds["test"]["text"], "label": ds["test"]["label"]}).to_csv(b77_te, index=False)
    tasks["BANKING77"] = {
        "type": "single",
        "metric": "macro_f1",
        "train": str(b77_tr),
        "test": str(b77_te),
    }
    print("[✓] BANKING77 ready")

    # 3. MRPC
    mrpc_tr = out_path / "mrpc_train.csv"
    mrpc_te = out_path / "mrpc_test.csv"
    if not (mrpc_tr.exists() and mrpc_te.exists()):
        print("Downloading MRPC...")
        ds = load_dataset("nyu-mll/glue", "mrpc")
        pd.DataFrame({
            "sentence1": ds["train"]["sentence1"],
            "sentence2": ds["train"]["sentence2"],
            "label": ds["train"]["label"],
        }).to_csv(mrpc_tr, index=False)
        pd.DataFrame({
            "sentence1": ds["validation"]["sentence1"],
            "sentence2": ds["validation"]["sentence2"],
            "label": ds["validation"]["label"],
        }).to_csv(mrpc_te, index=False)
    tasks["MRPC"] = {
        "type": "pair",
        "metric": "macro_f1",
        "train": str(mrpc_tr),
        "test": str(mrpc_te),
    }
    print("[✓] MRPC ready")

    # 4. DEEHAN1866/WIC
    wic_tr = out_path / "wic_train.csv"
    wic_te = out_path / "wic_test.csv"
    if not (wic_tr.exists() and wic_te.exists()):
        print("Downloading DEEHAN1866/WIC...")
        try:
            ds = load_dataset("super_glue", "wic")
        except Exception:
            ds = load_dataset("aps/super_glue", "wic")
        pd.DataFrame({
            "sentence1": ds["train"]["sentence1"],
            "sentence2": ds["train"]["sentence2"],
            "label": ds["train"]["label"],
        }).to_csv(wic_tr, index=False)
        pd.DataFrame({
            "sentence1": ds["validation"]["sentence1"],
            "sentence2": ds["validation"]["sentence2"],
            "label": ds["validation"]["label"],
        }).to_csv(wic_te, index=False)
    tasks["DEEHAN1866/WIC"] = {
        "type": "pair",
        "metric": "macro_f1",
        "train": str(wic_tr),
        "test": str(wic_te),
    }
    print("[✓] DEEHAN1866/WIC ready")

    # 5. DGEM_FORMAT
    dgem_tr = out_path / "dgem_train.csv"
    dgem_te = out_path / "dgem_test.csv"
    if not (dgem_tr.exists() and dgem_te.exists()):
        print("Downloading DGEM_FORMAT (SciTail)...")
        try:
            ds = load_dataset("allenai/scitail", "dgem_format")
        except Exception:
            ds = load_dataset("scitail", "dgem_format")
        tr_labels = [map_scitail_label(l) for l in ds["train"]["label"]]
        te_labels = [map_scitail_label(l) for l in ds["test"]["label"]]
        pd.DataFrame({
            "sentence1": ds["train"]["premise"],
            "sentence2": ds["train"]["hypothesis"],
            "label": tr_labels,
        }).to_csv(dgem_tr, index=False)
        pd.DataFrame({
            "sentence1": ds["test"]["premise"],
            "sentence2": ds["test"]["hypothesis"],
            "label": te_labels,
        }).to_csv(dgem_te, index=False)
    tasks["DGEM_FORMAT"] = {
        "type": "pair",
        "metric": "macro_f1",
        "train": str(dgem_tr),
        "test": str(dgem_te),
    }
    print("[✓] DGEM_FORMAT ready")

    # 6. SICK (Direct download from Zenodo SemEval benchmark)
    sick_tr = out_path / "sick_train.csv"
    sick_te = out_path / "sick_test.csv"
    if not (sick_tr.exists() and sick_te.exists()):
        print("Downloading SICK from Zenodo benchmark...")
        sick_url = "https://zenodo.org/record/2787612/files/SICK.zip?download=1"
        req = urllib.request.urlopen(sick_url)
        with zipfile.ZipFile(io.BytesIO(req.read())) as z:
            with z.open("SICK.txt") as f:
                df = pd.read_csv(f, sep="\t")
        sick_label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
        df["mapped_label"] = df["entailment_label"].str.lower().map(sick_label_map)
        train_df = df[df["SemEval_set"] == "TRAIN"]
        test_df = df[df["SemEval_set"] == "TEST"]
        pd.DataFrame({
            "sentence1": train_df["sentence_A"],
            "sentence2": train_df["sentence_B"],
            "label": train_df["mapped_label"],
        }).to_csv(sick_tr, index=False)
        pd.DataFrame({
            "sentence1": test_df["sentence_A"],
            "sentence2": test_df["sentence_B"],
            "label": test_df["mapped_label"],
        }).to_csv(sick_te, index=False)
    tasks["SICK"] = {
        "type": "pair",
        "metric": "macro_f1",
        "train": str(sick_tr),
        "test": str(sick_te),
    }
    print("[✓] SICK ready")

    # 7. STSB
    stsb_tr = out_path / "stsb_train.csv"
    stsb_te = out_path / "stsb_test.csv"
    if not (stsb_tr.exists() and stsb_te.exists()):
        print("Downloading STSB...")
        ds = load_dataset("nyu-mll/glue", "stsb")
        pd.DataFrame({
            "sentence1": ds["train"]["sentence1"],
            "sentence2": ds["train"]["sentence2"],
            "label": ds["train"]["label"],
        }).to_csv(stsb_tr, index=False)
        pd.DataFrame({
            "sentence1": ds["validation"]["sentence1"],
            "sentence2": ds["validation"]["sentence2"],
            "label": ds["validation"]["label"],
        }).to_csv(stsb_te, index=False)
    tasks["STSB"] = {
        "type": "pair_regression",
        "metric": "spearman",
        "train": str(stsb_tr),
        "test": str(stsb_te),
    }
    print("[✓] STSB ready")

    return tasks


if __name__ == "__main__":
    tasks = download_and_prepare_eval_datasets()
    print(f"\nSuccessfully prepared {len(tasks)}/7 evaluation datasets.")
