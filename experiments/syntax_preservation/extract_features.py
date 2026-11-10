"""Extract layer 0-6 features for the syntax-preservation probes: UD EWT word vectors and tweet [CLS] vectors.

For one encoder (base TinyBERT or a trained checkpoint) this writes, to
experiments/syntax_preservation/features/<name>/:

    ud_{split}_layer{l}.npy     (n_words, 768) float32, the FIRST-SUBWORD vector of
                                every UD word, hidden_states[l][:, i, :]
    ud_{split}_upos.npy         (n_words,) int8, UPOS ids in ud_data.UPOS_TAGS order
    ud_{split}_xpos.npy         (n_words,) int16, XPOS ids in meta["xpos_vocab"] order
    ud_{split}_sentence.npy     (n_words,) int32, index of the sentence each word is in
    tweet_{split}_layer{l}.npy  (n_rows, 768) float32, the [CLS] vector, hidden_states[l][:, 0, :]
    tweet_{split}_labels.npy    (n_rows,) int64, sentiment labels
    meta.json                   settings, row counts, timings, completed parts

for UD splits train/dev/test, tweet splits train/test (data/tweet_*.csv), and
layers l = 0..6 (0 is the embedding output). Rows are in the original data order.

Design choices:
- Memory stays flat. Each batch's vectors are copied straight into .npy files
  opened as memory maps; nothing keeps a reference to the batch tensors. (On CPU,
  a NumPy view of a slice pins the whole hidden-state tensor, which is how
  layerwise_probe/extract_cls.py runs out of memory on a CPU-only machine.)
- Batches are formed after sorting sentences by token length, so little padding
  is computed. Padding positions are masked in attention, so the vectors match
  per-sentence encoding up to float rounding (checked by the tests).
- Each part (e.g. ud_train) is written under *.partial.npy names and renamed only
  when complete, and meta.json records completed parts. A rerun skips completed
  parts, so an interrupted run loses at most the part it was working on.
- No UD sentence is truncated: the run stops if one exceeds the model's 512
  positions. Words that the tokenizer maps to no subword at all are dropped
  together with their labels and counted in meta.json.
- Tweets use the layerwise_probe settings (truncation at 256 tokens), so tweet [CLS] vectors
  are the ones layerwise_probe probed.

Example, from the repository root:
    python experiments/syntax_preservation/extract_features.py --name base --limit 50          # quick smoke test
    python experiments/syntax_preservation/extract_features.py --name base
    python experiments/syntax_preservation/extract_features.py --name mipic_s42 \\
        --checkpoint MRL_LEXIS_H11_checkpoints/h11_probe/runs/seed42/checkpoints/mipic/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.syntax_preservation import ud_data  # noqa: E402

LAYERS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
HIDDEN = 768
UD_SPLITS = ud_data.SPLITS  # ("train", "dev", "test")
TWEET_SPLITS = {
    "train": REPO_ROOT / "data" / "tweet_train.csv",
    "test": REPO_ROOT / "data" / "tweet_test.csv",
}
TWEET_MAX_LENGTH = 256  # as in layerwise_probe/extract_cls.py and run_experiments.py
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "syntax_preservation" / "features"


# ---------------------------------------------------------------------------
# Helpers that do not need a model (unit-tested directly)
# ---------------------------------------------------------------------------

def sorted_batches(lengths: Sequence[int], batch_size: int) -> Iterator[np.ndarray]:
    """Yield index arrays covering range(len(lengths)), longest items first.

    Sorting by length keeps padding small; longest-first makes any memory
    problem show up in the first batch rather than at the end of a long run.
    The sort is stable, so ties keep their data order and runs are repeatable.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    order = np.argsort(-np.asarray(lengths), kind="stable")
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def first_subword_positions(word_ids: Sequence[Optional[int]], n_words: int) -> List[int]:
    """Token position of each word's first subword; -1 for a word with no subword.

    ``word_ids`` is the fast tokenizer's ``BatchEncoding.word_ids()``: the word
    index of each token, None for special tokens and padding.
    """
    positions = [-1] * n_words
    for position, word in enumerate(word_ids):
        if word is not None and positions[word] == -1:
            positions[word] = position
    return positions


@dataclass
class UDPlan:
    """Where each UD sentence's words go, decided before any model is run."""
    positions: List[List[int]]   # per sentence: first-subword positions of the KEPT words
    keep: List[List[bool]]       # per sentence: whether each word was kept
    offsets: np.ndarray          # row offset of each sentence's first kept word; len = n_sentences + 1
    lengths: List[int]           # token length of each sentence, specials included
    dropped_words: int


def plan_ud_split(tokenizer, sentences: Sequence[ud_data.Sentence], max_positions: int) -> UDPlan:
    positions, keep, lengths, kept_counts = [], [], [], []
    too_long = []
    for index, sentence in enumerate(sentences):
        encoded = tokenizer(list(sentence.words), is_split_into_words=True, truncation=False)
        length = len(encoded["input_ids"])
        if length > max_positions:
            too_long.append((sentence.sent_id, length))
        first = first_subword_positions(encoded.word_ids(), len(sentence.words))
        mask = [p >= 0 for p in first]
        positions.append([p for p in first if p >= 0])
        keep.append(mask)
        lengths.append(length)
        kept_counts.append(sum(mask))
    if too_long:
        raise ValueError(f"{len(too_long)} UD sentences exceed {max_positions} tokens "
                         f"(first: {too_long[0]}); refusing to truncate")
    offsets = np.concatenate([[0], np.cumsum(kept_counts)]).astype(np.int64)
    dropped = sum(len(s.words) for s in sentences) - int(offsets[-1])
    return UDPlan(positions, keep, offsets, lengths, dropped)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Output handling: memory-mapped .npy parts with atomic completion
# ---------------------------------------------------------------------------

class PartWriter:
    """Writes the .npy files of one part (e.g. ud_train) and renames them when complete."""

    def __init__(self, target: Path, prefix: str, n_rows: int):
        self.target = target
        self.prefix = prefix
        self.final: Dict[str, Path] = {}
        self.layers: Dict[int, np.memmap] = {}
        for layer in LAYERS:
            path = self._partial(f"layer{layer}")
            self.layers[layer] = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(n_rows, HIDDEN))
        self.filled = np.zeros(n_rows, dtype=bool)

    def _partial(self, stem: str) -> Path:
        self.final[stem] = self.target / f"{self.prefix}_{stem}.npy"
        return self.target / f"{self.prefix}_{stem}.partial.npy"

    def write_rows(self, rows: np.ndarray, per_layer: Dict[int, np.ndarray]) -> None:
        for layer, values in per_layer.items():
            # Fancy-index assignment copies into the memmap; no reference is kept.
            self.layers[layer][rows] = values
        self.filled[rows] = True

    def save_array(self, stem: str, array: np.ndarray) -> None:
        np.save(self._partial(stem), array)

    def commit(self) -> None:
        if not self.filled.all():
            raise RuntimeError(f"{self.prefix}: {int((~self.filled).sum())} rows were never written")
        for layer in LAYERS:
            if not np.isfinite(self.layers[layer]).all():
                raise ValueError(f"{self.prefix} layer {layer}: non-finite values")
            self.layers[layer].flush()
        # Drop every reference so the memmaps close; Windows can't rename an open file.
        self.layers.clear()
        for stem, final in self.final.items():
            os.replace(self.target / f"{self.prefix}_{stem}.partial.npy", final)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _forward(encoder, encoded) -> Tuple[torch.Tensor, ...]:
    device = next(encoder.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = encoder(**encoded, output_hidden_states=True, return_dict=True)
    if len(output.hidden_states) != len(LAYERS):
        raise ValueError(f"expected {len(LAYERS)} hidden states, got {len(output.hidden_states)}")
    return output.hidden_states


def extract_ud_split(encoder, tokenizer, sentences, plan: UDPlan, writer: PartWriter,
                     batch_size: int, progress: Callable[[int], None]) -> None:
    for batch in sorted_batches(plan.lengths, batch_size):
        words = [list(sentences[i].words) for i in batch]
        encoded = tokenizer(words, is_split_into_words=True, truncation=False,
                            padding=True, return_tensors="pt")
        batch_index, token_index, rows = [], [], []
        for slot, sentence in enumerate(batch):
            positions = plan.positions[sentence]
            # Batched tokenization must agree with the plan (right padding only appends).
            first = first_subword_positions(encoded.word_ids(slot), len(sentences[sentence].words))
            if [p for p in first if p >= 0] != positions:
                raise RuntimeError(f"subword positions changed under batching for sentence {sentence}")
            batch_index.extend([slot] * len(positions))
            token_index.extend(positions)
            rows.extend(range(plan.offsets[sentence], plan.offsets[sentence + 1]))
        with torch.no_grad():
            hidden = _forward(encoder, encoded)
        b = torch.as_tensor(batch_index)
        t = torch.as_tensor(token_index)
        writer.write_rows(
            np.asarray(rows, dtype=np.int64),
            {layer: hidden[layer][b, t, :].float().cpu().numpy() for layer in LAYERS},
        )
        progress(len(batch))


def extract_tweet_split(encoder, tokenizer, texts: List[str], writer: PartWriter,
                        batch_size: int, progress: Callable[[int], None]) -> None:
    lengths = [len(ids) for ids in tokenizer(texts, truncation=True, max_length=TWEET_MAX_LENGTH)["input_ids"]]
    for batch in sorted_batches(lengths, batch_size):
        encoded = tokenizer([texts[i] for i in batch], truncation=True, max_length=TWEET_MAX_LENGTH,
                            padding=True, return_tensors="pt")
        with torch.no_grad():
            hidden = _forward(encoder, encoded)
        writer.write_rows(batch, {layer: hidden[layer][:, 0, :].float().cpu().numpy() for layer in LAYERS})
        progress(len(batch))


def read_tweets(csv_path: Path, limit: Optional[int]) -> Tuple[List[str], np.ndarray]:
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain 'text' and 'label' columns.")
    if limit is not None:
        frame = frame.head(limit)
    # Same casting as evaluate.ClassificationDataset and layerwise_probe/extract_cls.py.
    return frame["text"].astype(str).tolist(), frame["label"].astype(np.int64).to_numpy()


def estimate_bytes(n_ud_words: int, n_tweets: int) -> int:
    return (n_ud_words + n_tweets) * HIDDEN * 4 * len(LAYERS)


def _progress_printer(label: str, total: int) -> Callable[[int], None]:
    state = {"done": 0, "start": time.perf_counter(), "last": 0.0}

    def update(n: int) -> None:
        state["done"] += n
        now = time.perf_counter()
        if now - state["last"] >= 30 or state["done"] == total:
            state["last"] = now
            rate = state["done"] / max(now - state["start"], 1e-9)
            eta = (total - state["done"]) / rate if rate > 0 else float("nan")
            print(f"  {label}: {state['done']}/{total} ({rate:.0f}/s, about {eta / 60:.1f} min left)",
                  flush=True)

    return update


def load_meta(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_meta(path: Path, meta: dict) -> None:
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    os.replace(partial, path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract layer 0-6 UD word and tweet [CLS] features for the syntax-preservation experiment.")
    parser.add_argument("--name", required=True, help="Feature folder name, e.g. base or mipic_s42.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Omit for base TinyBERT.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Use only the first N sentences/rows of each split (smoke tests).")
    parser.add_argument("--parts", nargs="+", default=None,
                        help="Only these parts, e.g. ud_dev tweet_test (default: all five).")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice).")
    args = parser.parse_args(argv)

    from layerwise_probe.load_encoder import load_encoder  # read-only reuse of the layerwise_probe loader

    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = args.out_dir / args.name
    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "meta.json"
    meta = load_meta(meta_path)

    settings = {
        "name": args.name,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "limit": args.limit,
        "layers": list(LAYERS),
        "tweet_max_length": TWEET_MAX_LENGTH,
        "ud_release": ud_data.UD_RELEASE,
        "dtype": "float32",
    }
    if meta.get("settings") not in (None, settings):
        raise ValueError(f"{meta_path} was written with different settings:\n  {meta['settings']}\n"
                         f"now:\n  {settings}\nUse another --name or delete {target}.")
    if args.checkpoint is not None and "checkpoint_sha256" not in meta:
        print(f"Hashing {args.checkpoint} ...", flush=True)
        meta["checkpoint_sha256"] = file_sha256(args.checkpoint)
    meta.setdefault("completed", {})
    meta["settings"] = settings

    ud = {split: ud_data.load_split(split, check_counts=args.limit is None) for split in UD_SPLITS}
    if args.limit is not None:
        ud = {split: sentences[:args.limit] for split, sentences in ud.items()}
    xpos_vocab = ud_data.tag_vocab([s.xpos for sentences in ud.values() for s in sentences])
    if meta.get("xpos_vocab") not in (None, list(xpos_vocab)):
        raise ValueError("XPOS vocabulary differs from the one in meta.json")
    meta["xpos_vocab"] = list(xpos_vocab)
    tweets = {split: read_tweets(path, args.limit) for split, path in TWEET_SPLITS.items()}

    all_parts = [f"ud_{s}" for s in UD_SPLITS] + [f"tweet_{s}" for s in TWEET_SPLITS]
    parts = args.parts or all_parts
    unknown = sorted(set(parts) - set(all_parts))
    if unknown:
        raise ValueError(f"unknown parts {unknown}; choose from {all_parts}")
    def complete_on_disk(part: str) -> bool:
        stems = [f"layer{l}" for l in LAYERS] + (["upos", "xpos", "sentence"] if part.startswith("ud_") else ["labels"])
        return part in meta["completed"] and all((target / f"{part}_{s}.npy").exists() for s in stems)

    todo = [p for p in parts if not complete_on_disk(p)]
    for part in parts:
        if part not in todo:
            print(f"{part}: already complete, skipping")
    if not todo:
        save_meta(meta_path, meta)
        print(f"Nothing to do in {target}")
        return

    encoder, tokenizer = load_encoder(args.checkpoint, device=device)
    max_positions = encoder.config.max_position_embeddings
    print(f"Device: {device}, torch threads: {torch.get_num_threads()}", flush=True)

    plans = {s: plan_ud_split(tokenizer, ud[s], max_positions) for s in UD_SPLITS if f"ud_{s}" in todo}
    needed = estimate_bytes(sum(int(p.offsets[-1]) for p in plans.values()),
                            sum(len(tweets[s][0]) for s in TWEET_SPLITS if f"tweet_{s}" in todo))
    free = shutil.disk_usage(target).free
    print(f"Writing about {needed / 2**30:.2f} GiB to {target} ({free / 2**30:.1f} GiB free)", flush=True)
    if needed > free:
        raise OSError(f"not enough disk space: need {needed / 2**30:.2f} GiB, have {free / 2**30:.1f} GiB")

    for part in todo:
        kind, split = part.split("_", 1)
        start = time.perf_counter()
        if kind == "ud":
            sentences, plan = ud[split], plans[split]
            n_rows = int(plan.offsets[-1])
            writer = PartWriter(target, part, n_rows)
            extract_ud_split(encoder, tokenizer, sentences, plan, writer, args.batch_size,
                             _progress_printer(part, len(sentences)))
            kept = [(tag, x, i) for i, s in enumerate(sentences)
                    for tag, x, k in zip(s.upos, s.xpos, plan.keep[i]) if k]
            writer.save_array("upos", np.asarray(ud_data.encode_tags([k[0] for k in kept], ud_data.UPOS_TAGS), dtype=np.int8))
            writer.save_array("xpos", np.asarray(ud_data.encode_tags([k[1] for k in kept], xpos_vocab), dtype=np.int16))
            writer.save_array("sentence", np.asarray([k[2] for k in kept], dtype=np.int32))
            info = {"sentences": len(sentences), "rows": n_rows, "dropped_words": plan.dropped_words,
                    "max_tokens": max(plan.lengths)}
        else:
            texts, labels = tweets[split]
            n_rows = len(texts)
            writer = PartWriter(target, part, n_rows)
            extract_tweet_split(encoder, tokenizer, texts, writer, args.batch_size,
                                _progress_printer(part, n_rows))
            writer.save_array("labels", labels)
            info = {"rows": n_rows}
        writer.commit()
        info["seconds"] = round(time.perf_counter() - start, 1)
        info["date"] = datetime.now().isoformat(timespec="seconds")
        info["device"] = device
        meta["completed"][part] = info
        save_meta(meta_path, meta)
        print(f"{part}: {n_rows} rows in {info['seconds']:.0f}s"
              + (f", {info['dropped_words']} words dropped (no subword)" if kind == "ud" else ""), flush=True)

    print(f"Saved features to {target}")


if __name__ == "__main__":
    main()
