"""UD English EWT: download, verify and parse the CoNLL-U files for the POS probe.

The release is pinned (r2.13) and each file is checked against a SHA-256 hash,
so every model and seed is probed on byte-identical data.

Parsing keeps only syntactic words, the lines whose ID is a plain integer:
    "3-4" lines are multiword-token ranges (e.g. "don't" -> "do" + "n't"); the
          words they span are listed separately, so the range line is skipped.
    "8.1" lines are empty nodes of enhanced dependencies; they have no surface
          form and are skipped.
This is the standard word-level view used for UPOS and XPOS tagging.

Example, from the repository root:
    python H2/ud_data.py            # download if needed, print statistics
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "H2" / "data" / "ud_ewt"

UD_RELEASE = "r2.13"
UD_URL = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/{release}/{filename}"
SPLITS = ("train", "dev", "test")
SHA256 = {
    "train": "34b201268743515b040b529da95d8b8fa874fcecac0279768bcfa796a1a9090e",
    "dev": "bea3f4b869539cf343a35c53148c85737f0c78231ee9321f73e0ae26f387394f",
    "test": "1655a2756f3db5386ad0fb08bef94a018827cc1c67ba097f16b9bde93ad2a188",
}
# Sentence and word counts of r2.13, checked after parsing.
EXPECTED_COUNTS = {
    "train": (12544, 204577),
    "dev": (2001, 25149),
    "test": (2077, 25094),
}

# The 17 Universal POS tags (https://universaldependencies.org/u/pos/), in a fixed
# order so label ids are identical for every model, seed and split.
UPOS_TAGS: Tuple[str, ...] = (
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
)


@dataclass(frozen=True)
class Sentence:
    sent_id: str
    words: Tuple[str, ...]
    upos: Tuple[str, ...]
    xpos: Tuple[str, ...]


def split_filename(split: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return f"en_ewt-ud-{split}.conllu"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Return the local path of one split, downloading it first if needed.

    The file is downloaded to a temporary name and only renamed into place once
    its hash matches, so an interrupted or altered download never looks valid.
    """
    path = data_dir / split_filename(split)
    if path.exists():
        actual = sha256_of(path)
        if actual != SHA256[split]:
            raise ValueError(f"{path} has SHA-256 {actual}, expected {SHA256[split]} "
                             f"(UD_English-EWT {UD_RELEASE}); delete it to re-download")
        return path

    data_dir.mkdir(parents=True, exist_ok=True)
    url = UD_URL.format(release=UD_RELEASE, filename=split_filename(split))
    partial = path.with_name(path.name + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
        handle.write(response.read())
    actual = sha256_of(partial)
    if actual != SHA256[split]:
        partial.unlink()
        raise ValueError(f"downloaded {split} has SHA-256 {actual}, expected {SHA256[split]}")
    partial.replace(path)
    return path


def parse_conllu(text: str) -> List[Sentence]:
    """Parse CoNLL-U text into sentences of (word, UPOS, XPOS), skipping ranges and empty nodes."""
    sentences: List[Sentence] = []
    sent_id: Optional[str] = None
    words: List[str] = []
    upos: List[str] = []
    xpos: List[str] = []

    def flush() -> None:
        nonlocal sent_id, words, upos, xpos
        if words:
            sentences.append(Sentence(
                sent_id=sent_id if sent_id is not None else f"#{len(sentences)}",
                words=tuple(words), upos=tuple(upos), xpos=tuple(xpos),
            ))
        sent_id, words, upos, xpos = None, [], [], []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            raise ValueError(f"line {line_no}: expected 10 tab-separated fields, got {len(fields)}")
        token_id = fields[0]
        if "-" in token_id or "." in token_id:
            continue
        if not token_id.isdigit():
            raise ValueError(f"line {line_no}: unexpected token id {token_id!r}")
        if int(token_id) != len(words) + 1:
            raise ValueError(f"line {line_no}: word id {token_id} out of order (expected {len(words) + 1})")
        words.append(fields[1])
        upos.append(fields[3])
        xpos.append(fields[4])
    flush()
    return sentences


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR, check_counts: bool = True) -> List[Sentence]:
    path = ensure_split(split, data_dir)
    sentences = parse_conllu(path.read_text(encoding="utf-8"))
    unknown = {tag for s in sentences for tag in s.upos} - set(UPOS_TAGS)
    if unknown:
        raise ValueError(f"{path.name}: UPOS tags outside the 17 UD tags: {sorted(unknown)}")
    if check_counts:
        got = (len(sentences), sum(len(s.words) for s in sentences))
        if got != EXPECTED_COUNTS[split]:
            raise ValueError(f"{path.name}: parsed {got} (sentences, words), expected {EXPECTED_COUNTS[split]}")
    return sentences


def load_all(data_dir: Path = DEFAULT_DATA_DIR) -> Dict[str, List[Sentence]]:
    return {split: load_split(split, data_dir) for split in SPLITS}


def tag_vocab(tag_sets: Sequence[Sequence[str]]) -> Tuple[str, ...]:
    """Sorted union of tags; used for XPOS, whose inventory is not fixed by UD."""
    return tuple(sorted({tag for tags in tag_sets for tag in tags}))


def encode_tags(tags: Sequence[str], vocab: Sequence[str]) -> List[int]:
    index = {tag: i for i, tag in enumerate(vocab)}
    missing = sorted(set(tags) - index.keys())
    if missing:
        raise ValueError(f"tags not in vocabulary: {missing}")
    return [index[tag] for tag in tags]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Download, verify and summarise UD English EWT.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args(argv)

    data = load_all(args.data_dir)
    print(f"UD_English-EWT {UD_RELEASE} in {args.data_dir}")
    for split, sentences in data.items():
        upos = Counter(tag for s in sentences for tag in s.upos)
        xpos = {tag for s in sentences for tag in s.xpos}
        n_words = sum(upos.values())
        majority, majority_count = upos.most_common(1)[0]
        print(f"  {split:5}: {len(sentences):6} sentences, {n_words:7} words, "
              f"{len(upos)} UPOS / {len(xpos)} XPOS tags, "
              f"majority UPOS {majority} = {100 * majority_count / n_words:.1f}%")


if __name__ == "__main__":
    main()
