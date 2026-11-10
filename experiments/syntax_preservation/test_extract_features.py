"""Tests for the syntax-preservation feature extraction.

Run from the repository root:  python -m pytest experiments/syntax_preservation/test_extract_features.py -q
Needs the base TinyBERT weights (downloaded from Hugging Face on first use) and
the UD EWT files (python experiments/syntax_preservation/ud_data.py). Runs on CPU in about a minute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.syntax_preservation import extract_features as ef  # noqa: E402
from experiments.syntax_preservation import ud_data  # noqa: E402

LIMIT = 8
BATCH = 3  # several batches of unequal length, so padding and reordering are exercised

needs_ud = pytest.mark.skipif(
    not all((ud_data.DEFAULT_DATA_DIR / ud_data.split_filename(s)).exists() for s in ud_data.SPLITS),
    reason="UD EWT not downloaded; run python experiments/syntax_preservation/ud_data.py",
)


# --------------------------------------------------------------------------- helpers

def test_sorted_batches_cover_everything_longest_first():
    lengths = [3, 9, 1, 9, 5, 2, 7]
    batches = list(ef.sorted_batches(lengths, 3))
    flat = np.concatenate(batches)
    assert sorted(flat.tolist()) == list(range(len(lengths)))
    assert [len(b) for b in batches] == [3, 3, 1]
    assert flat.tolist() == [1, 3, 6, 4, 0, 5, 2]  # by length, ties in data order
    with pytest.raises(ValueError):
        list(ef.sorted_batches(lengths, 0))


def test_first_subword_positions():
    # [CLS] w0 w1a w1b w1c [SEP] [PAD]; w2 has no subword
    word_ids = [None, 0, 1, 1, 1, None, None]
    assert ef.first_subword_positions(word_ids, 3) == [1, 2, -1]


# --------------------------------------------------------------------------- model fixtures

@pytest.fixture(scope="module")
def model():
    from experiments.layerwise_probe.load_encoder import load_encoder
    return load_encoder(None, device="cpu")


def test_plan_points_at_first_subwords_and_drops_empty_words(model):
    _, tokenizer = model
    sentence = ud_data.Sentence("t", ("Unbelievably", "​", "good", "."),
                                ("ADV", "X", "ADJ", "PUNCT"), ("RB", "NFP", "JJ", "."))
    plan = ef.plan_ud_split(tokenizer, [sentence], max_positions=512)
    assert plan.keep == [[True, False, True, True]]
    assert plan.dropped_words == 1
    assert plan.offsets.tolist() == [0, 3]
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer(list(sentence.words), is_split_into_words=True)["input_ids"])
    first, good, period = plan.positions[0]
    assert first == 1  # right after [CLS]
    assert not tokens[first].startswith("##")
    assert all(t.startswith("##") for t in tokens[first + 1:good])  # the rest of "Unbelievably"
    assert (tokens[good], tokens[period]) == ("good", ".")


def test_plan_refuses_to_truncate(model):
    _, tokenizer = model
    long = ud_data.Sentence("long", ("word",) * 20, ("NOUN",) * 20, ("NN",) * 20)
    with pytest.raises(ValueError, match="refusing to truncate"):
        ef.plan_ud_split(tokenizer, [long], max_positions=10)


# --------------------------------------------------------------------------- end to end

@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    out = tmp_path_factory.mktemp("features")
    ef.main(["--name", "t", "--limit", str(LIMIT), "--batch-size", str(BATCH), "--out-dir", str(out)])
    return out / "t"


def _single(encoder, tokenizer, **kwargs):
    with torch.no_grad():
        return encoder(**tokenizer(**kwargs, return_tensors="pt"), output_hidden_states=True).hidden_states


@needs_ud
def test_ud_rows_match_per_sentence_encoding(extracted, model):
    encoder, tokenizer = model
    sentences = ud_data.load_split("dev")[:LIMIT]
    upos = np.load(extracted / "ud_dev_upos.npy")
    owner = np.load(extracted / "ud_dev_sentence.npy")
    assert upos.tolist() == ud_data.encode_tags([t for s in sentences for t in s.upos], ud_data.UPOS_TAGS)
    assert owner.tolist() == [i for i, s in enumerate(sentences) for _ in s.words]

    layers = {l: np.load(extracted / f"ud_dev_layer{l}.npy") for l in ef.LAYERS}
    assert all(a.shape == (len(upos), ef.HIDDEN) and a.dtype == np.float32 for a in layers.values())
    row = 0
    for sentence in sentences:  # each sentence encoded alone, no padding, in data order
        hidden = _single(encoder, tokenizer, text=list(sentence.words), is_split_into_words=True)
        ids = tokenizer(list(sentence.words), is_split_into_words=True).word_ids()
        positions = ef.first_subword_positions(ids, len(sentence.words))
        for l in ef.LAYERS:
            expected = hidden[l][0, positions, :].numpy()
            np.testing.assert_allclose(layers[l][row:row + len(positions)], expected, atol=1e-4, rtol=0)
        row += len(positions)
    assert row == len(upos)


def test_tweet_rows_match_per_tweet_encoding(extracted, model):
    encoder, tokenizer = model
    texts, labels = ef.read_tweets(ef.TWEET_SPLITS["test"], LIMIT)
    assert np.load(extracted / "tweet_test_labels.npy").tolist() == labels.tolist()
    layers = {l: np.load(extracted / f"tweet_test_layer{l}.npy") for l in ef.LAYERS}
    for i, text in enumerate(texts):
        hidden = _single(encoder, tokenizer, text=text, truncation=True, max_length=ef.TWEET_MAX_LENGTH)
        for l in ef.LAYERS:
            np.testing.assert_allclose(layers[l][i], hidden[l][0, 0, :].numpy(), atol=1e-4, rtol=0)
    # Sanity: layer 0 [CLS] is the same vector for every sentence (embedding + position 0).
    np.testing.assert_allclose(layers[0], np.broadcast_to(layers[0][0], layers[0].shape), atol=1e-5)


def test_meta_records_parts_and_rerun_skips_them(extracted, capsys):
    meta = json.loads((extracted / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["completed"]) == {"ud_train", "ud_dev", "ud_test", "tweet_train", "tweet_test"}
    assert meta["settings"]["limit"] == LIMIT
    assert not list(extracted.glob("*.partial*"))

    before = (extracted / "ud_dev_layer3.npy").stat().st_mtime_ns
    ef.main(["--name", "t", "--limit", str(LIMIT), "--out-dir", str(extracted.parent)])
    assert "Nothing to do" in capsys.readouterr().out
    assert (extracted / "ud_dev_layer3.npy").stat().st_mtime_ns == before


def test_changed_settings_are_refused(extracted):
    with pytest.raises(ValueError, match="different settings"):
        ef.main(["--name", "t", "--limit", str(LIMIT + 1), "--out-dir", str(extracted.parent)])
