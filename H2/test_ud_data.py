"""Tests for the UD EWT loader.

Run from the repository root:  python -m pytest H2/test_ud_data.py -q
The parser tests are offline. The last test checks the real files and is skipped
when they have not been downloaded yet (python H2/ud_data.py does that).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from H2 import ud_data  # noqa: E402

SAMPLE = """\
# sent_id = s1
# text = I don't know.
1-2\tdon't\t_\t_\t_\t_\t_\t_\t_\t_
1\tI\tI\tPRON\tPRP\t_\t3\tnsubj\t_\t_
2\tdo\tdo\tAUX\tVBP\t_\t3\taux\t_\t_
3\tn't\tnot\tPART\tRB\t_\t4\tadvmod\t_\t_
4\tknow\tknow\tVERB\tVB\t_\t0\troot\t_\t_
4.1\tknows\t_\t_\t_\t_\t_\t_\t_\t_
5\t.\t.\tPUNCT\t.\t_\t4\tpunct\t_\t_

# sent_id = s2
1\tHello\thello\tINTJ\tUH\t_\t0\troot\t_\t_
"""


def test_parse_skips_ranges_and_empty_nodes():
    sentences = ud_data.parse_conllu(SAMPLE)
    assert [s.sent_id for s in sentences] == ["s1", "s2"]
    first = sentences[0]
    assert first.words == ("I", "do", "n't", "know", ".")
    assert first.upos == ("PRON", "AUX", "PART", "VERB", "PUNCT")
    assert first.xpos == ("PRP", "VBP", "RB", "VB", ".")
    assert sentences[1].words == ("Hello",)


def test_parse_rejects_malformed_lines():
    with pytest.raises(ValueError, match="10 tab-separated fields"):
        ud_data.parse_conllu("1\tonly\tthree\n")
    with pytest.raises(ValueError, match="out of order"):
        ud_data.parse_conllu("2\tx\tx\tX\tX\t_\t0\troot\t_\t_\n")


def test_encode_tags_uses_fixed_upos_order():
    assert ud_data.encode_tags(["ADJ", "X", "NOUN"], ud_data.UPOS_TAGS) == [0, 16, 7]
    with pytest.raises(ValueError, match="not in vocabulary"):
        ud_data.encode_tags(["NOT_A_TAG"], ud_data.UPOS_TAGS)


def test_tag_vocab_is_sorted_union():
    assert ud_data.tag_vocab([("NN", "DT"), ("VB", "NN")]) == ("DT", "NN", "VB")


def test_corrupt_file_is_rejected(tmp_path):
    (tmp_path / ud_data.split_filename("dev")).write_text(SAMPLE, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        ud_data.ensure_split("dev", tmp_path)


@pytest.mark.skipif(
    not all((ud_data.DEFAULT_DATA_DIR / ud_data.split_filename(s)).exists() for s in ud_data.SPLITS),
    reason="UD EWT not downloaded; run python H2/ud_data.py",
)
def test_real_files_match_pinned_release():
    data = ud_data.load_all()  # verifies hashes, counts and the UPOS inventory
    assert {s: len(v) for s, v in data.items()} == {s: c[0] for s, c in ud_data.EXPECTED_COUNTS.items()}
    assert {tag for s in data["train"] for tag in s.upos} == set(ud_data.UPOS_TAGS)
