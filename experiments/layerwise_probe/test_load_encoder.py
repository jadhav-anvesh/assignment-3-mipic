"""Checks that load_encoder recovers the exact encoder weights from every checkpoint format.

Run from the repository root:  python -m pytest experiments/layerwise_probe/test_load_encoder.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.layerwise_probe.load_encoder import load_encoder  # noqa: E402


@pytest.fixture(scope="module")
def base_state():
    encoder, _ = load_encoder(None, device="cpu")
    return {k: v.clone() for k, v in encoder.state_dict().items()}


def _assert_matches(encoder, reference):
    loaded = encoder.state_dict()
    assert loaded.keys() == reference.keys()
    for key, value in reference.items():
        assert torch.equal(loaded[key].cpu(), value), key


def _wrap(state, prefix):
    wrapped = {prefix + k: v for k, v in state.items()}
    wrapped["pic_loss_fn.infonce_modules.dummy.weight"] = torch.zeros(3, 3)
    return wrapped


def test_base_model_is_eval_with_hidden_states(base_state):
    encoder, tokenizer = load_encoder(None, device="cpu")
    assert not encoder.training
    assert encoder.config.output_hidden_states
    out = encoder(**tokenizer(["a test sentence"], return_tensors="pt"))
    assert len(out.hidden_states) == 7


@pytest.mark.parametrize("prefix", ["encoder.", "core.encoder."])
def test_wrapper_checkpoint(tmp_path, base_state, prefix, capsys):
    path = tmp_path / "wrapper.pt"
    torch.save(_wrap(base_state, prefix), path)
    encoder, _ = load_encoder(path, device="cpu")
    _assert_matches(encoder, base_state)
    assert "dropped 1 non-encoder keys" in capsys.readouterr().out


def test_raw_checkpoint(tmp_path, base_state):
    # Raw keys include "encoder.layer.*"; they must not be mistaken for a wrapper.
    path = tmp_path / "raw.pt"
    torch.save(base_state, path)
    encoder, _ = load_encoder(path, device="cpu")
    _assert_matches(encoder, base_state)


def test_training_state_format(tmp_path, base_state):
    path = tmp_path / "training_state.pt"
    torch.save({"epoch": 1, "model_state_dict": _wrap(base_state, "core.encoder."), "history": []}, path)
    encoder, _ = load_encoder(path, device="cpu")
    _assert_matches(encoder, base_state)


def test_strict_mismatch_raises(tmp_path, base_state):
    broken = dict(base_state)
    broken.pop("pooler.dense.weight")
    path = tmp_path / "broken.pt"
    torch.save(broken, path)
    with pytest.raises(RuntimeError):
        load_encoder(path, device="cpu")


def test_loads_onto_cuda_when_available(base_state):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    encoder, _ = load_encoder(None)
    assert next(encoder.parameters()).is_cuda
