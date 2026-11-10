"""Load a bare TinyBERT-6L encoder from any checkpoint format run_experiments.py writes.

Formats handled (see run_experiments.py and the model wrappers):
    raw AutoModel state dict   mrl         keys like "embeddings.*", "encoder.layer.*", "pooler.*"
    "encoder." wrapper         mrl_sia,    keys like "encoder.embeddings.*", plus loss-module
                               mrl_pic     weights ("sia_loss_fn.*", "pic_loss_fn.*")
    "core.encoder." wrapper    mipic       TinyBERT_MIPIC nests everything under self.core
    training_state.pt          any         the above, inside {"model_state_dict": ...}

The format is detected from where the embedding table lives, not from a bare
"encoder." prefix: a raw BertModel also has "encoder.layer.*" keys, so stripping
"encoder." from it would silently produce a broken model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from models.tinybert import TINYBERT_6L_MODEL_NAME

# Prefixes in priority order. The marker key decides which one a checkpoint uses.
_WRAPPER_PREFIXES = ("core.encoder.", "encoder.")
_EMBEDDING_MARKER = "embeddings.word_embeddings.weight"


def _read_checkpoint(path: Path) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # training_state.pt also stores scheduler state, which the safe loader can
        # reject. These are our own training outputs, so a full unpickle is fine.
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return state


def _extract_encoder_state(
    state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Return (encoder state dict, detected format name)."""
    for prefix in _WRAPPER_PREFIXES:
        if prefix + _EMBEDDING_MARKER in state:
            stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            return stripped, f"wrapper ({prefix!r} prefix)"
    if _EMBEDDING_MARKER in state:
        return dict(state), "raw AutoModel"
    raise ValueError(
        "Could not find the TinyBERT embedding table in the checkpoint; expected one of "
        + ", ".join(repr(p + _EMBEDDING_MARKER) for p in ("",) + _WRAPPER_PREFIXES)
    )


def load_encoder(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[object, object]:
    """Return (encoder, tokenizer) for TinyBERT-6L, in eval mode.

    checkpoint_path=None gives the base pretrained model. device=None picks CUDA
    when available, else CPU. The checkpoint is loaded with strict=True, so any
    missing or unexpected encoder key raises.
    """
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(TINYBERT_6L_MODEL_NAME)
    encoder = AutoModel.from_pretrained(TINYBERT_6L_MODEL_NAME, output_hidden_states=True)

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        state = _read_checkpoint(path)
        encoder_state, fmt = _extract_encoder_state(state)

        # Older transformers versions saved position_ids as a persistent buffer;
        # current ones do not, so strict loading would fail on that key alone.
        expected = set(encoder.state_dict().keys())
        for key in [k for k in encoder_state if k.endswith("position_ids") and k not in expected]:
            del encoder_state[key]

        encoder.load_state_dict(encoder_state, strict=True)
        dropped = len(state) - len(encoder_state)
        print(
            f"load_encoder: {path.name} [{fmt}] loaded {len(encoder_state)} keys, "
            f"dropped {dropped} non-encoder keys"
        )

    encoder.to(device)
    encoder.eval()
    return encoder, tokenizer
