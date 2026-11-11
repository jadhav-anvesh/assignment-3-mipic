"""Embedding extraction using TinyBERT backbone and MRL readout."""

from typing import List, Optional
import numpy as np
import torch
from models.tinybert import mrl_sentence_embedding


def extract_embeddings(
    model: torch.nn.Module,
    texts: List[str],
    tokenizer: object,
    batch_size: int = 64,
    max_length: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract full 768-dim sentence embeddings for a list of text strings."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded, output_hidden_states=True, return_dict=True)
            emb = mrl_sentence_embedding(output)
            all_embs.append(emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)

