"""Dataset and collate utilities for multi-task MRL training."""

from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

# Canonical task ordering matching the 7 tasks in data/final_data.csv
CANONICAL_TASKS: List[str] = [
    "TWEET_EVAL_SENT",
    "BANKING77",
    "MRPC",
    "DEEHAN1866/WIC",
    "DGEM_FORMAT",
    "SICK",
    "STSB",
]

TASK_TO_ID: Dict[str, int] = {task: i for i, task in enumerate(CANONICAL_TASKS)}
ID_TO_TASK: Dict[int, str] = {i: task for i, task in enumerate(CANONICAL_TASKS)}


class MultiTaskTextDataset(Dataset):
    """Dataset providing text and stable integer task source ID."""

    def __init__(self, csv_path: str | Path, task_to_id: Dict[str, int] = TASK_TO_ID):
        df = pd.read_csv(csv_path)
        self.samples: List[Tuple[str, int]] = []
        for _, row in df.iterrows():
            source = str(row["source"]).strip()
            if source not in task_to_id:
                raise ValueError(f"Unknown source '{source}' not in task_to_id mapping.")
            self.samples.append((str(row["text"]), task_to_id[source]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]

    def get_source_ids(self) -> List[int]:
        """Return list of all source IDs for sampler initialization."""
        return [sample[1] for sample in self.samples]


def create_collate_fn(tokenizer: object, max_length: int = 256):
    """Creates a collate function that batches and tokenizes texts efficiently."""
    def collate_fn(batch: List[Tuple[str, int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        texts, source_ids = zip(*batch)
        encoded = tokenizer(
            list(texts),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return encoded, torch.tensor(source_ids, dtype=torch.long)

    return collate_fn

