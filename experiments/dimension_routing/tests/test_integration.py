"""Integration test verifying end-to-end compatibility with repo data."""

import pytest
import torch
from pathlib import Path

from routing_v2.src.data.dataset import MultiTaskTextDataset, create_collate_fn, TASK_TO_ID
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.training.losses import compute_training_loss, compute_nested_loss
from routing_v2.src.experiments.gradient_analysis import compute_gradient_cosine_matrix


def test_end_to_end_data_and_losses_on_real_csv():
    csv_path = Path("data/final_data.csv")
    if not csv_path.exists():
        pytest.skip("data/final_data.csv not found")

    dataset = MultiTaskTextDataset(csv_path)
    assert len(dataset) == 20240

    source_ids = dataset.get_source_ids()
    sampler = BalancedTaskBatchSampler(source_ids, examples_per_task=2, seed=42, num_tasks=7)
    assert len(sampler) == 2347 // 2  # min task is SICK with 2347

    # Sample one batch
    batch_indices = next(iter(sampler))
    assert len(batch_indices) == 14  # 7 tasks * 2

    batch_samples = [dataset[i] for i in batch_indices]
    batch_sources = torch.tensor([s[1] for s in batch_samples], dtype=torch.long)

    # Simulated 768-dim embeddings for the batch
    torch.manual_seed(42)
    z1 = torch.randn(14, 768)
    z2 = torch.randn(14, 768)

    blocks = [
        (0, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 768),
    ]
    task_id_to_block = {i: i for i in range(7)}

    # Test all 6 objective modes
    for mode in (
        "global",
        "query_masked",
        "task_isolated",
        "global_matched_query",
        "query_masked_cross_k3",
        "query_masked_mixed_k3",
    ):
        loss, block_losses = compute_training_loss(
            z1=z1,
            z2=z2,
            source_ids=batch_sources,
            blocks=blocks,
            mode=mode,
            task_id_to_block=task_id_to_block,
            matched_k=1,
            step_seed=42,
        )
        assert loss is not None
        assert loss.item() > 0
        assert len(block_losses) == 7

    # Test nested MRL loss
    nested_loss, dim_losses = compute_nested_loss(z1, z2)
    assert nested_loss.item() > 0
    assert len(dim_losses) == 7


def test_gradient_analysis_on_real_batch():
    # Verify autograd gradient extraction on shared parameters
    torch.manual_seed(42)
    shared_param = torch.nn.Parameter(torch.randn(14, 768))
    z1 = shared_param * 1.5
    z2 = shared_param * 0.8
    source_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    blocks = [(0, 110), (110, 220), (220, 330), (330, 440), (440, 550), (550, 659), (659, 768)]
    task_id_to_block = {i: i for i in range(7)}

    # Compute task-specific losses
    task_losses = {}
    for tid in range(7):
        mask = (source_ids == tid)
        start, end = blocks[tid]
        q = torch.nn.functional.normalize(z1[mask, start:end], dim=-1)
        k = torch.nn.functional.normalize(z2[mask, start:end], dim=-1)
        logits = torch.matmul(q, k.T) / 0.05
        task_losses[tid] = torch.nn.functional.cross_entropy(logits, torch.arange(2))

    cos_mat, norms = compute_gradient_cosine_matrix(task_losses, [shared_param])
    assert cos_mat.shape == (7, 7)
    # Diagonal should be 1.0 (self cosine similarity)
    for i in range(7):
        assert pytest.approx(cos_mat[i, i], abs=1e-3) == 1.0
