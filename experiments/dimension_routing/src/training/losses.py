"""Loss functions for MRL and task-specialized dimension routing.

Implements the six-condition E1 objective suite:
- C1: global (all 28 queries, all 27 negatives)
- C2: query_masked (4 assigned queries, all 27 negatives)
- C3: task_isolated (4 assigned queries, 3 same-task negatives)
- C4: global_matched_query (4 mixed queries across tasks, all 27 negatives)
- C5: query_masked_cross_k3 (4 assigned queries, 3 cross-task negatives)
- C6: query_masked_mixed_k3 (4 assigned queries, 3 negatives sampled from full 27 pool)

Also implements:
- compute_nested_loss: genuine Matryoshka prefix learning on z[:, :dim]
- deterministic subsampling via torch.Generator
- strict positive-key exclusion
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib
import torch
import torch.nn.functional as F

DEFAULT_NESTED_DIMS: Tuple[int, ...] = (16, 32, 64, 128, 256, 512, 768)

C4_TASK_SCHEDULE: Tuple[Tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (1, 2, 3, 4),
    (2, 3, 4, 5),
    (3, 4, 5, 6),
    (4, 5, 6, 0),
    (5, 6, 0, 1),
    (6, 0, 1, 2),
)


def derive_seed(*args: Any) -> int:
    """Derive a stable 31-bit integer seed across platforms and processes using sha256."""
    key = "_".join(str(a) for a in args).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


def sample_k_negatives(
    candidate_indices: List[int],
    k: int,
    generator: Optional[torch.Generator] = None,
) -> List[int]:
    """Deterministically sample k negative indices from candidates."""
    if len(candidate_indices) <= k:
        return candidate_indices
    perm = torch.randperm(len(candidate_indices), generator=generator)
    return [candidate_indices[p.item()] for p in perm[:k]]


def compute_block_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    source_ids: torch.Tensor,
    block_idx: int,
    blocks: List[Tuple[int, int]],
    mode: str,
    block_to_task_id: Dict[int, int],
    temperature: float = 0.05,
    matched_k: Optional[int] = None,
    step_seed: Optional[int] = None,
    c4_query_indices: Optional[torch.Tensor] = None,
    base_seed: int = 42,
    epoch_idx: int = 0,
    step: int = 0,
) -> Optional[torch.Tensor]:
    """Compute InfoNCE contrastive loss for one embedding block slice under one of 6 conditions."""
    device = z1.device
    batch_size = z1.shape[0]
    start, end = blocks[block_idx]

    # -------------------------------------------------------------
    # C1: Standard Global Objective (all queries, all keys)
    # -------------------------------------------------------------
    if mode == "global":
        q = F.normalize(z1[:, start:end], dim=-1)
        k = F.normalize(z2[:, start:end], dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        labels = torch.arange(batch_size, device=device)
        return F.cross_entropy(logits, labels)

    # -------------------------------------------------------------
    # C4: Global Matched Query (4 queries from distinct tasks via rotating schedule, all 27 keys)
    # -------------------------------------------------------------
    elif mode == "global_matched_query":
        if c4_query_indices is not None:
            query_indices_tensor = c4_query_indices
        else:
            query_indices_tensor = torch.arange(min(4, batch_size), device=device)

        q = F.normalize(z1[query_indices_tensor, start:end], dim=-1)
        k = F.normalize(z2[:, start:end], dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        # Positive key is at the query's original row in full batch
        labels = query_indices_tensor
        return F.cross_entropy(logits, labels)

    # For assigned-task modes (C2, C3, C5, C6):
    if block_idx not in block_to_task_id:
        raise KeyError(f"Block index {block_idx} has no assigned task in block_to_task_id mapping.")

    target_task_id = block_to_task_id[block_idx]
    mask = (source_ids == target_task_id)
    if not mask.any():
        return None

    # -------------------------------------------------------------
    # C2: Query-Masked (4 assigned queries, all 27 keys)
    # -------------------------------------------------------------
    if mode == "query_masked":
        q = F.normalize(z1[mask, start:end], dim=-1)
        k = F.normalize(z2[:, start:end], dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        labels = torch.nonzero(mask, as_tuple=True)[0]
        return F.cross_entropy(logits, labels)

    # -------------------------------------------------------------
    # C3: Task-Isolated (4 assigned queries, 3 same-task keys)
    # -------------------------------------------------------------
    elif mode == "task_isolated":
        q = F.normalize(z1[mask, start:end], dim=-1)
        k = F.normalize(z2[mask, start:end], dim=-1)
        num_task_samples = q.size(0)
        logits = torch.matmul(q, k.T) / temperature
        # Positive pair is at identical index in task-isolated subset
        labels = torch.arange(num_task_samples, device=device)
        return F.cross_entropy(logits, labels)

    # -------------------------------------------------------------
    # C5: Query-Masked Cross-K3 (4 assigned queries, 3 strictly cross-task negatives)
    # -------------------------------------------------------------
    elif mode in ("query_masked_cross_k3", "query_masked_matched_k"):
        k_count = matched_k if matched_k is not None else 3
        q = F.normalize(z1[mask, start:end], dim=-1)
        k = F.normalize(z2[:, start:end], dim=-1)
        orig_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()

        # Candidates strictly from other tasks
        cross_candidates = (~mask).nonzero(as_tuple=True)[0].tolist()

        losses = []
        for q_idx, orig_idx in enumerate(orig_indices):
            # Derive sampling seed deterministically per (base_seed, epoch, step, block, query, "C5_negatives")
            q_seed = derive_seed(base_seed, epoch_idx, step, block_idx, q_idx, "C5_negatives")
            generator = torch.Generator(device="cpu").manual_seed(q_seed)
            sampled_negs = sample_k_negatives(cross_candidates, k_count, generator=generator)
            cand_k = torch.cat([k[orig_idx : orig_idx + 1], k[sampled_negs]], dim=0)
            logits = torch.matmul(q[q_idx : q_idx + 1], cand_k.T) / temperature
            # Target is index 0 (positive key placed at position 0)
            loss_i = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
            losses.append(loss_i)

        return torch.stack(losses).mean()

    # -------------------------------------------------------------
    # C6: Query-Masked Mixed-K3 (4 assigned queries, 3 negatives sampled from all 27)
    # -------------------------------------------------------------
    elif mode == "query_masked_mixed_k3":
        k_count = matched_k if matched_k is not None else 3
        q = F.normalize(z1[mask, start:end], dim=-1)
        k = F.normalize(z2[:, start:end], dim=-1)
        orig_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
        all_batch_indices = list(range(batch_size))

        losses = []
        for q_idx, orig_idx in enumerate(orig_indices):
            # Derive sampling seed deterministically per (base_seed, epoch, step, block, query, "C6_negatives")
            q_seed = derive_seed(base_seed, epoch_idx, step, block_idx, q_idx, "C6_negatives")
            generator = torch.Generator(device="cpu").manual_seed(q_seed)
            # Candidates from full 27 pool strictly excluding the query's own positive key
            candidate_negs = [idx for idx in all_batch_indices if idx != orig_idx]
            sampled_negs = sample_k_negatives(candidate_negs, k_count, generator=generator)
            cand_k = torch.cat([k[orig_idx : orig_idx + 1], k[sampled_negs]], dim=0)
            logits = torch.matmul(q[q_idx : q_idx + 1], cand_k.T) / temperature
            loss_i = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=device))
            losses.append(loss_i)

        return torch.stack(losses).mean()

    else:
        raise ValueError(f"Unknown objective mode '{mode}'. Choose from C1-C6.")


def compute_training_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    source_ids: torch.Tensor,
    blocks: List[Tuple[int, int]],
    mode: str,
    task_id_to_block: Dict[int, int],
    temperature: float = 0.05,
    matched_k: Optional[int] = None,
    step_seed: Optional[int] = None,
    loss_reduction: str = "mean",
    base_seed: int = 42,
    epoch_idx: int = 0,
    step: int = 0,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute aggregate loss across all blocks using mean or sum reduction."""
    block_to_task_id = {b: t for t, b in task_id_to_block.items()}
    losses = []
    block_losses: Dict[int, float] = {}

    # Pre-select deterministic rotating queries for C4 across all blocks in this batch
    c4_query_indices = None
    if mode == "global_matched_query":
        schedule_idx = step % len(C4_TASK_SCHEDULE)
        selected_tasks = C4_TASK_SCHEDULE[schedule_idx]
        c4_seed = derive_seed(base_seed, epoch_idx, step, "C4_queries")
        c4_gen = torch.Generator(device="cpu").manual_seed(c4_seed)
        query_indices = []
        for tid in selected_tasks:
            task_rows = (source_ids == tid).nonzero(as_tuple=True)[0].tolist()
            if task_rows:
                perm = torch.randperm(len(task_rows), generator=c4_gen)
                query_indices.append(task_rows[perm[0].item()])
        if len(query_indices) == 4:
            c4_query_indices = torch.tensor(query_indices, device=z1.device, dtype=torch.long)

    for block_idx in range(len(blocks)):
        loss = compute_block_loss(
            z1=z1,
            z2=z2,
            source_ids=source_ids,
            block_idx=block_idx,
            blocks=blocks,
            mode=mode,
            block_to_task_id=block_to_task_id,
            temperature=temperature,
            matched_k=matched_k,
            step_seed=step_seed,
            c4_query_indices=c4_query_indices,
            base_seed=base_seed,
            epoch_idx=epoch_idx,
            step=step,
        )
        if loss is not None:
            losses.append(loss)
            block_losses[block_idx] = float(loss.item())

    if len(losses) == 0:
        raise RuntimeError("No block losses were computed. Check batch composition and assignments.")

    if loss_reduction == "sum":
        total_loss = torch.stack(losses).sum()
    else:
        total_loss = torch.stack(losses).mean()
    return total_loss, block_losses


def compute_nested_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.05,
    dimensions: Sequence[int] = DEFAULT_NESTED_DIMS,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """Compute genuine Matryoshka nested prefix InfoNCE loss on z[:, :dim]."""
    device = z1.device
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size, device=device)
    losses = []
    dim_losses: Dict[int, float] = {}

    for dim in dimensions:
        q = F.normalize(z1[:, :dim], dim=-1)
        k = F.normalize(z2[:, :dim], dim=-1)
        logits = torch.matmul(q, k.T) / temperature
        loss = F.cross_entropy(logits, labels)
        losses.append(loss)
        dim_losses[dim] = float(loss.item())

    total_loss = torch.stack(losses).mean()
    return total_loss, dim_losses
