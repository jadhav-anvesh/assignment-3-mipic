"""E4: Multi-Task Gradient Interference and Cosine Similarity Analysis.

Computes task-specific gradients on a common 768D pooled sentence representation
with respect to shared TinyBERT encoder parameters.
Supports hierarchical statistical reporting across batches and seeds.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer, mrl_sentence_embedding
from routing_v2.src.data.dataset import MultiTaskTextDataset, create_collate_fn, CANONICAL_TASKS, TASK_TO_ID
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.training.checkpoint import load_initial_weights
from routing_v2.src.utils.seed import set_seed
from routing_v2.src.utils.logging import ExperimentLogger


def compute_task_gradient_vector(
    loss: torch.Tensor,
    parameters: List[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute and flatten the gradient vector with respect to shared parameters."""
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    flattened = []
    for p, g in zip(parameters, grads):
        if g is not None:
            flattened.append(g.contiguous().view(-1))
        else:
            flattened.append(torch.zeros_like(p).contiguous().view(-1))
    return torch.cat(flattened)


def compute_gradient_cosine_matrix(
    task_losses: Dict[int, torch.Tensor],
    parameters: List[torch.nn.Parameter],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute pairwise gradient cosine similarity matrix between tasks."""
    task_ids = sorted(task_losses.keys())
    num_tasks = len(task_ids)
    grad_vectors: Dict[int, torch.Tensor] = {}
    grad_norms: Dict[int, float] = {}

    for idx, tid in enumerate(task_ids):
        is_last = (idx == num_tasks - 1)
        g = compute_task_gradient_vector(task_losses[tid], parameters, retain_graph=not is_last)
        norm = torch.norm(g).item()
        grad_norms[tid] = float(norm)
        # Avoid zero or non-finite gradient issues
        if norm > 1e-12 and not np.isnan(norm) and not np.isinf(norm):
            grad_vectors[tid] = g / norm
        else:
            grad_vectors[tid] = torch.zeros_like(g)

    cosine_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    for i, tid_i in enumerate(task_ids):
        for j, tid_j in enumerate(task_ids):
            if i == j:
                cosine_matrix[i, j] = 1.0
            else:
                cos_sim = torch.dot(grad_vectors[tid_i], grad_vectors[tid_j]).item()
                cosine_matrix[i, j] = float(cos_sim) if not np.isnan(cos_sim) else 0.0

    return cosine_matrix, grad_norms


def evaluate_batch_gradients_768d(
    model: torch.nn.Module,
    enc: Dict[str, torch.Tensor],
    source_ids: torch.Tensor,
    objective_type: str = "task_isolated",
    temperature: float = 0.05,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute task gradients on full 768D representation for one batch."""
    model.train()  # SimCSE two-view dropout active
    out1 = model(**enc, output_hidden_states=True, return_dict=True)
    out2 = model(**enc, output_hidden_states=True, return_dict=True)

    z1 = mrl_sentence_embedding(out1)  # [B, 768]
    z2 = mrl_sentence_embedding(out2)  # [B, 768]

    unique_tasks = sorted(torch.unique(source_ids).tolist())
    task_losses: Dict[int, torch.Tensor] = {}

    for tid in unique_tasks:
        mask = (source_ids == tid)
        if not mask.any():
            continue

        q = F.normalize(z1[mask, :768], dim=-1)

        if objective_type == "task_isolated":
            # Task-isolated: keys strictly from same task
            k = F.normalize(z2[mask, :768], dim=-1)
            logits = torch.matmul(q, k.T) / temperature
            labels = torch.arange(q.size(0), device=z1.device)
            loss_t = F.cross_entropy(logits, labels)
        elif objective_type == "query_masked":
            # Query-masked: keys from full batch
            k = F.normalize(z2[:, :768], dim=-1)
            logits = torch.matmul(q, k.T) / temperature
            labels = torch.nonzero(mask, as_tuple=True)[0]
            loss_t = F.cross_entropy(logits, labels)
        else:
            raise ValueError(f"Unknown objective_type: {objective_type}")

        task_losses[tid] = loss_t

    shared_params = [p for p in model.parameters() if p.requires_grad]
    return compute_gradient_cosine_matrix(task_losses, shared_params)


def run_gradient_analysis(
    config: Dict[str, Any],
    checkpoints: Dict[str, str],
    output_dir: str | Path = "outputs/e4_gradient_analysis",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute E4: Hierarchical multi-batch gradient interference analysis across checkpoints."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(base_out / "e4_run_log.json")

    analysis_cfg = config.get("analysis", {})
    num_batches = int(analysis_cfg.get("num_batches", 20))
    objective_types = analysis_cfg.get("objectives", ["task_isolated", "query_masked"])
    seeds = config.get("reproducibility", {}).get("seeds", [42, 123, 456])

    dataset_path = config.get("data", {}).get("training_csv", "data/final_data.csv")
    dataset = MultiTaskTextDataset(dataset_path)
    source_ids = dataset.get_source_ids()

    tokenizer = load_tinybert_tokenizer()
    collate = create_collate_fn(tokenizer, max_length=config.get("model", {}).get("max_length", 256))

    all_results: Dict[str, Any] = {}

    for cp_name, cp_path in checkpoints.items():
        logger.log(f"\n{'='*70}\nE4 Gradient Analysis on Checkpoint: {cp_name}\n{'='*70}")
        all_results[cp_name] = {}

        for obj_type in objective_types:
            logger.log(f"  Analyzing objective type: {obj_type}")
            seed_matrices: List[np.ndarray] = []
            seed_level_summaries: Dict[int, Any] = {}

            for seed in seeds:
                set_seed(seed)
                sampler = BalancedTaskBatchSampler(
                    source_ids=source_ids,
                    examples_per_task=4,
                    seed=seed,
                    num_tasks=len(CANONICAL_TASKS),
                )
                loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

                model, _ = load_tinybert_6l(device=device)
                if cp_path and Path(cp_path).exists():
                    load_initial_weights(model, cp_path)

                batch_cosines: List[np.ndarray] = []
                batch_norms_list: List[Dict[int, float]] = []

                for b_idx, (enc, s_ids) in enumerate(loader):
                    if b_idx >= num_batches:
                        break
                    enc = {k: v.to(device) for k, v in enc.items()}
                    s_ids = s_ids.to(device)

                    cos_mat, norms = evaluate_batch_gradients_768d(
                        model=model,
                        enc=enc,
                        source_ids=s_ids,
                        objective_type=obj_type,
                    )
                    batch_cosines.append(cos_mat)
                    batch_norms_list.append(norms)

                batch_stack = np.stack(batch_cosines, axis=0)  # [num_batches, 7, 7]
                seed_mean = np.mean(batch_stack, axis=0)
                seed_std = np.std(batch_stack, axis=0)

                seed_matrices.append(seed_mean)
                seed_level_summaries[seed] = {
                    "mean_matrix": seed_mean.tolist(),
                    "std_matrix": seed_std.tolist(),
                    "num_batches": len(batch_cosines),
                }

            # Hierarchical aggregation across seeds
            grand_mean = np.mean(np.stack(seed_matrices, axis=0), axis=0)
            grand_std = np.std(np.stack(seed_matrices, axis=0), axis=0)

            all_results[cp_name][obj_type] = {
                "grand_mean_matrix": grand_mean.tolist(),
                "across_seed_std": grand_std.tolist(),
                "tasks": CANONICAL_TASKS,
                "per_seed": seed_level_summaries,
            }

    with open(base_out / "gradient_interference_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.log(f"E4 Gradient Interference saved to: {base_out / 'gradient_interference_summary.json'}")
    return all_results
