"""Unified trainer for MRL and specialization experiments."""

from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.tinybert import mrl_sentence_embedding
from routing_v2.src.training.losses import compute_training_loss, compute_nested_loss
from routing_v2.src.utils.logging import ExperimentLogger


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    objective_mode: str,
    dimension_mode: str,
    blocks: List[Tuple[int, int]],
    task_id_to_block: Dict[int, int],
    nested_dims: Tuple[int, ...] = (16, 32, 64, 128, 256, 512, 768),
    temperature: float = 0.05,
    max_grad_norm: float = 1.0,
    matched_k: Optional[int] = None,
    loss_reduction: str = "mean",
    logger: Optional[ExperimentLogger] = None,
    epoch_idx: int = 0,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Train the model for one epoch using the specified objective and dimension mode."""
    model.train()
    total_loss = 0.0
    num_updates = 0
    start_time = time.time()
    accumulated_block_losses: Dict[int, float] = {}
    loss_trajectory: List[Dict[str, Any]] = []
    penultimate_window_losses: List[float] = []
    final_window_losses: List[float] = []
    final_dim_losses: Dict[str, float] = {}

    for step, batch in enumerate(loader):
        # Support both (enc, source_ids) and (enc1, enc2, source_ids)
        if len(batch) == 2:
            enc, source_ids = batch
            enc = {k: v.to(device) for k, v in enc.items()}
            source_ids = source_ids.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Two independent forward passes with dropout active
            out1 = model(**enc, output_hidden_states=True, return_dict=True)
            out2 = model(**enc, output_hidden_states=True, return_dict=True)
        else:
            enc1, enc2, source_ids = batch
            enc1 = {k: v.to(device) for k, v in enc1.items()}
            enc2 = {k: v.to(device) for k, v in enc2.items()}
            source_ids = source_ids.to(device)
            optimizer.zero_grad(set_to_none=True)

            out1 = model(**enc1, output_hidden_states=True, return_dict=True)
            out2 = model(**enc2, output_hidden_states=True, return_dict=True)

        z1 = mrl_sentence_embedding(out1)
        z2 = mrl_sentence_embedding(out2)

        if dimension_mode == "nested":
            batch_loss, step_dim_losses = compute_nested_loss(
                z1=z1,
                z2=z2,
                temperature=temperature,
                dimensions=nested_dims,
            )
            for d, l in step_dim_losses.items():
                accumulated_block_losses[d] = accumulated_block_losses.get(d, 0.0) + l
        else:
            batch_loss, step_block_losses = compute_training_loss(
                z1=z1,
                z2=z2,
                source_ids=source_ids,
                blocks=blocks,
                mode=objective_mode,
                task_id_to_block=task_id_to_block,
                temperature=temperature,
                matched_k=matched_k,
                step_seed=int((base_seed * 1000003 + epoch_idx * 10007 + step) & 0x7FFFFFFF),
                loss_reduction=loss_reduction,
                base_seed=base_seed,
                epoch_idx=epoch_idx,
                step=step,
            )
            for b, l in step_block_losses.items():
                accumulated_block_losses[b] = accumulated_block_losses.get(b, 0.0) + l

        batch_loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += batch_loss.item()
        num_updates += 1

        # Sample trajectory every 50 steps, plus step 1 and the final step
        if (step + 1) % 50 == 0 or (step + 1) == 1 or (step + 1) == len(loader):
            current_dim_losses = step_dim_losses if dimension_mode == "nested" else step_block_losses
            final_dim_losses = {str(k): float(v) for k, v in current_dim_losses.items()}
            trajectory_entry = {
                "step": step + 1,
                "batch_loss": float(batch_loss.item()),
                "running_avg_loss": float(total_loss / num_updates),
                "dim_losses": final_dim_losses,
            }
            loss_trajectory.append(trajectory_entry)

        # Track penultimate 10% (80%-90%) and final 10% (90%-100%) for convergence drift analysis
        if int(0.8 * len(loader)) <= (step + 1) < int(0.9 * len(loader)):
            penultimate_window_losses.append(float(batch_loss.item()))
        elif (step + 1) >= int(0.9 * len(loader)):
            final_window_losses.append(float(batch_loss.item()))

        if (step + 1) % 50 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            avg_so_far = total_loss / num_updates
            if logger:
                logger.log(
                    f"  Epoch {epoch_idx + 1} | Step {step + 1}/{len(loader)} | "
                    f"Loss: {batch_loss.item():.4f} (Avg: {avg_so_far:.4f}) | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    avg_epoch_loss = total_loss / max(1, num_updates)
    normalized_block_losses = {
        b: loss_sum / max(1, num_updates)
        for b, loss_sum in accumulated_block_losses.items()
    }
    final_window_avg = float(np.mean(final_window_losses)) if final_window_losses else avg_epoch_loss
    penultimate_window_avg = (
        float(np.mean(penultimate_window_losses)) if penultimate_window_losses else final_window_avg
    )
    final_loss_drift = float(final_window_avg - penultimate_window_avg)

    return {
        "epoch": epoch_idx + 1,
        "num_updates": num_updates,
        "average_loss": avg_epoch_loss,
        "block_losses": normalized_block_losses,
        "final_step_dim_losses": final_dim_losses,
        "elapsed_seconds": time.time() - start_time,
        "loss_trajectory": loss_trajectory,
        "final_window_avg_loss": final_window_avg,
        "penultimate_window_avg_loss": penultimate_window_avg,
        "final_loss_drift": final_loss_drift,
    }

