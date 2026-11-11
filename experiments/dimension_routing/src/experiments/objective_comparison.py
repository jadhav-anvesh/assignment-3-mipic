"""E1: Objective Comparison Runner (Global vs Query-Masked vs Task-Isolated)."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer
from routing_v2.src.data.dataset import MultiTaskTextDataset, create_collate_fn, TASK_TO_ID
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.training.checkpoint import save_initial_weights, load_initial_weights
from routing_v2.src.training.trainer import train_one_epoch
from routing_v2.src.evaluation.evaluate import evaluate_all_tasks
from routing_v2.src.utils.seed import set_seed
from routing_v2.src.utils.logging import ExperimentLogger


def run_objective_comparison(
    config: Dict[str, Any],
    eval_tasks_config: Dict[str, Any],
    output_dir: str | Path = "outputs/e1_objective_comparison",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute E1 experiment: Global vs Query-Masked vs Task-Isolated across seeds."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    training_cfg = config["training"]
    objective_cfg = config["objective"]
    modes: List[str] = objective_cfg["modes"]
    seeds: List[int] = training_cfg["seeds"]
    blocks = [tuple(b) for b in config["dimensions"]["blocks"]]
    task_assignment = config["assignment"]
    task_id_to_block = {TASK_TO_ID[task]: blk for task, blk in task_assignment.items()}

    dataset_path = config["data"]["training_csv"]
    dataset = MultiTaskTextDataset(dataset_path)
    source_ids = dataset.get_source_ids()

    tokenizer = load_tinybert_tokenizer()
    collate = create_collate_fn(tokenizer, max_length=config["model"]["max_length"])

    results_summary: Dict[str, Any] = {}

    for seed in seeds:
        set_seed(seed)
        seed_init_path = base_out / f"initial_tinybert_seed_{seed}.pt"

        # Initialize base model and save initial weights for seed pairing
        base_model, _ = load_tinybert_6l()
        save_initial_weights(base_model, seed_init_path)

        for mode in modes:
            run_dir = base_out / mode / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger = ExperimentLogger(run_dir / "train_log.json")
            logger.log(f"\n{'='*70}\nStarting E1: mode={mode} | seed={seed}\n{'='*70}")

            # Instantiate model with exact initial checkpoint for this seed
            model, _ = load_tinybert_6l(device=device)
            load_initial_weights(model, seed_init_path)

            if training_cfg.get("sampler_type") == "random_shuffle":
                g = torch.Generator()
                g.manual_seed(seed)
                loader = DataLoader(
                    dataset,
                    batch_size=training_cfg.get("batch_size", 16),
                    shuffle=True,
                    generator=g,
                    collate_fn=collate,
                )
                sampler = None
            else:
                sampler = BalancedTaskBatchSampler(
                    source_ids=source_ids,
                    examples_per_task=training_cfg.get("examples_per_task_per_batch", 4),
                    seed=seed,
                    num_tasks=len(TASK_TO_ID),
                )
                loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training_cfg["learning_rate"]),
                weight_decay=float(training_cfg["weight_decay"]),
            )
            epochs = int(training_cfg["epochs"])
            total_steps = len(loader) * epochs
            warmup_steps = int(training_cfg["warmup_ratio"] * total_steps)

            scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

            # Training loop
            for ep in range(epochs):
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(ep)
                epoch_stats = train_one_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    objective_mode=mode,
                    dimension_mode=config["experiment"]["dimension_mode"],
                    blocks=blocks,
                    task_id_to_block=task_id_to_block,
                    temperature=float(training_cfg["temperature"]),
                    max_grad_norm=float(training_cfg["max_grad_norm"]),
                    matched_k=training_cfg.get("matched_k"),
                    loss_reduction=objective_cfg.get("loss_reduction", "mean"),
                    logger=logger,
                    epoch_idx=ep,
                    base_seed=seed,
                )
                logger.log_step(epoch_stats)

            # Cross-evaluation across all blocks and tasks
            logger.log("Running cross-evaluation...")
            tune_c = config.get("evaluation", {}).get("tune_c", False)
            matrix = evaluate_all_tasks(
                model=model,
                tokenizer=tokenizer,
                tasks_config=eval_tasks_config,
                dimension_mode=config["experiment"]["dimension_mode"],
                blocks=blocks,
                device=device,
                tune_c=tune_c,
                seed=seed,
            )

            # Compile run metadata and metrics
            run_metadata = {
                "experiment": config["experiment"]["name"],
                "mode": mode,
                "seed": seed,
                "epochs": epochs,
                "optimizer_updates": total_steps,
                "blocks": blocks,
                "assignment": task_assignment,
                "metrics_matrix": matrix,
            }

            with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(run_metadata, f, indent=2)

            with open(run_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

            results_summary.setdefault(mode, {})[seed] = matrix

    return results_summary

