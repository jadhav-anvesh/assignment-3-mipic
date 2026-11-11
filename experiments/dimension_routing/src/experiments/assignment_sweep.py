"""E2: Task-to-Dimension Assignment Permutation Runner (Cyclic Rotations)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer
from routing_v2.src.data.dataset import MultiTaskTextDataset, create_collate_fn, CANONICAL_TASKS, TASK_TO_ID
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.training.checkpoint import save_initial_weights, load_initial_weights
from routing_v2.src.training.trainer import train_one_epoch
from routing_v2.src.evaluation.evaluate import evaluate_all_tasks
from routing_v2.src.utils.seed import set_seed
from routing_v2.src.utils.logging import ExperimentLogger

# Seven almost-equal contiguous blocks spanning all 768 dimensions
# Note: blocks 0-4 are 110D; blocks 5-6 are 109D.
POSITION_BLOCKS: List[Tuple[int, int]] = [
    (0, 110),
    (110, 220),
    (220, 330),
    (330, 440),
    (440, 550),
    (550, 659),
    (659, 768),
]

AUXILIARY_PERMUTATIONS: Dict[str, List[int]] = {
    "perm_aux_1": [2, 0, 4, 1, 6, 3, 5],
    "perm_aux_2": [5, 3, 1, 6, 0, 4, 2],
    "perm_aux_3": [4, 6, 0, 5, 2, 1, 3],
}


def create_cyclic_assignment(tasks: List[str], rotation: int) -> Dict[str, int]:
    """Rotate task assignments cyclically across position blocks."""
    num_tasks = len(tasks)
    return {task: (idx + rotation) % num_tasks for idx, task in enumerate(tasks)}


def create_assignment(tasks: List[str], spec: Any) -> Dict[str, int]:
    """Create task assignment mapping for cyclic rotation, named auxiliary permutation, or list."""
    if isinstance(spec, int):
        return create_cyclic_assignment(tasks, spec)
    elif isinstance(spec, str) and spec in AUXILIARY_PERMUTATIONS:
        perm = AUXILIARY_PERMUTATIONS[spec]
        return {task: perm[idx] for idx, task in enumerate(tasks)}
    elif isinstance(spec, (list, tuple)):
        return {task: spec[idx] for idx, task in enumerate(tasks)}
    else:
        raise ValueError(f"Unknown assignment specification: {spec}")


def run_assignment_sweep(
    config: Dict[str, Any],
    eval_tasks_config: Dict[str, Any],
    output_dir: str | Path = "outputs/e2_assignment_sweep",
    rotations: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute E2: Sweep task assignments over equal-capacity blocks across seeds."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Assert auxiliary permutations are strictly non-cyclic and valid bijections
    cyclic_perms = [tuple((i + r) % len(CANONICAL_TASKS) for i in range(len(CANONICAL_TASKS))) for r in range(len(CANONICAL_TASKS))]
    for name, perm in AUXILIARY_PERMUTATIONS.items():
        assert tuple(perm) not in cyclic_perms, f"Auxiliary permutation {name} duplicates a cyclic rotation!"
        assert sorted(perm) == list(range(len(CANONICAL_TASKS))), f"Auxiliary permutation {name} is not a valid bijection!"

    exp_cfg = config.get("experiment", {})
    if rotations is None:
        sweep_items: List[Any] = list(range(len(CANONICAL_TASKS)))
    else:
        sweep_items = list(rotations)

    if exp_cfg.get("include_auxiliary_permutations", False):
        for aux_name in AUXILIARY_PERMUTATIONS.keys():
            if aux_name not in sweep_items:
                sweep_items.append(aux_name)

    training_cfg = config["training"]
    seeds: List[int] = training_cfg.get("seeds", [training_cfg.get("seed", 42)])
    blocks = POSITION_BLOCKS

    dataset_path = config["data"]["training_csv"]
    dataset = MultiTaskTextDataset(dataset_path)
    source_ids = dataset.get_source_ids()

    tokenizer = load_tinybert_tokenizer()
    collate = create_collate_fn(tokenizer, max_length=config["model"]["max_length"])

    results: Dict[str, Any] = {}

    for seed in seeds:
        set_seed(seed)
        # Shared initial model checkpoint for this seed
        init_path = base_out / f"initial_tinybert_seed_{seed}.pt"
        base_model, _ = load_tinybert_6l()
        save_initial_weights(base_model, init_path)

        for item in sweep_items:
            # Crucial: Reset RNG state for each paired rotation/permutation within this seed
            # so that both the batch sequence and model dropout RNG are identical across rotations!
            set_seed(seed)

            assignment = create_assignment(CANONICAL_TASKS, spec=item)
            task_id_to_block = {TASK_TO_ID[task]: blk for task, blk in assignment.items()}

            item_tag = f"rotation_{item}" if isinstance(item, int) else str(item)
            rot_dir = base_out / f"seed_{seed}" / item_tag
            rot_dir.mkdir(parents=True, exist_ok=True)
            logger = ExperimentLogger(rot_dir / "train_log.json")
            logger.log(f"\n{'='*70}\nE2 Assignment Sweep: Seed {seed} | Assignment: {item_tag}\nAssignment: {assignment}\n{'='*70}")

            # Every rotation trains a fresh model starting from the identical initial seed weights!
            model, _ = load_tinybert_6l(device=device)
            load_initial_weights(model, init_path)

            sampler = BalancedTaskBatchSampler(
                source_ids=source_ids,
                examples_per_task=training_cfg["examples_per_task_per_batch"],
                seed=seed,
                num_tasks=len(CANONICAL_TASKS),
            )
            loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training_cfg["learning_rate"]),
                weight_decay=float(training_cfg["weight_decay"]),
            )
            epochs = int(training_cfg["epochs"])
            total_steps = len(loader) * epochs

            scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=int(training_cfg["warmup_ratio"] * total_steps),
                num_training_steps=total_steps,
            )

            # Reset PyTorch and CUDA RNG state immediately prior to training updates
            # so that model dropout stochasticity is identical across paired rotations of this seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for ep in range(epochs):
                sampler.set_epoch(ep)
                train_one_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=device,
                    objective_mode="query_masked",
                    dimension_mode="disjoint",
                    blocks=blocks,
                    task_id_to_block=task_id_to_block,
                    temperature=float(training_cfg["temperature"]),
                    max_grad_norm=float(training_cfg["max_grad_norm"]),
                    logger=logger,
                    epoch_idx=ep,
                    base_seed=seed,
                )

            matrix = evaluate_all_tasks(
                model=model,
                tokenizer=tokenizer,
                tasks_config=eval_tasks_config,
                dimension_mode="disjoint",
                blocks=blocks,
                device=device,
                tune_c=config.get("evaluation", {}).get("tune_c", False),
                seed=seed,
            )

            run_data = {
                "seed": seed,
                "rotation": item if isinstance(item, int) else None,
                "rotation_id": item_tag,
                "is_auxiliary": not isinstance(item, int),
                "assignment": assignment,
                "blocks": blocks,
                "metrics": matrix,
            }
            with open(rot_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2)

            results.setdefault(f"seed_{seed}", {})[item_tag] = run_data

    with open(base_out / "e2_sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
