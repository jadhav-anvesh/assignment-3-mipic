"""E3: Genuine Nested-Prefix MRL Profile Runner.

Optimizes all 7 nested prefixes jointly during training, computes absolute retention
deltas (Delta F1 and Delta rho), and extracts effective dimension d* per task.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from models.tinybert import load_tinybert_6l, load_tinybert_tokenizer
from routing_v2.src.data.dataset import MultiTaskTextDataset, create_collate_fn, CANONICAL_TASKS
from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.training.checkpoint import save_initial_weights, load_initial_weights
from routing_v2.src.training.trainer import train_one_epoch
from routing_v2.src.evaluation.evaluate import evaluate_all_tasks
from routing_v2.src.utils.seed import set_seed
from routing_v2.src.utils.logging import ExperimentLogger

NESTED_DIMS: Tuple[int, ...] = (16, 32, 64, 128, 256, 512, 768)


def compute_effective_dimension(
    val_scores: Dict[str, float],
    test_scores: Optional[Dict[str, float]] = None,
    is_regression: bool = False,
    tolerance: float = 0.01,
) -> Any:
    """Select effective dimension d* on validation performance only, and evaluate on test.
    
    If test_scores is None, runs in backward-compatible mode returning (d*, deltas_from_768).
    If test_scores is provided, returns comprehensive dict with validation-selected d*
    and test-set evaluation at that dimension.
    """
    v_768 = val_scores.get("768", 0.0)

    # Backward-compatible 2-tuple return if only single score dict is provided
    if test_scores is None:
        deltas = {str(d): val_scores.get(str(d), 0.0) - v_768 for d in NESTED_DIMS}
        eff_d = 768
        for d in NESTED_DIMS:
            if val_scores.get(str(d), 0.0) >= (v_768 - tolerance):
                eff_d = d
                break
        return eff_d, deltas

    t_768 = test_scores.get("768", 0.0)

    val_deltas = {str(d): val_scores.get(str(d), 0.0) - v_768 for d in NESTED_DIMS}
    test_deltas = {str(d): test_scores.get(str(d), 0.0) - t_768 for d in NESTED_DIMS}

    val_ret = {str(d): (val_scores.get(str(d), 0.0) / v_768 * 100.0) if v_768 > 0 else 0.0 for d in NESTED_DIMS}
    test_ret = {str(d): (test_scores.get(str(d), 0.0) / t_768 * 100.0) if t_768 > 0 else 0.0 for d in NESTED_DIMS}

    # 1. First-crossing dimension on validation
    first_d = 768
    for d in NESTED_DIMS:
        if val_scores.get(str(d), 0.0) >= (v_768 - tolerance):
            first_d = d
            break

    # 2. Sustained dimension on validation (remains satisfied for all d' >= d)
    sustained_d = 768
    for i, d in enumerate(NESTED_DIMS):
        if all(val_scores.get(str(d_prime), 0.0) >= (v_768 - tolerance) for d_prime in NESTED_DIMS[i:]):
            sustained_d = d
            break

    # 3. 95% retention dimension on validation
    d_95 = 768
    for d in NESTED_DIMS:
        if val_scores.get(str(d), 0.0) >= (v_768 * 0.95):
            d_95 = d
            break

    # 4. 99% retention dimension on validation
    d_99 = 768
    for d in NESTED_DIMS:
        if val_scores.get(str(d), 0.0) >= (v_768 * 0.99):
            d_99 = d
            break

    return {
        "val_selected_d_tau": first_d,
        "val_selected_d_sustained": sustained_d,
        "val_selected_d_95": d_95,
        "val_selected_d_99": d_99,
        "test_score_at_val_selected_d": test_scores.get(str(first_d), 0.0),
        "test_score_at_sustained_d": test_scores.get(str(sustained_d), 0.0),
        "test_score_at_95_d": test_scores.get(str(d_95), 0.0),
        "test_score_768": t_768,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "val_deltas": val_deltas,
        "test_deltas": test_deltas,
        "val_retentions": val_ret,
        "test_retentions": test_ret,
    }


def run_nested_experiment(
    config: Dict[str, Any],
    eval_tasks_config: Dict[str, Any],
    output_dir: str | Path = "outputs/e3_nested_dimensions",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Execute E3: Train nested-prefix MRL and evaluate accuracy profile across prefix sizes."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_out = Path(output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    training_cfg = config["training"]
    seeds: List[int] = training_cfg.get("seeds", [training_cfg.get("seed", 42)])

    dataset_path = config["data"]["training_csv"]
    dataset = MultiTaskTextDataset(dataset_path)
    source_ids = dataset.get_source_ids()

    tokenizer = load_tinybert_tokenizer()
    collate = create_collate_fn(tokenizer, max_length=config["model"]["max_length"])

    results_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        set_seed(seed)
        seed_dir = base_out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        init_path = base_out / f"initial_tinybert_seed_{seed}.pt"
        base_model, _ = load_tinybert_6l()
        save_initial_weights(base_model, init_path)

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

        logger = ExperimentLogger(seed_dir / "train_log.json")
        logger.log(f"\n{'='*70}\nE3 Nested Prefix MRL: Seed {seed} | Dimensions {NESTED_DIMS}\n{'='*70}")

        for ep in range(epochs):
            sampler.set_epoch(ep)
            stats = train_one_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                objective_mode="global",
                dimension_mode="nested",
                blocks=[],
                task_id_to_block={},
                nested_dims=NESTED_DIMS,
                temperature=float(training_cfg["temperature"]),
                max_grad_norm=float(training_cfg["max_grad_norm"]),
                logger=logger,
                epoch_idx=ep,
                base_seed=seed,
            )
            logger.log_step(stats)

        # Save model checkpoint for reproducible probe re-evaluations
        torch.save(model.state_dict(), seed_dir / "final_model.pt")
        ckpt_meta = {
            "model_name": config["model"]["name"],
            "embedding_dim": config["model"]["embedding_dim"],
            "max_length": config["model"]["max_length"],
            "seed": seed,
            "nested_dims": list(NESTED_DIMS),
            "total_updates": total_steps,
            "epochs": epochs,
            "learning_rate": float(training_cfg["learning_rate"]),
            "warmup_ratio": float(training_cfg["warmup_ratio"]),
            "weight_decay": float(training_cfg["weight_decay"]),
            "temperature": float(training_cfg["temperature"]),
            "max_grad_norm": float(training_cfg["max_grad_norm"]),
            "examples_per_task_per_batch": training_cfg["examples_per_task_per_batch"],
            "batch_size": training_cfg.get("batch_size", 28),
        }
        with open(seed_dir / "checkpoint_meta.json", "w", encoding="utf-8") as f:
            json.dump(ckpt_meta, f, indent=2)

        # 1. Primary evaluation: train-only CV tuning for C with out-of-fold validation scores
        logger.log("Running primary evaluation (tune_c=True, validation-based d* selection)...")
        matrix_test, matrix_val, matrix_c = evaluate_all_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks_config=eval_tasks_config,
            dimension_mode="nested",
            nested_dims=NESTED_DIMS,
            device=device,
            tune_c=True,
            seed=seed,
            return_val_scores=True,
            return_c=True,
            normalize_prefixes=True,
        )

        # 2. Sensitivity evaluation: fixed C=1.0 probe across dimensions
        logger.log("Running sensitivity evaluation (fixed C=1.0)...")
        matrix_fixed_c = evaluate_all_tasks(
            model=model,
            tokenizer=tokenizer,
            tasks_config=eval_tasks_config,
            dimension_mode="nested",
            nested_dims=NESTED_DIMS,
            device=device,
            tune_c=False,
            seed=seed,
            return_val_scores=False,
            normalize_prefixes=True,
        )

        # Compute effective dimensions strictly using validation performance,
        # then evaluate test performance at that validation-selected dimension!
        task_analysis = {}
        for task_name in matrix_test.keys():
            val_scores = matrix_val[task_name]
            test_scores = matrix_test[task_name]
            is_reg = (eval_tasks_config.get(task_name, {}).get("type") == "pair_regression")
            analysis = compute_effective_dimension(
                val_scores=val_scores,
                test_scores=test_scores,
                is_regression=is_reg,
                tolerance=float(config.get("experiment", {}).get("tolerance", 0.01)),
            )
            task_analysis[task_name] = analysis

        seed_output = {
            "seed": seed,
            "nested_dims": list(NESTED_DIMS),
            "metrics": matrix_test,
            "val_metrics": matrix_val,
            "metrics_fixed_c": matrix_fixed_c,
            "selected_c": matrix_c,
            "analysis": task_analysis,
            "epoch_average_loss": stats.get("average_loss", 0.0),
            "prefix_losses": stats.get("block_losses", {}),
            "final_step_dim_losses": stats.get("final_step_dim_losses", {}),
            "loss_trajectory": stats.get("loss_trajectory", []),
            "final_window_avg_loss": stats.get("final_window_avg_loss", 0.0),
            "penultimate_window_avg_loss": stats.get("penultimate_window_avg_loss", 0.0),
            "final_loss_drift": stats.get("final_loss_drift", 0.0),
        }
        with open(seed_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(seed_output, f, indent=2)

        results_by_seed[f"seed_{seed}"] = seed_output

    with open(base_out / "e3_nested_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_by_seed, f, indent=2)

    return results_by_seed
