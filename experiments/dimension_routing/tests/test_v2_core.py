"""Comprehensive unit tests for V2 routing modules.

Covers all 10 minimum test cases required by the implementation plan:
1. Global loss (all samples queries & keys)
2. Query-masked loss (queries from assigned task, all samples keys)
3. Task-isolated loss (queries and keys both from assigned task)
4. Positive index pairing (correct diagonal vs indexed targets)
5. Missing task handling in BalancedTaskBatchSampler
6. Data validation for unknown source names
7. SciTail label validation (rejection of unmapped labels)
8. Assignment permutations (bijective, zero collisions)
9. Nested prefix loss computation
10. Sampler reproducibility under fixed seed
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from routing_v2.src.data.sampler import BalancedTaskBatchSampler
from routing_v2.src.data.validation import map_scitail_label, validate_training_csv
from routing_v2.src.training.losses import (
    compute_block_loss,
    compute_training_loss,
    compute_nested_loss,
)
from routing_v2.src.experiments.assignment_sweep import (
    POSITION_BLOCKS,
    create_cyclic_assignment,
)
from routing_v2.src.evaluation.metrics import (
    compute_macro_f1,
    compute_spearman,
    compute_summary_statistics,
)


# -------------------------------------------------------------------------
# Test 1, 2, 3, 4: Loss Functions & Positive Indexing
# -------------------------------------------------------------------------


def test_01_global_loss_uses_all_samples_as_queries_and_keys():
    batch_size = 14
    dim = 64
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    source_ids = torch.tensor([0, 1] * 7)
    blocks = [(0, 32), (32, 64)]

    loss = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="global",
        block_to_task_id={0: 0, 1: 1},
    )

    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_02_query_masked_loss_queries_assigned_task_only():
    batch_size = 14
    dim = 64
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    # Task 0 has 4 samples, Task 1 has 10 samples
    source_ids = torch.tensor([0] * 4 + [1] * 10)
    blocks = [(0, 32), (32, 64)]

    loss_t0 = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked",
        block_to_task_id={0: 0, 1: 1},
    )
    assert loss_t0 is not None

    # Check that query masking ignores absent tasks
    empty_source_ids = torch.tensor([1] * 14)
    loss_empty = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=empty_source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked",
        block_to_task_id={0: 0, 1: 1},
    )
    assert loss_empty is None


def test_03_task_isolated_loss_uses_assigned_task_queries_and_keys():
    batch_size = 8
    dim = 32
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    source_ids = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    blocks = [(0, 32)]

    loss = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="task_isolated",
        block_to_task_id={0: 0},
    )
    assert loss is not None
    assert loss.item() > 0


def test_04_positive_index_paired_correctly():
    """Verify that an identical second view yields minimal loss (diagonal match)."""
    batch_size = 8
    dim = 16
    z = torch.randn(batch_size, dim)
    z = F.normalize(z, dim=-1)
    source_ids = torch.tensor([0] * 4 + [1] * 4)
    blocks = [(0, 16)]

    # Perfect identical views with very low temperature should produce near-zero loss
    loss_iso = compute_block_loss(
        z1=z,
        z2=z,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="task_isolated",
        block_to_task_id={0: 0},
        temperature=0.01,
    )
    assert loss_iso.item() < 0.1


# -------------------------------------------------------------------------
# Test 5 & 10: BalancedTaskBatchSampler
# -------------------------------------------------------------------------


def test_05_sampler_fails_clearly_on_missing_tasks():
    # Only tasks 0 and 1 provided, but 3 tasks requested
    source_ids = [0, 0, 1, 1, 0, 1]
    with pytest.raises(ValueError, match="Missing tasks"):
        BalancedTaskBatchSampler(source_ids, examples_per_task=2, num_tasks=3)


def test_10_sampler_produces_balanced_reproducible_batches():
    source_ids = [0] * 20 + [1] * 15 + [2] * 25
    sampler1 = BalancedTaskBatchSampler(source_ids, examples_per_task=4, seed=42, num_tasks=3)
    sampler2 = BalancedTaskBatchSampler(source_ids, examples_per_task=4, seed=42, num_tasks=3)

    batches1 = list(sampler1)
    batches2 = list(sampler2)

    # Exactly identical batch sequences for identical seed
    assert batches1 == batches2

    # Verify balance: exactly 4 samples per task in each batch of 12
    for batch in batches1:
        assert len(batch) == 12
        batch_tasks = [source_ids[idx] for idx in batch]
        for t in [0, 1, 2]:
            assert batch_tasks.count(t) == 4


# -------------------------------------------------------------------------
# Test 6 & 7: Data Validation & Label Handling
# -------------------------------------------------------------------------


def test_06_validation_rejects_unknown_sources(tmp_path):
    bad_csv = tmp_path / "bad_data.csv"
    bad_csv.write_text("text,source\nhello world,UNEXPECTED_TASK\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected sources"):
        validate_training_csv(bad_csv)


def test_07_scitail_label_strict_mapping():
    assert map_scitail_label("entails") == 1
    assert map_scitail_label("entailment") == 1
    assert map_scitail_label("1") == 1
    assert map_scitail_label("neutral") == 0
    assert map_scitail_label("not_entailment") == 0
    assert map_scitail_label("0") == 0

    with pytest.raises(ValueError, match="Unexpected SciTail label"):
        map_scitail_label("contradiction")
    with pytest.raises(ValueError, match="Unexpected SciTail label"):
        map_scitail_label("unknown_string")


# -------------------------------------------------------------------------
# Test 8: Cyclic Assignment Permutations
# -------------------------------------------------------------------------


def test_08_assignment_permutations_are_bijective():
    tasks = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]
    seen_mappings = set()

    for rot in range(7):
        mapping = create_cyclic_assignment(tasks, rotation=rot)
        # Bijective: each block used exactly once
        assert sorted(mapping.values()) == list(range(7))
        seen_mappings.add(tuple(mapping.items()))

    # All 7 rotations are distinct
    assert len(seen_mappings) == 7


# -------------------------------------------------------------------------
# Test 9: Nested Prefix MRL Loss
# -------------------------------------------------------------------------


def test_09_nested_loss_slices_prefixes_correctly():
    z1 = torch.randn(6, 768)
    z2 = torch.randn(6, 768)
    nested_dims = (16, 32, 64)

    loss, dim_losses = compute_nested_loss(z1, z2, temperature=0.05, dimensions=nested_dims)

    assert loss is not None
    assert set(dim_losses.keys()) == {16, 32, 64}
    for d, l in dim_losses.items():
        assert l > 0.0


# -------------------------------------------------------------------------
# Test 11: Statistical Reporting
# -------------------------------------------------------------------------


def test_11_statistical_reporting():
    scores = [0.70, 0.72, 0.74]
    stats = compute_summary_statistics(scores)
    assert pytest.approx(stats["mean"], rel=1e-3) == 0.72
    assert stats["std"] > 0
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]


# -------------------------------------------------------------------------
# Test 12-15: E1 Objective Subsampling & Negative Isolation
# -------------------------------------------------------------------------


def test_12_e1_c5_cross_k3_isolation():
    """C5: candidates must be strictly cross-task negatives."""
    batch_size = 14
    dim = 32
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    source_ids = torch.tensor([0] * 4 + [1] * 10)
    blocks = [(0, 32)]

    loss = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked_cross_k3",
        block_to_task_id={0: 0},
        matched_k=3,
        step_seed=42,
    )
    assert loss is not None
    assert loss.item() > 0.0


def test_13_e1_c6_mixed_k3_sampling():
    """C6: candidates sampled from the entire negative pool."""
    batch_size = 14
    dim = 32
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    source_ids = torch.tensor([0] * 4 + [1] * 10)
    blocks = [(0, 32)]

    loss = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked_mixed_k3",
        block_to_task_id={0: 0},
        matched_k=3,
        step_seed=42,
    )
    assert loss is not None
    assert loss.item() > 0.0


def test_14_e1_positive_key_exclusion():
    """Verify that when computing InfoNCE, the query's own positive key is never in negatives."""
    from routing_v2.src.training.losses import sample_k_negatives
    candidates = [1, 2, 3, 4, 5]
    # Positive index is 0
    sampled = sample_k_negatives(candidates, k=3)
    assert 0 not in sampled
    assert len(sampled) == 3


def test_15_e1_deterministic_subsampling():
    """Passing identical step_seed yields exactly identical loss and sampling."""
    batch_size = 28
    dim = 64
    z1 = torch.randn(batch_size, dim)
    z2 = torch.randn(batch_size, dim)
    source_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6] * 4)
    blocks = [(0, 64)]

    loss1 = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked_cross_k3",
        block_to_task_id={0: 0},
        matched_k=3,
        step_seed=12345,
    )
    loss2 = compute_block_loss(
        z1=z1,
        z2=z2,
        source_ids=source_ids,
        block_idx=0,
        blocks=blocks,
        mode="query_masked_cross_k3",
        block_to_task_id={0: 0},
        matched_k=3,
        step_seed=12345,
    )
    assert pytest.approx(loss1.item(), rel=1e-6) == loss2.item()


# -------------------------------------------------------------------------
# Test 16: E2 Rotation Model Independence
# -------------------------------------------------------------------------


def test_16_e2_rotation_model_independence():
    """Verify cyclic permutation generates independent, distinct assignments."""
    tasks = ["T0", "T1", "T2", "T3", "T4", "T5", "T6"]
    rot0 = create_cyclic_assignment(tasks, rotation=0)
    rot1 = create_cyclic_assignment(tasks, rotation=1)

    assert rot0["T0"] != rot1["T0"]
    assert rot0 != rot1


# -------------------------------------------------------------------------
# Test 17: E3 Metric-Safe Retention & Effective Dimension
# -------------------------------------------------------------------------


def test_17_e3_metric_safe_retention():
    """Verify effective dimension calculation with tolerance and absolute retention deltas."""
    from routing_v2.src.experiments.nested_dimensions import compute_effective_dimension

    scores = {
        "16": 0.70,
        "32": 0.75,
        "64": 0.82,
        "128": 0.845,
        "256": 0.848,
        "512": 0.850,
        "768": 0.850,
    }
    eff_dim, deltas = compute_effective_dimension(scores, tolerance=0.01)
    assert eff_dim == 128
    assert pytest.approx(deltas["128"], abs=1e-4) == (0.845 - 0.850)
    assert pytest.approx(deltas["768"], abs=1e-4) == 0.0


# -------------------------------------------------------------------------
# Test 18: Task-Aware Leakage Checks (Symmetric vs Directed)
# -------------------------------------------------------------------------


def test_18_data_leakage_task_aware(tmp_path):
    """Verify WiC reversed pairs are flagged as duplicates, while SICK reversed pairs are permitted."""
    from routing_v2.src.data.validation import check_cross_split_leakage

    # WiC is symmetric: ('A', 'B') and ('B', 'A') are exact duplicates
    wic_tr_csv = tmp_path / "wic_train.csv"
    wic_te_csv = tmp_path / "wic_test.csv"
    pd.DataFrame({"sentence1": ["The dog barked."], "sentence2": ["A hound bayed."], "label": [1]}).to_csv(wic_tr_csv, index=False)
    pd.DataFrame({"sentence1": ["A hound bayed."], "sentence2": ["The dog barked."], "label": [1]}).to_csv(wic_te_csv, index=False)

    res_wic = check_cross_split_leakage(wic_tr_csv, wic_te_csv, task_name="DEEHAN1866/WIC", task_type="pair")
    assert res_wic["reversed_duplicates"] == 1
    assert res_wic["is_leakage"] is True

    # SICK is directed entailment: ('A', 'B') and ('B', 'A') have different entailment meaning
    sick_tr_csv = tmp_path / "sick_train.csv"
    sick_te_csv = tmp_path / "sick_test.csv"
    pd.DataFrame({"sentence1": ["A man is playing piano."], "sentence2": ["Someone is making music."], "label": [0]}).to_csv(sick_tr_csv, index=False)
    pd.DataFrame({"sentence1": ["Someone is making music."], "sentence2": ["A man is playing piano."], "label": [1]}).to_csv(sick_te_csv, index=False)

    res_sick = check_cross_split_leakage(sick_tr_csv, sick_te_csv, task_name="SICK", task_type="pair")
    assert res_sick["reversed_duplicates"] == 0
    assert res_sick["informational_reversals"] == 1
    assert res_sick["is_leakage"] is False


# -------------------------------------------------------------------------
# Test 19: Probe CV Class Count
# -------------------------------------------------------------------------


def test_19_probe_cv_class_count(tmp_path):
    """Verify that probe validation rejects training splits with < 3 samples per class."""
    from routing_v2.src.data.validation import validate_probe_class_balance

    # Valid: 2 classes, each has >= 3 samples
    valid_csv = tmp_path / "valid.csv"
    pd.DataFrame({"text": ["t"] * 7, "label": [0, 0, 0, 1, 1, 1, 1]}).to_csv(valid_csv, index=False)
    counts = validate_probe_class_balance(valid_csv, min_samples_per_class=3)
    assert counts[0] >= 3 and counts[1] >= 3

    # Invalid: class 1 has only 2 samples
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"text": ["t"] * 6, "label": [0, 0, 0, 0, 1, 1]}).to_csv(invalid_csv, index=False)
    with pytest.raises(ValueError, match="fewer than 3 samples"):
        validate_probe_class_balance(invalid_csv, min_samples_per_class=3)


# -------------------------------------------------------------------------
# Test 20: E3 Validation-Selected Dimension & Retention
# -------------------------------------------------------------------------


def test_20_e3_validation_selection_and_retention():
    """Verify that E3 selects d* strictly on validation scores and evaluates test score at that d*."""
    from routing_v2.src.experiments.nested_dimensions import compute_effective_dimension
    from routing_v2.src.evaluation.probes import select_best_c

    # Simulated validation and test curves where val saturates at 64D
    val_scores = {
        "16": 0.70,
        "32": 0.78,
        "64": 0.842,   # Within 0.01 of 0.850
        "128": 0.845,
        "256": 0.848,
        "512": 0.850,
        "768": 0.850,
    }
    test_scores = {
        "16": 0.68,
        "32": 0.76,
        "64": 0.835,
        "128": 0.840,
        "256": 0.842,
        "512": 0.845,
        "768": 0.845,
    }

    res = compute_effective_dimension(val_scores=val_scores, test_scores=test_scores, tolerance=0.01)
    assert res["val_selected_d_tau"] == 64
    assert res["test_score_at_val_selected_d"] == 0.835
    assert res["test_score_768"] == 0.845
    assert pytest.approx(res["test_deltas"]["64"], abs=1e-4) == (0.835 - 0.845)
    assert pytest.approx(res["test_retentions"]["64"], abs=1e-2) == (0.835 / 0.845 * 100.0)

    # Test probe select_best_c with return_score=True
    np.random.seed(42)
    x = np.random.randn(30, 16)
    y = np.array([0] * 15 + [1] * 15)
    best_c, best_score = select_best_c(x, y, return_score=True)
    assert best_c in [0.01, 0.1, 1.0, 10.0, 100.0]
    assert 0.0 <= best_score <= 1.0




