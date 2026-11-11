# MRL Dimension Routing V2

## Overview
`routing_v2` is the systematic, modular refactor of the Matryoshka Representation Learning (MRL) dimension routing experiments. It resolves the methodological issues identified in the V1 implementation audit:

1. **Balanced Multi-Task Sampler**: Guarantees exactly $m$ samples per task in every training batch ($B = 7m$).
2. **Three Controlled Objective Modes**:
   - `global`: All examples are contrastive queries and keys.
   - `query_masked`: Assigned task examples are queries, all batch examples are keys.
   - `task_isolated`: Queries and keys are strictly restricted to the assigned task.
3. **Disjoint vs. Nested Architecture**: Clear separation between disjoint blockwise routing (`global_disjoint`, `query_masked`) and genuine nested-prefix Matryoshka learning (`global_nested` on $z_{0:d}$).
4. **Equal-Capacity Positional Blocks (E2)**: Seven 110-dim contiguous blocks spanning 768D to isolate coordinate position bias from capacity bottlenecks.
5. **Reproducibility & Rigor**: Pre-saved initial weights, paired seeds across conditions, stratified probe validation tuning, and multi-seed confidence reporting.

---

## Directory Structure

```text
routing_v2/
├── configs/
│   ├── e0_faithful_v1.yaml        # Reproduce Assignment 2 with identical task assignments
│   ├── e0_controlled_v1.yaml      # E0: Balanced stratified sampling baseline
│   ├── e1_objectives.yaml         # E1: Global vs Query-Masked vs Task-Isolated
│   ├── e2_assignment.yaml         # E2: Seven cyclic permutations over equal blocks
│   └── e3_nested_dimensions.yaml  # E3: Genuine nested-prefix MRL evaluation
├── data/
│   └── manifests/
│       └── dataset_manifest.json  # Checksums, schemas, and split accounting
├── src/
│   ├── data/
│   │   ├── dataset.py             # MultiTaskTextDataset and efficient collator
│   │   ├── sampler.py             # BalancedTaskBatchSampler (m samples per task)
│   │   └── validation.py          # Strict data and SciTail label validation
│   ├── training/
│   │   ├── checkpoint.py          # Seeded initial weights and checkpointing
│   │   ├── losses.py              # InfoNCE: global, query_masked, task_isolated, nested
│   │   └── trainer.py             # Unified, clean training loop
│   ├── evaluation/
│   │   ├── embeddings.py          # Fast TinyBERT embedding extraction
│   │   ├── metrics.py             # Macro-F1, Spearman correlation, uncertainty
│   │   ├── probes.py              # Logistic regression with validation C-tuning
│   │   └── evaluate.py            # Cross-evaluation engine across all blocks/tasks
│   ├── experiments/
│   │   ├── objective_comparison.py# E1 experiment runner
│   │   ├── assignment_sweep.py    # E2 cyclic assignment runner
│   │   ├── nested_dimensions.py   # E3 nested prefix runner
│   │   └── gradient_analysis.py   # E4 task gradient cosine matrix
│   └── utils/
│       ├── config.py              # YAML/JSON loader
│       ├── logging.py             # Experiment progress and metric logger
│       └── seed.py                # Deterministic seed utilities
├── scripts/
│   ├── validate_data.py           # Pre-flight data validation CLI
│   ├── run_experiment.py          # Unified experiment execution entrypoint
│   └── aggregate_results.py       # Aggregate metrics across seeds
└── tests/
    └── test_v2_core.py            # Comprehensive unit test suite (10 checks)
```

---

## Quickstart

### 1. Validate Data & Generate Manifest
```bash
python routing_v2/scripts/validate_data.py --training_csv data/final_data.csv
```

### 2. Run Tests
```bash
python -m pytest routing_v2/tests/test_v2_core.py -v
```

### 3. Run Experiments
```bash
# E1: Objective comparison (Global vs Query-Masked vs Task-Isolated)
python routing_v2/scripts/run_experiment.py --config routing_v2/configs/e1_objectives.yaml

# E2: Task-to-Dimension cyclic sweep
python routing_v2/scripts/run_experiment.py --config routing_v2/configs/e2_assignment.yaml

# E3: Nested Prefix MRL
python routing_v2/scripts/run_experiment.py --config routing_v2/configs/e3_nested_dimensions.yaml
```

### 4. Aggregate Results Across Seeds
```bash
python routing_v2/scripts/aggregate_results.py --output_dir routing_v2/outputs/e1_objective_comparison
```

