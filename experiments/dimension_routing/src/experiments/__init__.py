"""Experiments package for V2 routing."""

from routing_v2.src.experiments.objective_comparison import run_objective_comparison
from routing_v2.src.experiments.assignment_sweep import (
    POSITION_BLOCKS,
    create_cyclic_assignment,
    run_assignment_sweep,
)
from routing_v2.src.experiments.nested_dimensions import (
    NESTED_DIMS,
    run_nested_experiment,
)
from routing_v2.src.experiments.gradient_analysis import (
    compute_gradient_cosine_matrix,
)

__all__ = [
    "run_objective_comparison",
    "POSITION_BLOCKS",
    "create_cyclic_assignment",
    "run_assignment_sweep",
    "NESTED_DIMS",
    "run_nested_experiment",
    "compute_gradient_cosine_matrix",
]

