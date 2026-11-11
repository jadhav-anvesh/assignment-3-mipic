import sys
from pathlib import Path

# Add repository root to sys.path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Also add dimension_routing directory so direct imports work
pkg_dir = Path(__file__).resolve().parents[1]
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

import experiments.dimension_routing as dimension_routing
sys.modules.setdefault("routing_v2", dimension_routing)
