"""Where the learning-curve data lives.

Default: <repo>/data (git-ignored). /shared/datasets is the intended home but is mounted read-only in the
agent container; set CURVEFAKER_DATA to point elsewhere (e.g. /shared/datasets/learning-curves after a move).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CURVEFAKER_DATA", REPO / "data"))
