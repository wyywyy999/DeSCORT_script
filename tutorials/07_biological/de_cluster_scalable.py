#!/usr/bin/env python3
"""Compatibility entry point; implementation: descort.biological.de_cluster_scalable."""

# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

from pathlib import Path
import sys

# Support direct execution from a source checkout without installing DeSCORT.
_SOURCE = Path(__file__).resolve().parents[2] / 'src'
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from descort.biological.de_cluster_scalable import *  # noqa: F401,F403
