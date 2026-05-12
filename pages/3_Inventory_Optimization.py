"""Root-level Streamlit page wrapper for Inventory Optimization."""

from __future__ import annotations

import runpy
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "pages" / "3_Inventory_Optimization.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
