"""Root-level Streamlit page wrapper for MLOps Monitoring."""

from __future__ import annotations

import runpy
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "pages" / "4_MLOps_Monitoring.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
