"""Root-level Streamlit page wrapper for Customer Intelligence."""

from __future__ import annotations

import runpy
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "pages" / "2_Customer_Intelligence.py"

runpy.run_path(str(APP_PATH), run_name="__main__")
