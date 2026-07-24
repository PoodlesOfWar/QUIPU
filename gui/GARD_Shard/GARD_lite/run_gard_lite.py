#!/usr/bin/env python3
"""One-click launcher for GARD Lite."""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)

from gard_lite_engine import run_server

if __name__ == "__main__":
    run_server(port=8780, open_browser=True)
