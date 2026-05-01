#!/usr/bin/env python3
"""Submit-ready entry point for the final World Cup pipeline."""
from pathlib import Path
import sys

FINAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINAL_DIR))

from src.pipeline import main


if __name__ == "__main__":
    main()
