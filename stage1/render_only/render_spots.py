"""Command-line entry point for the render-only deliverable."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
from spot_renderer import main  # noqa: E402


if __name__ == "__main__":
    main()
