from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - direct script execution helper
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AI.AI_expert.ai import AI
from AI.main import main


if __name__ == "__main__":
    main(AI)
