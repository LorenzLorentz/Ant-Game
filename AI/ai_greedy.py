from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_IMPL_DIR = Path(__file__).with_name("ai_greedy")


def _load_impl(module_name: str):
    qualified_name = f"_agent_tradition_ai_greedy_{module_name}"
    cached = sys.modules.get(qualified_name)
    if cached is not None:
        return cached
    module_path = _IMPL_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load greedy implementation module {module_name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


AI = _load_impl("ai").AI
GreedySession = _load_impl("runtime").GreedySession
_to_greedy_info = _load_impl("runtime")._to_greedy_info
_to_sdk_operation = _load_impl("runtime")._to_sdk_operation


__all__ = ["AI", "GreedySession", "_to_greedy_info", "_to_sdk_operation"]
