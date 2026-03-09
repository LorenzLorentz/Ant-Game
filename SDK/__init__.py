from SDK.actions import ActionBundle, ActionCatalog
from SDK.backend import (
    BackendState,
    ForecastOperation,
    ForecastSimulator,
    ForecastState,
    GameState,
    MatchRuntime,
    PythonBackendState,
    build_forecast_state,
    create_python_backend_state,
)
from SDK.features import FeatureExtractor

__all__ = [
    "ActionBundle",
    "ActionCatalog",
    "AntWarParallelEnv",
    "BackendState",
    "FeatureExtractor",
    "ForecastOperation",
    "ForecastSimulator",
    "ForecastState",
    "GameState",
    "MatchRuntime",
    "PythonBackendState",
    "build_forecast_state",
    "create_python_backend_state",
    "env",
]


def __getattr__(name: str):
    if name not in {"AntWarParallelEnv", "env"}:
        raise AttributeError(f"module 'SDK' has no attribute {name!r}")
    from SDK.training import AntWarParallelEnv, env

    globals().update(
        {
            "AntWarParallelEnv": AntWarParallelEnv,
            "env": env,
        }
    )
    return globals()[name]
