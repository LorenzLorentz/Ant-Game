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
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.utils.features import FeatureExtractor

__all__ = [
    "ActionBundle",
    "ActionCatalog",
    "AntWarParallelEnv",
    "AntWarSequentialEnv",
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
    if name not in {"AntWarParallelEnv", "AntWarSequentialEnv", "env"}:
        raise AttributeError(f"module 'SDK' has no attribute {name!r}")
    from SDK.training import AntWarParallelEnv, AntWarSequentialEnv, env

    globals().update(
        {
            "AntWarParallelEnv": AntWarParallelEnv,
            "AntWarSequentialEnv": AntWarSequentialEnv,
            "env": env,
        }
    )
    return globals()[name]
