from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    return value


class TrainingLogger:
    def __init__(self, base_dir: str | Path, run_name: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        stem = run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_dir = self._allocate_run_dir(stem)
        self.text_log_path = self.run_dir / "train.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self._events_fp = self.events_path.open("a", encoding="utf-8")
        self.logger = self._build_logger()

    def _allocate_run_dir(self, stem: str) -> Path:
        candidate = self.base_dir / stem
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            candidate = self.base_dir / f"{stem}-{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _build_logger(self) -> logging.Logger:
        logger_name = f"agent_tradition.training.{self.run_dir.as_posix()}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._events_fp.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self._events_fp.flush()

    def log_config(self, payload: dict[str, Any]) -> None:
        self._write_json(self.config_path, payload)
        self.log_event("config", payload)
        self.logger.info("training run initialized run_dir=%s", self.run_dir)

    def log_batch_start(self, batch_index: int, total_batches: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "batch_start",
            {
                "batch_index": batch_index,
                "total_batches": total_batches,
                **payload,
            },
        )
        self.logger.info(
            "batch-start batch=%s/%s episodes=%s iterations=%s max_depth=%s max_rounds=%s checkpoint=%s",
            batch_index + 1,
            total_batches,
            payload.get("episodes"),
            payload.get("search_iterations"),
            payload.get("max_depth"),
            payload.get("max_rounds"),
            payload.get("checkpoint_path"),
        )

    def log_episode_start(self, batch_index: int, episode_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "episode_start",
            {
                "batch_index": batch_index,
                "episode_index": episode_index,
                **payload,
            },
        )
        self.logger.info(
            "episode-start batch=%s episode=%s seed=%s max_rounds=%s",
            batch_index,
            episode_index,
            payload.get("seed"),
            payload.get("max_rounds"),
        )

    def log_episode_progress(self, batch_index: int, episode_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "episode_progress",
            {
                "batch_index": batch_index,
                "episode_index": episode_index,
                **payload,
            },
        )
        self.logger.info(
            "episode-progress batch=%s episode=%s round=%s/%s decision=%s actor=%s bundles=%s elapsed=%.1fs last_search=%.2fs avg_search=%.2fs eta<=%.1fs samples_p0=%s samples_p1=%s",
            batch_index,
            episode_index,
            payload.get("round_index"),
            payload.get("max_rounds"),
            payload.get("decision_count"),
            payload.get("actor"),
            payload.get("bundle_count"),
            float(payload.get("elapsed_s", 0.0)),
            float(payload.get("last_search_s", 0.0)),
            float(payload.get("avg_search_s", 0.0)),
            float(payload.get("eta_upper_bound_s", 0.0)),
            payload.get("samples_player_0"),
            payload.get("samples_player_1"),
        )

    def log_episode(self, batch_index: int, episode_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "episode",
            {
                "batch_index": batch_index,
                "episode_index": episode_index,
                **payload,
            },
        )
        self.logger.info(
            "episode batch=%s episode=%s rounds=%s winner=%s reward_p0=%.4f reward_p1=%.4f",
            batch_index,
            episode_index,
            payload.get("rounds"),
            payload.get("winner"),
            float(payload.get("reward_player_0", 0.0)),
            float(payload.get("reward_player_1", 0.0)),
        )

    def log_batch_metrics(self, batch_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "batch_metrics",
            {
                "batch_index": batch_index,
                **payload,
            },
        )
        self.logger.info(
            "batch=%s policy_loss=%.4f value_loss=%.4f entropy=%.4f eval_win_rate=%.4f selfplay_rounds=%.2f batch_elapsed=%.1fs samples=%s",
            batch_index,
            float(payload.get("policy_loss", 0.0)),
            float(payload.get("value_loss", 0.0)),
            float(payload.get("entropy", 0.0)),
            float(payload.get("eval_win_rate", 0.0)),
            float(payload.get("mean_episode_rounds", 0.0)),
            float(payload.get("batch_elapsed_s", 0.0)),
            payload.get("samples"),
        )

    def log_evaluation_start(self, batch_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "evaluation_start",
            {
                "batch_index": batch_index,
                **payload,
            },
        )
        self.logger.info(
            "eval-start batch=%s episodes=%s",
            batch_index,
            payload.get("eval_episodes"),
        )

    def log_evaluation_episode(self, batch_index: int, episode_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "evaluation_episode",
            {
                "batch_index": batch_index,
                "episode_index": episode_index,
                **payload,
            },
        )
        self.logger.info(
            "eval-episode batch=%s episode=%s trained_side=%s winner=%s rounds=%s elapsed=%.1fs running_win_rate=%.3f",
            batch_index,
            episode_index,
            payload.get("trained_side"),
            payload.get("winner"),
            payload.get("rounds"),
            float(payload.get("elapsed_s", 0.0)),
            float(payload.get("running_win_rate", 0.0)),
        )

    def log_checkpoint(self, batch_index: int, checkpoint_path: str | Path) -> None:
        payload = {
            "batch_index": batch_index,
            "checkpoint_path": str(checkpoint_path),
        }
        self.log_event("checkpoint", payload)
        self.logger.info("checkpoint batch=%s path=%s", batch_index, checkpoint_path)

    def log_summary(self, payload: dict[str, Any]) -> None:
        self._write_json(self.summary_path, payload)
        self.log_event("summary", payload)
        self.logger.info("training completed")

    def log_error(self, message: str) -> None:
        self.log_event("error", {"message": message})
        self.logger.error("%s", message)

    def close(self) -> None:
        self._events_fp.close()
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
