from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "SDK" / "train_mcts.py"


@dataclass(frozen=True, slots=True)
class SweepProfile:
    name: str
    description: str
    args: tuple[str, ...]


PROFILES: dict[str, SweepProfile] = {
    "balanced": SweepProfile(
        name="balanced",
        description="Baseline league training with stable promotion pressure.",
        args=(
            "--batches", "32",
            "--episodes", "16",
            "--search-iterations", "96",
            "--max-depth", "5",
            "--max-rounds", "192",
            "--opponent-pool-size", "6",
            "--selfplay-mirror-ratio", "0.35",
            "--selfplay-heuristic-ratio", "0.25",
            "--promotion-episodes", "8",
            "--promotion-win-rate", "0.56",
        ),
    ),
    "champion_hunter": SweepProfile(
        name="champion_hunter",
        description="Lower mirror ratio and stricter promotion to push anti-champion strength.",
        args=(
            "--batches", "28",
            "--episodes", "18",
            "--search-iterations", "96",
            "--max-depth", "5",
            "--max-rounds", "192",
            "--opponent-pool-size", "8",
            "--selfplay-mirror-ratio", "0.20",
            "--selfplay-heuristic-ratio", "0.20",
            "--promotion-episodes", "10",
            "--promotion-win-rate", "0.58",
        ),
    ),
    "deep_search": SweepProfile(
        name="deep_search",
        description="Fewer batches but heavier search, useful on strong GPUs for quality-first data.",
        args=(
            "--batches", "24",
            "--episodes", "14",
            "--search-iterations", "128",
            "--max-depth", "6",
            "--max-rounds", "192",
            "--opponent-pool-size", "6",
            "--selfplay-mirror-ratio", "0.30",
            "--selfplay-heuristic-ratio", "0.20",
            "--promotion-episodes", "8",
            "--promotion-win-rate", "0.56",
        ),
    ),
    "broad_league": SweepProfile(
        name="broad_league",
        description="Larger opponent pool and more episodes to improve robustness against historical drift.",
        args=(
            "--batches", "36",
            "--episodes", "20",
            "--search-iterations", "96",
            "--max-depth", "5",
            "--max-rounds", "192",
            "--opponent-pool-size", "10",
            "--selfplay-mirror-ratio", "0.25",
            "--selfplay-heuristic-ratio", "0.20",
            "--promotion-episodes", "10",
            "--promotion-win-rate", "0.57",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a matrix of Ant Game training experiments.")
    parser.add_argument(
        "--profiles",
        type=str,
        default="balanced,champion_hunter,deep_search",
        help="Comma-separated profile names.",
    )
    parser.add_argument("--runs-per-profile", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--run-dir", type=str, default="checkpoints/runs")
    parser.add_argument("--export-dir", type=str, default="AI/sweep_exports")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=(),
        help="Additional arguments forwarded to train_mcts.py after '--extra-args'.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _profile_order(raw: str) -> list[SweepProfile]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    missing = [name for name in requested if name not in PROFILES]
    if missing:
        raise SystemExit(f"Unknown profile(s): {', '.join(missing)}")
    return [PROFILES[name] for name in requested]


def _command_for_run(
    profile: SweepProfile,
    run_index: int,
    seed: int,
    run_dir: str,
    export_dir: str,
    python_executable: str,
    extra_args: tuple[str, ...],
    sweep_name: str,
) -> list[str]:
    export_path = Path(export_dir) / f"{profile.name}-run{run_index:02d}.npz"
    return [
        python_executable,
        str(TRAIN_SCRIPT),
        *profile.args,
        "--seed", str(seed),
        "--run-dir", run_dir,
        "--run-name", f"{sweep_name}-{profile.name}-run{run_index:02d}",
        "--export-agent-model", str(export_path),
        *extra_args,
    ]


def main() -> None:
    args = parse_args()
    profiles = _profile_order(args.profiles)
    sweep_name = f"sweep-{_timestamp()}"
    commands: list[list[str]] = []
    seed_cursor = args.base_seed
    for profile in profiles:
        for run_index in range(1, args.runs_per_profile + 1):
            commands.append(
                _command_for_run(
                    profile=profile,
                    run_index=run_index,
                    seed=seed_cursor,
                    run_dir=args.run_dir,
                    export_dir=args.export_dir,
                    python_executable=args.python,
                    extra_args=tuple(args.extra_args),
                    sweep_name=sweep_name,
                )
            )
            seed_cursor += 1

    print(f"Sweep name: {sweep_name}")
    for command in commands:
        print(" ".join(command))

    if args.dry_run:
        return

    for index, command in enumerate(commands, start=1):
        print(f"\n=== Running experiment {index}/{len(commands)} ===")
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
