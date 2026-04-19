from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import struct
import subprocess

from SDK.backend.engine import GameState
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = REPO_ROOT / "game"
GAME_BIN = GAME_DIR / "output" / "main"


@lru_cache(maxsize=1)
def _ensure_game_binary() -> None:
    subprocess.run(["make"], cwd=GAME_DIR, check=True, capture_output=True, text=True)


def _packet(message: object) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _prefixed_text_packet(text: str) -> str:
    payload = text.encode("utf-8")
    return (struct.pack(">I", len(payload)) + payload).decode("latin1")


def _run_game(input_packets: bytes) -> subprocess.CompletedProcess[bytes]:
    _ensure_game_binary()
    return subprocess.run(
        [str(GAME_BIN)],
        cwd=GAME_DIR,
        input=input_packets,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _operations_text(operations: list[Operation]) -> str:
    if not operations:
        return "0\n"
    lines = [str(len(operations))]
    for operation in operations:
        lines.append(" ".join(str(token) for token in operation.to_protocol_tokens()))
    return "\n".join(lines) + "\n"


def _run_game_replay(
    *,
    replay_path: Path,
    seed: int,
    movement_policy: str,
    rounds0: list[list[Operation]],
    rounds1: list[list[Operation]],
    extra_config: dict[str, object] | None = None,
) -> list[dict]:
    config: dict[str, object] = {"random_seed": seed, "movement_policy": movement_policy}
    if extra_config:
        config.update(extra_config)
    packets = [
        _packet(
            {
                "player_list": [1, 1],
                "player_num": 2,
                "config": config,
                "replay": str(replay_path),
            }
        )
    ]
    round_count = max(len(rounds0), len(rounds1))
    for round_index in range(round_count):
        packets.append(
            _packet(
                {
                    "player": 0,
                    "content": _prefixed_text_packet(
                        _operations_text(rounds0[round_index] if round_index < len(rounds0) else [])
                    ),
                    "time": 0,
                }
            )
        )
        packets.append(
            _packet(
                {
                    "player": 1,
                    "content": _prefixed_text_packet(
                        _operations_text(rounds1[round_index] if round_index < len(rounds1) else [])
                    ),
                    "time": 0,
                }
            )
        )
    packets.append(
        _packet(
            {
                "player": -1,
                "content": json.dumps({"player": 0, "error": 0}),
                "time": 0,
            }
        )
    )

    completed = _run_game(b"".join(packets))
    stderr = completed.stderr.decode("utf-8", errors="replace")
    assert completed.returncode == 0
    assert "read from judger error" not in stderr
    return json.loads(replay_path.read_text())


def _operation_from_replay_json(raw: dict) -> Operation:
    op_type = OperationType(int(raw["type"]))
    if op_type in (
        OperationType.BUILD_TOWER,
        OperationType.USE_LIGHTNING_STORM,
        OperationType.USE_EMP_BLASTER,
        OperationType.USE_DEFLECTOR,
        OperationType.USE_EMERGENCY_EVASION,
    ):
        return Operation(op_type, int(raw["pos"]["x"]), int(raw["pos"]["y"]))
    if op_type == OperationType.UPGRADE_TOWER:
        return Operation(op_type, int(raw["id"]), int(raw["args"]))
    if op_type == OperationType.DOWNGRADE_TOWER:
        return Operation(op_type, int(raw["id"]))
    return Operation(op_type)


def _round_operations_from_replay(path: Path, *, player: int) -> tuple[int, list[list[Operation]]]:
    replay = json.loads(path.read_text())
    return int(replay[0]["seed"]), [
        [_operation_from_replay_json(raw) for raw in entry.get(f"op{player}", [])]
        for entry in replay
    ]


def _cpp_tower_hp_by_round(replay: list[dict], *, x: int, y: int, player: int) -> list[int | None]:
    towers_by_id: dict[int, dict] = {}
    hp_by_round: list[int | None] = []
    for entry in replay:
        for tower in entry.get("round_state", {}).get("towers", []):
            if tower["type"] == -1:
                towers_by_id.pop(int(tower["id"]), None)
                continue
            towers_by_id[int(tower["id"])] = tower
        hp = None
        for tower in towers_by_id.values():
            if (
                int(tower["player"]) == player
                and int(tower["pos"]["x"]) == x
                and int(tower["pos"]["y"]) == y
            ):
                hp = int(tower["hp"])
                break
        hp_by_round.append(hp)
    return hp_by_round


def _cpp_player_ant_snapshots(replay: list[dict], *, player: int) -> list[dict[int, tuple[int, int, int, int, int]]]:
    snapshots: list[dict[int, tuple[int, int, int, int, int]]] = []
    for entry in replay:
        ants = {
            int(ant["id"]): (
                int(ant["pos"]["x"]),
                int(ant["pos"]["y"]),
                int(ant["kind"]),
                int(ant["behavior"]),
                int(ant["hp"]),
            )
            for ant in entry.get("round_state", {}).get("ants", [])
            if int(ant["player"]) == player
        }
        snapshots.append(ants)
    return snapshots


def _python_round_snapshots(
    *,
    seed: int,
    movement_policy: str,
    rounds0: list[list[Operation]],
    rounds1: list[list[Operation]],
    tower_x: int,
    tower_y: int,
    tower_player: int,
    ant_player: int,
) -> tuple[list[int | None], list[dict[int, tuple[int, int, int, int, int]]]]:
    state = GameState.initial(seed=seed, movement_policy=movement_policy)
    round_count = max(len(rounds0), len(rounds1))
    tower_hp_by_round: list[int | None] = []
    ant_snapshots: list[dict[int, tuple[int, int, int, int, int]]] = []
    for round_index in range(round_count):
        state.resolve_turn(
            rounds0[round_index] if round_index < len(rounds0) else [],
            rounds1[round_index] if round_index < len(rounds1) else [],
        )
        tower = next(
            (
                item
                for item in state.towers
                if item.player == tower_player and item.x == tower_x and item.y == tower_y
            ),
            None,
        )
        tower_hp_by_round.append(None if tower is None else int(tower.hp))
        ant_snapshots.append(
            {
                ant.ant_id: (
                    int(ant.x),
                    int(ant.y),
                    int(ant.kind),
                    int(ant.behavior),
                    int(ant.hp),
                )
                for ant in state.ants
                if ant.player == ant_player
            }
        )
    return tower_hp_by_round, ant_snapshots


def test_cpp_game_accepts_null_random_seed(tmp_path: Path) -> None:
    replay_path = tmp_path / "null-seed-replay.json"
    init_packet = _packet(
        {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": None},
            "replay": str(replay_path),
        }
    )
    error_packet = _packet(
        {
            "player": -1,
            "content": json.dumps({"player": 0, "error": 0}),
            "time": 0,
        }
    )

    completed = _run_game(init_packet + error_packet)
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0
    assert "type_error" not in stderr
    assert replay_path.exists()


def test_cpp_game_accepts_movement_policy_toggle(tmp_path: Path) -> None:
    replay_path = tmp_path / "legacy-policy-replay.json"
    init_packet = _packet(
        {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": 5, "movement_policy": "legacy"},
            "replay": str(replay_path),
        }
    )
    error_packet = _packet(
        {
            "player": -1,
            "content": json.dumps({"player": 0, "error": 0}),
            "time": 0,
        }
    )

    completed = _run_game(init_packet + error_packet)
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0
    assert "error" not in stderr.lower()
    assert replay_path.exists()


def test_cpp_game_decodes_length_prefixed_ai_operations(tmp_path: Path) -> None:
    replay_path = tmp_path / "prefixed-ops-replay.json"
    init_packet = _packet(
        {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": 7},
            "replay": str(replay_path),
        }
    )
    round0_packet = _packet(
        {
            "player": 0,
            "content": _prefixed_text_packet("1\n11 6 9\n"),
            "time": 0,
        }
    )
    round1_packet = _packet(
        {
            "player": 1,
            "content": _prefixed_text_packet("0\n"),
            "time": 0,
        }
    )
    error_packet = _packet(
        {
            "player": -1,
            "content": json.dumps({"player": 0, "error": 0}),
            "time": 0,
        }
    )

    completed = _run_game(init_packet + round0_packet + round1_packet + error_packet)
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0
    assert "Undefined type" not in stderr
    assert "read from judger error" not in stderr

    replay = json.loads(replay_path.read_text())
    assert any(op["type"] == 11 for entry in replay for op in entry.get("op0", []))
    assert any(
        tower["pos"]["x"] == 6 and tower["pos"]["y"] == 9 and "hp" in tower
        for entry in replay
        for tower in entry.get("round_state", {}).get("towers", [])
    )
    assert any(
        ant.get("kind") == 0
        for entry in replay
        for ant in entry.get("round_state", {}).get("ants", [])
    )
    assert all("weaponCooldowns" in entry.get("round_state", {}) for entry in replay)
    assert all("activeEffects" in entry.get("round_state", {}) for entry in replay)
    assert all("pheromone" in entry.get("round_state", {}) for entry in replay)


def test_cpp_game_rule_illegal_remains_fatal_by_default(tmp_path: Path) -> None:
    replay = _run_game_replay(
        replay_path=tmp_path / "strict-rule-illegal-replay.json",
        seed=11,
        movement_policy="enhanced",
        rounds0=[[]],
        rounds1=[[Operation(OperationType.BUILD_TOWER, 6, 9)]],
    )

    assert any("IA" in str(entry.get("round_state", {}).get("message", "")) for entry in replay)
    assert any(
        "TowerBuild" in str(entry.get("round_state", {}).get("error", ""))
        for entry in replay
    )


def test_cpp_game_can_cold_handle_rule_illegal_when_enabled(tmp_path: Path) -> None:
    replay = _run_game_replay(
        replay_path=tmp_path / "soft-rule-illegal-replay.json",
        seed=11,
        movement_policy="enhanced",
        rounds0=[[]],
        rounds1=[[Operation(OperationType.BUILD_TOWER, 6, 9)]],
        extra_config={"cold_handle_rule_illegal": True},
    )

    assert replay[0]["round_state"]["winner"] == -1
    assert "IA" not in str(replay[0]["round_state"].get("message", ""))
    assert "ignored" in str(replay[0]["round_state"].get("error", ""))


def test_cpp_game_replays_official_sample2_without_ia_after_active_tower_pricing_fix(tmp_path: Path) -> None:
    sample_path = REPO_ROOT / "sample2.json"
    seed0, rounds0 = _round_operations_from_replay(sample_path, player=0)
    seed1, rounds1 = _round_operations_from_replay(sample_path, player=1)

    assert seed0 == seed1
    replay = _run_game_replay(
        replay_path=tmp_path / "sample2-regression-replay.json",
        seed=seed0,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
    )

    first_25_rounds = replay[:25]
    assert all("IA" not in str(entry.get("round_state", {}).get("message", "")) for entry in first_25_rounds)
    assert any(
        tower["player"] == 1 and tower["pos"]["x"] == 10 and tower["pos"]["y"] == 11 and tower["type"] == 0
        for tower in first_25_rounds[-1]["round_state"].get("towers", [])
    )


def test_cpp_game_worker_tower_pressure_matches_python_on_frontline_basic_tower(tmp_path: Path) -> None:
    rounds = 10
    rounds0 = [[] for _ in range(rounds)]
    rounds1 = [[Operation(OperationType.BUILD_TOWER, 12, 9)]] + [[] for _ in range(rounds - 1)]

    replay = _run_game_replay(
        replay_path=tmp_path / "worker-frontline-tower-replay.json",
        seed=62,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
    )
    cpp_tower_hp = _cpp_tower_hp_by_round(replay[:rounds], x=12, y=9, player=1)
    cpp_ants = _cpp_player_ant_snapshots(replay[:rounds], player=0)
    py_tower_hp, py_ants = _python_round_snapshots(
        seed=62,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
        tower_x=12,
        tower_y=9,
        tower_player=1,
        ant_player=0,
    )

    for round_index in (0, 1, 2, 3, 9):
        assert cpp_tower_hp[round_index] == py_tower_hp[round_index]
        assert cpp_ants[round_index][0] == py_ants[round_index][0]


def test_cpp_game_basic_tower_does_not_hit_distance_two_target(tmp_path: Path) -> None:
    rounds = 13
    rounds0 = [[] for _ in range(rounds)]
    rounds1 = [[Operation(OperationType.BUILD_TOWER, 12, 9)]] + [[] for _ in range(rounds - 1)]

    replay = _run_game_replay(
        replay_path=tmp_path / "basic-range-regression-replay.json",
        seed=7,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
    )
    cpp_ants = _cpp_player_ant_snapshots(replay[:rounds], player=0)

    x, y, _, _, hp = cpp_ants[11][0]
    assert (x, y, hp) == (11, 9, 20)

    x, y, _, _, hp = cpp_ants[12][0]
    assert (x, y, hp) == (11, 8, 15)


def test_cpp_game_combat_tower_attack_matches_python_on_frontline_basic_tower(tmp_path: Path) -> None:
    rounds = 18
    rounds0 = [[] for _ in range(rounds)]
    rounds1 = [[Operation(OperationType.BUILD_TOWER, 12, 9)]] + [[] for _ in range(rounds - 1)]

    replay = _run_game_replay(
        replay_path=tmp_path / "combat-frontline-tower-replay.json",
        seed=126,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
    )
    cpp_tower_hp = _cpp_tower_hp_by_round(replay[:rounds], x=12, y=9, player=1)
    cpp_ants = _cpp_player_ant_snapshots(replay[:rounds], player=0)
    py_tower_hp, py_ants = _python_round_snapshots(
        seed=126,
        movement_policy="enhanced",
        rounds0=rounds0,
        rounds1=rounds1,
        tower_x=12,
        tower_y=9,
        tower_player=1,
        ant_player=0,
    )

    for round_index in (14, 15, 16, 17):
        assert cpp_tower_hp[round_index] == py_tower_hp[round_index]
        assert cpp_ants[round_index][2] == py_ants[round_index][2]
