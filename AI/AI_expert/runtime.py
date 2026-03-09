from __future__ import annotations

try:
    from common import MatchSession
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import MatchSession

try:
    from protocol import ProtocolIO
except ModuleNotFoundError as exc:
    if exc.name != "protocol":
        raise
    from AI.protocol import ProtocolIO

from SDK.constants import PHEROMONE_SCALE, OperationType as SDKOperationType
from SDK.model import Operation as SDKOperation
from SDK.runtime import MatchRuntime

try:
    from antwar.core import (
        MAP_SIZE,
        Ant,
        AntState,
        Base,
        BuildingType,
        GameInfo,
        Operation,
        SuperWeapon,
        SuperWeaponType,
        Tower,
        TowerType,
    )
except ModuleNotFoundError as exc:
    if exc.name != "antwar":
        raise
    from AI.AI_expert.antwar.core import (
        MAP_SIZE,
        Ant,
        AntState,
        Base,
        BuildingType,
        GameInfo,
        Operation,
        SuperWeapon,
        SuperWeaponType,
        Tower,
        TowerType,
    )


def _to_sdk_operation(operation: Operation) -> SDKOperation:
    return SDKOperation(SDKOperationType(int(operation.type)), int(operation.arg0), int(operation.arg1))


def _to_expert_info(state) -> GameInfo:
    info = GameInfo(state.seed)
    info.round = state.round_index
    info.coins = list(state.coins)
    info.old_count = list(state.old_count)
    info.die_count = list(state.die_count)
    info.next_ant_id = state.next_ant_id
    info.next_tower_id = state.next_tower_id
    info.super_weapon_cd = state.weapon_cooldowns.astype(int).tolist()

    info.bases = [
        Base(
            player=base.player,
            x=base.x,
            y=base.y,
            hp=base.hp,
            gen_speed_level=base.generation_level,
            ant_level=base.ant_level,
        )
        for base in state.bases
    ]

    info.building_tag = [[BuildingType.EMPTY for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
    for base in info.bases:
        info.building_tag[base.x][base.y] = BuildingType.BASE

    info.towers = []
    for tower in state.towers:
        info.towers.append(
            Tower(
                id=tower.tower_id,
                player=tower.player,
                x=tower.x,
                y=tower.y,
                type=TowerType(int(tower.tower_type)),
                cd=tower.display_cooldown(),
            )
        )
        info.building_tag[tower.x][tower.y] = BuildingType.TOWER

    info.ants = [
        Ant(
            id=ant.ant_id,
            player=ant.player,
            x=ant.x,
            y=ant.y,
            hp=ant.hp,
            level=ant.level,
            age=ant.age,
            state=AntState(int(ant.status)),
            evasion=2 if ant.evasion else 0,
            deflector=bool(ant.deflector),
            path=list(ant.path),
        )
        for ant in state.ants
    ]

    info.pheromone = [
        [
            [state.pheromone[player, x, y] / float(PHEROMONE_SCALE) for y in range(MAP_SIZE)]
            for x in range(MAP_SIZE)
        ]
        for player in range(2)
    ]

    info.super_weapons = []
    for effect in state.active_effects:
        weapon = SuperWeapon(
            type=SuperWeaponType(int(effect.weapon_type)),
            player=effect.player,
            x=effect.x,
            y=effect.y,
        )
        weapon.left_time = effect.remaining_turns
        info.super_weapons.append(weapon)
    return info


class ExpertSession(MatchSession):
    def __init__(self, agent, io: ProtocolIO | None = None) -> None:
        self.agent = agent
        self.io = io or ProtocolIO()
        player, seed = self.io.recv_init()
        self.runtime = MatchRuntime.create(player=player, seed=seed, prefer_native=False)

    @property
    def player(self) -> int:
        return self.runtime.player

    def perform_self_turn(self) -> None:
        operations = [_to_sdk_operation(operation) for operation in self.agent(self.player, _to_expert_info(self.runtime.state))]
        self.runtime.apply_self_operations(operations)
        self.io.send_operations(operations)

    def receive_opponent_turn(self) -> bool:
        try:
            opponent_operations = self.io.recv_operations()
        except Exception:
            return False
        self.runtime.apply_opponent_operations(opponent_operations)
        return True

    def sync_round(self) -> bool:
        round_state = self.io.recv_round_state()
        if round_state is None:
            return False
        self.runtime.finish_round(round_state)
        return True

