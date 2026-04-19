from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from SDK.utils.constants import (
    ANT_AGE_LIMIT,
    ANT_GENERATION_CYCLE,
    ANT_MAX_HP,
    AntBehavior,
    HIGHLAND_CELLS,
    MAP_PROPERTY,
    MAP_SIZE,
    MAX_ACTIONS,
    MAX_ROUND,
    PHEROMONE_SCALE,
    PLAYER_BASES,
    Terrain,
    SuperWeaponType,
    SUPER_WEAPON_STATS,
)
from SDK.utils.geometry import hex_distance
from SDK.backend.state import BackendState
from SDK.utils.turns import DecisionContext


@dataclass(slots=True)
class StateFeatures:
    values: np.ndarray
    named: dict[str, float]


class FeatureExtractor:
    def __init__(self, max_actions: int = MAX_ACTIONS) -> None:
        self.max_actions = max_actions

    @staticmethod
    def _world_pos(x: int, y: int) -> tuple[float, float]:
        return x + 0.5 * (y & 1), y * math.sqrt(3) / 2.0

    @staticmethod
    def _angle_delta(angle: float, target: float) -> float:
        return (angle - target + math.pi) % (2 * math.pi) - math.pi

    def _base_arc_coverage(self, state: BackendState, player: int) -> float:
        my_towers = state.towers_of(player)
        if not my_towers:
            return 0.0
        base_x, base_y = PLAYER_BASES[player]
        enemy_x, enemy_y = PLAYER_BASES[1 - player]
        world_base_x, world_base_y = self._world_pos(base_x, base_y)
        world_enemy_x, world_enemy_y = self._world_pos(enemy_x, enemy_y)
        forward_angle = math.atan2(world_enemy_y - world_base_y, world_enemy_x - world_base_x)
        target_offsets = (-30.0, 0.0, 30.0)
        tolerance = math.radians(20.0)
        covered = {target: False for target in target_offsets}
        for tower in my_towers:
            tower_world_x, tower_world_y = self._world_pos(tower.x, tower.y)
            angle = math.atan2(tower_world_y - world_base_y, tower_world_x - world_base_x)
            if abs(self._angle_delta(angle, forward_angle)) > math.pi / 2:
                continue
            for target in target_offsets:
                target_angle = forward_angle + math.radians(target)
                if abs(self._angle_delta(angle, target_angle)) <= tolerance:
                    covered[target] = True
        return float(sum(1 for hit in covered.values() if hit) / len(covered))

    def _tower_spacing_score(self, state: BackendState, player: int) -> float:
        towers = state.towers_of(player)
        if len(towers) <= 1:
            return 0.0
        penalty = 0.0
        distant_pair = False
        for index, tower in enumerate(towers[:-1]):
            for other in towers[index + 1 :]:
                gap = hex_distance(tower.x, tower.y, other.x, other.y)
                if gap <= 3:
                    penalty += 5.0
                elif gap <= 6:
                    penalty += 2.0
                else:
                    distant_pair = True
        if len(towers) >= 3 and not distant_pair:
            penalty += 20.0
        return float(-penalty / math.sqrt(max(len(towers), 1)))

    def summarize(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
    ) -> StateFeatures:
        del context
        enemy = 1 - player
        my_towers = state.towers_of(player)
        enemy_towers = state.towers_of(enemy)
        my_ants = state.ants_of(player)
        enemy_ants = state.ants_of(enemy)
        my_base_x, my_base_y = PLAYER_BASES[player]
        enemy_base_x, enemy_base_y = PLAYER_BASES[enemy]

        my_front = min((hex_distance(ant.x, ant.y, enemy_base_x, enemy_base_y) for ant in my_ants), default=32)
        enemy_front = min((hex_distance(ant.x, ant.y, my_base_x, my_base_y) for ant in enemy_ants), default=32)
        enemy_progress = np.mean(
            [float(ANT_AGE_LIMIT) - ant.age - 1.5 * hex_distance(ant.x, ant.y, my_base_x, my_base_y) for ant in enemy_ants],
            dtype=np.float32,
        ) if enemy_ants else 0.0
        my_progress = np.mean(
            [float(ANT_AGE_LIMIT) - ant.age - 1.5 * hex_distance(ant.x, ant.y, enemy_base_x, enemy_base_y) for ant in my_ants],
            dtype=np.float32,
        ) if my_ants else 0.0
        tower_level_sum = sum(tower.level for tower in my_towers)
        enemy_tower_level_sum = sum(tower.level for tower in enemy_towers)
        safe_coin = max(state.coins[player] - state.safe_coin_threshold(player), 0)
        hp_delta = float(state.bases[player].hp - state.bases[enemy].hp)
        kill_delta = float(state.die_count[enemy] - state.die_count[player])
        old_delta = float(state.old_count[enemy] - state.old_count[player])
        tower_spread = state.tower_spread_score(player)
        slot_fill_ratio = len(my_towers) / max(len(HIGHLAND_CELLS[player]), 1)
        hostile_distance = float(state.nearest_ant_distance(player))
        base_cycle = ANT_GENERATION_CYCLE[0]
        best_cycle = min(ANT_GENERATION_CYCLE)
        cycle_span = max(base_cycle - best_cycle, 1e-6)
        generation_value = (base_cycle - ANT_GENERATION_CYCLE[state.bases[player].generation_level]) / cycle_span
        base_ant_hp = ANT_MAX_HP[0]
        best_ant_hp = ANT_MAX_HP[-1]
        ant_hp_span = max(best_ant_hp - base_ant_hp, 1)
        ant_value = (ANT_MAX_HP[state.bases[player].ant_level] - base_ant_hp) / ant_hp_span
        base_arc_coverage = self._base_arc_coverage(state, player)
        tower_spacing = self._tower_spacing_score(state, player)

        named = {
            "round_ratio": state.round_index / MAX_ROUND,
            "hp_delta": hp_delta,
            "coin_ratio": state.coins[player] / max(state.coins[enemy], 1),
            "safe_coin": float(safe_coin),
            "frontline_advantage": float(enemy_front - my_front),
            "enemy_front_distance": float(enemy_front),
            "my_front_distance": float(my_front),
            "enemy_progress": float(enemy_progress),
            "my_progress": float(my_progress),
            "tower_count": float(len(my_towers)),
            "enemy_tower_count": float(len(enemy_towers)),
            "tower_level_sum": float(tower_level_sum),
            "enemy_tower_level_sum": float(enemy_tower_level_sum),
            "kill_delta": kill_delta,
            "old_delta": old_delta,
            "tower_spread": float(tower_spread),
            "slot_fill_ratio": float(slot_fill_ratio),
            "generation_level": float(generation_value),
            "ant_level": float(ant_value),
            "hostile_distance": hostile_distance,
            "base_arc_coverage": float(base_arc_coverage),
            "tower_spacing": float(tower_spacing),
        }
        values = np.array(list(named.values()), dtype=np.float32)
        return StateFeatures(values=values, named=named)

    def encode_board(self, state: BackendState, player: int) -> np.ndarray:
        enemy = 1 - player
        board = np.zeros((28, MAP_SIZE, MAP_SIZE), dtype=np.float32)
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                terrain = MAP_PROPERTY[x][y]
                if terrain == Terrain.PATH:
                    board[0, x, y] = 1.0
                elif terrain == Terrain.PLAYER0_HIGHLAND:
                    board[1, x, y] = 1.0 if player == 0 else 0.5
                elif terrain == Terrain.PLAYER1_HIGHLAND:
                    board[2, x, y] = 1.0 if player == 1 else 0.5
                elif terrain == Terrain.BARRIER:
                    board[3, x, y] = 1.0
                board[14, x, y] = state.pheromone[player, x, y] / float(12 * PHEROMONE_SCALE)
                board[15, x, y] = state.pheromone[enemy, x, y] / float(12 * PHEROMONE_SCALE)
        for tower in state.towers:
            channel = 4 if tower.player == player else 5
            board[channel, tower.x, tower.y] = 1.0
            board[6, tower.x, tower.y] = tower.level / 2.0
            board[7, tower.x, tower.y] = tower.attack_range / 6.0
            board[8, tower.x, tower.y] = tower.display_cooldown() / 6.0
            board[9, tower.x, tower.y] = tower.damage / 50.0
        for ant in state.ants:
            channel = 10 if ant.player == player else 11
            board[channel, ant.x, ant.y] += ant.hp / max(ant.max_hp, 1)
            board[12, ant.x, ant.y] += ant.level / 2.0
            board[13, ant.x, ant.y] += ant.age / float(ANT_AGE_LIMIT)
            if ant.frozen:
                board[16, ant.x, ant.y] += 1.0
            if ant.behavior == AntBehavior.RANDOM:
                board[17, ant.x, ant.y] += 1.0
            elif ant.behavior == AntBehavior.BEWITCHED:
                board[18, ant.x, ant.y] += 1.0
            elif ant.behavior == AntBehavior.CONTROL_FREE:
                board[19, ant.x, ant.y] += 1.0
        for effect in state.active_effects:
            base_channel = {
                SuperWeaponType.LIGHTNING_STORM: 20,
                SuperWeaponType.EMP_BLASTER: 22,
                SuperWeaponType.DEFLECTOR: 24,
                SuperWeaponType.EMERGENCY_EVASION: 26,
            }[effect.weapon_type]
            channel = base_channel if effect.player == player else base_channel + 1
            radius = SUPER_WEAPON_STATS[effect.weapon_type].attack_range
            strength = effect.remaining_turns / max(SUPER_WEAPON_STATS[effect.weapon_type].duration, 1)
            for x in range(MAP_SIZE):
                for y in range(MAP_SIZE):
                    if MAP_PROPERTY[x][y] == Terrain.VOID:
                        continue
                    if effect.in_range(x, y):
                        board[channel, x, y] = max(board[channel, x, y], strength)
        return board

    def encode_stats(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
    ) -> np.ndarray:
        if context is None:
            context = DecisionContext.for_player(player)
        enemy = 1 - player
        features = self.summarize(state, player, context=context).values
        extras = np.array(
            [
                state.coins[player] / 300.0,
                state.coins[enemy] / 300.0,
                state.bases[player].hp / 50.0,
                state.bases[enemy].hp / 50.0,
                state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM]
                / SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.EMP_BLASTER]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.DEFLECTOR]
                / SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cooldown,
                state.weapon_cooldowns[player, SuperWeaponType.EMERGENCY_EVASION]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.LIGHTNING_STORM]
                / SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.EMP_BLASTER]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.DEFLECTOR]
                / SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cooldown,
                state.weapon_cooldowns[enemy, SuperWeaponType.EMERGENCY_EVASION]
                / SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cooldown,
                state.super_weapon_usage[player] / 16.0,
                state.super_weapon_usage[enemy] / 16.0,
                float(player == 0),
                float(player == 1),
                float(context.to_play == 0),
                float(context.to_play == 1),
                float(context.settles_after_action),
                float(context.opponent_already_acted),
            ],
            dtype=np.float32,
        )
        return np.concatenate([features, extras], dtype=np.float32)

    def encode_observation(
        self,
        state: BackendState,
        player: int,
        action_mask: np.ndarray,
        context: DecisionContext | None = None,
    ) -> dict[str, np.ndarray]:
        if context is None:
            context = DecisionContext.for_player(player)
        return {
            "board": self.encode_board(state, player),
            "stats": self.encode_stats(state, player, context=context),
            "action_mask": action_mask.astype(np.int8, copy=False),
        }

    def flatten_observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        board = observation["board"].reshape(-1)
        stats = observation["stats"].reshape(-1)
        return np.concatenate([board, stats], dtype=np.float32)

    def evaluate(
        self,
        state: BackendState,
        player: int,
        context: DecisionContext | None = None,
    ) -> float:
        summary = self.summarize(state, player, context=context).named
        value = 0.0
        value += summary["hp_delta"] * 15.0
        value += summary["coin_ratio"] * 2.5
        value += summary["safe_coin"] * 0.05
        value += summary["frontline_advantage"] * 1.8
        value += summary["my_progress"] * 0.7
        value -= summary["enemy_progress"] * 0.9
        value += summary["kill_delta"] * 3.0
        value -= summary["old_delta"] * 1.5
        value += summary["tower_level_sum"] * 5.0
        value -= summary["enemy_tower_level_sum"] * 3.0
        value += summary["tower_spread"]
        value += summary["base_arc_coverage"] * 10.0
        value += summary["tower_spacing"] * 0.8
        value += summary["hostile_distance"] * 0.4
        value += summary["generation_level"] * 6.0
        value += summary["ant_level"] * 8.0
        if state.terminal:
            if state.winner == player:
                value += 10000.0
            elif state.winner == 1 - player:
                value -= 10000.0
        return value
