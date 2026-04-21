from __future__ import annotations
from SDK.utils.geometry import hex_distance
from SDK.utils.constants import PLAYER_BASES

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.utils.actions import ActionBundle
from SDK.backend import BackendState

class ExampleAgent(BaseAgent):
    """
    自定义的启发式 AI：发育与爆兵流
    """

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        try:
            from SDK.backend.model import Operation
            from SDK.utils.constants import OperationType
            from SDK.utils.actions import ActionBundle
            
            bundles = bundles or self.list_bundles(state, player)
            if not bundles:
                return ActionBundle(name="hold")
            if len(bundles) <= 1:
                return bundles[0]

            # ==========================================
            # 📡 战术雷达：动态评估我方半场的安全度
            # ==========================================
            enemy_ants = state.ants_of(1 - player)
            enemy_towers = state.towers_of(1 - player)
            base_x, base_y = PLAYER_BASES[player]
            
            danger_level = 0
            for ant in enemy_ants:
                if hex_distance(base_x, base_y, ant.x, ant.y) <= 10:
                    danger_level += 1
            for tower in enemy_towers:
                # 敌方如果在我们家附近建塔，危险度极高！
                if hex_distance(base_x, base_y, tower.x, tower.y) <= 12:
                    danger_level += 3

            # 【动态经济流】：只有在半场绝对安全时，才去贪科技！
            if state.round_index < 180 and danger_level == 0:
                op_gen = Operation(OperationType.UPGRADE_GENERATION_SPEED)
                if state.can_apply_operation(player, op_gen):
                    return ActionBundle("rush-gen", (op_gen,), 999.0, ("base",))
                
                op_ant = Operation(OperationType.UPGRADE_GENERATED_ANT)
                if state.can_apply_operation(player, op_ant):
                    return ActionBundle("rush-ant", (op_ant,), 999.0, ("base",))

            # 【紧急防守】：如果兵临城下（距离<=4），疯狂找AOE、冰冻塔和超级武器
            if danger_level > 0:
                my_base_dist = min([hex_distance(base_x, base_y, ant.x, ant.y) for ant in enemy_ants], default=999)
                if my_base_dist <= 4:
                    for bundle in bundles:
                        if "weapon" in bundle.tags or "tower:31" in bundle.tags or "tower:20" in bundle.tags:
                            return bundle

            # 默认回退：交给底层引擎去挑选出最优动作
            shortlist = bundles[1 : min(len(bundles), 15)]
            best = max(shortlist, key=lambda bundle: (bundle.score, -len(bundle.operations)), default=None)
            return best or bundles[0]

        except Exception as e:
            import sys, traceback
            print(f"[CRITICAL ERROR] AI survived a crash in Round {state.round_index}!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            from SDK.utils.actions import ActionBundle
            return ActionBundle(name="hold")
class AI(ExampleAgent):
    pass