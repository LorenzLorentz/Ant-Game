from __future__ import annotations

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
from AI.common import BaseAgent

from SDK.actions import ActionBundle
from SDK.backend.state import BackendState


class RandomAgent(BaseAgent):
    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if len(bundles) <= 1:
            return bundles[0]
        # Bias very slightly away from the mandatory no-op so the agent explores the action space.
        pool = bundles[1:] if len(bundles) > 1 else bundles
        return self.rng.choice(pool)


class AI(RandomAgent):
    pass
