from __future__ import annotations

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.actions import ActionBundle
from SDK.backend import BackendState


class ExampleAgent(BaseAgent):
    """Minimal reference agent.

    Replace `choose_bundle()` with your own strategy logic. The backend already
    owns rule simulation and operation validation, so this file should only rank
    the candidate bundles you want to send.
    """

    def choose_bundle(self, state: BackendState, player: int, bundles: list[ActionBundle] | None = None) -> ActionBundle:
        bundles = bundles or self.list_bundles(state, player)
        if len(bundles) <= 1:
            return bundles[0]

        # Core agent logic belongs here: read `state`, inspect `bundles`, and
        # return the bundle you want to execute. This example just prefers the
        # strongest non-noop candidate among a short shortlist.
        shortlist = bundles[1 : min(len(bundles), 8)]
        best = max(shortlist, key=lambda bundle: (bundle.score, -len(bundle.operations)), default=None)
        return best or bundles[0]


class AI(ExampleAgent):
    pass
