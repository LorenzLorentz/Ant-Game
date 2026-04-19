from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AI.ai_example import AI as ExampleAgent
from SDK.backend import create_python_backend_state
from SDK.training import AntWarSequentialEnv
from SDK.utils.actions import ActionCatalog


class ExampleTrainerGuide:
    """Tiny guide script that shows the intended training integration points."""

    def __init__(self, seed: int, max_actions: int) -> None:
        self.seed = seed
        self.max_actions = max_actions

    def train_one_batch(self) -> dict[str, object]:
        state = create_python_backend_state(seed=self.seed)
        catalog = ActionCatalog(max_actions=self.max_actions)
        bundles = catalog.build(state, 0)
        agent = ExampleAgent(seed=self.seed, max_actions=self.max_actions)
        chosen_index = agent.choose_action_index(state, 0, bundles=bundles)

        env = AntWarSequentialEnv(seed=self.seed, max_actions=self.max_actions)
        try:
            observations, infos = env.reset(seed=self.seed)
            for agent in env.agent_iter(max_iter=2):
                _, _, termination, truncation, _ = env.last()
                env.step(None if termination or truncation else 0)
        finally:
            env.close()

        return {
            "backend_entrypoint": "SDK.backend",
            "training_entrypoint": "SDK.training",
            "agent_logic_file": "AI/ai_example.py",
            "agent_logic_hook": "AI.choose_bundle()",
            "trainer_logic_hook": "ExampleTrainerGuide.train_one_batch()",
            "initial_bundle_count": len(bundles),
            "chosen_action_index": chosen_index,
            "observation_keys": sorted(observations["player_0"].keys()),
            "reset_backend": infos["player_0"]["backend"],
            "note": "Use this script as a template. Keep rules in SDK/backend and write your algorithm in the trainer hook.",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show how to hook custom training code into the framework.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the example backend/environment.")
    parser.add_argument("--max-actions", type=int, default=32, help="Candidate action budget for the example.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guide = ExampleTrainerGuide(seed=args.seed, max_actions=args.max_actions)
    print(json.dumps(guide.train_one_batch(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
