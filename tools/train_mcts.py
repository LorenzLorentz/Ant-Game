from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AI.ai_mcts import AI as MCTSAgent
from SDK.training import AntWarParallelEnv


@dataclass(slots=True)
class EpisodeSummary:
    seed: int
    rounds: int
    winner: int | None
    reward_player_0: float
    reward_player_1: float


class MCTSSelfPlayScaffold:
    """Reference self-play loop for the MCTS agent.

    Search behavior still lives in `AI/ai_mcts.py`. If you want to train a
    learnable policy or value head, the place to add parameter updates is
    `update_from_episodes()` below.
    """

    def __init__(self, episodes: int, iterations: int, max_depth: int, seed: int, max_rounds: int) -> None:
        self.episodes = episodes
        self.iterations = iterations
        self.max_depth = max_depth
        self.seed = seed
        self.max_rounds = max_rounds

    def build_agent(self, seed: int) -> MCTSAgent:
        return MCTSAgent(iterations=self.iterations, max_depth=self.max_depth, seed=seed)

    def collect_episode(self, seed: int) -> EpisodeSummary:
        env = AntWarParallelEnv(seed=seed)
        try:
            _, infos = env.reset(seed=seed)
            agents = [
                self.build_agent(seed * 2),
                self.build_agent(seed * 2 + 1),
            ]
            total_reward = [0.0, 0.0]
            rounds = 0
            while env.agents and rounds < self.max_rounds:
                actions = {}
                for player, agent_name in enumerate(env.possible_agents):
                    bundles = infos[agent_name]["bundles"]
                    actions[agent_name] = agents[player].choose_action_index(env.state, player, bundles=bundles)
                _, rewards, terminations, truncations, infos = env.step(actions)
                total_reward[0] += rewards["player_0"]
                total_reward[1] += rewards["player_1"]
                rounds += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
            return EpisodeSummary(
                seed=seed,
                rounds=rounds,
                winner=env.state.winner,
                reward_player_0=round(total_reward[0], 4),
                reward_player_1=round(total_reward[1], 4),
            )
        finally:
            env.close()

    def update_from_episodes(self, episodes: list[EpisodeSummary]) -> dict[str, object]:
        del episodes
        return {
            "status": "scaffold-only",
            "next_step": "Implement your optimizer or distillation update inside update_from_episodes().",
            "policy_logic": "Keep search and bundle ranking in AI/ai_mcts.py.",
            "backend_logic": "Do not reimplement rules here; use SDK.backend and SDK.training only.",
        }

    def run(self) -> dict[str, object]:
        episodes = [self.collect_episode(self.seed + index) for index in range(self.episodes)]
        winners = [episode.winner for episode in episodes]
        return {
            "episodes": self.episodes,
            "iterations": self.iterations,
            "max_depth": self.max_depth,
            "avg_rounds": sum(episode.rounds for episode in episodes) / max(len(episodes), 1),
            "player_0_wins": sum(1 for winner in winners if winner == 0),
            "player_1_wins": sum(1 for winner in winners if winner == 1),
            "draws": sum(1 for winner in winners if winner is None),
            "update_hint": self.update_from_episodes(episodes),
            "samples": [asdict(episode) for episode in episodes[: min(len(episodes), 3)]],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCTS self-play training scaffold.")
    parser.add_argument("--episodes", type=int, default=2, help="Number of self-play episodes to collect.")
    parser.add_argument("--iterations", type=int, default=24, help="MCTS iterations per decision.")
    parser.add_argument("--max-depth", type=int, default=3, help="Rollout depth for the reference MCTS agent.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for self-play collection.")
    parser.add_argument("--max-rounds", type=int, default=128, help="Hard cap for each scaffold episode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scaffold = MCTSSelfPlayScaffold(
        episodes=args.episodes,
        iterations=args.iterations,
        max_depth=args.max_depth,
        seed=args.seed,
        max_rounds=args.max_rounds,
    )
    print(json.dumps(scaffold.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
