from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 9通道全景视野
class AntWarValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 19 * 19, 256)
        self.fc2 = nn.Linear(256, 1) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) 
        return x

from SDK.utils.constants import PLAYER_BASES

def state_to_tensor(state, player: int) -> torch.Tensor:
    tensor = torch.zeros((9, 19, 19), dtype=torch.float32)
    enemy = 1 - player
    for ant in state.ants_of(player):
        if 0 <= ant.x < 19 and 0 <= ant.y < 19: tensor[0, ant.x, ant.y] += ant.hp / 100.0  
    for ant in state.ants_of(enemy):
        if 0 <= ant.x < 19 and 0 <= ant.y < 19: tensor[1, ant.x, ant.y] += ant.hp / 100.0
    for tower in state.towers_of(player):
        if 0 <= tower.x < 19 and 0 <= tower.y < 19: tensor[2, tower.x, tower.y] = tower.level / 3.0 
    for tower in state.towers_of(enemy):
        if 0 <= tower.x < 19 and 0 <= tower.y < 19: tensor[3, tower.x, tower.y] = tower.level / 3.0

    my_base_x, my_base_y = PLAYER_BASES[player]
    en_base_x, en_base_y = PLAYER_BASES[enemy]
    if 0 <= my_base_x < 19 and 0 <= my_base_y < 19:
        tensor[4, my_base_x, my_base_y] = 1.0
        tensor[5, my_base_x, my_base_y] = state.bases[player].hp / 50.0
        tensor[6, my_base_x, my_base_y] = (state.bases[player].generation_level + state.bases[player].ant_level) / 6.0
    if 0 <= en_base_x < 19 and 0 <= en_base_y < 19:
        tensor[4, en_base_x, en_base_y] = -1.0
        tensor[5, en_base_x, en_base_y] = state.bases[enemy].hp / 50.0
        tensor[7, en_base_x, en_base_y] = (state.bases[enemy].generation_level + state.bases[enemy].ant_level) / 6.0
    
    tensor[8, :, :] = (state.coins[player] - state.coins[enemy]) / 200.0
    return tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from AI.ai_mcts import AI as MCTSAgent
from SDK.training import AntWarParallelEnv

@dataclass(slots=True)
class EpisodeSummary:
    seed: int
    rounds: int
    winner: int | None
    reward_player_0: float
    reward_player_1: float
    states: list[object] = field(default_factory=list) 

class MCTSSelfPlayScaffold:
    def __init__(self, episodes: int, iterations: int, max_depth: int, seed: int, max_rounds: int) -> None:
        self.episodes = episodes
        self.iterations = iterations
        self.max_depth = max_depth
        self.seed = seed
        self.max_rounds = max_rounds
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AntWarValueNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() 
        
        if os.path.exists("latest_model.pth"):
            try:
                self.model.load_state_dict(torch.load("latest_model.pth", map_location=self.device, weights_only=True))
            except RuntimeError:
                print("⚠️ 重置模型结构...")

    def build_agent(self, seed: int) -> MCTSAgent:
        return MCTSAgent(iterations=self.iterations, max_depth=self.max_depth, seed=seed)

    def collect_episode(self, seed: int) -> EpisodeSummary:
        env = AntWarParallelEnv(seed=seed)
        try:
            _, infos = env.reset(seed=seed)
            agents = [self.build_agent(seed * 2), self.build_agent(seed * 2 + 1)]
            total_reward = [0.0, 0.0]
            rounds = 0
            episode_states = [] 
            
            while env.agents and rounds < self.max_rounds:
                actions = {}
                for player, agent_name in enumerate(env.possible_agents):
                    bundles = infos[agent_name]["bundles"]
                    actions[agent_name] = agents[player].choose_action_index(env.state, player, bundles=bundles)
                
                episode_states.append(env.state.clone())
                _, rewards, terminations, truncations, infos = env.step(actions)
                total_reward[0] += rewards["player_0"]
                total_reward[1] += rewards["player_1"]
                rounds += 1
                if all(terminations.values()) or all(truncations.values()): break
                    
            return EpisodeSummary(seed=seed, rounds=rounds, winner=env.state.winner,
                                  reward_player_0=round(total_reward[0], 4), reward_player_1=round(total_reward[1], 4),
                                  states=episode_states)
        finally:
            env.close()

    def update_from_episodes(self, episodes) -> dict[str, object]:
        self.model.train()
        total_loss = 0.0
        valid_samples = 0
        
        for episode in episodes:
            if episode.winner == 0: target_value = 1.0  
            elif episode.winner == 1: target_value = -1.0 
            else:
                # 即使平局，也根据最终奖励强制给出胜率倾向
                r0 = getattr(episode, 'reward_player_0', 0)
                r1 = getattr(episode, 'reward_player_1', 0)
                if r0 > r1 + 50: target_value = 0.8  
                elif r1 > r0 + 50: target_value = -0.8
                else: target_value = 0.0
            
            for state_data in getattr(episode, 'states', []):
                if state_data is not None:
                    state_tensor = state_to_tensor(state_data, player=0).unsqueeze(0).to(self.device)
                    target_tensor = torch.tensor([[target_value]], dtype=torch.float32).to(self.device)
                    
                    prediction = self.model(state_tensor)
                    loss = self.criterion(prediction, target_tensor)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    valid_samples += 1
            
        avg_loss = total_loss / valid_samples if valid_samples > 0 else 0.0
        torch.save(self.model.state_dict(), "latest_model.pth")
        return {"status": "training_active", "valid_samples": valid_samples, "avg_loss": avg_loss}

    def run(self) -> dict[str, object]:
        episodes = [self.collect_episode(self.seed + index) for index in range(self.episodes)]
        winners = [episode.winner for episode in episodes]
        safe_samples = []
        for episode in episodes[:3]:
            ep_dict = asdict(episode)
            ep_dict.pop('states', None)
            safe_samples.append(ep_dict)

        return {"avg_rounds": sum(episode.rounds for episode in episodes) / max(len(episodes), 1),
                "player_0_wins": sum(1 for w in winners if w == 0),
                "player_1_wins": sum(1 for w in winners if w == 1),
                "draws": sum(1 for w in winners if w is None),
                "update_hint": self.update_from_episodes(episodes), "samples": safe_samples}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=512) # 真正的大后期！
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    scaffold = MCTSSelfPlayScaffold(args.episodes, args.iterations, args.max_depth, args.seed, args.max_rounds)
    print(json.dumps(scaffold.run(), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()