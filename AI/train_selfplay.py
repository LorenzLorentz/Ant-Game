import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None  # type: ignore

from SDK-antgame import pettingzoo_env as ant_env  # type: ignore


def make_env(render_mode=None):
    return ant_env.env(render_mode=render_mode)


class AntPolicy(nn.Module):
    """
    针对 AntGame C++ PettingZoo 环境的简单策略网络：
    - 输入：raw_state_bytes (4096,) 归一化到 [0,1]
    - 输出：对 MultiDiscrete 各维度（type, x, y, id, args）的 logits
    """

    def __init__(self, action_nvec: np.ndarray):
        super().__init__()
        self.action_nvec = action_nvec.astype(int)
        in_dim = 4096
        hid = 512
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, hid),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hid, int(n)) for n in self.action_nvec]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: (B, 4096)
        返回：长度为 len(nvec) 的 logits 列表，每个 logits 形状为 (B, n_i)。
        """
        h = self.backbone(x)
        return [head(h) for head in self.heads]


def obs_to_tensor(obs) -> np.ndarray:
    """
    将环境观测转换为 1D numpy 向量：
    - 直接使用 raw_state_bytes/255.0 作为输入特征。
    """
    raw = obs["raw_state_bytes"].astype(np.float32) / 255.0
    return raw


def logits_to_action(
    logits_list: List[torch.Tensor], action_nvec: np.ndarray
) -> Tuple[np.ndarray, torch.Tensor, bool]:
    """
    从各维 logits 中采样动作，并返回：
    - action: np.ndarray[int]，长度与 nvec 相同
    - logp_sum: 所有维度 log-prob 之和，用于 REINFORCE
    - is_end_turn: 是否选择了 type==0（结束本轮）
    """
    actions = []
    logps = []
    for logits, n in zip(logits_list, action_nvec):
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()  # (B,)
        actions.append(a)
        logps.append(dist.log_prob(a))

    # 假定 batch=1
    act_vec = torch.stack(actions, dim=-1)[0]  # (D,)
    logp_vec = torch.stack(logps, dim=-1)[0]
    logp_sum = logp_vec.sum()
    action_np = act_vec.cpu().numpy().astype(np.int64)
    is_end = int(action_np[0]) == 0
    return action_np, logp_sum, is_end


def train_selfplay(episodes: int, save_path: str, seed: int | None = None):
    """
    使用 C++ AntGame PettingZoo 环境进行自博弈 REINFORCE 训练：
    - 每局为完整一盘游戏，终局时胜者奖励 +1，败者 -1。
    - 策略共享：两名玩家共用同一个策略网络。
    """
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to run training.")

    env = make_env(render_mode=None)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # 从环境中提取动作空间结构
    a_space = env.action_space(env.possible_agents[0])
    action_nvec = np.array(a_space.nvec, dtype=np.int64)

    policy = AntPolicy(action_nvec)
    opt = optim.Adam(policy.parameters(), lr=3e-4)

    ep_returns: List[float] = []

    for ep in range(episodes):
        env.reset()
        logps = defaultdict(list)    # agent -> [logp_sum...]
        rewards = defaultdict(float) # agent -> total_reward

        while True:
            agent = env.agent_selection
            obs = env.observe(agent)

            x_np = obs_to_tensor(obs)
            x = torch.from_numpy(x_np).float().unsqueeze(0)  # (1,4096)
            logits_list = policy(x)
            action, logp, is_end = logits_to_action(logits_list, action_nvec)
            logps[agent].append(logp)

            env.step(action)

            # 回合内 reward 目前只来自终局 +1/-1
            if any(env.terminations.values()) or any(env.truncations.values()):
                for a in env.possible_agents:
                    rewards[a] += env.rewards[a]
                break

        # REINFORCE 损失
        loss = torch.tensor(0.0)
        for a in env.possible_agents:
            if len(logps[a]) == 0:
                continue
            R = rewards[a]
            lp = torch.stack(logps[a]).sum()
            loss = loss - R * lp

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_ret = sum(rewards.values())
        ep_returns.append(total_ret)
        if (ep + 1) % 10 == 0:
            print(f"[ep {ep+1}] avg return(last10)={np.mean(ep_returns[-10:]):.3f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"[TRAIN] Saved policy to {save_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Self-play training with C++ AntGame PettingZoo env")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--save", type=str, default="AI/antgame_selfplay/model.pt")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    train_selfplay(args.episodes, args.save, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())

