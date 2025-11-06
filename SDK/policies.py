import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

from dataclasses import dataclass
from logic.constant import row, col


@dataclass
class PolicyConfig:
    lr: float = 1e-3
    hidden: int = 256


class SmallPolicy(nn.Module):  # type: ignore[misc]
    """Tiny policy over reduced action set for both agents.
    Actions: 0 EndTurn, 1..4 = move army from main-general tile U,D,L,R with num=1.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        inp = row * col * 4 + 2 + 2 + 1
        self.net = nn.Sequential(
            nn.Linear(inp, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 5),
        )

    def forward(self, x):
        return self.net(x)


def obs_to_tensor(obs) -> np.ndarray:
    owner = obs["board_owner"].astype(np.float32)
    army = np.log1p(obs["board_army"].astype(np.float32))
    terrain = obs["board_terrain"].astype(np.float32) / 2.0
    gens = (obs["generals_owner"].astype(np.float32) + 1.0) / 2.0
    coins = obs["coins"].astype(np.float32) / 100.0
    rest = obs["rest_moves"].astype(np.float32) / 5.0
    cur = np.array([obs["current_player"]], dtype=np.float32)
    feat = np.concatenate(
        [owner.reshape(-1), army.reshape(-1), terrain.reshape(-1), gens.reshape(-1), coins, rest, cur]
    )
    return feat


def logits_to_action(logits, obs):
    probs = torch.softmax(logits, dim=-1)
    m = torch.distributions.Categorical(probs)
    a = m.sample()
    logp = m.log_prob(a)
    action_id = int(a.item())
    act = np.zeros(8, dtype=np.int64)
    if action_id == 0:
        act[0] = 0
        return act, logp
    gens = obs["generals_owner"]
    curp = int(obs["current_player"])
    pos = np.argwhere(gens == curp)
    if len(pos) == 0:
        act[0] = 0
        return act, logp
    x, y = pos[0]
    act[0] = 1
    act[1] = x
    act[2] = y
    dir_map = {1: 0, 2: 1, 3: 2, 4: 3}
    act[3] = dir_map.get(action_id, 0)
    act[4] = 1
    return act, logp


def make_optimizer(policy: SmallPolicy, cfg: PolicyConfig):
    return optim.Adam(policy.parameters(), lr=cfg.lr)

