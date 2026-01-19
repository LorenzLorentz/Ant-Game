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
    """Policy over action types and parameters.
    Outputs: type_logits (9), x_logits (row), y_logits (col), dir_logits (4), num_logits (5), gid_logits (10)
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        inp = row * col * 4 + 2 + 2 + 1
        self.net = nn.Sequential(
            nn.Linear(inp, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, 9 + row + col + 4 + 5 + 10),
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
    # Split logits
    type_logits = logits[:9]
    x_logits = logits[9:9+row]
    y_logits = logits[9+row:9+row+col]
    dir_logits = logits[9+row+col:9+row+col+4]
    num_logits = logits[9+row+col+4:9+row+col+4+5]
    gid_logits = logits[9+row+col+4+5:]

    # Sample type
    type_probs = torch.softmax(type_logits, dim=-1)
    type_m = torch.distributions.Categorical(type_probs)
    atype = int(type_m.sample().item())  # 0-8
    type_logp = type_m.log_prob(torch.tensor(atype))

    act = np.zeros(8, dtype=np.int64)
    if atype == 0:
        act[0] = 0  # EndTurn
        # 标记为直接结束回合
        return act, type_logp, True

    act[0] = atype  # 1-8

    # Sample common parameters
    x_probs = torch.softmax(x_logits, dim=-1)
    x_m = torch.distributions.Categorical(x_probs)
    x = int(x_m.sample().item())
    x_logp = x_m.log_prob(torch.tensor(x))

    y_probs = torch.softmax(y_logits, dim=-1)
    y_m = torch.distributions.Categorical(y_probs)
    y = int(y_m.sample().item())
    y_logp = y_m.log_prob(torch.tensor(y))

    dir_probs = torch.softmax(dir_logits, dim=-1)
    dir_m = torch.distributions.Categorical(dir_probs)
    d = int(dir_m.sample().item())
    dir_logp = dir_m.log_prob(torch.tensor(d))

    num_probs = torch.softmax(num_logits, dim=-1)
    num_m = torch.distributions.Categorical(num_probs)
    n = int(num_m.sample().item()) + 1  # 1-5
    num_logp = num_m.log_prob(torch.tensor(n - 1))

    gid_probs = torch.softmax(gid_logits, dim=-1)
    gid_m = torch.distributions.Categorical(gid_probs)
    gid = int(gid_m.sample().item())
    gid_logp = gid_m.log_prob(torch.tensor(gid))

    # 合法性检查辅助函数
    def has_main_general(obs):
        curp = int(obs["current_player"])
        gens = obs["generals_owner"]
        return np.any(gens == curp)

    # Construct action based on type
    if atype == 1:  # ArmyMove: [x, y, direction(1..4), num]
        if not has_main_general(obs):
            act[0] = 0
            return act, type_logp, True
        act[1] = x
        act[2] = y
        act[3] = d + 1  # 1-4
        act[4] = n
        logp = type_logp + x_logp + y_logp + dir_logp + num_logp
        return act, logp, False
    elif atype == 2:  # GeneralMove: [general_id, dest_x, dest_y]
        if not has_main_general(obs):
            act[0] = 0
            return act, type_logp, True
        act[1] = gid
        act[2] = x
        act[3] = y
        logp = type_logp + gid_logp + x_logp + y_logp
        return act, logp, False
    elif atype == 3:  # LevelUp: [general_id, upgrade_type 1..3]
        if not has_main_general(obs):
            act[0] = 0
            return act, type_logp, True
        act[1] = gid
        act[2] = (d % 3) + 1  # 1-3
        logp = type_logp + gid_logp + dir_logp
        return act, logp, False
    elif atype == 4:  # GeneralSkill: [general_id, skill_type, ...]
        if not has_main_general(obs):
            act[0] = 0
            return act, type_logp, True
        act[1] = gid
        act[2] = (d % 2) + 1  # 1-2 for now
        act[3] = x  # target x
        act[4] = y  # target y
        logp = type_logp + gid_logp + dir_logp + x_logp + y_logp
        return act, logp, False
    elif atype == 5:  # TechUpdate: [tech_type 0..?]
        act[1] = d  # 0-3
        logp = type_logp + dir_logp
        return act, logp, False
    elif atype == 6:  # SuperWeapon: [weapon_type, ...]
        act[1] = (d % 4) + 1  # 1-4
        act[2] = x
        act[3] = y
        logp = type_logp + dir_logp + x_logp + y_logp
        return act, logp, False
    elif atype == 7:  # CallGenerals: [params]
        act[1] = x
        act[2] = y
        logp = type_logp + x_logp + y_logp
        return act, logp, False
    elif atype == 8:  # GiveUp
        logp = type_logp
        return act, logp, True
    else:
        # Fallback to EndTurn
        act[0] = 0
        logp = type_logp
        return act, logp, True


def make_optimizer(policy: SmallPolicy, cfg: PolicyConfig):
    return optim.Adam(policy.parameters(), lr=cfg.lr)

