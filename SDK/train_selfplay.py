import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

from SDK import env as make_env
from SDK.policies import PolicyConfig, SmallPolicy, obs_to_tensor, logits_to_action, make_optimizer


def train(episodes: int, save_path: str, seed: int | None = None):
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to run training.")

    env = make_env(render_mode=None)
    if seed is not None:
        np.random.seed(seed)

    cfg = PolicyConfig()
    policy = SmallPolicy(cfg)
    opt = make_optimizer(policy, cfg)

    ep_returns = []
    for ep in range(episodes):
        env.reset()
        logps = defaultdict(list)
        rewards = defaultdict(float)

        while True:
            agent = env.agent_selection
            obs = env.observe(agent)
            x = torch.from_numpy(obs_to_tensor(obs)).float().unsqueeze(0)
            logits = policy(x)[0]
            action, logp = logits_to_action(logits, obs)
            logps[agent].append(logp)
            env.step(action)
            if any(env.terminations.values()) or any(env.truncations.values()):
                for a in env.possible_agents:
                    rewards[a] += env.rewards[a]
                break

        loss = 0.0
        for a in env.possible_agents:
            if len(logps[a]) == 0:
                continue
            R = rewards[a]
            lp = torch.stack(logps[a]).sum()
            loss = loss - R * lp

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_returns.append(sum(rewards.values()))
        if (ep + 1) % 10 == 0:
            print(f"[ep {ep+1}] avg return(last10)={np.mean(ep_returns[-10:]):.3f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"Saved policy to {save_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Self-play training with PettingZoo env")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--save", type=str, default="AI/selfplay/model.pt")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    train(args.episodes, args.save, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
