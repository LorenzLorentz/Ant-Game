import argparse
import sys
from pathlib import Path
import numpy as np
import tqdm

# make sure parent path contains SDK
sys.path.insert(0, str(Path(__file__).parent.parent))

# the new driver does not require PyTorch; it relies solely on the pettingzoo env
from SDK import pettingzoo_env



import AI.ai_greedy as ai_greedy

def _ops_to_actions(ops):
    """Convert list of op lists from ai_greedy to vectors for env.step()."""
    actions = []
    for op in ops:
        vec = [0, 0, 0, 0, 0]
        if len(op) >= 1:
            vec[0] = int(op[0])
        if len(op) >= 3:
            vec[1] = int(op[1])
            vec[2] = int(op[2])
        if len(op) >= 4:
            vec[3] = int(op[3])
        if len(op) >= 5:
            vec[4] = int(op[4])
        actions.append(np.array(vec, dtype=np.int64))
    return actions


def selfplay(episodes: int, seed: int):
    env = pettingzoo_env.env(render_mode=None)
    returns = []
    if seed is None:
        seed = np.random.randint(0, 1000000)
    print(f"Starting self-play for {episodes} episodes with seed={seed}...")
    for ep in range(episodes):
        print(f"Episode {ep+1}/{episodes}")
        env.reset(seed=seed)
        print(f"Initial state: {env._last_state}")
        while True:
            if any(env.terminations.values()) or any(env.truncations.values()):
                returns.append(sum(env.rewards.values()))
                print(f"Episode {ep+1} ended with returns: {env.rewards}")
                break
            agent = env.agent_selection
            seat = 0 if agent == "player_0" else 1
            state = env._last_state.get("round_state", env._last_state) or {}
            ops = ai_greedy.policy(state, seat)
            for action in _ops_to_actions(ops):
                env.step(action)
                if any(env.terminations.values()) or any(env.truncations.values()):
                    break
            # end turn after issuing ops (or immediately if none)
            env.step(np.array([0, 0, 0, 0, 0], dtype=np.int64))
        if (ep + 1) % 10 == 0:
            avg = np.mean(returns[-10:])
            print(f"[ep {ep+1}] last10 avg return={avg:.3f}")
    return returns

def train_selfplay(episodes: int, save_path: str, seed: int):
    # delegate to the greedy selfplay driver; parameters kept for compatibility
    return selfplay(episodes, seed)


def main(argv=None):
    p = argparse.ArgumentParser(description="Greedy-policy self-play with pettingzoo AntGame env")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)
    selfplay(args.episodes, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())

