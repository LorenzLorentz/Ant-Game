import argparse

try:
    import torch
except Exception:
    torch = None  # type: ignore

from SDK import env as make_env
from SDK.policies import PolicyConfig, SmallPolicy, obs_to_tensor, logits_to_action


def load_policy(path: str) -> SmallPolicy:
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to run evaluation.")
    policy = SmallPolicy(PolicyConfig())
    state = torch.load(path, map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()
    return policy


def run_episode(policy: SmallPolicy) -> int:
    env = make_env(render_mode=None)
    env.reset()
    while True:
        agent = env.agent_selection
        obs = env.observe(agent)
        import torch as T

        x = T.from_numpy(obs_to_tensor(obs)).float().unsqueeze(0)
        with T.no_grad():
            logits = policy(x)[0]
        action, _ = logits_to_action(logits, obs)
        env.step(action)
        if any(env.terminations.values()) or any(env.truncations.values()):
            if env.rewards["player_0"] > env.rewards["player_1"]:
                return 1
            elif env.rewards["player_0"] < env.rewards["player_1"]:
                return -1
            else:
                return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained self-play policy")
    p.add_argument("--model", type=str, default="AI/selfplay/model.pt")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args(argv)
    policy = load_policy(args.model)
    wins = losses = draws = 0
    for _ in range(args.episodes):
        r = run_episode(policy)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            draws += 1
    print(f"Results over {args.episodes}: wins={wins}, losses={losses}, draws={draws}")


if __name__ == "__main__":
    raise SystemExit(main())
