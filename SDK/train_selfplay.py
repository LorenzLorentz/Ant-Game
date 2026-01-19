import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
except Exception:
    torch = None  # type: ignore

from SDK import env as make_env
from SDK.policies import PolicyConfig, SmallPolicy, obs_to_tensor, logits_to_action, make_optimizer



def train_legal(episodes: int, save_path: str, seed: int | None = None):
    """第一阶段：只奖励合法操作"""
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to run training.")
    env = make_env(render_mode=None)
    if seed is not None:
        np.random.seed(seed)
    cfg = PolicyConfig()
    policy = SmallPolicy(cfg)
    opt = make_optimizer(policy, cfg)
    ep_returns = []
    action_type_counter = np.zeros(9, dtype=int)
    for ep in tqdm.tqdm(range(episodes)):
        env.reset()
        logps = defaultdict(list)
        rewards = defaultdict(float)
        entropies = defaultdict(list)
        epsilon = max(0.7 * (1 - ep / (episodes * 0.9)), 0.01)  # 前90%线性衰减，最低0.01
        ban8 = ep < int(episodes * 0.5)  # 前50%训练完全禁止8
        while True:
            agent = env.agent_selection
            obs = env.observe(agent)
            x = torch.from_numpy(obs_to_tensor(obs)).float().unsqueeze(0)
            logits = policy(x)[0]
            # 计算熵
            type_logits = logits[:9]
            type_probs = torch.softmax(type_logits, dim=-1)
            entropy = -torch.sum(type_probs * torch.log(type_probs + 1e-8))
            # 激进探索：前半程完全禁止8，后半程epsilon-greedy
            if ban8:
                logits = logits.clone()
                logits[0] = -1e9  # type_logits[0]对应EndTurn/8
            elif np.random.rand() < epsilon:
                # 随机采样非8动作
                type_probs_np = type_probs.detach().cpu().numpy()
                type_probs_np[0] = 0
                type_probs_np = type_probs_np / type_probs_np.sum()
                atype = np.random.choice(np.arange(9), p=type_probs_np)
                logits = logits.clone()
                logits[:9] = -1000
                logits[atype] = 1000
            action, logp, is_direct_end = logits_to_action(logits, obs)
            logps[agent].append(logp)
            entropies[agent].append(entropy)
            # 统计动作类型分布
            action_type_counter[action[0]] += 1
            env.step(action)
            # 合法非8+2，非法-2，8为-10
            if is_direct_end:
                env.rewards[agent] = -10.0
            elif env.rewards[agent] < 0:
                env.rewards[agent] = -2.0
            else:
                env.rewards[agent] = 2.0
            if any(env.terminations.values()) or any(env.truncations.values()):
                for a in env.possible_agents:
                    rewards[a] += env.rewards[a]
                break
        loss = 0.0
        entropy_loss = 0.0
        for a in env.possible_agents:
            if len(logps[a]) == 0:
                continue
            R = rewards[a]
            lp = torch.stack(logps[a]).sum()
            ent = torch.stack(entropies[a]).mean()
            loss = loss - R * lp
            entropy_loss = entropy_loss + ent
        # 熵正则项鼓励探索（系数提升）
        loss = loss - 0.05 * entropy_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        ep_returns.append(sum(rewards.values()))
        if (ep + 1) % 10 == 0:
            print(f"[ep {ep+1}] avg LEGAL return(last10)={np.mean(ep_returns[-10:]):.3f}")
            print("Action type count (0=EndTurn/8):", action_type_counter)
            action_type_counter[:] = 0
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"[LEGAL] Saved policy to {save_path}")

def train_win(episodes: int, save_path: str, seed: int | None = None, legal_model_path: str | None = None):
    """第二阶段：加载合法性模型，继续训练胜负奖励"""
    if torch is None:
        raise RuntimeError("PyTorch not installed. Please install torch to run training.")
    env = make_env(render_mode=None)
    if seed is not None:
        np.random.seed(seed)
    cfg = PolicyConfig()
    policy = SmallPolicy(cfg)
    if legal_model_path is not None:
        policy.load_state_dict(torch.load(legal_model_path, map_location="cpu"))
        print(f"Loaded legal model from {legal_model_path}")
    opt = make_optimizer(policy, cfg)
    ep_returns = []
    for ep in tqdm.tqdm(range(episodes)):
        env.reset()
        logps = defaultdict(list)
        rewards = defaultdict(float)
        while True:
            agent = env.agent_selection
            obs = env.observe(agent)
            x = torch.from_numpy(obs_to_tensor(obs)).float().unsqueeze(0)
            logits = policy(x)[0]
            action, logp, is_direct_end = logits_to_action(logits, obs)
            logps[agent].append(logp)
            env.step(action)
            # 只用环境原有奖励（胜负）
            if is_direct_end:
                env.rewards[agent] = -1.0
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
            print(f"[ep {ep+1}] avg WIN return(last10)={np.mean(ep_returns[-10:]):.3f}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"[WIN] Saved policy to {save_path}")



def main(argv=None):
    p = argparse.ArgumentParser(description="Self-play training with PettingZoo env")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--save", type=str, default="AI/selfplay/model.pt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stage", type=str, choices=["legal", "win"], default="win", help="train stage: legal or win")
    p.add_argument("--legal_model", type=str, default="AI/selfplay/legal.pt", help="path to legal model for win stage")
    args = p.parse_args(argv)
    if args.stage == "legal":
        train_legal(args.episodes, args.save, args.seed)
    else:
        train_win(args.episodes, args.save, args.seed, args.legal_model)


if __name__ == "__main__":
    raise SystemExit(main())
