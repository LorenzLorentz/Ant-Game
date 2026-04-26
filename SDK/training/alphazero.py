from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np

from SDK.alphazero import (
    PolicyValueNet,
    PolicyValueNetConfig,
    PriorGuidedMCTS,
    SearchConfig,
    build_policy_value_net,
    infer_observation_dim,
)
from SDK.training.env import AntWarSequentialEnv
from SDK.training.logging_utils import TrainingLogger
from SDK.utils.actions import ActionCatalog
from SDK.utils.features import FeatureExtractor


@dataclass(slots=True)
class AlphaZeroTrainerConfig:
    batches: int = 1
    episodes: int = 4
    learning_rate: float = 1e-3
    value_weight: float = 1.0
    l2_weight: float = 1e-5
    search_iterations: int = 48
    max_depth: int = 4
    c_puct: float = 1.25
    root_action_limit: int = 16
    child_action_limit: int = 10
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    prior_mix: float = 0.7
    value_mix: float = 0.7
    value_scale: float = 350.0
    root_temperature: float = 1.0
    temperature_drop_round: int = 96
    seed: int = 0
    max_rounds: int = 128
    max_actions: int = 96
    hidden_dim: int = 128
    hidden_dim2: int = 64
    checkpoint_path: str = "checkpoints/ai_mcts_latest.npz"
    resume_from: str | None = None
    evaluation_episodes: int = 2
    promotion_episodes: int = 6
    promotion_win_rate: float = 0.55
    champion_path: str = "checkpoints/ai_mcts_champion.npz"
    progress_log_decisions: int = 8
    progress_log_seconds: float = 5.0
    opponent_pool_size: int = 6
    selfplay_mirror_ratio: float = 0.4
    selfplay_heuristic_ratio: float = 0.2


@dataclass(slots=True)
class SelfPlaySample:
    observation: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    bootstrap_value: float
    round_index: int


@dataclass(slots=True)
class SelfPlayBatch:
    observations: np.ndarray
    masks: np.ndarray
    policies: np.ndarray
    values: np.ndarray


@dataclass(slots=True)
class EpisodeSummary:
    seed: int
    rounds: int
    winner: int | None
    reward_player_0: float
    reward_player_1: float
    outcome_player_0: float
    outcome_player_1: float
    trained_side: int
    opponent_kind: str


@dataclass(slots=True)
class OpponentSpec:
    kind: str
    search: PriorGuidedMCTS


class AlphaZeroSelfPlayTrainer:
    def __init__(
        self,
        env_factory,
        config: AlphaZeroTrainerConfig | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.config = config or AlphaZeroTrainerConfig()
        self.logger = logger
        self.rng = random.Random(self.config.seed)
        self.feature_extractor = FeatureExtractor(max_actions=self.config.max_actions)
        self.action_catalog = ActionCatalog(max_actions=self.config.max_actions, feature_extractor=self.feature_extractor)
        self.model = self._build_or_resume_model()
        self.search = PriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=True),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.eval_search = PriorGuidedMCTS(
            model=self.model,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.heuristic_search = PriorGuidedMCTS(
            model=None,
            search_config=self._build_search_config(exploration=False),
            feature_extractor=self.feature_extractor,
            action_catalog=self.action_catalog,
        )
        self.champion_model = self._load_checkpoint_model(Path(self.config.champion_path))
        self.champion_search = (
            PriorGuidedMCTS(
                model=self.champion_model,
                search_config=self._build_search_config(exploration=False),
                feature_extractor=self.feature_extractor,
                action_catalog=self.action_catalog,
            )
            if self.champion_model is not None
            else None
        )
        self.opponent_pool: list[OpponentSpec] = []
        self.refresh_opponent_pool()

    def _build_search_config(self, exploration: bool) -> SearchConfig:
        return SearchConfig(
            iterations=self.config.search_iterations,
            max_depth=self.config.max_depth,
            c_puct=self.config.c_puct,
            root_action_limit=self.config.root_action_limit,
            child_action_limit=self.config.child_action_limit,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon if exploration else 0.0,
            prior_mix=self.config.prior_mix,
            value_mix=self.config.value_mix,
            value_scale=self.config.value_scale,
            seed=self.config.seed,
        )

    def _build_or_resume_model(self) -> PolicyValueNet:
        resume_path = Path(self.config.resume_from) if self.config.resume_from else None
        if resume_path is not None and resume_path.exists():
            model = PolicyValueNet.from_checkpoint(resume_path)
            expected_obs_dim = infer_observation_dim(self.feature_extractor, self.config.max_actions)
            if model.action_dim != self.config.max_actions:
                raise ValueError(
                    f"checkpoint action_dim={model.action_dim} does not match max_actions={self.config.max_actions}"
                )
            if model.obs_dim != expected_obs_dim:
                raise ValueError(
                    f"checkpoint obs_dim={model.obs_dim} does not match current feature obs_dim={expected_obs_dim}"
                )
            return model
        return build_policy_value_net(
            feature_extractor=self.feature_extractor,
            action_dim=self.config.max_actions,
            config=PolicyValueNetConfig(
                hidden_dim=self.config.hidden_dim,
                hidden_dim2=self.config.hidden_dim2,
                seed=self.config.seed,
            ),
        )

    def _load_checkpoint_model(self, path: Path) -> PolicyValueNet | None:
        try:
            model = PolicyValueNet.from_checkpoint(path)
        except (OSError, ValueError, KeyError):
            return None
        expected_obs_dim = infer_observation_dim(self.feature_extractor, self.config.max_actions)
        if model.action_dim != self.config.max_actions or model.obs_dim != expected_obs_dim:
            return None
        return model

    def _checkpoint_candidates(self) -> list[Path]:
        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint_dir = checkpoint_path.parent
        candidates = list(checkpoint_dir.glob("*.npz"))
        candidates = [path for path in candidates if path.exists()]
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates

    def refresh_opponent_pool(self) -> None:
        pool: list[OpponentSpec] = [OpponentSpec(kind="heuristic", search=self.heuristic_search)]
        if self.champion_search is not None:
            pool.append(OpponentSpec(kind="champion", search=self.champion_search))
        checkpoint_candidates = self._checkpoint_candidates()
        for path in checkpoint_candidates:
            if Path(path) == Path(self.config.checkpoint_path):
                continue
            if Path(path) == Path(self.config.champion_path):
                continue
            model = self._load_checkpoint_model(path)
            if model is None:
                continue
            pool.append(
                OpponentSpec(
                    kind=f"checkpoint:{path.stem}",
                    search=PriorGuidedMCTS(
                        model=model,
                        search_config=self._build_search_config(exploration=False),
                        feature_extractor=self.feature_extractor,
                        action_catalog=self.action_catalog,
                    ),
                )
            )
            if len(pool) - 1 >= self.config.opponent_pool_size:
                break
        self.opponent_pool = pool

    def _refresh_champion_search(self) -> None:
        self.champion_model = self._load_checkpoint_model(Path(self.config.champion_path))
        self.champion_search = (
            PriorGuidedMCTS(
                model=self.champion_model,
                search_config=self._build_search_config(exploration=False),
                feature_extractor=self.feature_extractor,
                action_catalog=self.action_catalog,
            )
            if self.champion_model is not None
            else None
        )

    def _sample_selfplay_matchup(self) -> tuple[int | None, OpponentSpec | None]:
        roll = self.rng.random()
        if roll < self.config.selfplay_mirror_ratio:
            return None, None
        trained_side = self.rng.randrange(2)
        non_heuristic = [spec for spec in self.opponent_pool if spec.kind != "heuristic"]
        if roll < self.config.selfplay_mirror_ratio + self.config.selfplay_heuristic_ratio or not non_heuristic:
            return trained_side, self.opponent_pool[0]
        return trained_side, self.rng.choice(non_heuristic)

    def _temperature_for_round(self, round_index: int) -> float:
        if round_index >= self.config.temperature_drop_round:
            return 1e-6
        return self.config.root_temperature

    def _should_log_progress(self, decision_count: int, now: float, last_log_time: float) -> bool:
        if decision_count <= 0:
            return False
        if decision_count == 1:
            return True
        decision_interval = self.config.progress_log_decisions
        if decision_interval > 0 and decision_count % decision_interval == 0:
            return True
        time_interval = self.config.progress_log_seconds
        if time_interval > 0.0 and now - last_log_time >= time_interval:
            return True
        return False

    def _value_target(self, env: AntWarSequentialEnv, player: int) -> float:
        if env.state.terminal:
            if env.state.winner is None:
                return 0.0
            return 1.0 if env.state.winner == player else -1.0
        raw = self.feature_extractor.evaluate(env.state, player, context=env.decision_context)
        return float(np.tanh(raw / self.config.value_scale))

    def _sample_value_target(
        self,
        sample: SelfPlaySample,
        terminal_value: float,
        final_round_index: int,
    ) -> float:
        remaining_rounds = max(final_round_index - sample.round_index, 0)
        horizon_discount = 0.997 ** remaining_rounds
        discounted_outcome = terminal_value * horizon_discount
        progress = min(max(sample.round_index / max(self.config.max_rounds, 1), 0.0), 1.0)
        bootstrap_mix = 0.35 * (1.0 - progress)
        outcome_mix = 1.0 - bootstrap_mix
        target = outcome_mix * discounted_outcome + bootstrap_mix * sample.bootstrap_value
        return float(max(min(target, 1.0), -1.0))

    def collect_episode(
        self,
        seed: int,
        batch_index: int | None = None,
        episode_index: int | None = None,
    ) -> tuple[SelfPlayBatch, EpisodeSummary]:
        trained_side, opponent_spec = self._sample_selfplay_matchup()
        opponent_kind = "mirror" if opponent_spec is None else opponent_spec.kind
        env = self.env_factory(seed=seed)
        try:
            env.reset(seed=seed)
            traces = {agent: [] for agent in env.possible_agents}
            total_reward = {agent: 0.0 for agent in env.possible_agents}
            decision_count = 0
            recent_search_times: list[float] = []
            episode_start = time.perf_counter()
            last_progress_time = episode_start
            if self.logger is not None and batch_index is not None and episode_index is not None:
                self.logger.log_episode_start(
                    batch_index=batch_index,
                    episode_index=episode_index,
                    payload={
                        "seed": seed,
                        "max_rounds": self.config.max_rounds,
                    },
                )
            for agent_name in env.agent_iter():
                current, reward, termination, truncation, info = env.last()
                total_reward[agent_name] += float(reward)
                if agent_name == "player_0" and env.state.round_index >= self.config.max_rounds:
                    total_reward["player_1"] += float(env.rewards.get("player_1", 0.0))
                    break
                if termination or truncation:
                    env.step(None)
                    continue

                player = env.player_index(agent_name)
                bundles = info["bundles"]
                search_start = time.perf_counter()
                is_trained_actor = trained_side is None or player == trained_side
                active_search = self.search if (is_trained_actor or opponent_spec is None) else opponent_spec.search
                temperature = self._temperature_for_round(env.state.round_index) if is_trained_actor else 1e-6
                add_root_noise = is_trained_actor and opponent_spec is None
                result = active_search.search(
                    env.state,
                    player,
                    bundles=bundles,
                    context=env.decision_context,
                    temperature=temperature,
                    add_root_noise=add_root_noise,
                )
                search_elapsed = time.perf_counter() - search_start
                recent_search_times.append(search_elapsed)
                if len(recent_search_times) > 16:
                    recent_search_times.pop(0)
                if is_trained_actor:
                    traces[agent_name].append(
                        SelfPlaySample(
                            observation=self.feature_extractor.flatten_observation(current),
                            mask=current["action_mask"].astype(np.float32),
                            policy=result.policy.copy(),
                            bootstrap_value=float(result.root_value),
                            round_index=env.state.round_index,
                        )
                    )
                decision_count += 1
                now = time.perf_counter()
                if (
                    self.logger is not None
                    and batch_index is not None
                    and episode_index is not None
                    and self._should_log_progress(decision_count, now, last_progress_time)
                ):
                    elapsed = now - episode_start
                    avg_search_s = sum(recent_search_times) / max(len(recent_search_times), 1)
                    avg_decision_s = elapsed / max(decision_count, 1)
                    max_decisions = max(self.config.max_rounds, 1) * 2
                    eta_upper_bound_s = max(max_decisions - decision_count, 0) * avg_decision_s
                    self.logger.log_episode_progress(
                        batch_index=batch_index,
                        episode_index=episode_index,
                        payload={
                            "round_index": env.state.round_index,
                            "max_rounds": self.config.max_rounds,
                            "decision_count": decision_count,
                            "actor": agent_name,
                            "bundle_count": len(bundles),
                            "elapsed_s": elapsed,
                            "last_search_s": search_elapsed,
                            "avg_search_s": avg_search_s,
                            "eta_upper_bound_s": eta_upper_bound_s,
                            "samples_player_0": len(traces["player_0"]),
                            "samples_player_1": len(traces["player_1"]),
                            "reward_player_0": round(total_reward["player_0"], 4),
                            "reward_player_1": round(total_reward["player_1"], 4),
                            "opponent_kind": opponent_kind,
                            "trained_side": trained_side,
                        },
                    )
                    last_progress_time = now
                env.step(result.action_index)

            player_targets = {
                "player_0": self._value_target(env, 0),
                "player_1": self._value_target(env, 1),
            }
            observation_rows = []
            mask_rows = []
            policy_rows = []
            value_rows = []
            for agent_name in env.possible_agents:
                for sample in traces[agent_name]:
                    observation_rows.append(sample.observation)
                    mask_rows.append(sample.mask)
                    policy_rows.append(sample.policy)
                    value_rows.append(
                        self._sample_value_target(
                            sample,
                            terminal_value=player_targets[agent_name],
                            final_round_index=env.state.round_index,
                        )
                    )

            batch = SelfPlayBatch(
                observations=np.asarray(observation_rows, dtype=np.float32),
                masks=np.asarray(mask_rows, dtype=np.float32),
                policies=np.asarray(policy_rows, dtype=np.float32),
                values=np.asarray(value_rows, dtype=np.float32),
            )
            summary = EpisodeSummary(
                seed=seed,
                rounds=env.state.round_index,
                winner=env.state.winner,
                reward_player_0=round(total_reward["player_0"], 4),
                reward_player_1=round(total_reward["player_1"], 4),
                outcome_player_0=round(player_targets["player_0"], 4),
                outcome_player_1=round(player_targets["player_1"], 4),
                trained_side=-1 if trained_side is None else trained_side,
                opponent_kind=opponent_kind,
            )
            return batch, summary
        finally:
            env.close()

    def _selfplay_metrics(self, summaries: list[EpisodeSummary]) -> dict[str, float]:
        if not summaries:
            return {
                "mean_episode_rounds": 0.0,
                "selfplay_draw_rate": 0.0,
                "selfplay_player_0_win_rate": 0.0,
                "selfplay_player_1_win_rate": 0.0,
                "mean_reward_player_0": 0.0,
                "mean_reward_player_1": 0.0,
            }
        episodes = float(len(summaries))
        player_0_wins = sum(1 for summary in summaries if summary.winner == 0)
        player_1_wins = sum(1 for summary in summaries if summary.winner == 1)
        draws = sum(1 for summary in summaries if summary.winner is None)
        return {
            "mean_episode_rounds": float(sum(summary.rounds for summary in summaries) / episodes),
            "selfplay_draw_rate": float(draws / episodes),
            "selfplay_player_0_win_rate": float(player_0_wins / episodes),
            "selfplay_player_1_win_rate": float(player_1_wins / episodes),
            "mean_reward_player_0": float(sum(summary.reward_player_0 for summary in summaries) / episodes),
            "mean_reward_player_1": float(sum(summary.reward_player_1 for summary in summaries) / episodes),
        }

    def _matchup_metrics(self, summaries: list[EpisodeSummary]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        grouped: dict[str, list[EpisodeSummary]] = {}
        side_grouped: dict[int, list[EpisodeSummary]] = {}
        for summary in summaries:
            grouped.setdefault(summary.opponent_kind, []).append(summary)
            side_grouped.setdefault(summary.trained_side, []).append(summary)

        for opponent_kind, items in grouped.items():
            prefix = opponent_kind.replace(":", "_").replace("-", "_")
            total = float(len(items))
            trained_wins = 0.0
            draws = 0.0
            mean_rounds = float(sum(item.rounds for item in items) / total)
            mean_reward = 0.0
            for item in items:
                if item.winner is None:
                    draws += 1.0
                elif item.trained_side == -1:
                    if item.winner == 0:
                        trained_wins += 0.5
                    elif item.winner == 1:
                        trained_wins += 0.5
                elif item.winner == item.trained_side:
                    trained_wins += 1.0
                if item.trained_side == 0:
                    mean_reward += item.reward_player_0
                elif item.trained_side == 1:
                    mean_reward += item.reward_player_1
                else:
                    mean_reward += 0.5 * (item.reward_player_0 + item.reward_player_1)
            metrics[f"matchup_{prefix}_episodes"] = total
            metrics[f"matchup_{prefix}_win_rate"] = trained_wins / total
            metrics[f"matchup_{prefix}_draw_rate"] = draws / total
            metrics[f"matchup_{prefix}_avg_rounds"] = mean_rounds
            metrics[f"matchup_{prefix}_mean_reward"] = mean_reward / total

        for trained_side, items in side_grouped.items():
            total = float(len(items))
            if total <= 0:
                continue
            side_key = "mirror" if trained_side == -1 else f"side_{trained_side}"
            wins = sum(1.0 for item in items if item.winner is not None and item.winner == trained_side)
            draws = sum(1.0 for item in items if item.winner is None)
            metrics[f"trained_{side_key}_episodes"] = total
            metrics[f"trained_{side_key}_win_rate"] = wins / total if trained_side != -1 else 0.0
            metrics[f"trained_{side_key}_draw_rate"] = draws / total
        return metrics

    def _merge_batches(self, batches: list[SelfPlayBatch]) -> SelfPlayBatch:
        return SelfPlayBatch(
            observations=np.concatenate([batch.observations for batch in batches], axis=0),
            masks=np.concatenate([batch.masks for batch in batches], axis=0),
            policies=np.concatenate([batch.policies for batch in batches], axis=0),
            values=np.concatenate([batch.values for batch in batches], axis=0),
        )

    def update_from_batch(self, batch: SelfPlayBatch) -> dict[str, float]:
        metrics = self.model.update(
            observations=batch.observations,
            masks=batch.masks,
            policy_targets=batch.policies,
            value_targets=batch.values,
            learning_rate=self.config.learning_rate,
            value_weight=self.config.value_weight,
            l2_weight=self.config.l2_weight,
        )
        metrics["samples"] = float(len(batch.values))
        metrics["mean_target_value"] = float(np.mean(batch.values))
        return metrics

    def _play_evaluation_episode(self, seed: int, trained_side: int) -> tuple[int | None, int]:
        env = self.env_factory(seed=seed)
        try:
            env.reset(seed=seed)
            for agent_name in env.agent_iter():
                current, _, termination, truncation, info = env.last()
                del current
                if agent_name == "player_0" and env.state.round_index >= self.config.max_rounds:
                    break
                if termination or truncation:
                    env.step(None)
                    continue
                player = env.player_index(agent_name)
                bundles = info["bundles"]
                if player == trained_side:
                    result = self.eval_search.search(
                        env.state,
                        player,
                        bundles=bundles,
                        context=env.decision_context,
                        temperature=1e-6,
                    )
                else:
                    result = self.heuristic_search.search(
                        env.state,
                        player,
                        bundles=bundles,
                        context=env.decision_context,
                        temperature=1e-6,
                    )
                env.step(result.action_index)
            return env.state.winner, env.state.round_index
        finally:
            env.close()

    def _play_search_match(
        self,
        seed: int,
        challenger_side: int,
        challenger_search: PriorGuidedMCTS,
        defender_search: PriorGuidedMCTS,
    ) -> tuple[int | None, int]:
        env = self.env_factory(seed=seed)
        try:
            env.reset(seed=seed)
            for agent_name in env.agent_iter():
                current, _, termination, truncation, info = env.last()
                del current
                if agent_name == "player_0" and env.state.round_index >= self.config.max_rounds:
                    break
                if termination or truncation:
                    env.step(None)
                    continue
                player = env.player_index(agent_name)
                bundles = info["bundles"]
                active_search = challenger_search if player == challenger_side else defender_search
                result = active_search.search(
                    env.state,
                    player,
                    bundles=bundles,
                    context=env.decision_context,
                    temperature=1e-6,
                )
                env.step(result.action_index)
            return env.state.winner, env.state.round_index
        finally:
            env.close()

    def evaluate_against_heuristic(
        self,
        num_episodes: int | None = None,
        batch_index: int | None = None,
    ) -> dict[str, float]:
        games = num_episodes if num_episodes is not None else self.config.evaluation_episodes
        if games <= 0:
            return {"eval_episodes": 0.0, "eval_win_rate": 0.0, "eval_draw_rate": 0.0}
        if self.logger is not None and batch_index is not None:
            self.logger.log_evaluation_start(batch_index=batch_index, payload={"eval_episodes": games})
        trained_wins = 0
        draws = 0
        total_rounds = 0
        for episode_index in range(games):
            trained_side = episode_index % 2
            evaluation_start = time.perf_counter()
            winner, rounds = self._play_evaluation_episode(self.config.seed + 10_000 + episode_index, trained_side)
            total_rounds += rounds
            if winner is None:
                draws += 1
            elif winner == trained_side:
                trained_wins += 1
            if self.logger is not None and batch_index is not None:
                self.logger.log_evaluation_episode(
                    batch_index=batch_index,
                    episode_index=episode_index,
                    payload={
                        "trained_side": trained_side,
                        "winner": winner,
                        "rounds": rounds,
                        "elapsed_s": time.perf_counter() - evaluation_start,
                        "running_win_rate": float(trained_wins / (episode_index + 1)),
                    },
                )
        return {
            "eval_episodes": float(games),
            "eval_win_rate": float(trained_wins / games),
            "eval_draw_rate": float(draws / games),
            "eval_avg_rounds": float(total_rounds / games),
        }

    def evaluate_against_champion(
        self,
        num_episodes: int | None = None,
        batch_index: int | None = None,
    ) -> dict[str, float]:
        games = num_episodes if num_episodes is not None else self.config.promotion_episodes
        if self.champion_search is None or games <= 0:
            return {
                "promotion_episodes": float(games),
                "promotion_win_rate": 1.0,
                "promotion_draw_rate": 0.0,
                "promotion_avg_rounds": 0.0,
                "promoted": 1.0,
            }
        challenger_wins = 0
        draws = 0
        total_rounds = 0
        for episode_index in range(games):
            challenger_side = episode_index % 2
            winner, rounds = self._play_search_match(
                seed=self.config.seed + 20_000 + episode_index,
                challenger_side=challenger_side,
                challenger_search=self.eval_search,
                defender_search=self.champion_search,
            )
            total_rounds += rounds
            if winner is None:
                draws += 1
            elif winner == challenger_side:
                challenger_wins += 1
            if self.logger is not None and batch_index is not None:
                self.logger.log_evaluation_episode(
                    batch_index=batch_index,
                    episode_index=episode_index,
                    payload={
                        "evaluation_kind": "promotion",
                        "trained_side": challenger_side,
                        "winner": winner,
                        "rounds": rounds,
                        "running_win_rate": float(challenger_wins / (episode_index + 1)),
                        "elapsed_s": 0.0,
                    },
                )
        win_rate = float(challenger_wins / games)
        promoted = 1.0 if win_rate >= self.config.promotion_win_rate else 0.0
        return {
            "promotion_episodes": float(games),
            "promotion_win_rate": win_rate,
            "promotion_draw_rate": float(draws / games),
            "promotion_avg_rounds": float(total_rounds / games),
            "promoted": promoted,
        }

    def save_checkpoint(self) -> str:
        self.model.save(self.config.checkpoint_path)
        return str(Path(self.config.checkpoint_path))

    def maybe_promote_champion(self, metrics: dict[str, float]) -> bool:
        champion_path = Path(self.config.champion_path)
        should_promote = self.champion_search is None or metrics.get("promoted", 0.0) >= 1.0
        if should_promote:
            champion_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(champion_path)
            self._refresh_champion_search()
            self.refresh_opponent_pool()
            return True
        self.refresh_opponent_pool()
        return False

    def train(self, num_batches: int | None = None) -> tuple[list[dict[str, float]], list[EpisodeSummary]]:
        updates = num_batches if num_batches is not None else self.config.batches
        history: list[dict[str, float]] = []
        samples: list[EpisodeSummary] = []
        for batch_index in range(updates):
            batch_start = time.perf_counter()
            if self.logger is not None:
                self.logger.log_batch_start(
                    batch_index=batch_index,
                    total_batches=updates,
                    payload={
                        "episodes": self.config.episodes,
                        "search_iterations": self.config.search_iterations,
                        "max_depth": self.config.max_depth,
                        "max_rounds": self.config.max_rounds,
                        "checkpoint_path": self.config.checkpoint_path,
                    },
                )
            episode_batches = []
            episode_summaries = []
            selfplay_start = time.perf_counter()
            for episode_offset in range(self.config.episodes):
                seed = self.config.seed + batch_index * 1_000 + episode_offset
                batch, summary = self.collect_episode(seed=seed, batch_index=batch_index, episode_index=episode_offset)
                episode_batches.append(batch)
                episode_summaries.append(summary)
                if self.logger is not None:
                    self.logger.log_episode(batch_index=batch_index, episode_index=episode_offset, payload=asdict(summary))
            selfplay_elapsed_s = time.perf_counter() - selfplay_start
            merged = self._merge_batches(episode_batches)
            update_start = time.perf_counter()
            metrics = self.update_from_batch(merged)
            metrics["update_elapsed_s"] = float(time.perf_counter() - update_start)
            metrics["batch"] = float(batch_index)
            metrics["episodes"] = float(self.config.episodes)
            metrics["checkpoint_saved"] = 1.0
            metrics["selfplay_elapsed_s"] = float(selfplay_elapsed_s)
            metrics.update(self._selfplay_metrics(episode_summaries))
            metrics.update(self._matchup_metrics(episode_summaries))
            checkpoint_path = self.save_checkpoint()
            evaluation_start = time.perf_counter()
            metrics.update(self.evaluate_against_heuristic(batch_index=batch_index))
            metrics.update(self.evaluate_against_champion(batch_index=batch_index))
            metrics["champion_promoted"] = 1.0 if self.maybe_promote_champion(metrics) else 0.0
            metrics["champion_path"] = self.config.champion_path
            metrics["evaluation_elapsed_s"] = float(time.perf_counter() - evaluation_start)
            metrics["batch_elapsed_s"] = float(time.perf_counter() - batch_start)
            history.append(metrics)
            samples.extend(episode_summaries)
            if self.logger is not None:
                self.logger.log_batch_metrics(batch_index=batch_index, payload=metrics)
                self.logger.log_checkpoint(batch_index=batch_index, checkpoint_path=checkpoint_path)
        return history, samples
