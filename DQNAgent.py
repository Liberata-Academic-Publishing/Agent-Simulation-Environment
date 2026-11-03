"""Deep Q-network agent for the single-review peer-review market.

Uses a nine-feature observation and a small ReLU MLP with experience replay
and a target network. Numpy-only (no PyTorch dependency). The stateless
tabular agent in ``QLearningAgent`` does not share this observation.

The action values are a dueling head: one stream for the state's value and one
for each action's advantage, so a review that pays nothing can sit below writing
without both actions having to store the whole return. Targets are Double DQN
over an n-step return. The online network picks the next action and the target
network scores it. A zero-share review adds nothing to that return.

Training is offline-ish: transitions are stored in a replay buffer and the
network is updated in mini-batches via ``train_dqn.py`` or online during a
simulation run when ``learning=True``.
"""

from __future__ import annotations

import math
import pickle
import random
from collections import deque

import numpy as np

from Agent import (
    Agent,
    CONTINUOUS_CLAIM,
    CONTINUOUS_RESEARCH,
    CONTINUOUS_RESEARCH_FINISH,
    CONTINUOUS_REVIEW,
    ownership_papers,
)
from Paper import MIN_REVIEW_EFFORT_THRESHOLD, Paper
from QLearningAgent import NUM_ACTIONS, QAction, QLearningAgent
from config import SIM

NUM_FEATURES = 9
EFFORT_FEATURE_SCALE = 5.0
SHARE_FEATURE_MIN = SIM.rl_share_feature_min
SHARE_FEATURE_MAX = SIM.rl_share_feature_max


def share_feature(share: float) -> float:
    if share <= 0.0:
        return 0.0
    low = math.log10(SHARE_FEATURE_MIN)
    high = math.log10(SHARE_FEATURE_MAX)
    scaled = (math.log10(share) - low) / (high - low)
    return min(max(scaled, 0.0), 1.0)


def progress_potential(agent) -> float:
    weight = float(SIM.rl_progress_potential)
    if weight == 0.0:
        return 0.0
    threshold = float(agent.paper_completion_threshold())
    if threshold <= 0.0:
        return 0.0
    fraction = min(max(float(agent.paper_progress) / threshold, 0.0), 1.0)
    return weight * fraction


def snapshot_holdings(agent) -> dict[int, tuple[float, float, float]]:
    holdings: dict[int, tuple[float, float, float]] = {}
    for paper in ownership_papers():
        share = float(paper.share_distribution.get(agent, 0.0))
        if share:
            holdings[id(paper)] = (
                share,
                float(paper.current_ac),
                float(paper.accrual_rate),
            )
    return holdings


def action_caused_deltas(agent) -> tuple[float, float]:
    delta_ac = agent.academic_capital - agent._last_capital
    delta_accrual = agent._portfolio_accrual_rate() - agent._last_accrual_rate
    holdings = getattr(agent, "_last_holdings", None) or {}
    if not holdings:
        return delta_ac, delta_accrual
    exogenous_ac = 0.0
    exogenous_accrual = 0.0
    for paper in ownership_papers():
        previous = holdings.get(id(paper))
        if previous is None:
            continue
        old_share, old_ac, old_accrual = previous
        exogenous_ac += old_share * (float(paper.current_ac) - old_ac)
        exogenous_accrual += old_share * (float(paper.accrual_rate) - old_accrual)
    return delta_ac - exogenous_ac, delta_accrual - exogenous_accrual


def ac_percentile_rank(agent_ac: float, capitals: list[float]) -> float:
    if not capitals:
        return 0.0
    n = len(capitals)
    if n == 1:
        return 1.0
    less = sum(1 for value in capitals if value < agent_ac)
    equal = sum(1 for value in capitals if value == agent_ac)
    return (less + 0.5 * equal) / n


def _he_init(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(2.0 / max(fan_in, 1))
    return np.random.randn(fan_in, fan_out) * limit


def _copy_params(params: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(W.copy(), b.copy()) for W, b in params]


def _forward_trunk(
    x: np.ndarray, params: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """ReLU trunk. Returns the last hidden state, pre-activations, and inputs."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    inputs: list[np.ndarray] = [x]
    pre_relu: list[np.ndarray] = []
    h = x
    for weight, bias in params:
        z = h @ weight + bias
        pre_relu.append(z)
        h = np.maximum(0.0, z)
        inputs.append(h)
    return h, pre_relu, inputs


def _dueling_q(
    hidden: np.ndarray,
    value_head: tuple[np.ndarray, np.ndarray],
    advantage_head: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Q = V + A - mean(A). The mean keeps the advantage identified."""
    value_w, value_b = value_head
    advantage_w, advantage_b = advantage_head
    value = hidden @ value_w + value_b
    advantage = hidden @ advantage_w + advantage_b
    return value + advantage - advantage.mean(axis=1, keepdims=True)


def double_dqn_targets(
    q_online_next: np.ndarray,
    q_target_next: np.ndarray,
    legal_masks: np.ndarray,
    rewards: np.ndarray,
    bootstraps: np.ndarray,
) -> np.ndarray:
    """Score the online network's greedy legal action with the target network."""
    masked = np.where(legal_masks > 0.0, q_online_next, -np.inf)
    greedy = np.argmax(masked, axis=1)
    rows = np.arange(len(rewards))
    chosen = q_target_next[rows, greedy]
    valid = np.isfinite(masked[rows, greedy])
    chosen = np.where(valid, chosen, 0.0)
    return rewards + bootstraps * chosen


_HUBER_DELTA = 1.0
_GRAD_CLIP = 5.0


def _huber_grad(errors: np.ndarray) -> np.ndarray:
    """Gradient of mean Huber loss with respect to the prediction."""
    clipped = np.clip(errors, -_HUBER_DELTA, _HUBER_DELTA)
    return clipped / max(len(errors), 1)


def _clip_grad(grad: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(grad))
    if norm > _GRAD_CLIP:
        return grad * (_GRAD_CLIP / norm)
    return grad


class DQNBackend:
    """Dueling MLP with replay, a target network, and Double DQN targets."""

    def __init__(
        self,
        *,
        hidden_size: int = SIM.dqn_hidden_size,
        hidden_layers: int = SIM.dqn_hidden_layers,
        lr: float = SIM.dqn_lr,
        gamma: float = SIM.dqn_gamma,
        replay_capacity: int = SIM.dqn_replay_capacity,
        batch_size: int = SIM.dqn_batch_size,
        target_sync: int = SIM.dqn_target_sync,
    ):
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.train_steps = 0
        self.replay: deque[tuple] = deque(maxlen=replay_capacity)
        self.params = self._build_trunk()
        width = self.hidden_size if self.hidden_layers else NUM_FEATURES
        self.value_head = (np.zeros((width, 1), dtype=np.float64), np.zeros(1))
        self.advantage_head = (
            np.zeros((width, NUM_ACTIONS), dtype=np.float64),
            np.zeros(NUM_ACTIONS, dtype=np.float64),
        )
        self.target_params = _copy_params(self.params)
        self.target_value_head = _copy_params([self.value_head])[0]
        self.target_advantage_head = _copy_params([self.advantage_head])[0]

    def _build_trunk(self) -> list[tuple[np.ndarray, np.ndarray]]:
        sizes = [NUM_FEATURES]
        for _ in range(self.hidden_layers):
            sizes.append(self.hidden_size)
        params: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            params.append((_he_init(fan_in, fan_out), np.zeros(fan_out, dtype=np.float64)))
        return params

    def _q_bundle(self, features: np.ndarray, params, value_head, advantage_head):
        single = features.ndim == 1
        hidden, pre_relu, inputs = _forward_trunk(features, params)
        q = _dueling_q(hidden, value_head, advantage_head)
        if single:
            q = q.reshape(-1)
        return q, hidden, pre_relu, inputs

    def q_values(self, features: np.ndarray) -> np.ndarray:
        q, _, _, _ = self._q_bundle(
            features, self.params, self.value_head, self.advantage_head
        )
        return q

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_legal_mask: np.ndarray,
        *,
        done: bool,
        bootstrap: float | None = None,
    ) -> None:
        if bootstrap is None:
            bootstrap = 0.0 if done else self.gamma
        self.replay.append(
            (
                state.astype(np.float64, copy=True),
                int(action),
                float(reward),
                next_state.astype(np.float64, copy=True),
                next_legal_mask.astype(np.float64, copy=True),
                float(bootstrap),
            )
        )

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = random.sample(self.replay, self.batch_size)
        states = np.stack([item[0] for item in batch])
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float64)
        next_states = np.stack([item[3] for item in batch])
        legal_masks = np.stack([item[4] for item in batch])
        bootstraps = np.array([item[5] for item in batch], dtype=np.float64)

        q_pred_all, hidden, pre_relu, inputs = self._q_bundle(
            states, self.params, self.value_head, self.advantage_head
        )
        rows = np.arange(self.batch_size)
        q_pred = q_pred_all[rows, actions]

        q_online_next, _, _, _ = self._q_bundle(
            next_states, self.params, self.value_head, self.advantage_head
        )
        q_target_next, _, _, _ = self._q_bundle(
            next_states,
            self.target_params,
            self.target_value_head,
            self.target_advantage_head,
        )
        targets = double_dqn_targets(
            q_online_next, q_target_next, legal_masks, rewards, bootstraps
        )

        errors = q_pred - targets
        grad_q = np.zeros_like(q_pred_all)
        grad_q[rows, actions] = _huber_grad(errors)
        self._apply(hidden, pre_relu, inputs, grad_q)

        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_params = _copy_params(self.params)
            self.target_value_head = _copy_params([self.value_head])[0]
            self.target_advantage_head = _copy_params([self.advantage_head])[0]

        return float(np.mean(errors * errors))

    def _apply(self, hidden, pre_relu, inputs, grad_q: np.ndarray) -> None:
        """Backprop a dueling Q gradient into the advantage, value, and trunk."""
        grad_value = grad_q.sum(axis=1, keepdims=True)
        grad_advantage = grad_q - grad_q.mean(axis=1, keepdims=True)
        value_w, value_b = self.value_head
        advantage_w, advantage_b = self.advantage_head
        grad = grad_value @ value_w.T + grad_advantage @ advantage_w.T
        value_w -= self.lr * _clip_grad(hidden.T @ grad_value)
        value_b -= self.lr * _clip_grad(grad_value.sum(axis=0))
        advantage_w -= self.lr * _clip_grad(hidden.T @ grad_advantage)
        advantage_b -= self.lr * _clip_grad(grad_advantage.sum(axis=0))
        for layer_idx in reversed(range(len(self.params))):
            weight, bias = self.params[layer_idx]
            layer_input = inputs[layer_idx]
            grad = grad * (pre_relu[layer_idx] > 0)
            incoming = grad @ weight.T
            weight -= self.lr * _clip_grad(layer_input.T @ grad)
            bias -= self.lr * _clip_grad(grad.sum(axis=0))
            grad = incoming

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "hidden_size": self.hidden_size,
                    "hidden_layers": self.hidden_layers,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "target_sync": self.target_sync,
                    "params": self.params,
                    "value_head": self.value_head,
                    "advantage_head": self.advantage_head,
                    "target_params": self.target_params,
                    "target_value_head": self.target_value_head,
                    "target_advantage_head": self.target_advantage_head,
                    "train_steps": self.train_steps,
                },
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if "value_head" not in data:
            raise ValueError(f"DQN policy {path} is from before the dueling network")
        self.hidden_size = int(data["hidden_size"])
        self.hidden_layers = int(data["hidden_layers"])
        self.lr = float(data["lr"])
        self.gamma = float(data["gamma"])
        self.batch_size = int(data["batch_size"])
        self.target_sync = int(data["target_sync"])
        self.params = data["params"]
        self.value_head = data["value_head"]
        self.advantage_head = data["advantage_head"]
        self.target_params = data["target_params"]
        self.target_value_head = data["target_value_head"]
        self.target_advantage_head = data["target_advantage_head"]
        self.train_steps = int(data.get("train_steps", 0))


def make_dqn_backend(**kwargs) -> DQNBackend:
    return DQNBackend(**kwargs)


def legal_action_mask(legal: list[int]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.float64)
    for action in legal:
        mask[int(action)] = 1.0
    return mask


class DQNAgent(QLearningAgent):
    """Deep Q-network agent; inherits action logic and reward from ``QLearningAgent``."""

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_timesteps: int = 30,
        name: str | None = None,
        *,
        backend: DQNBackend | None = None,
        gamma: float = SIM.dqn_gamma,
        epsilon: float = SIM.dqn_epsilon,
        learning: bool = True,
        n_step: int | None = None,
    ):
        backend = backend if backend is not None else make_dqn_backend()
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            forecast_horizon_timesteps=forecast_horizon_timesteps,
            name=name,
            backend=backend,
            gamma=gamma,
            epsilon=epsilon,
            learning=learning,
        )
        self.n_step = max(1, int(SIM.dqn_n_step if n_step is None else n_step))
        self._nstep: deque[tuple] = deque()
        self._last_features: np.ndarray | None = None
        self._last_accrual_rate = 0.0
        self._last_rank = self._current_ac_rank()
        self._last_progress_potential = progress_potential(self)
        self._last_holdings = snapshot_holdings(self)

    @property
    def dqn_backend(self) -> DQNBackend:
        return self.backend  # type: ignore[return-value]

    def choose_marketplace_action(self) -> Paper | None:
        in_review = self.active_review_paper is not None
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_actions(in_review, best_paper is not None)
        self._learn_transition(features, legal)
        action = self._select_featured(features, legal)
        self._last_features = features
        self._last_action = int(action)
        self._remember_reward_baseline()
        self._pending_action = int(action)
        if action in (QAction.CLAIM, QAction.CLAIM_NEW):
            return best_paper
        return None

    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        in_review = self.active_review_paper is not None
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_continuous_actions(in_review, best_paper is not None)
        if self.learning and self._last_features is not None:
            self._learn_transition(features, legal)
        action = self._select_featured(features, legal)
        self._last_features = features
        self._last_action = int(action)
        self._remember_reward_baseline()
        if action in (QAction.CLAIM, QAction.CLAIM_NEW):
            return (CONTINUOUS_CLAIM, best_paper)
        if action == QAction.CONTINUE:
            return (CONTINUOUS_REVIEW, None)
        if action == QAction.RESEARCH_FINISH:
            return (CONTINUOUS_RESEARCH_FINISH, None)
        return (CONTINUOUS_RESEARCH, None)

    def _legal_continuous_actions(self, in_review: bool, has_reviewable: bool) -> list[int]:
        if in_review:
            actions = [QAction.CONTINUE, QAction.FINISH_WRITE]
            if has_reviewable:
                actions.append(QAction.CLAIM_NEW)
            return actions
        actions = [QAction.WRITE]
        if not self.continuous_publish_by_threshold():
            actions.append(QAction.RESEARCH_FINISH)
        if has_reviewable:
            actions.append(QAction.CLAIM)
        return actions

    def _legal_actions(self, in_review: bool, has_reviewable: bool) -> list[int]:
        if in_review:
            actions = [QAction.CONTINUE, QAction.FINISH_WRITE]
            if has_reviewable:
                actions.append(QAction.CLAIM_NEW)
            return actions
        actions = [QAction.WRITE]
        if has_reviewable:
            actions.append(QAction.CLAIM)
        return actions

    def _select_featured(self, features: np.ndarray, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return int(random.choice(legal))
        q = self.backend.q_values(features)
        return int(max(legal, key=lambda action: q[action]))

    def _best_reviewable(self, exclude: Paper | None = None) -> tuple[Paper | None, float, float]:
        reviewable = [
            paper
            for paper in Agent.all_papers
            if paper is not exclude and self._can_review(paper)
        ]
        if not reviewable:
            return None, 0.0, 0.0
        best = max(reviewable, key=self._score_claim)
        return best, self._prospective_share(best), best.current_ac

    def _features(self, best_share: float, best_ac: float) -> np.ndarray:
        num_reviewable = sum(1 for paper in Agent.all_papers if self._can_review(paper))
        in_review = self.active_review_paper is not None
        return np.array(
            [
                min(self.paper_progress / self.paper_completion_threshold(), 1.0),
                np.tanh(self.active_review_effort / EFFORT_FEATURE_SCALE),
                np.tanh(self.academic_capital / 100.0),
                np.tanh(self.peer_review_history / 10.0),
                np.tanh(num_reviewable / 10.0),
                share_feature(best_share),
                np.tanh(best_ac / 100.0),
                1.0 if in_review else 0.0,
                min(self.active_review_effort / max(MIN_REVIEW_EFFORT_THRESHOLD, 1.0), 1.0),
            ],
            dtype=np.float64,
        )

    def _peer_capitals(self) -> list[float]:
        return [agent.academic_capital for agent in Agent.all_agents]

    def _current_ac_rank(self) -> float:
        return ac_percentile_rank(self.academic_capital, self._peer_capitals())

    def _compute_reward(self) -> float:
        delta_ac, delta_accrual = action_caused_deltas(self)
        delta_rank = self._current_ac_rank() - self._last_rank
        return (
            SIM.rl_reward_ac_weight * delta_ac
            + SIM.rl_reward_rank_weight * delta_rank
            + SIM.rl_reward_accrual_weight * delta_accrual
            + self.gamma * progress_potential(self)
            - self._last_progress_potential
        )

    def _remember_reward_baseline(self) -> None:
        self._last_capital = self.academic_capital
        self._last_rank = self._current_ac_rank()
        self._last_accrual_rate = self._portfolio_accrual_rate()
        self._last_holdings = snapshot_holdings(self)
        self._last_progress_potential = progress_potential(self)

    def _portfolio_accrual_rate(self) -> float:
        total = 0.0
        for paper in ownership_papers():
            share = paper.share_distribution.get(self, 0.0)
            if share:
                total += share * paper.accrual_rate
        return total

    def _learn_transition(self, features: np.ndarray, legal: list[int]) -> None:
        if not self.learning or self._last_features is None:
            return
        self._nstep.append(
            (
                self._last_features.copy(),
                int(self._last_action),
                self._compute_reward(),
                features.copy(),
                legal_action_mask(legal),
            )
        )
        self._flush_nstep(episode_done=False)

    def _flush_nstep(self, episode_done: bool) -> None:
        """Store an n-step return. A zero reward adds nothing to it."""
        while self._nstep and (episode_done or len(self._nstep) >= self.n_step):
            length = min(self.n_step, len(self._nstep))
            window = list(self._nstep)[:length]
            returns = 0.0
            discount = 1.0
            for _, _, reward, _, _ in window:
                returns += discount * reward
                discount *= self.gamma
            reaches_end = episode_done and length == len(self._nstep)
            state, action, _, _, _ = window[0]
            _, _, _, next_state, mask = window[-1]
            self.dqn_backend.remember(
                state,
                action,
                returns,
                next_state,
                mask,
                done=reaches_end,
                bootstrap=0.0 if reaches_end else discount,
            )
            self.dqn_backend.train_step()
            self._nstep.popleft()
            if not episode_done:
                break

    def end_episode(self) -> None:
        if self.learning and self._last_features is not None:
            self._nstep.append(
                (
                    self._last_features.copy(),
                    int(self._last_action),
                    self._compute_reward(),
                    np.zeros(NUM_FEATURES, dtype=np.float64),
                    np.zeros(NUM_ACTIONS, dtype=np.float64),
                )
            )
        if self.learning:
            self._flush_nstep(episode_done=True)
        self._nstep.clear()
        self._last_features = None
        self._last_action = None
