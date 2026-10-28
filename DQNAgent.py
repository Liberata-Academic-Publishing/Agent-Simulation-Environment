"""Deep Q-network agent for the single-review peer-review market.

Same action space, feature vector, and reward function as ``QLearningAgent``,
but uses a small ReLU MLP with experience replay and a target network instead of
tabular or linear Q backends. Numpy-only (no PyTorch dependency).

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

import pickle
import random
from collections import deque

import numpy as np

from QLearningAgent import (
    NUM_ACTIONS,
    NUM_FEATURES,
    QLearningAgent,
)
from config import SIM


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

    @property
    def dqn_backend(self) -> DQNBackend:
        return self.backend  # type: ignore[return-value]

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
