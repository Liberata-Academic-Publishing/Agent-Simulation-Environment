"""Deep Q-network agent for the single-review peer-review market.

Same action space, feature vector, and reward function as ``QLearningAgent``,
but uses a small ReLU MLP with experience replay and a target network instead of
tabular or linear Q backends. Numpy-only (no PyTorch dependency).

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


def _forward(
    x: np.ndarray, params: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Return output Q-values, pre-ReLU activations, and layer inputs."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    inputs: list[np.ndarray] = [x]
    pre_relu: list[np.ndarray] = []
    h = x
    for layer_idx, (weight, bias) in enumerate(params):
        z = h @ weight + bias
        if layer_idx < len(params) - 1:
            pre_relu.append(z)
            h = np.maximum(0.0, z)
        else:
            h = z
        inputs.append(h)
    return h, pre_relu, inputs


def _apply_gradients(
    params: list[tuple[np.ndarray, np.ndarray]],
    pre_relu: list[np.ndarray],
    inputs: list[np.ndarray],
    grad_output: np.ndarray,
    lr: float,
) -> None:
    """Backprop ``grad_output`` (batch, out_dim) through the MLP."""
    grad = grad_output
    for layer_idx in reversed(range(len(params))):
        weight, bias = params[layer_idx]
        layer_input = inputs[layer_idx]
        grad_weight = layer_input.T @ grad / grad.shape[0]
        grad_bias = grad.mean(axis=0)
        weight -= lr * grad_weight
        bias -= lr * grad_bias
        if layer_idx > 0:
            grad = (grad @ weight.T) * (pre_relu[layer_idx - 1] > 0)


class DQNBackend:
    """Two-layer (configurable) MLP Q-network with replay and target net."""

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
        self.replay: deque[
            tuple[np.ndarray, int, float, np.ndarray, np.ndarray, bool]
        ] = deque(maxlen=replay_capacity)
        self.params = self._build_params()
        self.target_params = _copy_params(self.params)

    def _layer_sizes(self) -> list[int]:
        sizes = [NUM_FEATURES]
        for _ in range(self.hidden_layers):
            sizes.append(self.hidden_size)
        sizes.append(NUM_ACTIONS)
        return sizes

    def _build_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params: list[tuple[np.ndarray, np.ndarray]] = []
        for fan_in, fan_out in zip(self._layer_sizes(), self._layer_sizes()[1:]):
            params.append((_he_init(fan_in, fan_out), np.zeros(fan_out, dtype=np.float64)))
        return params

    def q_values(self, features: np.ndarray) -> np.ndarray:
        q, _, _ = _forward(features, self.params)
        return q.reshape(-1) if features.ndim == 1 else q

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_legal_mask: np.ndarray,
        *,
        done: bool,
    ) -> None:
        self.replay.append(
            (
                state.astype(np.float64, copy=True),
                int(action),
                float(reward),
                next_state.astype(np.float64, copy=True),
                next_legal_mask.astype(np.float64, copy=True),
                bool(done),
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
        dones = np.array([item[5] for item in batch], dtype=np.float64)

        q_pred_all, pre_relu, inputs = _forward(states, self.params)
        q_pred = q_pred_all[np.arange(self.batch_size), actions]

        q_next_all, _, _ = _forward(next_states, self.target_params)
        masked = np.where(legal_masks > 0.0, q_next_all, -np.inf)
        max_next = np.max(masked, axis=1)
        max_next = np.where(np.isfinite(max_next), max_next, 0.0)
        targets = rewards + self.gamma * max_next * (1.0 - dones)

        errors = q_pred - targets
        grad_output = np.zeros_like(q_pred_all)
        grad_output[np.arange(self.batch_size), actions] = (
            2.0 * errors / self.batch_size
        )
        _apply_gradients(self.params, pre_relu, inputs, grad_output, self.lr)

        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_params = _copy_params(self.params)

        return float(np.mean(errors * errors))

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
                    "target_params": self.target_params,
                    "train_steps": self.train_steps,
                },
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        num_features = data["params"][0][0].shape[0]
        if num_features != NUM_FEATURES:
            raise ValueError(
                f"policy has {num_features} features, expected {NUM_FEATURES}"
            )
        self.hidden_size = int(data["hidden_size"])
        self.hidden_layers = int(data["hidden_layers"])
        self.lr = float(data["lr"])
        self.gamma = float(data["gamma"])
        self.batch_size = int(data["batch_size"])
        self.target_sync = int(data["target_sync"])
        self.params = data["params"]
        self.target_params = data["target_params"]
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

    @property
    def dqn_backend(self) -> DQNBackend:
        return self.backend  # type: ignore[return-value]

    def _learn_transition(self, features: np.ndarray, legal: list[int]) -> None:
        if not self.learning or self._last_features is None:
            return
        reward = self._compute_reward()
        self.dqn_backend.remember(
            self._last_features,
            int(self._last_action),
            reward,
            features,
            legal_action_mask(legal),
            done=False,
        )
        self.dqn_backend.train_step()

    def end_episode(self) -> None:
        if self.learning and self._last_features is not None:
            reward = self._compute_reward()
            self.dqn_backend.remember(
                self._last_features,
                int(self._last_action),
                reward,
                np.zeros(NUM_FEATURES, dtype=np.float64),
                np.zeros(NUM_ACTIONS, dtype=np.float64),
                done=True,
            )
            self.dqn_backend.train_step()
        self._last_features = None
        self._last_action = None
