"""Stateless Q-learning agent for the continuous peer-review market.

One situation, one value per action the environment will actually carry out.
There are no state features. A claim attaches to any paper that can be
reviewed, because the action is the decision to review, not which paper.

Actions match the continuous phase:

* no review open: ``WRITE`` (research), ``CLAIM`` when a paper is listed,
  and ``RESEARCH_FINISH`` when publishing is by choice
* review open: ``FINISH_WRITE`` (finish the review, then research) and
  ``CONTINUE`` (another timestep of review)

``CLAIM_NEW`` stays in the table for the two-phase marketplace, where
claiming a second paper finalizes the current review. The continuous phase
turns that choice into ``CONTINUE``, so it is not offered there.

Reward is the change in academic capital since the previous decision, the
score of this simulation. The update is one-step Q-learning: the action just
taken moves toward ``reward + gamma * max Q(legal next actions)``.
"""

from __future__ import annotations

import pickle
import random
from enum import IntEnum

import numpy as np

from Agent import (
    Agent,
    CONTINUOUS_CLAIM,
    CONTINUOUS_RESEARCH,
    CONTINUOUS_RESEARCH_FINISH,
    CONTINUOUS_REVIEW,
)
from config import SIM
from HeuristicAgent import HeuristicAgent
from Paper import Paper


class QAction(IntEnum):
    WRITE = 0
    CLAIM = 1
    CONTINUE = 2
    FINISH_WRITE = 3
    CLAIM_NEW = 4
    RESEARCH_FINISH = 5


NUM_ACTIONS = len(QAction)


class TabularQ:
    """One Q-value per action. The same values are used on every turn."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self.q = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def q_values(self) -> np.ndarray:
        return self.q

    def update(self, action: int, target: float) -> None:
        action = int(action)
        self.q[action] += self.alpha * (target - self.q[action])

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"alpha": self.alpha, "q": self.q, "num_actions": NUM_ACTIONS},
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        q = data.get("q")
        if q is None:
            raise ValueError("policy is not a stateless Q table")
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (NUM_ACTIONS,):
            raise ValueError(
                f"policy has shape {q.shape}, expected {(NUM_ACTIONS,)}"
            )
        self.q = q
        if "alpha" in data:
            self.alpha = float(data["alpha"])


def make_backend(kind: str = "tabular", **kwargs):
    if kind != "tabular":
        raise ValueError(
            f"stateless Q-learning only supports a tabular table, got {kind!r}"
        )
    return TabularQ(**kwargs)


class QLearningAgent(HeuristicAgent):
    """Q-learning agent with a single value for each simulation action."""

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_timesteps: int = 30,
        name: str | None = None,
        *,
        backend=None,
        gamma: float = SIM.rl_gamma,
        epsilon: float = SIM.rl_epsilon,
        learning: bool = True,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            forecast_horizon_timesteps=forecast_horizon_timesteps,
            name=name,
        )
        self.backend = backend if backend is not None else make_backend("tabular")
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.learning = learning
        self._last_action: int | None = None
        self._pending_action = int(QAction.WRITE)
        self._last_capital = float(academic_capital)

    def choose_marketplace_action(self) -> Paper | None:
        legal = self._legal_marketplace_actions()
        self._learn(legal)
        action = self._select(legal)
        self._last_action = int(action)
        self._pending_action = int(action)
        self._remember_reward_baseline()
        if action in (QAction.CLAIM, QAction.CLAIM_NEW):
            return self._claim_target()
        return None

    def choose_work_action(self) -> tuple[str, Paper | None]:
        action = self._pending_action
        if action == QAction.CONTINUE:
            return "peer_review", self.active_review_paper
        if action == QAction.FINISH_WRITE:
            return "finish_review_write_paper", None
        return "write_paper", None

    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        legal = self._legal_continuous_actions()
        self._learn(legal)
        action = self._select(legal)
        self._last_action = int(action)
        self._pending_action = int(action)
        self._remember_reward_baseline()
        return self._as_continuous(action)

    def end_episode(self) -> None:
        """Terminal step: the last action has no following value."""
        if self.learning and self._last_action is not None:
            self.backend.update(self._last_action, self._compute_reward())
        self._last_action = None

    def _legal_continuous_actions(self) -> list[int]:
        if self.active_review_paper is not None:
            return [int(QAction.FINISH_WRITE), int(QAction.CONTINUE)]
        actions = [int(QAction.WRITE)]
        if not self.continuous_publish_by_threshold():
            actions.append(int(QAction.RESEARCH_FINISH))
        if self._claim_target() is not None:
            actions.append(int(QAction.CLAIM))
        return actions

    def _legal_marketplace_actions(self) -> list[int]:
        if self.active_review_paper is not None:
            actions = [int(QAction.FINISH_WRITE), int(QAction.CONTINUE)]
            if self._claim_target() is not None:
                actions.append(int(QAction.CLAIM_NEW))
            return actions
        actions = [int(QAction.WRITE)]
        if self._claim_target() is not None:
            actions.append(int(QAction.CLAIM))
        return actions

    def _as_continuous(self, action: int) -> tuple[str, Paper | None]:
        if action == QAction.CLAIM:
            paper = self._claim_target()
            if paper is not None:
                return (CONTINUOUS_CLAIM, paper)
        if action == QAction.CONTINUE:
            return (CONTINUOUS_REVIEW, None)
        if action == QAction.RESEARCH_FINISH:
            return (CONTINUOUS_RESEARCH_FINISH, None)
        return (CONTINUOUS_RESEARCH, None)

    def _claim_target(self) -> Paper | None:
        for paper in Agent.all_papers:
            if self._can_review(paper):
                return paper
        return None

    def _select(self, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return int(random.choice(legal))
        q = self.backend.q_values()
        return int(max(legal, key=lambda action: float(q[action])))

    def _learn(self, legal: list[int]) -> None:
        if not self.learning or self._last_action is None:
            return
        reward = self._compute_reward()
        nxt = 0.0
        if legal:
            q = self.backend.q_values()
            nxt = max(float(q[action]) for action in legal)
        self.backend.update(self._last_action, reward + self.gamma * nxt)

    def _compute_reward(self) -> float:
        return self.academic_capital - self._last_capital

    def _remember_reward_baseline(self) -> None:
        self._last_capital = self.academic_capital


class LowTalentQLearningAgent(QLearningAgent):
    """RL agent trained and evaluated at low intrinsic talent."""
