"""Discrete-mode Q-learning agent for the single-review peer-review market.

Standalone from ``QLearningAgent`` (continuous). When free, the agent learns
one of three marketplace actions each timestep:

* ``WRITE`` — pass and work-write
* ``BAD_FAITH_CLAIM`` — claim the best listed paper for a 1-timestep review
* ``GOOD_FAITH_CLAIM`` — claim the best listed paper for a 5-timestep review

Good/bad faith is chosen by Q, not by the heuristic forecast. Paper targeting
still uses inherited ``HeuristicAgent`` scoring. While a review is in progress
the environment skips marketplace decisions and auto-advances effort.

Policies save to ``policies/policy_<backend>_discrete.pkl``.
"""

from __future__ import annotations

import math
import pickle
import random
from collections import defaultdict
from enum import IntEnum

import numpy as np

from Agent import Agent, ownership_papers
from config import SIM
from HeuristicAgent import HeuristicAgent
from Paper import (
    BAD_FAITH_REVIEW,
    GOOD_FAITH_REVIEW,
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_PARADIGM_DISCRETE,
    Paper,
)

DISCRETE_REVIEW_PARADIGM = REVIEW_PARADIGM_DISCRETE

SHARE_FEATURE_MIN = SIM.rl_share_feature_min
SHARE_FEATURE_MAX = SIM.rl_share_feature_max


def share_feature(share: float) -> float:
    """Log-scale a review share into [0, 1]. No offer maps to 0."""
    if share <= 0.0:
        return 0.0
    low = math.log10(SHARE_FEATURE_MIN)
    high = math.log10(SHARE_FEATURE_MAX)
    scaled = (math.log10(share) - low) / (high - low)
    return min(max(scaled, 0.0), 1.0)


def progress_potential(agent) -> float:
    """Weight times how far the current manuscript has progressed."""
    weight = float(SIM.rl_progress_potential)
    if weight == 0.0:
        return 0.0
    threshold = float(agent.paper_completion_threshold())
    if threshold <= 0.0:
        return 0.0
    fraction = min(max(float(agent.paper_progress) / threshold, 0.0), 1.0)
    return weight * fraction


def snapshot_holdings(agent) -> dict[int, tuple[float, float, float]]:
    """Share, academic capital, and accrual rate of each paper ``agent`` owns."""
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
    """Capital and accrual changes caused by ownership the action changed."""
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


class WritingTrace:
    """Writing steps of the current manuscript.

    When the manuscript publishes, its reward is divided by the number of
    steps and each step is updated toward that share.
    """

    def __init__(self) -> None:
        self.steps: list[tuple] = []

    def visit(self, token, action: int) -> None:
        self.steps.append((token, int(action)))

    def apply_split(self, backend, reward: float) -> None:
        n = len(self.steps)
        if n == 0:
            return
        share = float(reward) / n
        for token, action in self.steps:
            backend.update_token(token, action, share)
        self.clear()

    def clear(self) -> None:
        self.steps.clear()


class DiscreteQAction(IntEnum):
    WRITE = 0
    BAD_FAITH_CLAIM = 1
    GOOD_FAITH_CLAIM = 2


NUM_DISCRETE_FEATURES = 9
NUM_DISCRETE_ACTIONS = len(DiscreteQAction)
EFFORT_FEATURE_SCALE = 5.0


def ac_percentile_rank(agent_ac: float, capitals: list[float]) -> float:
    if not capitals:
        return 0.0
    n = len(capitals)
    if n == 1:
        return 1.0
    less = sum(1 for value in capitals if value < agent_ac)
    equal = sum(1 for value in capitals if value == agent_ac)
    return (less + 0.5 * equal) / n


class DiscreteTabularQ:
    def __init__(self, alpha: float = 0.1, buckets: int = 5):
        self.alpha = alpha
        self.buckets = buckets
        self.table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_DISCRETE_ACTIONS, dtype=np.float64)
        )

    def _key(self, features: np.ndarray) -> tuple[int, ...]:
        clipped = np.clip(features, 0.0, 1.0)
        idx = np.minimum((clipped * self.buckets).astype(int), self.buckets - 1)
        return tuple(int(i) for i in idx)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.table[self._key(features)]

    def update(self, features: np.ndarray, action: int, target: float) -> None:
        q = self.table[self._key(features)]
        q[action] += self.alpha * (target - q[action])

    def trace_token(self, features: np.ndarray) -> tuple[int, ...]:
        return self._key(features)

    def add_error(
        self, token: tuple[int, ...], action: int, td_error: float, eligibility: float
    ) -> None:
        q = self.table[token]
        q[action] += self.alpha * td_error * eligibility

    def update_token(self, token: tuple[int, ...], action: int, target: float) -> None:
        q = self.table[token]
        q[action] += self.alpha * (target - q[action])

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "alpha": self.alpha,
                    "buckets": self.buckets,
                    "table": dict(self.table),
                    "review_paradigm": DISCRETE_REVIEW_PARADIGM,
                    "num_actions": NUM_DISCRETE_ACTIONS,
                },
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        _validate_discrete_policy_payload(data)
        self.alpha = data["alpha"]
        self.buckets = data["buckets"]
        self.table = defaultdict(
            lambda: np.zeros(NUM_DISCRETE_ACTIONS, dtype=np.float64),
            data["table"],
        )


class DiscreteLinearQ:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.W = np.zeros(
            (NUM_DISCRETE_ACTIONS, NUM_DISCRETE_FEATURES + 1), dtype=np.float64
        )

    @staticmethod
    def _augment(features: np.ndarray) -> np.ndarray:
        return np.append(features, 1.0)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.W @ self._augment(features)

    def update(self, features: np.ndarray, action: int, target: float) -> None:
        x = self._augment(features)
        pred = self.W[action] @ x
        self.W[action] += self.alpha * (target - pred) * x

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "W": self.W,
                    "alpha": self.alpha,
                    "review_paradigm": DISCRETE_REVIEW_PARADIGM,
                    "num_actions": NUM_DISCRETE_ACTIONS,
                },
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        _validate_discrete_policy_payload(data)
        self.alpha = data["alpha"]
        self.W = np.array(data["W"], dtype=np.float64)


def _validate_discrete_policy_payload(data: dict) -> None:
    paradigm = data.get("review_paradigm")
    if paradigm is not None and paradigm != DISCRETE_REVIEW_PARADIGM:
        raise ValueError(f"policy review_paradigm={paradigm!r} is not discrete")
    num_actions = data.get("num_actions")
    if num_actions is not None and int(num_actions) != NUM_DISCRETE_ACTIONS:
        raise ValueError(
            f"policy num_actions={num_actions} does not match discrete "
            f"({NUM_DISCRETE_ACTIONS})"
        )
    table = data.get("table")
    if table is not None:
        for values in table.values():
            if len(values) != NUM_DISCRETE_ACTIONS:
                raise ValueError("policy Q-table action dimension is not discrete")
    weights = data.get("W")
    if weights is not None and np.array(weights).shape[0] != NUM_DISCRETE_ACTIONS:
        raise ValueError("policy weight matrix action dimension is not discrete")


def make_discrete_backend(kind: str = "tabular", **kwargs):
    if kind == "tabular":
        return DiscreteTabularQ(**kwargs)
    if kind == "linear":
        return DiscreteLinearQ(**kwargs)
    raise ValueError(f"unknown discrete Q backend: {kind!r}")


class DiscreteQLearningAgent(HeuristicAgent):
    """Discrete-only RL agent with a 3-action marketplace policy."""

    requires_review_paradigm = DISCRETE_REVIEW_PARADIGM

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
        epsilon: float = 0.1,
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
        self.review_paradigm = DISCRETE_REVIEW_PARADIGM
        self.backend = (
            backend if backend is not None else make_discrete_backend("tabular")
        )
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = learning
        self.trace_lambda = float(SIM.rl_trace_lambda)
        self._writing_trace = WritingTrace()
        self._manuscript_published = False

        self._last_features: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_capital: float = academic_capital
        self._last_rank: float = self._current_ac_rank()
        self._last_accrual_rate: float = self._portfolio_accrual_rate()
        self._last_holdings: dict[int, tuple[float, float, float]] = snapshot_holdings(self)
        self._last_progress_potential: float = progress_potential(self)
        self._pending_review_kind: str | None = None

    def configure_review_paradigm(self, review_paradigm: str) -> None:
        super().configure_review_paradigm(review_paradigm)
        if self.review_paradigm != DISCRETE_REVIEW_PARADIGM:
            raise ValueError(
                f"{type(self).__name__} requires "
                f"review_paradigm={DISCRETE_REVIEW_PARADIGM!r}"
            )

    def choose_marketplace_action(self) -> Paper | None:
        best_paper, best_share, best_ac = self._best_reviewable()
        features = self._features(best_share, best_ac)
        legal = self._legal_actions(best_paper is not None)

        self._learn_transition(features, legal)
        action = self._select(features, legal)

        self._last_features = features
        self._last_action = int(action)
        self._remember_reward_baseline()
        self._pending_review_kind = None

        if action == DiscreteQAction.WRITE:
            return None
        if best_paper is None:
            return None
        if action == DiscreteQAction.BAD_FAITH_CLAIM:
            self._pending_review_kind = BAD_FAITH_REVIEW
            return best_paper
        if action == DiscreteQAction.GOOD_FAITH_CLAIM:
            self._pending_review_kind = GOOD_FAITH_REVIEW
            return best_paper
        return None

    def choose_review_kind(self, paper: Paper) -> str:
        if self._pending_review_kind is not None:
            kind = self._pending_review_kind
            self._pending_review_kind = None
            return kind
        return super().choose_review_kind(paper)

    def end_episode(self) -> None:
        if self.learning and self._last_features is not None:
            self._spread_return(None, None)
        self._writing_trace.clear()
        self._manuscript_published = False
        self._last_features = None
        self._last_action = None
        self._pending_review_kind = None

    def _legal_actions(self, has_reviewable: bool) -> list[int]:
        actions = [int(DiscreteQAction.WRITE)]
        if has_reviewable:
            actions.extend(
                [
                    int(DiscreteQAction.BAD_FAITH_CLAIM),
                    int(DiscreteQAction.GOOD_FAITH_CLAIM),
                ]
            )
        return actions

    def _select(self, features: np.ndarray, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.backend.q_values(features)
        return max(legal, key=lambda action: q[action])

    def _learn_transition(self, features: np.ndarray, legal: list[int]) -> None:
        if not self.learning or self._last_features is None:
            return
        self._spread_return(features, legal)

    def publish_paper(self) -> Paper:
        paper = super().publish_paper()
        self._manuscript_published = True
        return paper

    def finish_research_paper(self) -> Paper:
        paper = super().finish_research_paper()
        self._manuscript_published = True
        return paper

    def _spread_return(
        self, features: np.ndarray | None, legal: list[int] | None
    ) -> None:
        reward = self._compute_reward()
        published = self._manuscript_published
        writing = int(self._last_action) == int(DiscreteQAction.WRITE)
        if (
            writing
            and self.trace_lambda > 0.0
            and hasattr(self.backend, "update_token")
            and self._last_features is not None
        ):
            self._writing_trace.visit(
                self.backend.trace_token(self._last_features),
                int(self._last_action),
            )
            if published:
                self._writing_trace.apply_split(self.backend, reward)
                self._manuscript_published = False
            return
        if features is None or legal is None:
            max_next = 0.0
        else:
            q_next = self.backend.q_values(features)
            max_next = max((q_next[action] for action in legal), default=0.0)
        self.backend.update(
            self._last_features, self._last_action, reward + self.gamma * max_next
        )
        if published:
            self._manuscript_published = False

    def _peer_capitals(self) -> list[float]:
        return [agent.academic_capital for agent in Agent.all_agents]

    def _current_ac_rank(self) -> float:
        return ac_percentile_rank(self.academic_capital, self._peer_capitals())

    def _portfolio_accrual_rate(self) -> float:
        total = 0.0
        for paper in ownership_papers():
            share = paper.share_distribution.get(self, 0.0)
            if share:
                total += share * paper.accrual_rate
        return total

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

    def _best_reviewable(
        self, exclude: Paper | None = None
    ) -> tuple[Paper | None, float, float]:
        reviewable = [
            paper
            for paper in Agent.all_papers
            if paper is not exclude
            and self._can_review(paper)
        ]
        if not reviewable:
            return None, 0.0, 0.0
        best = max(reviewable, key=self._score_claim)
        share = self._prospective_share(best)
        return best, share, best.current_ac

    def _features(self, best_share: float, best_ac: float) -> np.ndarray:
        num_reviewable = sum(
            1
            for paper in Agent.all_papers
            if self._can_review(paper)
        )
        return np.array(
            [
                min(self.paper_progress / self.paper_completion_threshold(), 1.0),
                np.tanh(self.active_review_effort / EFFORT_FEATURE_SCALE),
                np.tanh(self.academic_capital / 100.0),
                np.tanh(self.peer_review_history / 10.0),
                np.tanh(num_reviewable / 10.0),
                share_feature(best_share),
                np.tanh(best_ac / 100.0),
                0.0,
                min(
                    self.active_review_effort / max(MIN_REVIEW_EFFORT_THRESHOLD, 1.0),
                    1.0,
                ),
            ],
            dtype=np.float64,
        )


class LowTalentDiscreteQLearningAgent(DiscreteQLearningAgent):
    """Discrete RL agent at low intrinsic talent; grouped separately in History."""
