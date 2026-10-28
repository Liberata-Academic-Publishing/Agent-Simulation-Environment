"""Reinforcement-learning agent for the single-review peer-review market.

Reward = weighted change in academic capital and portfolio accrual caused by
ownership this action gained or lost, plus weighted Δ AC percentile rank.
Citations on shares the agent already held are not part of the reward.
The agent learns *which action type* to take each timestep. Its observation
is only two buckets: how far the current manuscript has progressed, and how
much effort the active review has accumulated. The target paper for a claim
is still chosen by the inherited heuristic forecast, so the action space stays
tiny and the offer, capital, and reputation never enter the Q table.

The agent makes one decision per timestep, and the wiring depends on the active
review paradigm:

* **continuous** (default) -- one merged decision per timestep via
  ``choose_continuous_action``; the chosen action also applies that timestep's
  effort (claim+review, continue a review, research, or research+finish).
* **discrete** -- the two-phase flow: the decision is computed in the
  marketplace phase (``choose_marketplace_action``) and, when it is not a claim,
  a cached work-phase action is replayed by ``choose_work_action``.

Action space
------------
* free (no active review)  -> ``WRITE`` (research, no publish), ``RESEARCH_FINISH``
                              (research then finish/list the paper; continuous
                              only), or ``CLAIM`` (start reviewing a paper)
* active review            -> ``CONTINUE`` (one more timestep), ``FINISH_WRITE``
                              (finalize then research/write), or ``CLAIM_NEW``
                              (finalize the current review by claiming another
                              paper)

NOTE: the observation is the two progress buckets above, and the continuous
merged-phase overhaul added ``RESEARCH_FINISH`` (so ``NUM_ACTIONS`` grew).
Policies trained on the old 9-feature vector are incompatible and must be
retrained.

Library: numpy only. Two interchangeable Q backends (tabular + linear).
"""

from __future__ import annotations

import math
import pickle
import random
from collections import defaultdict
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
    WRITE = 0          # free: research own paper (continuous: no finish)
    CLAIM = 1          # free: claim and start reviewing the best listed paper
    CONTINUE = 2       # review: invest one more timestep in the active review
    FINISH_WRITE = 3   # review: finalize the active review, then research/write
    CLAIM_NEW = 4      # review: finalize by claiming another listed paper
    RESEARCH_FINISH = 5  # free (continuous): research, then finish/list the paper


# Observation is (manuscript bucket, review-effort bucket), encoded in [0, 1]
# so tabular quantization and the linear/DQN backends share one vector.
# Index 0 is manuscript progress. Index 1 is active review effort.
NUM_FEATURES = 2
MANUSCRIPT_FEATURE = 0
REVIEW_FEATURE = 1
NUM_ACTIONS = len(QAction)

# Range of the log-scaled review-share feature (see ``share_feature``).
# Discrete Q-learning still uses it. Continuous Q-learning does not.
SHARE_FEATURE_MIN = SIM.rl_share_feature_min
SHARE_FEATURE_MAX = SIM.rl_share_feature_max


def share_feature(share: float) -> float:
    """Log-scale a review share into [0, 1] for the price feature.

    Offers span several orders of magnitude (0.01% floor to tens of percent),
    and a linear feature puts nearly all of them in one tabular bucket. This
    maps ``SHARE_FEATURE_MIN`` -> 0 and ``SHARE_FEATURE_MAX`` -> 1 on a log10
    scale, clipping outside that range. No offer (share <= 0) maps to 0.
    """
    if share <= 0.0:
        return 0.0
    low = math.log10(SHARE_FEATURE_MIN)
    high = math.log10(SHARE_FEATURE_MAX)
    scaled = (math.log10(share) - low) / (high - low)
    return min(max(scaled, 0.0), 1.0)


def manuscript_bucket_count() -> int:
    return int(SIM.rl_manuscript_buckets)


def review_bucket_count() -> int:
    return int(SIM.rl_review_buckets)


def manuscript_progress_bucket(progress: float, threshold: float) -> int:
    """Bucket manuscript progress in ``[0, threshold]`` into ``0 .. n-1``."""
    buckets = manuscript_bucket_count()
    if threshold <= 0.0:
        return 0
    fraction = min(max(float(progress) / float(threshold), 0.0), 1.0)
    if fraction >= 1.0:
        return buckets - 1
    return min(int(fraction * buckets), buckets - 1)


def review_progress_bucket(effort: float) -> int:
    """Bucket active review effort. Zero effort is always bucket 0.

    At decision time a continuous review already includes the effort from the
    turn that started it, so bucket 0 means the agent is not reviewing.
    Positive effort is spread over buckets ``1 .. n-1`` up to
    ``rl_review_progress_cap``.
    """
    buckets = review_bucket_count()
    if float(effort) <= 0.0 or buckets <= 1:
        return 0
    cap = max(float(SIM.rl_review_progress_cap), 1e-9)
    fraction = min(max(float(effort) / cap, 0.0), 1.0)
    upper = buckets - 1
    return 1 + min(int(fraction * upper), upper - 1)


def bucket_feature(bucket: int, count: int) -> float:
    """Encode bucket ``b`` as ``(b + 0.5) / count`` so ``int(value * count)`` is ``b``."""
    count = max(int(count), 1)
    bounded = min(max(int(bucket), 0), count - 1)
    return (bounded + 0.5) / count


def progress_features(manuscript_bucket: int, review_bucket: int) -> np.ndarray:
    """Two-dimensional observation for one pair of progress buckets."""
    return np.array(
        [
            bucket_feature(manuscript_bucket, manuscript_bucket_count()),
            bucket_feature(review_bucket, review_bucket_count()),
        ],
        dtype=np.float64,
    )


def _combined_q(backend, backend_b, manuscript_bucket: int, review_bucket: int) -> np.ndarray:
    features = progress_features(manuscript_bucket, review_bucket)
    values = np.asarray(backend.q_values(features), dtype=np.float64)
    if backend_b is not None:
        values = values + np.asarray(backend_b.q_values(features), dtype=np.float64)
    return values


def write_claim_gaps(backend, backend_b=None) -> list[float]:
    """``Q(write) - Q(claim)`` in every manuscript bucket while not reviewing.

    Greedy selection adds the two tables when both are present. A positive gap
    means writing beats claiming. A zero gap is a tie, and the free-action
    order then keeps writing.
    """
    gaps = []
    for manuscript_bucket in range(manuscript_bucket_count()):
        values = _combined_q(backend, backend_b, manuscript_bucket, 0)
        gaps.append(float(values[QAction.WRITE] - values[QAction.CLAIM]))
    return gaps


def describe_progress_policy(backend, backend_b=None) -> str:
    """Q values the greedy policy sees on the progress grid."""
    lines = [
        "Progress-state Q (greedy sums the two tables when both exist).",
        "Free states, review bucket 0. Gap is Q(write) - Q(claim).",
    ]
    for manuscript_bucket, gap in enumerate(write_claim_gaps(backend, backend_b)):
        values = _combined_q(backend, backend_b, manuscript_bucket, 0)
        lines.append(
            f"  manuscript {manuscript_bucket}: "
            f"write={values[QAction.WRITE]:+.4f} "
            f"claim={values[QAction.CLAIM]:+.4f} "
            f"gap={gap:+.4f}"
        )
    lines.append("Review states. Gap is Q(finish) - Q(continue).")
    for review_bucket in range(1, review_bucket_count()):
        for manuscript_bucket in range(manuscript_bucket_count()):
            values = _combined_q(
                backend, backend_b, manuscript_bucket, review_bucket
            )
            gap = float(values[QAction.FINISH_WRITE] - values[QAction.CONTINUE])
            lines.append(
                f"  manuscript {manuscript_bucket} review {review_bucket}: "
                f"finish={values[QAction.FINISH_WRITE]:+.4f} "
                f"continue={values[QAction.CONTINUE]:+.4f} "
                f"gap={gap:+.4f}"
            )
    return "\n".join(lines)


def progress_potential(agent) -> float:
    """Φ(state): weight times how far the current manuscript has progressed.

    A function of state only, so γΦ(next) − Φ(now) does not change the best action.
    """
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
    for paper in Agent.all_papers:
        share = float(paper.share_distribution.get(agent, 0.0))
        if share:
            holdings[id(paper)] = (
                share,
                float(paper.current_ac),
                float(paper.accrual_rate),
            )
    return holdings


def action_caused_deltas(agent) -> tuple[float, float]:
    """Capital and accrual changes caused by ownership the action changed.

    Citations on shares the agent already held arrive whichever action it took.
    Those are removed, so a review is rewarded only for share it actually gained.
    """
    delta_ac = agent.academic_capital - agent._last_capital
    delta_accrual = agent._portfolio_accrual_rate() - agent._last_accrual_rate
    holdings = getattr(agent, "_last_holdings", None) or {}
    if not holdings:
        return delta_ac, delta_accrual
    exogenous_ac = 0.0
    exogenous_accrual = 0.0
    for paper in Agent.all_papers:
        previous = holdings.get(id(paper))
        if previous is None:
            continue
        old_share, old_ac, old_accrual = previous
        exogenous_ac += old_share * (float(paper.current_ac) - old_ac)
        exogenous_accrual += old_share * (float(paper.accrual_rate) - old_accrual)
    return delta_ac - exogenous_ac, delta_accrual - exogenous_accrual


def ac_percentile_rank(agent_ac: float, capitals: list[float]) -> float:
    """Tie-aware percentile rank in [0, 1] among ``capitals``.

    Uses average rank normalized by cohort size: 0 when everyone is ahead,
    1 when everyone is behind or tied at the top.
    """
    if not capitals:
        return 0.0
    n = len(capitals)
    if n == 1:
        return 1.0
    less = sum(1 for value in capitals if value < agent_ac)
    equal = sum(1 for value in capitals if value == agent_ac)
    return (less + 0.5 * equal) / n


# --------------------------------------------------------------------------- #
# Q backends: q_values(features) -> np.ndarray[NUM_ACTIONS]; update(f, a, target)
# --------------------------------------------------------------------------- #
class TabularQ:
    """Bucketized tabular Q-table. Simple, interpretable warm-up baseline.

    ``feature_buckets`` is one count per observation coordinate. Continuous
    Q-learning sets it to ``(manuscript buckets, review buckets)`` and encodes
    each coordinate as ``(bucket + 0.5) / count``, which this key recovers.
    Without it, every coordinate uses the single ``buckets`` count. That is
    the path the discrete share-feature test still uses.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        buckets: int = 5,
        feature_buckets: tuple[int, ...] | None = None,
    ):
        self.alpha = alpha
        self.buckets = int(buckets)
        if feature_buckets is None:
            self.feature_buckets = None
        else:
            self.feature_buckets = tuple(int(count) for count in feature_buckets)
        self.table: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64)
        )

    def _key(self, features: np.ndarray) -> tuple[int, ...]:
        clipped = np.clip(np.asarray(features, dtype=np.float64), 0.0, 1.0)
        if not self.feature_buckets:
            idx = np.minimum((clipped * self.buckets).astype(int), self.buckets - 1)
            return tuple(int(i) for i in idx)
        indices = []
        paired = min(len(clipped), len(self.feature_buckets))
        for value, count in zip(clipped[:paired], self.feature_buckets[:paired]):
            bucket = int(float(value) * int(count))
            if bucket >= count:
                bucket = count - 1
            if bucket < 0:
                bucket = 0
            indices.append(bucket)
        if len(clipped) > paired:
            extra = clipped[paired:]
            more = np.minimum((extra * self.buckets).astype(int), self.buckets - 1)
            indices.extend(int(i) for i in more)
        return tuple(indices)

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

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "alpha": self.alpha,
                    "buckets": self.buckets,
                    "feature_buckets": self.feature_buckets,
                    "table": dict(self.table),
                },
                fh,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.alpha = data["alpha"]
        self.buckets = data["buckets"]
        feature_buckets = data.get("feature_buckets")
        self.feature_buckets = (
            None if feature_buckets is None else tuple(int(count) for count in feature_buckets)
        )
        self.table = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=np.float64), data["table"]
        )


class LinearQ:
    """Linear function approximation: Q(s, a) = w_a . [features, 1]."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        # +1 column for a bias term.
        self.W = np.zeros((NUM_ACTIONS, NUM_FEATURES + 1), dtype=np.float64)

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
        np.save(path, self.W)

    def load(self, path: str) -> None:
        self.W = np.load(path if path.endswith(".npy") else path + ".npy")


def average_backends(primary, secondary):
    """Mean of two Q tables. Greedy actions match the sum used while learning."""
    if isinstance(primary, TabularQ):
        averaged = TabularQ(
            alpha=primary.alpha,
            buckets=primary.buckets,
            feature_buckets=getattr(primary, "feature_buckets", None),
        )
        zeros = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for key in set(primary.table) | set(secondary.table):
            averaged.table[key] = 0.5 * (
                np.asarray(primary.table.get(key, zeros), dtype=np.float64)
                + np.asarray(secondary.table.get(key, zeros), dtype=np.float64)
            )
        return averaged
    if isinstance(primary, LinearQ):
        averaged = LinearQ(alpha=primary.alpha)
        averaged.W = 0.5 * (primary.W + secondary.W)
        return averaged
    raise TypeError(f"cannot average {type(primary).__name__}")


def make_backend(kind: str = "tabular", **kwargs):
    if kind == "tabular":
        kwargs.setdefault(
            "feature_buckets",
            (manuscript_bucket_count(), review_bucket_count()),
        )
        return TabularQ(**kwargs)
    if kind == "linear":
        return LinearQ(**kwargs)
    raise ValueError(f"unknown Q backend: {kind!r}")


_TRACE_FLOOR = 1e-4

# Actions that advance the agent's own manuscript. A publication payment is
# carried back to these steps only.
WRITING_TRACE_ACTIONS = frozenset({
    int(QAction.WRITE),
    int(QAction.FINISH_WRITE),
    int(QAction.RESEARCH_FINISH),
})


class WritingTrace:
    """Replacing eligibility trace over writing actions."""

    def __init__(self) -> None:
        self.weights: dict[tuple, float] = {}

    def visit(self, token, action: int) -> None:
        self.weights = {
            key: value
            for key, value in self.weights.items()
            if key[0] != token
        }
        self.weights[(token, int(action))] = 1.0

    def apply(self, backend, td_error: float) -> None:
        stale = []
        for (token, action), eligibility in self.weights.items():
            if eligibility < _TRACE_FLOOR:
                stale.append((token, action))
                continue
            backend.add_error(token, action, td_error, eligibility)
        for key in stale:
            del self.weights[key]

    def decay(self, factor: float) -> None:
        if factor <= 0.0:
            self.weights.clear()
            return
        stale = []
        for key, eligibility in self.weights.items():
            updated = eligibility * factor
            if updated < _TRACE_FLOOR:
                stale.append(key)
            else:
                self.weights[key] = updated
        for key in stale:
            del self.weights[key]

    def clear(self) -> None:
        self.weights.clear()


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
class QLearningAgent(HeuristicAgent):
    """Q-learning agent. Subclasses ``HeuristicAgent`` to reuse its review-target
    selection and forecast helpers; only the action *type* is RL-driven."""

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
        backend_b=None,
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
        # Share one backend across agents for self-play; default to a private one.
        self.backend = backend if backend is not None else make_backend("tabular")
        self.backend_b = backend_b
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = learning
        self.trace_lambda = float(SIM.rl_trace_lambda)
        self._writing_trace = WritingTrace()
        self._writing_trace_b = WritingTrace()

        # Online-TD bookkeeping for the previous transition.
        self._last_features: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_capital: float = academic_capital
        self._last_rank: float = self._current_ac_rank()
        self._last_accrual_rate: float = self._portfolio_accrual_rate()
        self._last_holdings: dict[int, tuple[float, float, float]] = snapshot_holdings(self)
        self._last_progress_potential: float = progress_potential(self)

        # Decision cached in the marketplace phase, replayed in the work phase.
        self._pending_action: int = int(QAction.WRITE)
        self._pending_paper: Paper | None = None

    # ---- policy: one decision per timestep, made in the marketplace phase -- #
    def choose_marketplace_action(self) -> Paper | None:
        in_review = self.active_review_paper is not None
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_actions(in_review, best_paper is not None)

        # TD update for the transition that ended at this state.
        self._learn_transition(features, legal)

        action = self._select(features, legal)

        self._last_features = features
        self._last_action = int(action)
        self._remember_reward_baseline()
        self._pending_action = int(action)
        self._pending_paper = best_paper

        if action in (QAction.CLAIM, QAction.CLAIM_NEW):
            return best_paper
        return None

    def choose_work_action(self) -> tuple[str, Paper | None]:
        action = self._pending_action
        if action == QAction.CONTINUE:
            return "peer_review", self.active_review_paper
        if action == QAction.FINISH_WRITE:
            return "finish_review_write_paper", None
        return "write_paper", None

    # ---- continuous merged phase: one Q decision per timestep ------------- #
    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        in_review = self.active_review_paper is not None
        best_paper, best_share, best_ac = self._best_reviewable(
            exclude=self.active_review_paper
        )
        features = self._features(best_share, best_ac)
        legal = self._legal_continuous_actions(in_review, best_paper is not None)

        if self.learning and self._last_features is not None:
            self._learn_transition(features, legal)

        action = self._select(features, legal)

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

    def _legal_continuous_actions(
        self, in_review: bool, has_reviewable: bool
    ) -> list[int]:
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

    def _learn_transition(self, features: np.ndarray, legal: list[int]) -> None:
        """Credit the action that just finished, and earlier writing steps."""
        if not self.learning or self._last_features is None:
            return
        self._spread_return(features, legal)

    def _online_pair(self):
        """Table to update, table that scores the next action, and its trace.

        With two tables, one chooses the greedy next action and the other
        supplies its value, so a noisy high cell is not used as its own target.
        """
        if self.backend_b is None or random.random() < 0.5:
            return self.backend, self.backend_b, self._writing_trace
        return self.backend_b, self.backend, self._writing_trace_b

    def _spread_return(
        self, features: np.ndarray | None, legal: list[int] | None
    ) -> None:
        reward = self._compute_reward()
        online, offline, trace = self._online_pair()
        if features is None or legal is None:
            evaluated_next = 0.0
        else:
            q_online = online.q_values(features)
            greedy = max(legal, key=lambda action: q_online[action])
            if offline is None:
                evaluated_next = float(q_online[greedy])
            else:
                evaluated_next = float(offline.q_values(features)[greedy])
        current_q = float(online.q_values(self._last_features)[self._last_action])
        delta = reward + self.gamma * evaluated_next - current_q
        target = reward + self.gamma * evaluated_next
        writing = int(self._last_action) in WRITING_TRACE_ACTIONS
        if self.trace_lambda > 0.0 and hasattr(online, "add_error"):
            if writing:
                trace.visit(
                    online.trace_token(self._last_features),
                    int(self._last_action),
                )
                trace.apply(online, delta)
            else:
                online.update(self._last_features, self._last_action, target)
                trace.apply(online, delta)
            trace.decay(self.gamma * self.trace_lambda)
            return
        online.update(self._last_features, self._last_action, target)

    def end_episode(self) -> None:
        """Terminal flush: no future value, then drop the writing traces."""
        if self.learning and self._last_features is not None:
            self._spread_return(None, None)
        self._writing_trace.clear()
        self._writing_trace_b.clear()
        self._last_features = None
        self._last_action = None

    # ---- reward ----------------------------------------------------------- #
    def _peer_capitals(self) -> list[float]:
        return [agent.academic_capital for agent in Agent.all_agents]

    def _current_ac_rank(self) -> float:
        return ac_percentile_rank(self.academic_capital, self._peer_capitals())

    def _portfolio_accrual_rate(self) -> float:
        """Share-weighted sum of accrual rates on papers this agent owns."""
        total = 0.0
        for paper in Agent.all_papers:
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

    # ---- action helpers --------------------------------------------------- #
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

    def _select(self, features: np.ndarray, legal: list[int]) -> int:
        if self.learning and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.backend.q_values(features)
        if self.backend_b is not None:
            q = q + self.backend_b.q_values(features)
        return max(legal, key=lambda a: q[a])

    # ---- state ------------------------------------------------------------ #
    def _best_reviewable(
        self, exclude: Paper | None = None
    ) -> tuple[Paper | None, float, float]:
        """Highest-value claimable paper (reusing the heuristic score) plus its
        estimated review share and current AC."""
        reviewable = [
            p
            for p in Agent.all_papers
            if p is not exclude
            and self._can_review(p)
        ]
        if not reviewable:
            return None, 0.0, 0.0
        best = max(reviewable, key=self._score_claim)
        share = self._prospective_share(best)
        return best, share, best.current_ac

    def _features(self, best_share: float, best_ac: float) -> np.ndarray:
        """Manuscript and review progress only.

        ``best_share`` and ``best_ac`` stay in the signature because callers
        still pick a claim target. They are not part of the observation.
        """
        del best_share, best_ac
        return progress_features(
            manuscript_progress_bucket(
                self.paper_progress, self.paper_completion_threshold()
            ),
            review_progress_bucket(self.active_review_effort),
        )


class LowTalentQLearningAgent(QLearningAgent):
    """RL agent trained/evaluated at low intrinsic talent; grouped separately in History."""
