"""Single store for one simulation run: per-day metric series + the action log.

Pure standard library (no numpy/matplotlib), so recording always works. The
``Environment`` feeds it: ``record_step(env)`` once per day from ``nextstep()``,
and ``record_action(env, agent, record)`` for each agent turn from ``agentact()``.
Export with ``to_csv`` / ``to_json`` / ``to_dict``; visualize separately (see
``visualize.py``) by reading these series.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from Agent import portfolio_accrual_rate
from config import SIM
from Paper import (
    BAD_FAITH_REVIEW,
    GOOD_FAITH_REVIEW,
    REVIEW_BUMP_DECAY,
    forecast_decayed_accrual_gain,
    validate_review_bump_duration,
)

if TYPE_CHECKING:
    from Agent import ActionRecord
    from Environment import Environment

MetricFn = Callable[["Environment"], float]

COMPLETED_REVIEW_KINDS = frozenset({
    "bad_faith_review",
    "good_faith_review",
    "review_finished_write",
    "review_finished_peer_review",
    "review_stopped",
})

# Action kinds that complete a review but carry the *new* review's starting
# effort, so they must not be logged as a completed-review effort sample.
_NON_COMPLETION_REVIEW_KINDS = frozenset({"review_started"})

# A turn has exactly one unit of agent time.  These labels identify where that
# unit was spent; completion-only records (for example ``review_finished_peer_review``
# emitted immediately before a replacement claim) deliberately do not count as
# a second unit of work.
_WRITING_TIME_ACTIONS = frozenset({"write_paper", "review_finished_write"})
_REVIEW_TIME_ACTIONS = frozenset({
    "review_started",
    "review_continued",
    "bad_faith_review",
    "good_faith_review",
})

# These are derived from the action log, rather than from Environment state.
# They are initialized alongside ordinary scalar metrics so all exported series
# remain aligned with ``timesteps``.
_ACTION_DERIVED_METRICS = (
    "papers_published_this_timestep",
    "cumulative_papers_published",
    "paper_publication_rate_trailing_100",
    "supply_coverage_ratio_trailing_100",
    "market_empty_rate_trailing_100",
    "writing_effort_this_timestep",
    "writing_time_share_this_timestep",
    "review_time_share_this_timestep",
    "unallocated_time_share_this_timestep",
)


def gini(values: Iterable[float]) -> float:
    """Gini coefficient of non-negative values (0 = perfectly equal, →1 = unequal)."""
    nonneg = sorted(max(0.0, float(v)) for v in values)
    n = len(nonneg)
    total = sum(nonneg)
    if n == 0 or total == 0.0:
        return 0.0
    weighted = sum(i * value for i, value in enumerate(nonneg, start=1))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _papers_listed_this_timestep(env: "Environment") -> float:
    return float(
        sum(
            1
            for paper in env.papers
            if getattr(paper, "listed_timestep", None) == env.timestep
        )
    )


def _papers_claimed_this_timestep(env: "Environment") -> float:
    return float(
        sum(
            1
            for paper in env.papers
            if getattr(paper, "claimed_timestep", None) == env.timestep
        )
    )


def _papers_claimed_same_timestep(env: "Environment") -> float:
    return float(
        sum(
            1
            for paper in env.papers
            if getattr(paper, "claimed_timestep", None) is not None
            and getattr(paper, "claimed_timestep", None)
            == getattr(paper, "listed_timestep", None)
        )
    )


def _mean_time_on_market_claimed(env: "Environment") -> float:
    waits = [
        float(wait)
        for paper in env.papers
        for wait in [getattr(paper, "time_on_market_timesteps", None)]
        if wait is not None
    ]
    return sum(waits) / len(waits) if waits else 0.0


def _instant_claim_rate(env: "Environment") -> float:
    claimed = [
        paper
        for paper in env.papers
        if getattr(paper, "claimed_timestep", None) is not None
    ]
    if not claimed:
        return 0.0
    instant = sum(
        1
        for paper in claimed
        if getattr(paper, "claimed_timestep", None)
        == getattr(paper, "listed_timestep", None)
    )
    return instant / len(claimed)


def _mean_author_price_multiplier(env: "Environment") -> float:
    values = [
        float(getattr(agent, "review_offer_multiplier", 1.0))
        for agent in env.agents
    ]
    return sum(values) / len(values) if values else 1.0


def _papers_in_review(env: "Environment") -> float:
    """Papers already claimed but whose review has not yet finished."""
    return float(
        sum(
            1
            for paper in env.papers
            if getattr(paper, "review_claimed", False)
            and not getattr(paper, "reviewed", False)
        )
    )


def _review_backlog_total(env: "Environment") -> float:
    """All listed review work, whether still open or currently in progress."""
    return float(
        sum(
            1
            for paper in env.papers
            if (
                getattr(paper, "review_available", False)
                or (
                    getattr(paper, "review_claimed", False)
                    and not getattr(paper, "reviewed", False)
                )
            )
        )
    )


def review_benefit_at_completion(
    review_record: dict[str, Any],
    *,
    horizon_timesteps: float | None = None,
) -> dict[str, float] | None:
    """Per-review surplus at settlement using the agent forecast horizon.

    Unlike the cumulative ``author_net_*`` scalars (which re-value every past
    review with each paper's *current* AC), this freezes the trade at completion:
    incremental bump value minus the reviewer's share of forecast paper AC over
    ``horizon_timesteps`` (default ``SIM.forecast_horizon_timesteps``).
    """
    share = float(review_record.get("share", 0.0))
    if share <= 0.0:
        return None

    horizon = max(
        0.0,
        float(
            horizon_timesteps
            if horizon_timesteps is not None
            else SIM.forecast_horizon_timesteps
        ),
    )
    epsilon = float(review_record.get("epsilon", 0.0))
    a0 = float(review_record.get("current_ac_at_review", 0.0))
    base_rate_raw = review_record.get("base_accrual_rate")
    if base_rate_raw is not None:
        base_rate = float(base_rate_raw)
    else:
        accrual_rate = float(review_record.get("accrual_rate", 0.0))
        base_rate = (
            accrual_rate / (1.0 + epsilon) if epsilon > 0.0 else accrual_rate
        )

    baseline_ac = a0 + base_rate * horizon
    bumped_ac = a0 + forecast_decayed_accrual_gain(base_rate, epsilon, horizon)
    value_created = max(0.0, bumped_ac - baseline_ac)
    reviewer_benefit = share * bumped_ac
    author_net = value_created - reviewer_benefit
    return {
        "author_net": author_net,
        "reviewer_benefit": reviewer_benefit,
        "value_created": value_created,
    }


def _review_benefit_components(env: "Environment") -> dict[str, float]:
    """Good/bad-faith surplus split of every completed review, in AC units.

    For each reviewed paper, a review is a trade: the reviewer takes a ``share``
    of the *whole* paper (past and future AC), while the author gains an accrual
    bump (``epsilon``) on the paper's *future* accrual only. Relative to the
    no-review counterfactual:

        reviewer_benefit = share * final_ac
        value_created    = (final_ac - A0) * epsilon / (1 + epsilon)
        author_net       = value_created - reviewer_benefit   # < 0 => exploited

    where ``A0`` (``current_ac_at_review``) is the paper's AC at the instant the
    review finished, so ``final_ac - A0`` is exactly the post-review accrual (all
    at the bumped rate) and ``epsilon / (1 + epsilon)`` is the fraction of it the
    bump created. Sums are split by the review's good/bad-faith classification.
    """
    totals = {
        "reviewer_benefit_good": 0.0,
        "reviewer_benefit_bad": 0.0,
        "author_net_good": 0.0,
        "author_net_bad": 0.0,
        "value_created_good": 0.0,
        "value_created_bad": 0.0,
    }
    for paper in env.papers:
        records = getattr(paper, "review_records", None)
        if not records:
            continue
        final_ac = float(getattr(paper, "current_ac", 0.0))
        for record in records:
            share = float(record.get("share", 0.0))
            if share <= 0.0:
                # Sub-threshold reviews transfer no share and create no value.
                continue
            epsilon = float(record.get("epsilon", 0.0))
            a0 = float(record.get("current_ac_at_review", final_ac))
            reviewer_benefit = share * final_ac
            bump_duration = str(
                record.get("bump_duration", SIM.review_bump_duration)
            ).strip().lower()
            if (
                bump_duration == REVIEW_BUMP_DECAY
                or validate_review_bump_duration(bump_duration) == REVIEW_BUMP_DECAY
            ):
                base_rate = record.get("base_accrual_rate")
                review_t = record.get("review_completed_timestep")
                if base_rate is not None and review_t is not None:
                    elapsed = max(0, int(env.timestep) - int(review_t))
                    base_only = a0 + float(base_rate) * elapsed
                    value_created = max(0.0, final_ac - base_only)
                elif epsilon > 0.0:
                    value_created = (
                        max(0.0, final_ac - a0) * epsilon / (1.0 + epsilon)
                    )
                else:
                    value_created = 0.0
            else:
                value_created = (
                    max(0.0, final_ac - a0) * epsilon / (1.0 + epsilon)
                    if epsilon > 0.0
                    else 0.0
                )
            author_net = value_created - reviewer_benefit
            suffix = (
                "good" if record.get("review_kind") == GOOD_FAITH_REVIEW else "bad"
            )
            totals[f"reviewer_benefit_{suffix}"] += reviewer_benefit
            totals[f"value_created_{suffix}"] += value_created
            totals[f"author_net_{suffix}"] += author_net
    return totals


def _ac_by_source(env: "Environment") -> dict[str, float]:
    """Split current academic capital by how it was earned: writing vs reviewing.

    Every unit of AC is held by exactly one of a paper's shareholders. The
    paper's author holds AC earned by *writing*; any other shareholder holds AC
    earned by *peer reviewing* (their granted review share of the paper). The two
    sums add up to the system's ``total_capital``.
    """
    agent_set = set(env.agents)
    writing = 0.0
    review = 0.0
    for paper in env.papers:
        author = getattr(paper, "author", None)
        current_ac = float(getattr(paper, "current_ac", 0.0))
        for agent, share in paper.share_distribution.items():
            if agent not in agent_set:
                continue
            value = float(share) * current_ac
            if agent is author:
                writing += value
            else:
                review += value
    return {"writing_held_ac": writing, "review_held_ac": review}


def default_metrics() -> dict[str, MetricFn]:
    """Scalar, per-timestep metrics recorded by default (aggregates + review behavior)."""
    return {
        "total_capital": lambda env: sum(a.academic_capital for a in env.agents),
        "mean_capital": lambda env: (
            sum(a.academic_capital for a in env.agents) / len(env.agents)
            if env.agents
            else 0.0
        ),
        "max_capital": lambda env: max(
            (a.academic_capital for a in env.agents), default=0.0
        ),
        "capital_gini": lambda env: gini(a.academic_capital for a in env.agents),
        "num_papers": lambda env: float(len(env.papers)),
        "papers_on_market": lambda env: float(
            sum(1 for p in env.papers if getattr(p, "review_available", False))
        ),
        "papers_listed_this_timestep": _papers_listed_this_timestep,
        "papers_claimed_this_timestep": _papers_claimed_this_timestep,
        "papers_claimed_same_timestep": _papers_claimed_same_timestep,
        "mean_time_on_market_claimed": _mean_time_on_market_claimed,
        "instant_claim_rate": _instant_claim_rate,
        # ``papers_on_market`` is the open queue only.  These two distinguish
        # claimed work in progress from the total review-work pipeline.
        "papers_in_review": _papers_in_review,
        "review_backlog_total": _review_backlog_total,
        "completed_peer_reviews": lambda env: float(
            sum(getattr(p, "completed_peer_reviews", 0) for p in env.papers)
        ),
        "good_faith_reviews": lambda env: float(
            sum(
                1
                for p in env.papers
                for record in getattr(p, "review_records", [])
                if record.get("review_kind") == GOOD_FAITH_REVIEW
            )
        ),
        "bad_faith_reviews": lambda env: float(
            sum(
                1
                for p in env.papers
                for record in getattr(p, "review_records", [])
                if record.get("review_kind") == BAD_FAITH_REVIEW
            )
        ),
        "mean_peer_review_history": lambda env: (
            sum(getattr(a, "peer_review_history", 0.0) for a in env.agents)
            / len(env.agents)
            if env.agents
            else 0.0
        ),
        "mean_peer_review_epsilon": lambda env: (
            sum(getattr(a, "peer_review_epsilon_history", 0.0) for a in env.agents)
            / len(env.agents)
            if env.agents
            else 0.0
        ),
        "fair_market_price": lambda env: float(
            getattr(env, "fair_market_price", 0.0)
        ),
        "reviewer_pressure": lambda env: float(
            getattr(env, "reviewer_pressure", 0.0)
        ),
        "scarcity_multiplier": lambda env: float(
            getattr(env, "scarcity_multiplier", 1.0)
        ),
        "market_assignments": lambda env: float(
            getattr(env, "market_assignments", 0)
        ),
        "mean_author_price_multiplier": _mean_author_price_multiplier,
        # Reviewer-vs-author benefit split (AC), by good/bad faith. See
        # ``_review_benefit_components`` for the surplus decomposition.
        "reviewer_benefit_good": (
            lambda env: _review_benefit_components(env)["reviewer_benefit_good"]
        ),
        "reviewer_benefit_bad": (
            lambda env: _review_benefit_components(env)["reviewer_benefit_bad"]
        ),
        "author_net_good": (
            lambda env: _review_benefit_components(env)["author_net_good"]
        ),
        "author_net_bad": (
            lambda env: _review_benefit_components(env)["author_net_bad"]
        ),
        "value_created_good": (
            lambda env: _review_benefit_components(env)["value_created_good"]
        ),
        "value_created_bad": (
            lambda env: _review_benefit_components(env)["value_created_bad"]
        ),
        # Total AC currently held by source: writing (author shares) vs peer
        # review (reviewer shares). Sums to ``total_capital``.
        "writing_held_ac": lambda env: _ac_by_source(env)["writing_held_ac"],
        "review_held_ac": lambda env: _ac_by_source(env)["review_held_ac"],
    }


def mean_completed_review_effort(completed_reviews: list[tuple]) -> float:
    """Running mean effort across all finished reviews logged so far."""
    efforts = [float(row[3]) for row in completed_reviews if row[3] is not None]
    return sum(efforts) / len(efforts) if efforts else 0.0


class History:
    """Time-series + action log for a run.

    Series are kept aligned to ``self.days``: every agent/paper series has the
    same length, with papers that appear mid-run back-filled with ``0.0`` for the
    days before they existed.
    """

    def __init__(
        self,
        metrics: dict[str, MetricFn] | None = None,
        *,
        track_agents: bool = True,
        track_papers: bool = True,
    ):
        self.metrics = default_metrics() if metrics is None else dict(metrics)
        self.track_agents = track_agents
        self.track_papers = track_papers

        self.timesteps: list[int] = []
        self.scalars: dict[str, list[float]] = {name: [] for name in self.metrics}
        for name in _ACTION_DERIVED_METRICS:
            self.scalars.setdefault(name, [])
        self.agent_capital: dict[str, list[float]] = {}
        self.agent_accrual_rate: dict[str, list[float]] = {}
        self.agent_review_history: dict[str, list[float]] = {}
        self.agent_review_epsilon_history: dict[str, list[float]] = {}
        self.agent_groups: dict[str, str] = {}  # agent label -> class name
        self.agent_talent: dict[str, float] = {}
        self.paper_ac: dict[str, list[float]] = {}
        # Per-paper attributes (constant or final snapshot) for outcome charts.
        self.paper_quality: dict[str, float] = {}
        self.paper_authors: dict[str, str] = {}
        self.paper_reviewed: dict[str, bool] = {}
        # Reviewer ownership per paper, for the writing-vs-peer-review AC split.
        self.paper_reviewer: dict[str, str] = {}
        self.paper_reviewer_share: dict[str, float] = {}
        self.paper_writing_effort: dict[str, float] = {}
        self.paper_required_writing_effort: dict[str, float] = {}
        self.paper_accrual_rate: dict[str, float] = {}
        self.paper_first_seen_timestep: dict[str, int] = {}
        self.paper_listed_timestep: dict[str, int] = {}
        self.paper_claimed_timestep: dict[str, int] = {}
        self.paper_time_on_market: dict[str, int] = {}

        # Action log: one entry per agent turn.
        self.actions: list[tuple[int, str, str, str | None]] = []
        self.completed_reviews: list[
            tuple[int, str, str | None, float, str | None]
        ] = []
        self.writing_efforts: list[tuple[int, str, float, bool]] = []
        # (timestep, agreed reviewer share) for each accepted marketplace claim.
        self.accepted_review_claims: list[tuple[int, float]] = []
        # Per completed review: (timestep, faith suffix, author_net, reviewer_benefit,
        # value_created, share). Settlement uses ``review_benefit_at_completion``.
        self.review_benefit_events: list[
            tuple[int, str, float, float, float, float]
        ] = []
        self.action_counts: Counter[str] = Counter()
        self.agent_actions: dict[str, list[str]] = {}

        self._labels: dict[int, str] = {}
        self._used_labels: set[str] = set()
        self._agent_counter = 0
        self._paper_counter = 0

    @property
    def days(self) -> list[int]:
        """Backwards-compatible alias for the timestep axis."""
        return self.timesteps

    # ---- recording -------------------------------------------------------
    def record_step(self, env: "Environment") -> None:
        """Snapshot per-timestep metric series. Called from ``run_timestep()``."""
        self.timesteps.append(env.timestep)
        for name, fn in self.metrics.items():
            self.scalars[name].append(float(fn(env)))
        self._record_action_derived_metrics(env)
        self.scalars.setdefault("mean_completed_review_effort", []).append(
            mean_completed_review_effort(self.completed_reviews)
        )
        if self.track_agents:
            self._record_series(
                env.agents,
                self.agent_capital,
                lambda a: float(getattr(a, "academic_capital", 0.0)),
                "Agent",
            )
            self._record_series(
                env.agents,
                self.agent_accrual_rate,
                lambda a: portfolio_accrual_rate(a, env.papers),
                "Agent",
            )
            self._record_series(
                env.agents,
                self.agent_review_history,
                lambda a: float(getattr(a, "peer_review_history", 0.0)),
                "Agent",
            )
            self._record_series(
                env.agents,
                self.agent_review_epsilon_history,
                lambda a: float(getattr(a, "peer_review_epsilon_history", 0.0)),
                "Agent",
            )
            for agent in env.agents:
                label = self._label(agent, "Agent")
                self.agent_talent[label] = float(getattr(agent, "intrinsic_talent", 0.0))
        if self.track_papers:
            self._record_series(
                env.papers,
                self.paper_ac,
                lambda p: float(getattr(p, "current_ac", 0.0)),
                "Paper",
            )
            for paper in env.papers:
                label = self._label(paper, "Paper")
                self.paper_first_seen_timestep.setdefault(label, env.timestep)
                self.paper_authors[label] = self._label(paper.author, "Agent")
                self.paper_quality[label] = float(getattr(paper, "quality", 0.0))
                self.paper_reviewed[label] = bool(getattr(paper, "reviewed", False))
                reviewer = getattr(paper, "reviewer", None)
                if reviewer is not None:
                    self.paper_reviewer[label] = self._label(reviewer, "Agent")
                    self.paper_reviewer_share[label] = float(
                        paper.share_distribution.get(reviewer, 0.0)
                    )
                effort = getattr(paper, "writing_effort", None)
                if effort is not None:
                    self.paper_writing_effort[label] = float(effort)
                required = getattr(paper, "required_writing_effort", None)
                if required is not None:
                    self.paper_required_writing_effort[label] = float(required)
                self.paper_accrual_rate[label] = float(
                    getattr(paper, "accrual_rate", 0.0)
                )
                listed = getattr(paper, "listed_timestep", None)
                if listed is not None:
                    self.paper_listed_timestep[label] = int(listed)
                claimed = getattr(paper, "claimed_timestep", None)
                if claimed is not None:
                    self.paper_claimed_timestep[label] = int(claimed)
                wait = getattr(paper, "time_on_market_timesteps", None)
                if wait is not None:
                    self.paper_time_on_market[label] = int(wait)

    def _record_action_derived_metrics(self, env: "Environment") -> None:
        """Record publication and time-allocation diagnostics for this timestep.

        The environment records action outcomes before it calls ``record_step``.
        Using those outcomes makes publication counts exact (including a paper
        finished while ending a review) and prevents review progress records from
        being mistaken for cumulative review effort.
        """
        timestep = env.timestep
        turn_actions: dict[str, str] = {}
        for day, agent_label, kind, _ in self.actions:
            if day == timestep:
                # Some continuous actions emit a completion record followed by
                # the actual new unit of work. The final actionable record is
                # the one that consumes this agent's timestep.
                turn_actions[agent_label] = kind

        published = sum(
            1
            for day, _, _, was_published in self.writing_efforts
            if day == timestep and was_published
        )
        writing_effort = sum(
            effort
            for day, _, effort, _ in self.writing_efforts
            if day == timestep
        )
        writer_count = sum(
            1 for kind in turn_actions.values() if kind in _WRITING_TIME_ACTIONS
        )
        reviewer_count = sum(
            1 for kind in turn_actions.values() if kind in _REVIEW_TIME_ACTIONS
        )
        agent_count = len(env.agents)
        denominator = max(1, agent_count)

        self.scalars["papers_published_this_timestep"].append(float(published))
        previous_total = self.scalars["cumulative_papers_published"][-1] if (
            self.scalars["cumulative_papers_published"]
        ) else 0.0
        self.scalars["cumulative_papers_published"].append(previous_total + published)
        publication_series = self.scalars["papers_published_this_timestep"]
        window = publication_series[-100:]
        self.scalars["paper_publication_rate_trailing_100"].append(
            sum(window) / len(window) if window else 0.0
        )
        claims = self.scalars.get("papers_claimed_this_timestep", [])
        claimed_window = claims[-100:]
        claimed_total = sum(claimed_window)
        # A ratio of 1 denotes balanced realised flow. With no claims there is
        # no observed review demand, so record 0 rather than inventing balance.
        self.scalars["supply_coverage_ratio_trailing_100"].append(
            sum(window) / claimed_total if claimed_total > 0.0 else 0.0
        )
        market_series = self.scalars.get("papers_on_market", [])
        empty_now = 1.0 if not market_series or market_series[-1] <= 0.0 else 0.0
        empties = self.scalars["market_empty_rate_trailing_100"][-99:] + [empty_now]
        self.scalars["market_empty_rate_trailing_100"].append(
            sum(empties) / len(empties)
        )
        self.scalars["writing_effort_this_timestep"].append(float(writing_effort))
        self.scalars["writing_time_share_this_timestep"].append(
            writer_count / denominator
        )
        self.scalars["review_time_share_this_timestep"].append(
            reviewer_count / denominator
        )
        self.scalars["unallocated_time_share_this_timestep"].append(
            max(0.0, (agent_count - writer_count - reviewer_count) / denominator)
        )

    def record_action(self, env: "Environment", agent: Any, record: "ActionRecord") -> None:
        """Log one agent turn. Called during a timestep's marketplace/work phases,
        so the action belongs to the timestep currently being simulated."""
        timestep = env.timestep
        agent_label = self._label(agent, "Agent")
        paper_label = (
            self._label(record.paper, "Paper") if record.paper is not None else None
        )
        self.actions.append((timestep, agent_label, record.kind, paper_label))
        self.action_counts[record.kind] += 1
        if record.kind == "review_started" and record.paper is not None:
            price = float(getattr(record.paper, "agreed_review_share", 0.0))
            if price > 0.0:
                self.accepted_review_claims.append((timestep, price))
        if (
            record.review_effort is not None
            and record.kind in COMPLETED_REVIEW_KINDS
            and record.kind not in _NON_COMPLETION_REVIEW_KINDS
        ):
            effort = float(record.review_effort)
            review_kind = record.review_kind
            # Record every finished review, including early stops below the
            # reward threshold, so the effort distribution shows where agents
            # actually choose to stop. The reward cliff (sub-threshold reviews
            # earn nothing) lives in Paper, not in this recording gate.
            if effort > 0:
                self.completed_reviews.append(
                    (timestep, agent_label, paper_label, effort, review_kind)
                )
                if record.paper is not None:
                    records = getattr(record.paper, "review_records", None)
                    if records:
                        latest = records[-1]
                        benefit = review_benefit_at_completion(latest)
                        if benefit is not None:
                            kind = review_kind or latest.get("review_kind")
                            faith = (
                                "good"
                                if kind == GOOD_FAITH_REVIEW
                                else "bad"
                            )
                            self.review_benefit_events.append(
                                (
                                    timestep,
                                    faith,
                                    benefit["author_net"],
                                    benefit["reviewer_benefit"],
                                    benefit["value_created"],
                                    float(latest.get("share", 0.0)),
                                )
                            )
        if record.writing_effort is not None:
            self.writing_efforts.append(
                (
                    timestep,
                    agent_label,
                    float(record.writing_effort),
                    bool(record.published),
                )
            )
        suffix = f" of {paper_label}" if paper_label else ""
        self.agent_actions.setdefault(agent_label, []).append(
            f"timestep {timestep}: {record.kind}{suffix}"
        )

    def _record_series(
        self,
        entities: Iterable[Any],
        store: dict[str, list[float]],
        value_fn: Callable[[Any], float],
        prefix: str,
    ) -> None:
        target_len = len(self.days)
        for entity in entities:
            label = self._label(entity, prefix)
            if prefix == "Agent":
                self.agent_groups[label] = type(entity).__name__
            series = store.get(label)
            if series is None:
                series = [0.0] * (target_len - 1)  # back-fill days before it existed
                store[label] = series
            series.append(value_fn(entity))

    def _label(self, obj: Any, prefix: str) -> str:
        """Stable, unique display label for an agent/paper, cached by object id."""
        key = id(obj)
        cached = self._labels.get(key)
        if cached is not None:
            return cached

        if prefix == "Agent":
            self._agent_counter += 1
            default = f"Agent {self._agent_counter}"
        else:
            self._paper_counter += 1
            default = f"Paper {self._paper_counter}"

        name = getattr(obj, "name", None) or getattr(obj, "title", None) or default
        base, n = name, 2
        while name in self._used_labels:  # defend against duplicate names/titles
            name = f"{base} ({n})"
            n += 1

        self._used_labels.add(name)
        self._labels[key] = name
        return name

    # ---- export ----------------------------------------------------------
    def to_dict(self, *, max_paper_series: int | None = None) -> dict[str, Any]:
        # ``days``/``day`` are kept as aliases of the timestep axis so the static
        # gallery (which reads older runs too) keeps working unchanged.
        #
        # ``max_paper_series`` slims the gallery payload: the full per-paper AC
        # time series (by far the heaviest field for large runs) is kept only for
        # the highest-final-AC papers, while ``paper_final_ac`` always carries one
        # final value per paper so the quality-vs-AC scatter stays complete. Pass
        # ``None`` (the default) for a full, lossless export.
        paper_ac_full = {k: list(v) for k, v in self.paper_ac.items()}
        paper_final_ac = {
            k: (v[-1] if v else 0.0) for k, v in paper_ac_full.items()
        }
        if max_paper_series is not None and len(paper_ac_full) > max_paper_series:
            top = sorted(
                paper_ac_full,
                key=lambda k: paper_final_ac[k],
                reverse=True,
            )[:max_paper_series]
            paper_ac_out = {k: paper_ac_full[k] for k in top}
        else:
            paper_ac_out = paper_ac_full

        return {
            "timesteps": list(self.timesteps),
            "days": list(self.timesteps),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "agent_accrual_rate": {k: list(v) for k, v in self.agent_accrual_rate.items()},
            "agent_review_history": {
                k: list(v) for k, v in self.agent_review_history.items()
            },
            "agent_review_epsilon_history": {
                k: list(v) for k, v in self.agent_review_epsilon_history.items()
            },
            "agent_groups": dict(self.agent_groups),
            "agent_talent": dict(self.agent_talent),
            "paper_ac": paper_ac_out,
            "paper_final_ac": paper_final_ac,
            "paper_authors": dict(self.paper_authors),
            "paper_quality": dict(self.paper_quality),
            "paper_reviewed": dict(self.paper_reviewed),
            "paper_writing_effort": dict(self.paper_writing_effort),
            "paper_required_writing_effort": dict(self.paper_required_writing_effort),
            "paper_accrual_rate": dict(self.paper_accrual_rate),
            "paper_first_seen_timestep": dict(self.paper_first_seen_timestep),
            "paper_listed_timestep": dict(self.paper_listed_timestep),
            "paper_claimed_timestep": dict(self.paper_claimed_timestep),
            "paper_time_on_market": dict(self.paper_time_on_market),
            "actions": [
                {"timestep": d, "day": d, "agent": a, "kind": k, "paper": p}
                for (d, a, k, p) in self.actions
            ],
            "completed_reviews": [
                {
                    "timestep": d,
                    "day": d,
                    "agent": a,
                    "paper": p,
                    "effort": e,
                    "review_kind": k,
                }
                for (d, a, p, e, k) in self.completed_reviews
            ],
            "writing_efforts": [
                {
                    "timestep": d,
                    "day": d,
                    "agent": a,
                    "effort": e,
                    "published": p,
                }
                for (d, a, e, p) in self.writing_efforts
            ],
            "accepted_review_claims": [
                {"timestep": d, "day": d, "price": p}
                for (d, p) in self.accepted_review_claims
            ],
            "review_benefit_events": [
                {
                    "timestep": d,
                    "day": d,
                    "faith": faith,
                    "author_net": author_net,
                    "reviewer_benefit": reviewer_benefit,
                    "value_created": value_created,
                    "share": share,
                }
                for (
                    d,
                    faith,
                    author_net,
                    reviewer_benefit,
                    value_created,
                    share,
                ) in self.review_benefit_events
            ],
            "action_counts": dict(self.action_counts),
            "action_counts_by_timestep": self.action_counts_by_timestep(),
            "agent_group_summary": self.agent_group_summary(),
            "agent_outcome_summary": self.agent_outcome_summary(),
        }

    def _ac_from_reviewing_by_agent(self) -> dict[str, float]:
        """Final AC each agent holds via reviewer shares (earned by peer review)."""
        held: dict[str, float] = {}
        for paper_label, reviewer_label in self.paper_reviewer.items():
            share = self.paper_reviewer_share.get(paper_label, 0.0)
            series = self.paper_ac.get(paper_label, [])
            final_ac = float(series[-1]) if series else 0.0
            held[reviewer_label] = held.get(reviewer_label, 0.0) + share * final_ac
        return held

    def agent_group_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate final outcomes by concrete agent class."""
        reviewer_held = self._ac_from_reviewing_by_agent()
        groups: dict[str, dict[str, Any]] = {}

        def ensure(group: str) -> dict[str, Any]:
            if group not in groups:
                groups[group] = {
                    "agent_count": 0,
                    "total_final_capital": 0.0,
                    "min_final_capital": None,
                    "max_final_capital": None,
                    "papers_authored": 0,
                    "completed_reviews": 0,
                    "good_faith_reviews": 0,
                    "bad_faith_reviews": 0,
                    "total_review_effort": 0.0,
                    "total_peer_review_history": 0.0,
                    "total_peer_review_epsilon": 0.0,
                    "total_writing_effort": 0.0,
                    "total_ac_from_writing": 0.0,
                    "total_ac_from_reviewing": 0.0,
                    "actions": Counter(),
                }
            return groups[group]

        for agent_label, series in self.agent_capital.items():
            group = self.agent_groups.get(agent_label, "Agent")
            stats = ensure(group)
            final_capital = float(series[-1]) if series else 0.0
            stats["agent_count"] += 1
            stats["total_final_capital"] += final_capital
            current_min = stats["min_final_capital"]
            current_max = stats["max_final_capital"]
            stats["min_final_capital"] = (
                final_capital if current_min is None else min(current_min, final_capital)
            )
            stats["max_final_capital"] = (
                final_capital if current_max is None else max(current_max, final_capital)
            )

            from_reviewing = float(reviewer_held.get(agent_label, 0.0))
            stats["total_ac_from_reviewing"] += from_reviewing
            stats["total_ac_from_writing"] += max(0.0, final_capital - from_reviewing)

            review_series = self.agent_review_history.get(agent_label, [])
            if review_series:
                stats["total_peer_review_history"] += float(review_series[-1])
            epsilon_series = self.agent_review_epsilon_history.get(agent_label, [])
            if epsilon_series:
                stats["total_peer_review_epsilon"] += float(epsilon_series[-1])

        for author_label in self.paper_authors.values():
            group = self.agent_groups.get(author_label, "Agent")
            ensure(group)["papers_authored"] += 1

        for _, agent_label, _, effort, review_kind in self.completed_reviews:
            group = self.agent_groups.get(agent_label, "Agent")
            stats = ensure(group)
            stats["completed_reviews"] += 1
            stats["total_review_effort"] += float(effort)
            if review_kind == GOOD_FAITH_REVIEW:
                stats["good_faith_reviews"] += 1
            elif review_kind == BAD_FAITH_REVIEW:
                stats["bad_faith_reviews"] += 1

        for _, agent_label, effort, _ in self.writing_efforts:
            group = self.agent_groups.get(agent_label, "Agent")
            ensure(group)["total_writing_effort"] += float(effort)

        for _, agent_label, kind, _ in self.actions:
            group = self.agent_groups.get(agent_label, "Agent")
            ensure(group)["actions"][kind] += 1

        output: dict[str, dict[str, Any]] = {}
        for group, stats in groups.items():
            count = stats["agent_count"]
            reviews = stats["completed_reviews"]
            summary = dict(stats)
            summary["mean_final_capital"] = (
                stats["total_final_capital"] / count if count else 0.0
            )
            summary["mean_peer_review_history"] = (
                stats["total_peer_review_history"] / count if count else 0.0
            )
            summary["mean_peer_review_epsilon"] = (
                stats["total_peer_review_epsilon"] / count if count else 0.0
            )
            summary["average_review_effort"] = (
                stats["total_review_effort"] / reviews if reviews else 0.0
            )
            summary["good_faith_review_rate"] = (
                stats["good_faith_reviews"] / reviews if reviews else 0.0
            )
            summary["mean_good_faith_reviews"] = (
                stats["good_faith_reviews"] / count if count else 0.0
            )
            summary["mean_bad_faith_reviews"] = (
                stats["bad_faith_reviews"] / count if count else 0.0
            )
            summary["actions"] = dict(stats["actions"])
            if summary["min_final_capital"] is None:
                summary["min_final_capital"] = 0.0
            if summary["max_final_capital"] is None:
                summary["max_final_capital"] = 0.0
            output[group] = summary
        return output

    def agent_outcome_summary(self) -> list[dict[str, Any]]:
        """Per-agent final outcomes for interpreting top/bottom performers."""
        reviewer_held = self._ac_from_reviewing_by_agent()
        papers_authored = Counter(self.paper_authors.values())
        paper_quality_sum: Counter[str] = Counter()
        paper_effort_sum: Counter[str] = Counter()
        paper_required_sum: Counter[str] = Counter()
        review_counts: Counter[str] = Counter()
        good_counts: Counter[str] = Counter()
        bad_counts: Counter[str] = Counter()
        review_effort: Counter[str] = Counter()
        writing_effort: Counter[str] = Counter()
        actions_by_agent: dict[str, Counter[str]] = {}

        for _, agent_label, _, effort, review_kind in self.completed_reviews:
            review_counts[agent_label] += 1
            review_effort[agent_label] += float(effort)
            if review_kind == GOOD_FAITH_REVIEW:
                good_counts[agent_label] += 1
            elif review_kind == BAD_FAITH_REVIEW:
                bad_counts[agent_label] += 1

        for _, agent_label, effort, _ in self.writing_efforts:
            writing_effort[agent_label] += float(effort)

        for _, agent_label, kind, _ in self.actions:
            actions_by_agent.setdefault(agent_label, Counter())[kind] += 1

        for paper_label, author_label in self.paper_authors.items():
            if paper_label in self.paper_quality:
                paper_quality_sum[author_label] += float(self.paper_quality[paper_label])
            if paper_label in self.paper_writing_effort:
                paper_effort_sum[author_label] += float(
                    self.paper_writing_effort[paper_label]
                )
            if paper_label in self.paper_required_writing_effort:
                paper_required_sum[author_label] += float(
                    self.paper_required_writing_effort[paper_label]
                )

        agent_labels = set(self.agent_capital)
        agent_labels.update(self.agent_groups)
        agent_labels.update(papers_authored)
        agent_labels.update(review_counts)
        agent_labels.update(writing_effort)
        agent_labels.update(actions_by_agent)

        rows: list[dict[str, Any]] = []
        for agent_label in sorted(agent_labels):
            capital_series = self.agent_capital.get(agent_label, [])
            review_series = self.agent_review_history.get(agent_label, [])
            epsilon_series = self.agent_review_epsilon_history.get(agent_label, [])
            reviews = int(review_counts[agent_label])
            action_counter = actions_by_agent.get(agent_label, Counter())
            final_capital = float(capital_series[-1]) if capital_series else 0.0
            ac_from_reviewing = float(reviewer_held.get(agent_label, 0.0))
            rows.append({
                "agent": agent_label,
                "group": self.agent_groups.get(agent_label, "Agent"),
                "talent": float(self.agent_talent.get(agent_label, 0.0)),
                "final_capital": final_capital,
                "ac_from_writing": max(0.0, final_capital - ac_from_reviewing),
                "ac_from_reviewing": ac_from_reviewing,
                "papers_authored": int(papers_authored[agent_label]),
                "average_paper_quality": (
                    float(paper_quality_sum[agent_label]) / papers_authored[agent_label]
                    if papers_authored[agent_label]
                    else 0.0
                ),
                "average_paper_writing_effort": (
                    float(paper_effort_sum[agent_label]) / papers_authored[agent_label]
                    if papers_authored[agent_label]
                    else 0.0
                ),
                "average_required_writing_effort": (
                    float(paper_required_sum[agent_label]) / papers_authored[agent_label]
                    if papers_authored[agent_label]
                    else 0.0
                ),
                "completed_reviews": reviews,
                "good_faith_reviews": int(good_counts[agent_label]),
                "bad_faith_reviews": int(bad_counts[agent_label]),
                "total_review_effort": float(review_effort[agent_label]),
                "average_review_effort": (
                    float(review_effort[agent_label]) / reviews if reviews else 0.0
                ),
                "total_writing_effort": float(writing_effort[agent_label]),
                "peer_review_history": (
                    float(review_series[-1]) if review_series else 0.0
                ),
                "peer_review_epsilon": (
                    float(epsilon_series[-1]) if epsilon_series else 0.0
                ),
                "actions": dict(action_counter),
                "most_common_actions": [
                    {"kind": kind, "count": int(count)}
                    for kind, count in action_counter.most_common(3)
                ],
            })

        rows.sort(key=lambda row: row["final_capital"], reverse=True)
        return rows

    def to_gallery_dict(self, *, max_actions: int | None = None) -> dict[str, Any]:
        """Slim payload for the static gallery.

        The committed gallery renders the heavy static charts from PNGs (see
        ``visualize.render_gallery_charts``), so this carries only what the
        *animated* charts, the action feed/replay, and the side tables need:
        per-agent capital, scalar series, the decisions/action log, group
        summary, and each agent's final reputation (for the ranking table).
        The full, lossless history is kept locally in ``local_data/``.
        """
        actions = list(self.actions)
        total_actions = len(actions)
        actions_truncated = False
        if max_actions is not None and max_actions >= 0 and total_actions > max_actions:
            actions = actions[-max_actions:]
            actions_truncated = True

        return {
            "timesteps": list(self.timesteps),
            "days": list(self.timesteps),
            "scalars": {k: list(v) for k, v in self.scalars.items()},
            "agent_capital": {k: list(v) for k, v in self.agent_capital.items()},
            "agent_accrual_rate": {k: list(v) for k, v in self.agent_accrual_rate.items()},
            "agent_groups": dict(self.agent_groups),
            "agent_talent": dict(self.agent_talent),
            "accepted_review_claims": [
                {"timestep": d, "day": d, "price": p}
                for (d, p) in self.accepted_review_claims
            ],
            "agent_group_summary": self.agent_group_summary(),
            "agent_outcome_summary": self.agent_outcome_summary(),
            "action_counts": dict(self.action_counts),
            "total_actions": total_actions,
            "actions_truncated": actions_truncated,
            "gallery_action_limit": max_actions,
            "paper_authors": dict(self.paper_authors),
            "paper_quality": dict(self.paper_quality),
            "paper_writing_effort": dict(self.paper_writing_effort),
            "paper_required_writing_effort": dict(self.paper_required_writing_effort),
            "paper_accrual_rate": dict(self.paper_accrual_rate),
            "paper_first_seen_timestep": dict(self.paper_first_seen_timestep),
            "paper_listed_timestep": dict(self.paper_listed_timestep),
            "paper_claimed_timestep": dict(self.paper_claimed_timestep),
            "paper_time_on_market": dict(self.paper_time_on_market),
            "action_counts_by_timestep": self.action_counts_by_timestep(),
            # One value per agent: final share-weighted accrual-rate reputation,
            # so the ranking table keeps its "reliability" column without shipping
            # the full per-timestep reputation series (that chart is now a PNG).
            "agent_final_reputation": {
                label: (series[-1] if series else 0.0)
                for label, series in self.agent_review_history.items()
            },
            # Decisions/action log for the feed + replay. ``day`` is the only
            # timestep key the gallery reads, so the duplicate ``timestep`` is
            # dropped to keep this compact.
            "actions": [
                {"day": d, "agent": a, "kind": k, "paper": p}
                for (d, a, k, p) in actions
            ],
        }

    def action_counts_by_timestep(self) -> dict[int, dict[str, int]]:
        """Compact full action mix, safe for large gallery payloads."""
        by_timestep: dict[int, Counter[str]] = {}
        for timestep, _, kind, _ in self.actions:
            by_timestep.setdefault(int(timestep), Counter())[kind] += 1
        return {
            timestep: {kind: int(count) for kind, count in counts.items()}
            for timestep, counts in by_timestep.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "History":
        """Reconstruct a History from a saved (full) ``history.json`` dict.

        Lets ``visualize`` rebuild charts for any archived run. Accepts the
        full export produced by :meth:`to_dict`; missing fields default to
        empty so partial exports still load.
        """
        history = cls()
        timesteps = data.get("timesteps") or data.get("days") or []
        history.timesteps = [int(t) for t in timesteps]

        history.scalars = {
            k: [float(x) for x in v] for k, v in (data.get("scalars") or {}).items()
        }
        history.agent_capital = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_capital") or {}).items()
        }
        history.agent_accrual_rate = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_accrual_rate") or {}).items()
        }
        history.agent_review_history = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_review_history") or {}).items()
        }
        history.agent_review_epsilon_history = {
            k: [float(x) for x in v]
            for k, v in (data.get("agent_review_epsilon_history") or {}).items()
        }
        history.agent_groups = dict(data.get("agent_groups") or {})
        history.agent_talent = {
            k: float(v) for k, v in (data.get("agent_talent") or {}).items()
        }

        history.paper_ac = {
            k: [float(x) for x in v] for k, v in (data.get("paper_ac") or {}).items()
        }
        history.paper_quality = dict(data.get("paper_quality") or {})
        history.paper_authors = dict(data.get("paper_authors") or {})
        history.paper_reviewed = dict(data.get("paper_reviewed") or {})
        history.paper_writing_effort = {
            k: float(v) for k, v in (data.get("paper_writing_effort") or {}).items()
        }
        history.paper_required_writing_effort = dict(
            (k, float(v))
            for k, v in (data.get("paper_required_writing_effort") or {}).items()
        )
        history.paper_accrual_rate = {
            k: float(v) for k, v in (data.get("paper_accrual_rate") or {}).items()
        }
        history.paper_first_seen_timestep = {
            k: int(v)
            for k, v in (data.get("paper_first_seen_timestep") or {}).items()
        }
        history.paper_listed_timestep = {
            k: int(v)
            for k, v in (data.get("paper_listed_timestep") or {}).items()
        }
        history.paper_claimed_timestep = {
            k: int(v)
            for k, v in (data.get("paper_claimed_timestep") or {}).items()
        }
        history.paper_time_on_market = {
            k: int(v)
            for k, v in (data.get("paper_time_on_market") or {}).items()
        }

        def _ts(row: dict[str, Any]) -> int:
            return int(row.get("timestep", row.get("day", 0)))

        history.actions = [
            (_ts(r), r.get("agent"), r.get("kind"), r.get("paper"))
            for r in (data.get("actions") or [])
        ]
        history.completed_reviews = [
            (
                _ts(r),
                r.get("agent"),
                r.get("paper"),
                float(r.get("effort", 0.0)),
                r.get("review_kind"),
            )
            for r in (data.get("completed_reviews") or [])
        ]
        history.writing_efforts = [
            (
                _ts(r),
                r.get("agent"),
                float(r.get("effort", 0.0)),
                bool(r.get("published", False)),
            )
            for r in (data.get("writing_efforts") or [])
        ]
        history.accepted_review_claims = [
            (_ts(r), float(r.get("price", 0.0)))
            for r in (data.get("accepted_review_claims") or [])
            if float(r.get("price", 0.0)) > 0.0
        ]
        history.review_benefit_events = [
            (
                _ts(r),
                str(r.get("faith", "bad")),
                float(r.get("author_net", 0.0)),
                float(r.get("reviewer_benefit", 0.0)),
                float(r.get("value_created", 0.0)),
                float(r.get("share", 0.0)),
            )
            for r in (data.get("review_benefit_events") or [])
        ]

        counts = data.get("action_counts")
        if counts:
            history.action_counts = Counter(
                {k: int(v) for k, v in counts.items()}
            )
        else:
            history.action_counts = Counter(kind for _, _, kind, _ in history.actions)

        outcome = data.get("agent_outcome_summary")
        if outcome:
            history._agent_outcome_summary = list(outcome)
        final_rep = data.get("agent_final_reputation")
        if final_rep:
            history._agent_final_reputation = {
                k: float(v) for k, v in final_rep.items()
            }

        return history

    @classmethod
    def from_json(cls, path: str) -> "History":
        """Load a History saved by :meth:`to_json`."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self, path: str, *, max_paper_series: int | None = None) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(max_paper_series=max_paper_series), fh, indent=2)
        return path

    def to_csv(self, path: str) -> str:
        """Wide time-series: one row per timestep; columns are scalars + agents + papers."""
        scalar_names = list(self.scalars)
        agent_names = list(self.agent_capital)
        paper_names = list(self.paper_ac)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["timestep"] + scalar_names + agent_names + paper_names)
            for i, timestep in enumerate(self.timesteps):
                row: list[Any] = [timestep]
                row += [self.scalars[name][i] for name in scalar_names]
                row += [self._at(self.agent_capital[name], i) for name in agent_names]
                row += [self._at(self.paper_ac[name], i) for name in paper_names]
                writer.writerow(row)
        return path

    @staticmethod
    def _at(series: list[float], i: int) -> float:
        return series[i] if i < len(series) else 0.0
