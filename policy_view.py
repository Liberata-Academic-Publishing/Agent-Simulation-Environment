"""Greedy readout of a learned policy at fixed value intervals.

The continuous table stores one Q-value per action, so the choice is the same
at every interval. Feature-based policies (discrete Q, DQN) are queried with
the other inputs held at a baseline, and the greedy legal action is recorded
at each offered share, review effort, manuscript progress, or paper value.
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass

import numpy as np

from Agent import CONTINUOUS_PUBLISHING_THRESHOLD
from DiscreteQLearningAgent import DiscreteQAction
from Paper import MIN_REVIEW_EFFORT_THRESHOLD
from QLearningAgent import QAction
from config import SIM

POLICY_CHART_DESCRIPTION = (
    "Greedy RL policy and the action it picks at each share, effort, and progress"
)

_SHARE_CANDIDATES = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5)
_EFFORT_CANDIDATES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0)
_PROGRESS_CANDIDATES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
_PAPER_AC_CANDIDATES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)

CONTINUOUS_LABELS = (
    "write",
    "claim",
    "continue",
    "finish and write",
    "claim new",
    "finish research",
)
DISCRETE_LABELS = (
    "write",
    "bad-faith claim",
    "good-faith claim",
)


@dataclass(frozen=True)
class Sweep:
    """Q-values and the greedy action along one axis."""

    title: str
    xlabel: str
    labels: list[str]
    action_names: list[str]
    q: list[list[float]]
    chosen: list[str]
    marker_index: int | None = None
    marker_label: str | None = None


@dataclass(frozen=True)
class PolicyView:
    """Everything the policy chart and the printed table need."""

    title: str
    note: str
    action_labels: list[str]
    reference_q: list[float] | None
    offered: frozenset[str]
    situation_choices: list[tuple[str, str]]
    sweeps: list[Sweep]


def share_intervals() -> list[float]:
    """Offered-share points inside the configured price-feature range."""
    high = float(SIM.rl_share_feature_max)
    points = [value for value in _SHARE_CANDIDATES if value <= high + 1e-12]
    if high > 0 and all(abs(value - high) > 1e-12 for value in points):
        points.append(high)
    return points


def effort_intervals() -> list[float]:
    """Review-effort points, always including the good-faith threshold."""
    points = set(_EFFORT_CANDIDATES)
    points.add(float(SIM.good_faith_review_threshold))
    return sorted(points)


def baseline_share() -> float:
    """Share used when that axis is held fixed. One percent when it fits."""
    low = float(SIM.rl_share_feature_min)
    high = float(SIM.rl_share_feature_max)
    return min(max(0.01, low), high)


def decision_summary(labels: list[str], chosen: list[str]) -> str:
    """Compress a run of identical decisions into one readable phrase."""
    if not chosen:
        return "no legal action"
    if len(set(chosen)) == 1:
        return f"always {chosen[0]}"
    parts: list[str] = []
    start = 0
    for index in range(1, len(chosen) + 1):
        if index == len(chosen) or chosen[index] != chosen[start]:
            action = chosen[start]
            if start == index - 1:
                parts.append(f"{action} at {labels[start]}")
            else:
                parts.append(
                    f"{action} from {labels[start]} to {labels[index - 1]}"
                )
            start = index
    return "; ".join(parts)


def format_policy_report(view: PolicyView) -> str:
    """Plain-text table of Q-values and the greedy action at each interval."""
    lines = [view.title, view.note, ""]
    if view.reference_q is not None:
        lines.append("Q-values:")
        width = max(len(label) for label in view.action_labels)
        for label, value in zip(view.action_labels, view.reference_q):
            suffix = "" if label in view.offered else "  (not offered)"
            lines.append(f"  {label:<{width}}  {value: .4f}{suffix}")
        lines.append("")
    if view.situation_choices:
        lines.append("Greedy choice in each situation:")
        for situation, action in view.situation_choices:
            lines.append(f"  {situation}: {action}")
        lines.append("")
    for sweep in view.sweeps:
        lines.append(f"{sweep.title}: {decision_summary(sweep.labels, sweep.chosen)}")
        header = f"  {'value':<12}" + "".join(
            f"{name:>18}" for name in sweep.action_names
        )
        lines.append(header + f"{'decision':>20}")
        for index, label in enumerate(sweep.labels):
            cells = "".join(
                f"{sweep.q[action][index]:18.4f}"
                for action in range(len(sweep.action_names))
            )
            lines.append(f"  {label:<12}{cells}{sweep.chosen[index]:>20}")
        lines.append("")
    return "\n".join(lines).rstrip()


def view_from_backend(
    backend,
    *,
    title: str,
    publishing: str | None = None,
) -> PolicyView:
    """Read ``backend`` without changing it and record the greedy policy."""
    publishing = SIM.continuous_publishing if publishing is None else publishing
    kind = _kind(backend)
    labels = DISCRETE_LABELS if kind == "discrete" else CONTINUOUS_LABELS
    stateless = _is_stateless(backend)
    sweeps = _sweeps(backend, kind=kind, publishing=publishing, labels=labels)
    situations = _situations(backend, kind=kind, publishing=publishing, labels=labels)
    offered = frozenset(action for _, action in situations)
    for sweep in sweeps:
        offered = offered | frozenset(sweep.action_names)
    reference = None
    if stateless:
        reference = [float(value) for value in _q_values(backend, None)]
    return PolicyView(
        title=title,
        note=_note(kind, stateless),
        action_labels=list(labels[: len(reference)] if reference is not None else labels),
        reference_q=reference,
        offered=offered,
        situation_choices=situations,
        sweeps=sweeps,
    )


def views_for_agents(
    agents,
    *,
    publishing: str | None = None,
) -> list[tuple[PolicyView, str]]:
    """One view per RL cohort, named for the agent with the most capital."""
    from DQNAgent import DQNAgent
    from DiscreteQLearningAgent import (
        DiscreteQLearningAgent,
        LowTalentDiscreteQLearningAgent,
    )
    from QLearningAgent import LowTalentQLearningAgent, QLearningAgent

    specs = (
        (QLearningAgent, "RL policy", "agent_policy.png"),
        (DiscreteQLearningAgent, "Discrete RL policy", "agent_policy_discrete.png"),
        (DQNAgent, "DQN policy", "agent_policy_dqn.png"),
        (LowTalentQLearningAgent, "Low-talent RL policy", "agent_policy_low_talent.png"),
        (
            LowTalentDiscreteQLearningAgent,
            "Low-talent discrete RL policy",
            "agent_policy_discrete_low_talent.png",
        ),
    )
    found: list[tuple[PolicyView, str]] = []
    for cls, label, filename in specs:
        group = [agent for agent in agents if type(agent) is cls]
        if not group:
            continue
        best = max(group, key=lambda agent: agent.academic_capital)
        title = f"{label}: {best.name}"
        if len(group) > 1:
            title += f" (highest academic capital of {len(group)})"
        found.append(
            (
                view_from_backend(
                    best.backend, title=title, publishing=publishing
                ),
                filename,
            )
        )
    if len(found) == 1:
        return [(found[0][0], "agent_policy.png")]
    return found


def write_policy_views(
    views: list[tuple[PolicyView, str]],
    outdir: str,
    *,
    show: bool = False,
) -> dict[str, str]:
    """Print each policy table and save ``agent_policy*.png`` under ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    for view, _filename in views:
        print()
        print(format_policy_report(view))
    try:
        import visualize
    except ImportError as exc:
        print(f"Skipping policy chart ({exc}).")
        return {}
    paths: dict[str, str] = {}
    for view, filename in views:
        path = os.path.join(outdir, filename)
        visualize.plot_agent_policy(view, path, show=show)
        paths[os.path.splitext(filename)[0]] = path
    return paths


def emit_policy_chart(
    backend,
    *,
    title: str,
    outdir: str | None = None,
    filename: str = "agent_policy.png",
    publishing: str | None = None,
    show: bool = False,
) -> dict[str, str]:
    """Print and save one policy chart. Used at the end of a training run."""
    view = view_from_backend(backend, title=title, publishing=publishing)
    paths = write_policy_views(
        [(view, filename)],
        SIM.output_dir if outdir is None else outdir,
        show=show,
    )
    if paths:
        print("\nWrote policy chart:")
        for path in paths.values():
            print(f"- {path}  ({POLICY_CHART_DESCRIPTION})")
    return paths


def _note(kind: str, stateless: bool) -> str:
    if stateless:
        return (
            "One Q-value per action, used on every turn.\n"
            "The greedy action is the same at every offered share, "
            "review effort, and manuscript progress."
        )
    if kind == "discrete":
        return (
            "Other inputs are held at zero capital and zero review history, "
            "with one paper listed.\n"
            "A fixed share axis uses a 1% offer; a fixed progress axis is a new manuscript."
        )
    return (
        "Other inputs are held at zero capital and zero review history.\n"
        "The review-effort sweep is an open review with nothing else listed. "
        "A fixed share axis uses a 1% offer."
    )


def _situations(backend, *, kind: str, publishing: str, labels: tuple[str, ...]) -> list[tuple[str, str]]:
    share = baseline_share()
    if kind == "discrete":
        listed = _features(share=share, num_reviewable=1)
        empty = _features(share=0.0, num_reviewable=0)
        return [
            (
                f"paper listed at {_share_label(share)}, new manuscript",
                _choice(backend, listed, _discrete_legal(True), labels),
            ),
            (
                "nothing listed",
                _choice(backend, empty, _discrete_legal(False), labels),
            ),
        ]
    listed = _features(share=share, num_reviewable=1, in_review=False)
    empty = _features(share=0.0, num_reviewable=0, in_review=False)
    reviewing = _features(
        share=0.0, num_reviewable=0, in_review=True, review_effort=0.0
    )
    return [
        (
            f"paper listed at {_share_label(share)}, new manuscript",
            _choice(backend, listed, _free_legal(publishing, listed=True), labels),
        ),
        (
            "nothing listed",
            _choice(backend, empty, _free_legal(publishing, listed=False), labels),
        ),
        (
            "review open, nothing else listed",
            _choice(backend, reviewing, _review_legal(), labels),
        ),
    ]


def _sweeps(backend, *, kind: str, publishing: str, labels: tuple[str, ...]) -> list[Sweep]:
    share = baseline_share()
    if kind == "discrete":
        legal = _discrete_legal(True)
        return [
            _make_sweep(
                backend,
                title="Paper listed: write, bad faith, or good faith",
                xlabel="Offered share",
                values=share_intervals(),
                value_label=_share_label,
                legal=legal,
                labels=labels,
                features_at=lambda value: _features(share=value, num_reviewable=1),
            ),
            _make_sweep(
                backend,
                title="Paper listed: decision by manuscript progress",
                xlabel="Manuscript progress",
                values=list(_PROGRESS_CANDIDATES),
                value_label=_progress_label,
                legal=legal,
                labels=labels,
                features_at=lambda value: _features(
                    progress=value, share=share, num_reviewable=1
                ),
            ),
            _make_sweep(
                backend,
                title="Paper listed: decision by the paper's academic capital",
                xlabel="Listed paper academic capital",
                values=list(_PAPER_AC_CANDIDATES),
                value_label=_number_label,
                legal=legal,
                labels=labels,
                features_at=lambda value: _features(
                    paper_ac=value, share=share, num_reviewable=1
                ),
            ),
        ]
    free = _free_legal(publishing, listed=True)
    review = _review_legal()
    efforts = effort_intervals()
    threshold = float(SIM.good_faith_review_threshold)
    marker = next(
        (index for index, value in enumerate(efforts) if value == threshold),
        None,
    )
    return [
        _make_sweep(
            backend,
            title="Paper listed: write or claim",
            xlabel="Offered share",
            values=share_intervals(),
            value_label=_share_label,
            legal=free,
            labels=labels,
            features_at=lambda value: _features(share=value, num_reviewable=1),
        ),
        _make_sweep(
            backend,
            title="Review open: continue or finish",
            xlabel="Review effort (timesteps)",
            values=efforts,
            value_label=_number_label,
            legal=review,
            labels=labels,
            features_at=lambda value: _features(
                in_review=True, review_effort=value, num_reviewable=0, share=0.0
            ),
            marker_index=marker,
            marker_label=f"good faith ({threshold:g})",
        ),
        _make_sweep(
            backend,
            title="Paper listed: decision by manuscript progress",
            xlabel="Manuscript progress",
            values=list(_PROGRESS_CANDIDATES),
            value_label=_progress_label,
            legal=free,
            labels=labels,
            features_at=lambda value: _features(
                progress=value, share=share, num_reviewable=1
            ),
        ),
    ]


def _make_sweep(
    backend,
    *,
    title: str,
    xlabel: str,
    values: list[float],
    value_label,
    legal: list[int],
    labels: tuple[str, ...],
    features_at,
    marker_index: int | None = None,
    marker_label: str | None = None,
) -> Sweep:
    series = [[] for _ in legal]
    chosen: list[str] = []
    tick_labels: list[str] = []
    for value in values:
        q = _q_values(backend, features_at(value))
        action = _greedy(q, legal)
        chosen.append(labels[action])
        for index, legal_action in enumerate(legal):
            series[index].append(float(q[legal_action]))
        tick_labels.append(value_label(value))
    return Sweep(
        title=title,
        xlabel=xlabel,
        labels=tick_labels,
        action_names=[labels[action] for action in legal],
        q=series,
        chosen=chosen,
        marker_index=marker_index,
        marker_label=marker_label,
    )


def _choice(backend, features: np.ndarray, legal: list[int], labels: tuple[str, ...]) -> str:
    return labels[_greedy(_q_values(backend, features), legal)]


def _greedy(q: np.ndarray, legal: list[int]) -> int:
    return max(legal, key=lambda action: float(q[action]))


def _free_legal(publishing: str, *, listed: bool) -> list[int]:
    actions = [int(QAction.WRITE)]
    if publishing != CONTINUOUS_PUBLISHING_THRESHOLD:
        actions.append(int(QAction.RESEARCH_FINISH))
    if listed:
        actions.append(int(QAction.CLAIM))
    return actions


def _review_legal() -> list[int]:
    return [int(QAction.FINISH_WRITE), int(QAction.CONTINUE)]


def _discrete_legal(listed: bool) -> list[int]:
    actions = [int(DiscreteQAction.WRITE)]
    if listed:
        actions.extend(
            [
                int(DiscreteQAction.BAD_FAITH_CLAIM),
                int(DiscreteQAction.GOOD_FAITH_CLAIM),
            ]
        )
    return actions


def _features(
    *,
    progress: float = 0.0,
    review_effort: float = 0.0,
    capital: float = 0.0,
    history: float = 0.0,
    num_reviewable: float = 0.0,
    share: float = 0.0,
    paper_ac: float = 1.0,
    in_review: bool = False,
) -> np.ndarray:
    from DQNAgent import EFFORT_FEATURE_SCALE, share_feature

    threshold = max(float(MIN_REVIEW_EFFORT_THRESHOLD), 1.0)
    return np.array(
        [
            min(max(progress, 0.0), 1.0),
            math.tanh(review_effort / EFFORT_FEATURE_SCALE),
            math.tanh(capital / 100.0),
            math.tanh(history / 10.0),
            math.tanh(num_reviewable / 10.0),
            share_feature(share),
            math.tanh(paper_ac / 100.0),
            1.0 if in_review else 0.0,
            min(review_effort / threshold, 1.0),
        ],
        dtype=np.float64,
    )


def _q_values(backend, features: np.ndarray | None) -> np.ndarray:
    """Action values. A missing tabular bucket stays missing and scores as zero."""
    if _is_stateless(backend):
        return np.asarray(backend.q_values(), dtype=np.float64)
    if features is None:
        features = np.zeros(9, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    table = getattr(backend, "table", None)
    key_fn = getattr(backend, "_key", None)
    if table is not None and key_fn is not None:
        row = table.get(key_fn(features))
        if row is None:
            sample = next(iter(table.values()), None)
            width = len(sample) if sample is not None else len(DISCRETE_LABELS)
            return np.zeros(width, dtype=np.float64)
        return np.asarray(row, dtype=np.float64)
    return np.asarray(backend.q_values(features), dtype=np.float64)


def _is_stateless(backend) -> bool:
    parameters = list(inspect.signature(backend.q_values).parameters)
    return len(parameters) == 0


def _kind(backend) -> str:
    if _is_stateless(backend):
        return "continuous"
    q = _q_values(backend, np.zeros(9, dtype=np.float64))
    if len(q) == len(DISCRETE_LABELS):
        return "discrete"
    return "continuous"


def _share_label(share: float) -> str:
    if share <= 0:
        return "0%"
    return f"{share * 100:.3g}%"


def _progress_label(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def _number_label(value: float) -> str:
    return f"{value:g}"
