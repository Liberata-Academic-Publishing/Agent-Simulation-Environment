"""Matplotlib charts for a simulation ``History``.

This module is the *only* part that needs matplotlib; the recorder/exports in
``History`` stay dependency-free. Importing this module raises a clear error if
matplotlib is missing, so callers can ``try: import visualize`` and fall back to
the CSV/JSON export.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import Paper as paper_mod
from Paper import BAD_FAITH_REVIEW, GOOD_FAITH_REVIEW, MIN_REVIEW_EFFORT_THRESHOLD
from config import SIM

if TYPE_CHECKING:
    from History import History

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "visualize requires matplotlib. Install it with: pip install matplotlib"
    ) from exc

ACTION_COLORS = {
    "write_paper": "#60a5fa",
    "bad_faith_review": "#f87171",
    "good_faith_review": "#16a34a",
    "review_started": "#4ade80",
    "review_continued": "#22c55e",
    "review_finished_write": "#f59e0b",
    "review_finished_peer_review": "#a855f7",
    "review_stopped": "#16a34a",
    "review_unavailable": "#a78bfa",
    "idle": "#6b7280",
}

# Top-level decisions (matches run_simulation.DECISION_LABELS).
DECISION_LABELS = {
    "write_paper": "write_paper",
    "bad_faith_review": "bad_faith_review",
    "good_faith_review": "good_faith_review",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "review_unavailable": "start_review",
    "idle": "idle",
}

DECISION_COLORS = {
    "write_paper": "#60a5fa",
    "bad_faith_review": "#f87171",
    "good_faith_review": "#16a34a",
    "start_review": "#4ade80",
    "continue_review": "#22c55e",
    "finish_and_write": "#f59e0b",
    "finish_and_review": "#a855f7",
    "idle": "#6b7280",
}


def _finish(fig, path: str | None, show: bool) -> str | None:
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def _mean_completed_review_effort_series(history: "History") -> list[float]:
    """Cumulative mean effort of finished reviews at each timestep."""
    stored = history.scalars.get("mean_completed_review_effort")
    if stored is not None and len(stored) == len(history.days):
        return list(stored)

    by_day: dict[int, list[float]] = {}
    for ts, _, _, effort, _ in history.completed_reviews:
        by_day.setdefault(ts, []).append(float(effort))

    running_sum = 0.0
    running_count = 0
    series: list[float] = []
    for day in history.days:
        for effort in by_day.get(day, []):
            running_sum += effort
            running_count += 1
        series.append(running_sum / running_count if running_count else 0.0)
    return series


def _draw_mean_review_effort(ax, history: "History") -> None:
    """Running average effort invested in completed peer reviews over time."""
    series = _mean_completed_review_effort_series(history)
    if not any(series):
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(history.days, series, linewidth=2.0, color="#7c3aed", label="mean effort")
    ax.axhline(
        MIN_REVIEW_EFFORT_THRESHOLD,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"good-faith threshold ({MIN_REVIEW_EFFORT_THRESHOLD:g})",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean review effort (timesteps)")
    ax.legend(fontsize=8, loc="best")


def plot_mean_review_effort(
    history: "History", path: str | None = None, show: bool = False
):
    """Running mean effort of completed peer reviews over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_mean_review_effort(ax, history)
    ax.set_title("Average peer review effort over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_agent_capital(ax, history: "History", legend: bool = True) -> None:
    if not history.agent_capital:
        ax.text(0.5, 0.5, "No agent capital recorded", ha="center", va="center")
        ax.set_axis_off()
        return
    for label, series in history.agent_capital.items():
        ax.plot(history.days, series, linewidth=1.2, label=label)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    if legend and 0 < len(history.agent_capital) <= 30:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


def plot_agent_capital(history: "History", path: str | None = None, show: bool = False):
    """One line per agent: academic capital over time (the core view)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_agent_capital(ax, history)
    ax.set_title("Academic capital per agent over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _agent_number(label: str) -> str:
    """Extract the trailing agent index from labels like ``RL Agent 12``."""
    match = re.search(r"(\d+)\s*$", label)
    return match.group(1) if match else label


def _running_mean_review_effort_by_agent(
    history: "History",
) -> dict[str, list[float]]:
    """Per-agent running mean completed review effort, aligned to ``history.days``."""
    labels = list(history.agent_capital)
    if not labels:
        return {}

    running_sum = {label: 0.0 for label in labels}
    running_count = {label: 0 for label in labels}
    series = {label: [] for label in labels}
    reviews_by_day: dict[int, list[tuple[str, float]]] = {}
    for day, agent, _, effort, _ in history.completed_reviews:
        reviews_by_day.setdefault(int(day), []).append((agent, float(effort)))

    for day in history.days:
        for agent, effort in reviews_by_day.get(int(day), []):
            if agent not in running_sum:
                continue
            running_sum[agent] += effort
            running_count[agent] += 1
        for label in labels:
            count = running_count[label]
            series[label].append(running_sum[label] / count if count else 0.0)
    return series


def _annotate_agent_trajectories(ax, xs: list[float], ys: list[float], label: str) -> None:
    """Label the end of a trajectory with the agent's numeric id."""
    if not xs or not ys:
        return
    ax.annotate(
        _agent_number(label),
        (xs[-1], ys[-1]),
        fontsize=7,
        alpha=0.85,
        xytext=(3, 0),
        textcoords="offset points",
    )


def _agent_talent_legend_label(label: str, history: "History") -> str:
    """Legend entry: agent number plus intrinsic talent."""
    num = _agent_number(label)
    talent = history.agent_talent.get(label)
    if talent is not None:
        return f"{num} (talent={talent:.2f})"
    return num


def _sorted_agent_labels(labels) -> list[str]:
    def sort_key(label: str) -> tuple:
        num = _agent_number(label)
        try:
            return (int(num), label)
        except ValueError:
            return (9999, label)

    return sorted(labels, key=sort_key)


def _draw_talent_vs_ac(ax, history: "History") -> bool:
    """Each agent's portfolio AC accrual rate over time; legend maps agent to talent."""
    rates = history.agent_accrual_rate
    if not rates or not any(rates.values()):
        ax.text(
            0.5,
            0.5,
            "No portfolio accrual rate recorded\n(re-run simulation to populate)",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return False

    drew = False
    for label in _sorted_agent_labels(rates):
        series = rates.get(label)
        if not series:
            continue
        ax.plot(
            history.days,
            series,
            linewidth=1.0,
            alpha=0.75,
            label=_agent_talent_legend_label(label, history),
        )
        _annotate_agent_trajectories(ax, list(history.days), series, label)
        drew = True

    if not drew:
        ax.text(0.5, 0.5, "No portfolio accrual rate recorded", ha="center", va="center")
        ax.set_axis_off()
        return False

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Portfolio AC accrual rate")
    n_agents = len(rates)
    if 0 < n_agents <= 40:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    return True


def plot_talent_vs_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Portfolio AC accrual rate over time; legend shows each agent's talent."""
    fig, ax = plt.subplots(figsize=(14, 6))
    has_legend = _draw_talent_vs_ac(ax, history)
    ax.set_title("Portfolio AC accrual rate over time")
    if has_legend:
        fig.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        fig.tight_layout()
    return _finish(fig, path, show)


def _draw_mean_review_effort_vs_ac(ax, history: "History") -> None:
    """Each agent's capital trajectory in (running mean review effort, AC) space."""
    if not history.agent_capital:
        ax.text(0.5, 0.5, "No agent capital recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    effort_series = _running_mean_review_effort_by_agent(history)
    drew = False
    for label, ac_series in history.agent_capital.items():
        if not ac_series:
            continue
        efforts = effort_series.get(label, [])
        if len(efforts) != len(ac_series):
            continue
        ax.plot(efforts, ac_series, linewidth=1.0, alpha=0.65)
        _annotate_agent_trajectories(ax, efforts, ac_series, label)
        drew = True

    if not drew:
        ax.text(0.5, 0.5, "No agent capital recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.set_xlabel("Mean peer review effort (running)")
    ax.set_ylabel("Total academic capital")


def plot_mean_review_effort_vs_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Running mean review effort vs total academic capital per agent."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_mean_review_effort_vs_ac(ax, history)
    ax.set_title("Mean peer review effort vs total academic capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_agent_capital_by_group(
    history: "History", path: str | None = None, show: bool = False
):
    """Academic capital over time, with one color per agent class.

    Like ``plot_agent_capital`` but colors agents by their class with a single
    legend entry per group.
    """
    groups = history.agent_groups
    ordered_groups: list[str] = []
    for label in history.agent_capital:
        group = groups.get(label, "Agent")
        if group not in ordered_groups:
            ordered_groups.append(group)

    cmap = plt.get_cmap("tab10")
    color_for = {group: cmap(i % 10) for i, group in enumerate(ordered_groups)}

    fig, ax = plt.subplots(figsize=(11, 6))
    legended: set[str] = set()
    for label, series in history.agent_capital.items():
        group = groups.get(label, "Agent")
        legend_label = group if group not in legended else None
        legended.add(group)
        ax.plot(
            history.days,
            series,
            linewidth=1.2,
            alpha=0.8,
            color=color_for[group],
            label=legend_label,
        )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    ax.set_title("Academic capital per agent over time (by group)")
    if ordered_groups:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_agent_group_comparison(
    history: "History", path: str | None = None, show: bool = False
):
    """Executive summary by agent type: mean capital and review behavior."""
    summary = history.agent_group_summary()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if not summary:
        for ax in axes:
            ax.text(0.5, 0.5, "No agent group data", ha="center", va="center")
            ax.set_axis_off()
        return _finish(fig, path, show)

    groups = sorted(summary)
    mean_capital = [summary[g]["mean_final_capital"] for g in groups]
    good = [summary[g]["good_faith_reviews"] for g in groups]
    bad = [summary[g]["bad_faith_reviews"] for g in groups]
    avg_effort = [summary[g].get("average_review_effort", 0.0) for g in groups]
    x = range(len(groups))

    axes[0].bar(x, mean_capital, color="#60a5fa", edgecolor="#1e3a5f")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(groups, rotation=20, ha="right")
    axes[0].set_ylabel("Mean final AC")
    axes[0].set_title("Mean final capital by agent type")

    axes[1].bar(x, good, color="#16a34a", label="good faith")
    axes[1].bar(x, bad, bottom=good, color="#f87171", label="bad faith")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(groups, rotation=20, ha="right")
    axes[1].set_ylabel("Completed reviews")
    axes[1].set_title("Review outcomes by agent type")
    axes[1].legend(fontsize=8)

    axes[2].bar(x, avg_effort, color="#7c3aed", edgecolor="#4c1d95")
    axes[2].axhline(
        MIN_REVIEW_EFFORT_THRESHOLD,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"min effort ({MIN_REVIEW_EFFORT_THRESHOLD:g})",
    )
    axes[2].axhline(
        SIM.good_faith_review_threshold,
        color="#16a34a",
        linestyle=":",
        linewidth=1.2,
        label=f"good faith ({SIM.good_faith_review_threshold:g})",
    )
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(groups, rotation=20, ha="right")
    axes[2].set_ylabel("Mean review effort (timesteps)")
    axes[2].set_title("Average completed review effort")
    axes[2].legend(fontsize=8, loc="best")

    fig.tight_layout()
    return _finish(fig, path, show)


# Display order and short names for the overview figure; unknown classes are
# appended after these with their class name.
OVERVIEW_GROUPS = {
    "HeuristicAgent": "Heuristic",
    "RandomAgent": "Random",
    "ProbabilisticDiscreteAgent": "Probabilistic",
    "QLearningAgent": "RL",
    "DiscreteQLearningAgent": "RL (discrete)",
    "DQNAgent": "DQN",
    "LowTalentQLearningAgent": "Low-talent RL",
    "LowTalentDiscreteQLearningAgent": "Low-talent RL (discrete)",
}


def _good_faith_share_by_group_over_time(
    history: "History", groups: list[str], n_bins: int = 20
) -> tuple[list[float], dict[str, list[float | None]]]:
    """Per-group good-faith share of completed reviews in equal time bins."""
    horizon = max(history.timesteps) if history.timesteps else 1
    width = max(1.0, horizon / n_bins)
    centers = [width * (i + 0.5) for i in range(n_bins)]
    good = {g: [0] * n_bins for g in groups}
    total = {g: [0] * n_bins for g in groups}
    for timestep, agent_label, _, _, review_kind in history.completed_reviews:
        group = history.agent_groups.get(agent_label, "Agent")
        if group not in total:
            continue
        idx = min(n_bins - 1, int(timestep / width))
        total[group][idx] += 1
        if review_kind == GOOD_FAITH_REVIEW:
            good[group][idx] += 1
    shares = {
        g: [good[g][i] / total[g][i] if total[g][i] else None for i in range(n_bins)]
        for g in groups
    }
    return centers, shares


def plot_overview(history: "History", path: str | None = None, show: bool = False):
    """Four-panel headline figure: review faith, capital, behavior, inequality.

    (a) does good- or bad-faith review emerge over time, (b) academic capital
    by agent type split into writing vs reviewing, (c) review behavior by agent
    type, (d) within-group and overall capital inequality (Gini).
    """
    from History import gini

    summary = history.agent_group_summary()
    present = set(summary)
    groups = [g for g in OVERVIEW_GROUPS if g in present]
    groups += sorted(present - set(groups))
    names = [OVERVIEW_GROUPS.get(g, g) for g in groups]
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
    x = list(range(len(groups)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    horizon = history.timesteps[-1] if history.timesteps else 0
    fig.suptitle(
        f"Peer review under Liberata economics — {len(history.agent_capital)} "
        f"agents, {horizon} timesteps ({SIM.review_paradigm} review)",
        fontsize=14, fontweight="bold",
    )
    if not summary:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No agent group data", ha="center", va="center")
            ax.set_axis_off()
        return _finish(fig, path, show)

    # (a) Emergence: good-faith share of completed reviews over time.
    ax = axes[0, 0]
    centers, shares = _good_faith_share_by_group_over_time(history, groups)
    for g, name in zip(groups, names):
        pts = [(c, s) for c, s in zip(centers, shares[g]) if s is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o", markersize=3, color=colors[g], label=name)
    ax.axhline(0.5, color="#6b7280", linestyle=":", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Good-faith share of completed reviews")
    ax.set_title("(a) Does good-faith review emerge?")
    ax.legend(fontsize=8, loc="best")

    # (b) Academic capital by agent type, split by source.
    ax = axes[0, 1]
    count = [max(1, summary[g]["agent_count"]) for g in groups]
    from_writing = [summary[g]["total_ac_from_writing"] / n for g, n in zip(groups, count)]
    from_review = [summary[g]["total_ac_from_reviewing"] / n for g, n in zip(groups, count)]
    ax.bar(x, from_writing, color=[colors[g] for g in groups], label="from writing")
    # Reviewer shares are not serialized, so histories reloaded from JSON have
    # no writing/review split; show total capital only in that case.
    if any(from_review):
        ax.bar(
            x, from_review, bottom=from_writing, color=[colors[g] for g in groups],
            alpha=0.45, hatch="//", edgecolor="white", label="from peer review",
        )
        ax.legend(fontsize=8, loc="best")
    ax.set_ylim(0.0, max(w + r for w, r in zip(from_writing, from_review)) * 1.12 or 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Mean final academic capital per agent")
    ax.set_title("(b) Academic capital by agent type")

    # (c) Review behavior: completed reviews per agent, good vs bad faith.
    ax = axes[1, 0]
    good = [summary[g]["mean_good_faith_reviews"] for g in groups]
    bad = [summary[g]["mean_bad_faith_reviews"] for g in groups]
    ax.bar(x, good, color="#16a34a", label="good faith")
    ax.bar(x, bad, bottom=good, color="#f87171", label="bad faith")
    for i, g in enumerate(groups):
        rate = summary[g]["good_faith_review_rate"]
        ax.text(i, good[i] + bad[i], f"{rate:.0%} good", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, max(g + b for g, b in zip(good, bad)) * 1.12 or 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Completed reviews per agent")
    ax.set_title("(c) Review behavior by agent type")
    ax.legend(fontsize=8, loc="upper left")

    # (d) Inequality: Gini of final capital within each group and overall.
    ax = axes[1, 1]
    finals: dict[str, list[float]] = defaultdict(list)
    for label, series in history.agent_capital.items():
        finals[history.agent_groups.get(label, "Agent")].append(series[-1] if series else 0.0)
    within = [gini(finals[g]) for g in groups]
    overall = gini(v for vals in finals.values() for v in vals)
    ax.bar(x, within, color=[colors[g] for g in groups])
    ax.axhline(overall, color="black", linestyle="--", linewidth=1.2,
               label=f"all agents ({overall:.2f})")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Gini of final academic capital")
    ax.set_title("(d) Capital inequality")
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


def _draw_review_benefit(ax_left, ax_right, history: "History") -> None:
    """Reviewer-vs-author benefit over time, split by good/bad faith.

    Left: author net gain relative to the no-review counterfactual (the bad-faith
    line dipping below zero is the exploitation story). Right: the AC reviewers
    captured via their share.
    """
    scalars = history.scalars
    keys = (
        "author_net_good",
        "author_net_bad",
        "reviewer_benefit_good",
        "reviewer_benefit_bad",
    )
    if not any(any(scalars.get(k, [])) for k in keys):
        for ax in (ax_left, ax_right):
            ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
            ax.set_axis_off()
        return

    good_color, bad_color = "#16a34a", "#f87171"
    days = history.days

    ax_left.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax_left.plot(
        days, scalars.get("author_net_good", []),
        color=good_color, linewidth=2.0, label="good faith",
    )
    ax_left.plot(
        days, scalars.get("author_net_bad", []),
        color=bad_color, linewidth=2.0, label="bad faith",
    )
    ax_left.set_xlabel("Timestep")
    ax_left.set_ylabel("Author net gain vs no review (AC)")
    ax_left.set_title("Are authors better off being reviewed?")
    ax_left.legend(fontsize=8, loc="best")

    ax_right.plot(
        days, scalars.get("reviewer_benefit_good", []),
        color=good_color, linewidth=2.0, label="good faith",
    )
    ax_right.plot(
        days, scalars.get("reviewer_benefit_bad", []),
        color=bad_color, linewidth=2.0, label="bad faith",
    )
    ax_right.set_xlabel("Timestep")
    ax_right.set_ylabel("Reviewer captured value (AC)")
    ax_right.set_title("What reviewers capture")
    ax_right.legend(fontsize=8, loc="best")


def plot_review_benefit(history: "History", path: str | None = None, show: bool = False):
    """Reviewer vs author benefit (good vs bad faith); author_net < 0 = exploited."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_review_benefit(ax_left, ax_right, history)
    fig.suptitle("Reviewer vs author benefit (good vs bad faith)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


def _review_benefit_scalar_keys() -> tuple[str, ...]:
    return (
        "author_net_good",
        "author_net_bad",
        "reviewer_benefit_good",
        "reviewer_benefit_bad",
        "value_created_good",
        "value_created_bad",
    )


def _has_review_benefit(history: "History") -> bool:
    scalars = history.scalars
    return any(any(scalars.get(key, [])) for key in _review_benefit_scalar_keys())


def _review_benefit_total_series(
    history: "History",
) -> tuple[list[float], list[float], list[float]]:
    """Cumulative system totals (good + bad faith) aligned to ``history.days``."""
    scalars = history.scalars

    def combine(good_key: str, bad_key: str) -> list[float]:
        good = scalars.get(good_key, [])
        bad = scalars.get(bad_key, [])
        length = max(len(good), len(bad), len(history.days))
        totals: list[float] = []
        for index in range(length):
            good_value = float(good[index]) if index < len(good) else 0.0
            bad_value = float(bad[index]) if index < len(bad) else 0.0
            totals.append(good_value + bad_value)
        return totals

    value_created = combine("value_created_good", "value_created_bad")
    reviewer_benefit = combine("reviewer_benefit_good", "reviewer_benefit_bad")
    author_net = combine("author_net_good", "author_net_bad")
    return value_created, reviewer_benefit, author_net


# Writing earns the blue capital; peer review earns the purple capital.
WRITING_COLOR = "#2563eb"
REVIEW_COLOR = "#a855f7"


def _draw_review_surplus_aggregate(ax_left, ax_right, history: "History") -> None:
    """System-wide review surplus: absolute totals and how value is split."""
    value_created, reviewer_benefit, author_net = _review_benefit_total_series(history)
    if not any(value_created):
        for ax in (ax_left, ax_right):
            ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
            ax.set_axis_off()
        return

    days = history.days
    value_color = "#94a3b8"

    ax_left.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax_left.plot(
        days,
        value_created,
        color=value_color,
        linewidth=2.0,
        label="Value created",
    )
    ax_left.plot(
        days,
        reviewer_benefit,
        color=REVIEW_COLOR,
        linewidth=2.0,
        label="Reviewers capture",
    )
    ax_left.plot(
        days,
        author_net,
        color=WRITING_COLOR,
        linewidth=2.0,
        label="Authors net gain",
    )
    ax_left.set_xlabel("Timestep")
    ax_left.set_ylabel("Cumulative surplus (AC)")
    ax_left.set_title("How much value do reviews create and who keeps it?")
    ax_left.legend(fontsize=8, loc="best")

    author_pct: list[float] = []
    reviewer_pct: list[float] = []
    for created, author, reviewer in zip(value_created, author_net, reviewer_benefit):
        if created > 1e-9:
            author_pct.append(100.0 * author / created)
            reviewer_pct.append(100.0 * reviewer / created)
        else:
            author_pct.append(0.0)
            reviewer_pct.append(0.0)

    ax_right.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax_right.axhline(
        100.0,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
        label="100% of value created",
    )
    ax_right.plot(
        days,
        reviewer_pct,
        color=REVIEW_COLOR,
        linewidth=2.0,
        label="Reviewers' share",
    )
    ax_right.plot(
        days,
        author_pct,
        color=WRITING_COLOR,
        linewidth=2.0,
        label="Authors' share",
    )
    ax_right.set_xlabel("Timestep")
    ax_right.set_ylabel("Share of review value created (%)")
    ax_right.set_title("How the surplus is split")
    ax_right.legend(fontsize=8, loc="best")

    final_created = value_created[-1]
    final_reviewer = reviewer_benefit[-1]
    final_author = author_net[-1]
    if final_created > 1e-9:
        final_author_pct = 100.0 * final_author / final_created
        final_reviewer_pct = 100.0 * final_reviewer / final_created
    else:
        final_author_pct = 0.0
        final_reviewer_pct = 0.0

    if final_author >= 0.0 and final_reviewer > 0.0:
        outcome = "Both sides gain from reviews"
    elif final_author < 0.0:
        outcome = (
            f"Authors disadvantaged (net −{abs(final_author):.1f} AC vs no review)"
        )
    elif final_reviewer <= 0.0:
        outcome = "Reviewers capture no surplus"
    else:
        outcome = "Authors gain; reviewers break even"

    summary = (
        f"Final totals (AC)\n"
        f"Value created: {final_created:.1f}\n"
        f"Reviewers: {final_reviewer:.1f} ({final_reviewer_pct:.0f}%)\n"
        f"Authors net: {final_author:.1f} ({final_author_pct:.0f}%)\n"
        f"{outcome}"
    )
    ax_right.text(
        0.02,
        0.02,
        summary,
        transform=ax_right.transAxes,
        fontsize=8,
        color="#e6edf3",
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "#161d24",
            "edgecolor": "#6b7280",
            "alpha": 0.92,
        },
    )


def plot_review_surplus_aggregate(
    history: "History", path: str | None = None, show: bool = False
):
    """System-wide review surplus and split between authors and reviewers."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_review_surplus_aggregate(ax_left, ax_right, history)
    fig.suptitle("System-wide review surplus (authors vs reviewers)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


REVIEW_SURPLUS_BIN_SIZE = 10


def _review_surplus_binned_series(
    history: "History",
    *,
    bin_size: int = REVIEW_SURPLUS_BIN_SIZE,
) -> dict[str, tuple[list[float], list[float]]]:
    """Mean per-review surplus in each ``bin_size``-timestep window, by faith."""
    events = getattr(history, "review_benefit_events", None) or []
    max_t = max(history.timesteps) if history.timesteps else 0
    if max_t <= 0 or not events:
        return {}

    num_bins = (max_t - 1) // bin_size + 1
    author_good = [0.0] * num_bins
    author_bad = [0.0] * num_bins
    reviewer_good = [0.0] * num_bins
    reviewer_bad = [0.0] * num_bins
    counts_good = [0] * num_bins
    counts_bad = [0] * num_bins

    for timestep, faith, author_net, reviewer_benefit, _, _ in events:
        t = int(timestep)
        if t <= 0:
            continue
        b = (t - 1) // bin_size
        if not (0 <= b < num_bins):
            continue
        if faith == "good":
            author_good[b] += author_net
            reviewer_good[b] += reviewer_benefit
            counts_good[b] += 1
        else:
            author_bad[b] += author_net
            reviewer_bad[b] += reviewer_benefit
            counts_bad[b] += 1

    midpoint = bin_size / 2.0
    xs: list[float] = []
    series: dict[str, list[float]] = {
        "author_good": [],
        "author_bad": [],
        "reviewer_good": [],
        "reviewer_bad": [],
    }
    for b in range(num_bins):
        if counts_good[b] == 0 and counts_bad[b] == 0:
            continue
        xs.append(b * bin_size + midpoint)
        for key, totals, counts in (
            ("author_good", author_good, counts_good),
            ("author_bad", author_bad, counts_bad),
            ("reviewer_good", reviewer_good, counts_good),
            ("reviewer_bad", reviewer_bad, counts_bad),
        ):
            series[key].append(totals[b] / counts[b] if counts[b] else float("nan"))

    return {key: (xs, values) for key, values in series.items()}


def _draw_review_surplus_by_period(
    ax_left,
    ax_right,
    history: "History",
    *,
    bin_size: int = REVIEW_SURPLUS_BIN_SIZE,
) -> None:
    """Mean per-review surplus by time window (not cumulative)."""
    series = _review_surplus_binned_series(history, bin_size=bin_size)
    if not series.get("author_good", ([], []))[0]:
        for ax in (ax_left, ax_right):
            ax.text(
                0.5,
                0.5,
                "No per-review surplus recorded\n(re-run simulation to populate)",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        return

    xs = series["author_good"][0]
    good_color, bad_color = "#16a34a", "#f87171"
    horizon = int(SIM.forecast_horizon_timesteps)

    ax_left.axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    ax_left.plot(
        xs,
        series["author_good"][1],
        color=good_color,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="good faith",
    )
    ax_left.plot(
        xs,
        series["author_bad"][1],
        color=bad_color,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="bad faith",
    )
    ax_left.set_xlabel(f"Timestep ({bin_size}-step bin midpoint)")
    ax_left.set_ylabel(f"Mean author net gain per review (AC, {horizon}-step horizon)")
    ax_left.set_title("Do authors gain from reviews in each period?")
    ax_left.legend(fontsize=8, loc="best")

    ax_right.plot(
        xs,
        series["reviewer_good"][1],
        color=good_color,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="good faith",
    )
    ax_right.plot(
        xs,
        series["reviewer_bad"][1],
        color=bad_color,
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="bad faith",
    )
    ax_right.set_xlabel(f"Timestep ({bin_size}-step bin midpoint)")
    ax_right.set_ylabel(f"Mean reviewer capture per review (AC, {horizon}-step horizon)")
    ax_right.set_title("What reviewers earn per review in each period")
    ax_right.legend(fontsize=8, loc="best")


def _has_review_surplus_by_period(history: "History") -> bool:
    return bool(getattr(history, "review_benefit_events", None))


def plot_review_surplus_by_period(
    history: "History", path: str | None = None, show: bool = False
):
    """Mean per-review author/reviewer surplus in time bins (not cumulative)."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_review_surplus_by_period(ax_left, ax_right, history)
    fig.suptitle(
        "Review surplus by period (per review, not cumulative)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


def _draw_ac_source(ax_left, ax_right, history: "History") -> None:
    """Where academic capital comes from: writing papers vs peer review.

    Left: system total AC split over time (stacked area). Right: how that split
    distributes across agent types (stacked bar).
    """
    scalars = history.scalars
    writing = scalars.get("writing_held_ac", [])
    review = scalars.get("review_held_ac", [])
    if not (any(writing) or any(review)):
        for ax in (ax_left, ax_right):
            ax.text(0.5, 0.5, "No academic capital recorded", ha="center", va="center")
            ax.set_axis_off()
        return

    days = history.days
    ax_left.stackplot(
        days, writing, review,
        labels=["writing", "peer review"],
        colors=[WRITING_COLOR, REVIEW_COLOR],
        alpha=0.85,
    )
    ax_left.set_xlabel("Timestep")
    ax_left.set_ylabel("Academic capital (AC)")
    ax_left.set_title("Total AC by source over time")
    ax_left.legend(fontsize=8, loc="upper left")

    summary = history.agent_group_summary()
    groups = sorted(summary)
    writing_by_group = [float(summary[g].get("total_ac_from_writing", 0.0)) for g in groups]
    review_by_group = [float(summary[g].get("total_ac_from_reviewing", 0.0)) for g in groups]
    x = range(len(groups))
    ax_right.bar(x, writing_by_group, color=WRITING_COLOR, label="writing")
    ax_right.bar(
        x, review_by_group, bottom=writing_by_group, color=REVIEW_COLOR, label="peer review"
    )
    ax_right.set_xticks(list(x))
    ax_right.set_xticklabels(groups, rotation=20, ha="right")
    ax_right.set_ylabel("Final academic capital (AC)")
    ax_right.set_title("AC by source, per agent type")
    ax_right.legend(fontsize=8, loc="upper right")


def plot_ac_source(history: "History", path: str | None = None, show: bool = False):
    """Academic capital generated by writing vs peer review, overall and by agent type."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    _draw_ac_source(ax_left, ax_right, history)
    fig.suptitle("Academic capital: writing vs peer review", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


def _draw_system_aggregates(ax, history: "History") -> None:
    scalars = history.scalars
    for key in ("total_capital", "mean_capital", "max_capital"):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key.replace("_", " "))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Academic capital")
    ax.legend(loc="upper left", fontsize=8)
    if "capital_gini" in scalars:
        twin = ax.twinx()
        twin.plot(
            history.days,
            scalars["capital_gini"],
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="capital gini",
        )
        twin.set_ylabel("Gini (inequality)")
        twin.set_ylim(0.0, 1.0)
        twin.legend(loc="lower right", fontsize=8)


def plot_system_aggregates(history: "History", path: str | None = None, show: bool = False):
    """Total / mean / max capital, with the Gini inequality index on a twin axis."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_system_aggregates(ax, history)
    ax.set_title("System capital & inequality (Gini)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_behavior(history: "History", path: str | None = None, show: bool = False):
    """Cumulative good- vs bad-faith reviews (and paper count) over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    scalars = history.scalars
    for key in (
        "good_faith_reviews",
        "bad_faith_reviews",
        "completed_peer_reviews",
        "num_papers",
    ):
        if key in scalars:
            ax.plot(history.days, scalars[key], label=key)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Review behavior over time")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_paper_ac(history: "History", path: str | None = None, show: bool = False):
    """One thin line per paper: accrued capital (AC) over time (legend suppressed)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for series in history.paper_ac.values():
        ax.plot(history.days, series, linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Accrued capital (AC)")
    ax.set_title(f"Per-paper AC over time ({len(history.paper_ac)} papers)")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_marketplace(ax, history: "History") -> None:
    """Supply (papers waiting/listed/claimed) vs demand (reviews completed)."""
    scalars = history.scalars
    handles = []
    if "papers_on_market" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_on_market"],
            color="#f59e0b",
            label="papers on market",
        )
        handles.append(line)
    if "papers_listed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_listed_this_timestep"],
            color="#22c55e",
            linewidth=1.2,
            alpha=0.85,
            label="papers listed this timestep",
        )
        handles.append(line)
    if "papers_claimed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_claimed_this_timestep"],
            color="#ef4444",
            linewidth=1.2,
            alpha=0.85,
            label="papers claimed this timestep",
        )
        handles.append(line)
    if "papers_claimed_same_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_claimed_same_timestep"],
            color="#fb7185",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label="cumulative instant claims",
        )
        handles.append(line)
    ax.set_xlabel("Timestep")
    # Use the bright line colors for the axis labels too, so they read on both
    # the light runs/ PNGs and the transparent dark-themed gallery PNGs.
    ax.set_ylabel("Papers on market", color="#f59e0b")
    ax.tick_params(axis="y", labelcolor="#f59e0b")

    if "completed_peer_reviews" in scalars:
        twin = ax.twinx()
        (line,) = twin.plot(
            history.days,
            scalars["completed_peer_reviews"],
            color="#a855f7",
            linestyle="--",
            linewidth=1.6,
            label="cumulative reviews",
        )
        twin.set_ylabel("Reviews completed", color="#a855f7")
        twin.tick_params(axis="y", labelcolor="#a855f7")
        handles.append(line)

    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)


def plot_marketplace_activity(
    history: "History", path: str | None = None, show: bool = False
):
    """Market supply/listing/claim diagnostics against completed reviews."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace: supply, claims, and reviews completed")
    fig.tight_layout()
    return _finish(fig, path, show)


_MARKETPLACE_ZOOM_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (0, 100, "marketplace_0_100"),
    (200, 300, "marketplace_200_300"),
    (600, 700, "marketplace_600_700"),
    (1800, 1900, "marketplace_1800_1900"),
)


def _draw_marketplace_supply(
    ax, history: "History", *, xlim: tuple[int, int] | None = None
) -> None:
    """Supply/listing/claim series only (no cumulative reviews axis)."""
    scalars = history.scalars
    handles = []
    if "papers_on_market" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_on_market"],
            color="#f59e0b",
            label="papers on market",
        )
        handles.append(line)
    if "papers_listed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_listed_this_timestep"],
            color="#22c55e",
            linewidth=1.2,
            alpha=0.85,
            label="papers listed this timestep",
        )
        handles.append(line)
    if "papers_claimed_this_timestep" in scalars:
        (line,) = ax.plot(
            history.days,
            scalars["papers_claimed_this_timestep"],
            color="#ef4444",
            linewidth=1.2,
            alpha=0.85,
            label="papers claimed this timestep",
        )
        handles.append(line)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Count")
    if xlim is not None:
        ax.set_xlim(xlim)
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=8)


def plot_marketplace_zoom(
    history: "History",
    start: int,
    end: int,
    path: str | None = None,
    show: bool = False,
):
    """Zoomed marketplace supply view for a timestep window."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_marketplace_supply(ax, history, xlim=(start, end))
    ax.set_title(f"Review marketplace supply (timesteps {start}–{end})")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_marketplace_zoom_charts(
    history: "History", outdir: str, *, show: bool = False
) -> dict[str, str]:
    """Render the standard marketplace zoom windows into ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {}
    for start, end, name in _MARKETPLACE_ZOOM_WINDOWS:
        path = os.path.join(outdir, f"{name}.png")
        plot_marketplace_zoom(history, start, end, path, show=show)
        paths[name] = path
    return paths


ACCEPTED_REVIEW_PRICE_BIN_SIZE = 10


def _accepted_review_price_bin_series(
    history: "History",
    *,
    bin_size: int = ACCEPTED_REVIEW_PRICE_BIN_SIZE,
) -> tuple[list[float], list[float]]:
    """Mean agreed review share per ``bin_size``-timestep window."""
    claims = getattr(history, "accepted_review_claims", None) or []
    max_t = max(history.timesteps) if history.timesteps else 0
    if max_t <= 0:
        return [], []

    num_bins = (max_t - 1) // bin_size + 1
    sums = [0.0] * num_bins
    counts = [0] * num_bins
    for day, price in claims:
        t = int(day)
        if t <= 0:
            continue
        b = (t - 1) // bin_size
        if 0 <= b < num_bins:
            sums[b] += float(price)
            counts[b] += 1

    xs: list[float] = []
    ys: list[float] = []
    midpoint = bin_size / 2.0
    for b in range(num_bins):
        if counts[b] == 0:
            continue
        xs.append(b * bin_size + midpoint)
        ys.append(sums[b] / counts[b])
    return xs, ys


def _draw_accepted_review_price_binned(
    ax, history: "History", *, bin_size: int = ACCEPTED_REVIEW_PRICE_BIN_SIZE
) -> None:
    xs, ys = _accepted_review_price_bin_series(history, bin_size=bin_size)
    if not xs:
        if not getattr(history, "accepted_review_claims", None):
            msg = (
                "No accepted review prices recorded\n"
                "(re-run simulation to populate)"
            )
        else:
            msg = "No accepted review claims in this run"
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.set_axis_off()
        return
    ax.scatter(
        xs,
        ys,
        s=18,
        alpha=0.75,
        color="#60a5fa",
        edgecolors="#1e3a5f",
        linewidths=0.4,
    )
    ax.set_xlabel(f"Timestep ({bin_size}-step bin midpoint)")
    ax.set_ylabel("Mean accepted review price (share)")


def plot_accepted_review_price_binned(
    history: "History", path: str | None = None, show: bool = False
):
    """Scatter of mean agreed review share per 10-timestep bin."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_accepted_review_price_binned(ax, history)
    ax.set_title("Mean accepted peer review price (10-timestep bins)")
    fig.tight_layout()
    return _finish(fig, path, show)


REVIEW_FAITH_SHARE_PRICE_BINS = 8


def _paper_claim_share_prices(history: "History") -> dict[str, float]:
    """Map each reviewed paper to its agreed reviewer share at claim time."""
    claims_by_day: dict[int, list[float]] = defaultdict(list)
    for day, price in getattr(history, "accepted_review_claims", None) or []:
        claims_by_day[int(day)].append(float(price))

    started_by_day: dict[int, list[str]] = defaultdict(list)
    for day, _, kind, paper in history.actions:
        if kind == "review_started" and paper:
            started_by_day[int(day)].append(str(paper))

    paper_prices: dict[str, float] = {}
    for day in set(claims_by_day) | set(started_by_day):
        for paper, price in zip(started_by_day[day], claims_by_day[day]):
            paper_prices[paper] = price
    return paper_prices


def _review_faith_share_price_points(
    history: "History",
) -> list[tuple[float, str]]:
    """Return (share price, review_kind) for completed reviews with known price."""
    paper_prices = _paper_claim_share_prices(history)
    points: list[tuple[float, str]] = []
    for _, _, paper_label, _, review_kind in history.completed_reviews:
        if not paper_label or review_kind not in (GOOD_FAITH_REVIEW, BAD_FAITH_REVIEW):
            continue
        price = paper_prices.get(paper_label)
        if price is not None and price > 0.0:
            points.append((price, review_kind))
    return points


def _share_price_bin_edges(
    prices: list[float], *, n_bins: int
) -> tuple[list[float], int]:
    """Return bin edges and the effective bin count for share-price histograms."""
    p_min, p_max = min(prices), max(prices)
    if math.isclose(p_min, p_max):
        return [p_min - 0.0005, p_max + 0.0005], 1
    width = (p_max - p_min) / n_bins
    return [p_min + i * width for i in range(n_bins + 1)], n_bins


def _share_price_bin_index(price: float, edges: list[float], n_bins: int) -> int:
    if price >= edges[-1]:
        return n_bins - 1
    span = edges[-1] - edges[0]
    if span <= 0.0:
        return 0
    return min(n_bins - 1, max(0, int((price - edges[0]) / span * n_bins)))


def _draw_review_faith_by_share_price(
    ax_pct,
    ax_counts,
    history: "History",
    *,
    n_bins: int = REVIEW_FAITH_SHARE_PRICE_BINS,
) -> bool:
    """Good- vs bad-faith share by agreed review price bin."""
    points = _review_faith_share_price_points(history)
    if not points:
        for ax in (ax_pct, ax_counts):
            ax.text(
                0.5,
                0.5,
                "No completed reviews with share prices",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        return False

    good_color, bad_color = "#16a34a", "#f87171"
    prices = [p for p, _ in points]
    edges, n_bins = _share_price_bin_edges(prices, n_bins=n_bins)
    good_counts = [0] * n_bins
    bad_counts = [0] * n_bins
    for price, kind in points:
        idx = _share_price_bin_index(price, edges, n_bins)
        if kind == GOOD_FAITH_REVIEW:
            good_counts[idx] += 1
        else:
            bad_counts[idx] += 1

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    width = (edges[-1] - edges[0]) / max(n_bins, 1) * 0.85
    totals = [g + b for g, b in zip(good_counts, bad_counts)]
    good_pct = [100.0 * g / t if t else 0.0 for g, t in zip(good_counts, totals)]
    bad_pct = [100.0 * b / t if t else 0.0 for b, t in zip(bad_counts, totals)]

    ax_pct.bar(
        centers,
        good_pct,
        width=width,
        color=good_color,
        label="good faith",
        edgecolor="#14532d",
    )
    ax_pct.bar(
        centers,
        bad_pct,
        width=width,
        bottom=good_pct,
        color=bad_color,
        label="bad faith",
        edgecolor="#991b1b",
    )
    ax_pct.set_ylim(0, 100)
    ax_pct.set_xlabel("Agreed review share (bin center)")
    ax_pct.set_ylabel("Share of completed reviews (%)")
    ax_pct.set_title("Good vs bad faith by share price")
    ax_pct.legend(fontsize=8, loc="upper right")

    for center, good, bad, total in zip(centers, good_counts, bad_counts, totals):
        if total <= 0:
            continue
        ax_pct.text(
            center,
            102,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#8b98a5",
        )

    ax_counts.bar(
        centers,
        good_counts,
        width=width,
        color=good_color,
        label="good faith",
        edgecolor="#14532d",
    )
    ax_counts.bar(
        centers,
        bad_counts,
        width=width,
        bottom=good_counts,
        color=bad_color,
        label="bad faith",
        edgecolor="#991b1b",
    )
    ax_counts.set_xlabel("Agreed review share (bin center)")
    ax_counts.set_ylabel("Completed reviews")
    ax_counts.set_title("Review counts by share-price bin")
    ax_counts.legend(fontsize=8, loc="upper left")

    total_good = sum(good_counts)
    total_bad = sum(bad_counts)
    total_reviews = total_good + total_bad
    if total_reviews:
        ax_counts.text(
            0.98,
            0.97,
            f"overall: {100.0 * total_good / total_reviews:.1f}% good, "
            f"{100.0 * total_bad / total_reviews:.1f}% bad (n={total_reviews})",
            transform=ax_counts.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    return True


def plot_review_faith_by_share_price(
    history: "History", path: str | None = None, show: bool = False
):
    """Percentage of good- vs bad-faith reviews at different agreed share prices."""
    fig, (ax_pct, ax_counts) = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_review_faith_by_share_price(ax_pct, ax_counts, history)
    fig.suptitle(
        "Review faith vs agreed share price",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _finish(fig, path, show)


def _scalar_series(history: "History", key: str) -> list[float]:
    values = history.scalars.get(key) or []
    return [float(x) for x in values]


def _market_pricing_claim_points(
    history: "History",
) -> list[tuple[int, float]]:
    claims = getattr(history, "accepted_review_claims", None) or []
    points: list[tuple[int, float]] = []
    for timestep, price in claims:
        value = float(price)
        if value > 0.0:
            points.append((int(timestep), value))
    return points


def _aligned_scalar_plot(
    history: "History", key: str
) -> tuple[list[float], list[float]]:
    """Return (timesteps, values) for a scalar series aligned to ``history.days``."""
    values = _scalar_series(history, key)
    if not values:
        return [], []
    days = [float(d) for d in history.days[: len(values)]]
    return days, values


def _draw_market_pricing_dynamics(
    ax_top, ax_bottom, history: "History"
) -> None:
    """Within-run review pricing: market drivers and agreed claim prices."""
    if not history.days:
        for ax in (ax_top, ax_bottom):
            ax.text(0.5, 0.5, "No timestep data", ha="center", va="center")
            ax.set_axis_off()
        return

    top_handles: list[Any] = []
    fair_days, fair_prices = _aligned_scalar_plot(history, "fair_market_price")
    if fair_days:
        (line,) = ax_top.plot(
            fair_days,
            fair_prices,
            color="#60a5fa",
            linewidth=1.8,
            label="fair market price",
        )
        top_handles.append(line)

    claims = _market_pricing_claim_points(history)
    if claims:
        claim_x, claim_y = zip(*claims)
        scatter = ax_top.scatter(
            claim_x,
            claim_y,
            s=22,
            alpha=0.75,
            color="#fbbf24",
            edgecolors="#78350f",
            linewidths=0.4,
            label="accepted claim price",
            zorder=5,
        )
        top_handles.append(scatter)

    author_days, author_mult = _aligned_scalar_plot(
        history, "mean_author_price_multiplier"
    )
    if author_days:
        twin_top = ax_top.twinx()
        (line,) = twin_top.plot(
            author_days,
            author_mult,
            color="#34d399",
            linestyle="--",
            linewidth=1.4,
            label="mean author multiplier",
        )
        twin_top.set_ylabel("Author multiplier", color="#34d399")
        twin_top.tick_params(axis="y", labelcolor="#34d399")
        top_handles.append(line)

    ax_top.set_ylabel("Review share")
    ax_top.set_title("Offer baseline and agreed claim prices")
    if top_handles:
        ax_top.legend(
            top_handles,
            [h.get_label() for h in top_handles],
            loc="upper right",
            fontsize=8,
        )

    bottom_handles: list[Any] = []
    scarcity_days, scarcity = _aligned_scalar_plot(history, "scarcity_multiplier")
    if scarcity_days:
        (line,) = ax_bottom.plot(
            scarcity_days,
            scarcity,
            color="#f472b6",
            linewidth=1.8,
            label="scarcity multiplier",
        )
        bottom_handles.append(line)

    pressure_days, pressure = _aligned_scalar_plot(history, "reviewer_pressure")
    if pressure_days:
        twin_bottom = ax_bottom.twinx()
        (line,) = twin_bottom.plot(
            pressure_days,
            pressure,
            color="#a78bfa",
            linestyle="--",
            linewidth=1.4,
            label="reviewer pressure",
        )
        twin_bottom.set_ylabel("Reviewers / listed papers", color="#a78bfa")
        twin_bottom.tick_params(axis="y", labelcolor="#a78bfa")
        bottom_handles.append(line)

    ax_bottom.axhline(
        1.0,
        color="#6b7280",
        linestyle=":",
        linewidth=1.0,
        alpha=0.7,
    )
    ax_bottom.set_xlabel("Timestep")
    ax_bottom.set_ylabel("Scarcity multiplier", color="#f472b6")
    ax_bottom.tick_params(axis="y", labelcolor="#f472b6")
    ax_bottom.set_title(
        "Market pressure (pressure > 1 ⇒ more reviewers than papers ⇒ lower offers)"
    )
    if bottom_handles:
        ax_bottom.legend(
            bottom_handles,
            [h.get_label() for h in bottom_handles],
            loc="upper right",
            fontsize=8,
        )

    if not top_handles and not bottom_handles:
        for ax in (ax_top, ax_bottom):
            ax.text(
                0.5,
                0.5,
                "No pricing scalars recorded\n(re-run simulation to populate)",
                ha="center",
                va="center",
            )
            ax.set_axis_off()


def plot_market_pricing_dynamics(
    history: "History", path: str | None = None, show: bool = False
):
    """Review share pricing over timesteps: drivers, pressure, and claim prices."""
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    _draw_market_pricing_dynamics(ax_top, ax_bottom, history)
    fig.suptitle(
        "Dynamic review pricing during the run",
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, path, show)


def _paper_quality_ac_points(history: "History") -> list[tuple[float, float, bool]]:
    """(quality, final AC, reviewed?) for every paper that was tracked."""
    points: list[tuple[float, float, bool]] = []
    for label, series in history.paper_ac.items():
        quality = history.paper_quality.get(label)
        if quality is None or not series:
            continue
        points.append((quality, series[-1], history.paper_reviewed.get(label, False)))
    return points


def _paper_quality_citation_points(
    history: "History",
) -> list[tuple[float, float, bool]]:
    """(quality, citations received, reviewed?) for every paper that was tracked."""
    counts = getattr(history, "paper_citation_count", {})
    points: list[tuple[float, float, bool]] = []
    for label, count in counts.items():
        quality = history.paper_quality.get(label)
        if quality is None:
            continue
        points.append((quality, float(count), history.paper_reviewed.get(label, False)))
    return points


def _draw_quality_vs_ac(ax, history: "History") -> None:
    _draw_quality_scatter(
        ax, _paper_quality_ac_points(history), "Final accrued capital (AC)"
    )


def _draw_quality_scatter(
    ax, points: list[tuple[float, float, bool]], ylabel: str
) -> None:
    if not points:
        ax.text(0.5, 0.5, "No paper data", ha="center", va="center")
        ax.set_axis_off()
        return
    reviewed = [(q, a) for q, a, r in points if r]
    plain = [(q, a) for q, a, r in points if not r]
    if plain:
        qs, acs = zip(*plain)
        ax.scatter(qs, acs, s=16, alpha=0.45, color="#94a3b8", label="not reviewed")
    if reviewed:
        qs, acs = zip(*reviewed)
        ax.scatter(
            qs, acs, s=26, alpha=0.8, color="#a855f7",
            edgecolors="#4c1d95", label="reviewed",
        )
    ax.set_xlabel("Paper quality")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def plot_paper_quality_vs_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Does quality pay off? Final AC vs quality, split by whether reviewed."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_quality_vs_ac(ax, history)
    ax.set_title("Paper quality vs accrued capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_paper_quality_vs_citations(
    history: "History", path: str | None = None, show: bool = False
):
    """Citations received vs quality, split by whether reviewed."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_quality_scatter(
        ax, _paper_quality_citation_points(history), "Citations received"
    )
    ax.set_title("Paper quality vs citations")
    fig.tight_layout()
    return _finish(fig, path, show)


def _paper_quality_review_faith_points(
    history: "History",
) -> list[tuple[float, float, str]]:
    """Return (paper quality, review effort, review_kind) for completed reviews."""
    qualities = getattr(history, "paper_quality", {})
    points: list[tuple[float, float, str]] = []
    for _, _, paper_label, effort, review_kind in history.completed_reviews:
        if paper_label not in qualities or review_kind is None:
            continue
        points.append((float(qualities[paper_label]), float(effort), str(review_kind)))
    return points


def _draw_paper_quality_vs_review_faith(
    ax_scatter, ax_bins, history: "History", *, threshold: float
) -> bool:
    """Paper quality vs review faith: scatter by effort and binned good/bad counts."""
    points = _paper_quality_review_faith_points(history)
    if not points:
        for ax in (ax_scatter, ax_bins):
            ax.text(0.5, 0.5, "No completed reviews with paper quality", ha="center", va="center")
            ax.set_axis_off()
        return False

    good_color, bad_color = "#16a34a", "#f87171"
    good = [(q, e) for q, e, k in points if k == GOOD_FAITH_REVIEW]
    bad = [(q, e) for q, e, k in points if k == BAD_FAITH_REVIEW]

    if bad:
        qs, es = zip(*bad)
        ax_scatter.scatter(
            qs, es, s=22, alpha=0.55, color=bad_color,
            edgecolors="#991b1b", label="bad faith",
        )
    if good:
        qs, es = zip(*good)
        ax_scatter.scatter(
            qs, es, s=28, alpha=0.7, color=good_color,
            edgecolors="#14532d", label="good faith",
        )
    ax_scatter.axhline(
        threshold, color="#6b7280", linestyle="--", linewidth=1.0,
        label=f"good-faith threshold ({threshold:g})",
    )
    ax_scatter.set_xlabel("Paper quality")
    ax_scatter.set_ylabel("Review effort (timesteps)")
    ax_scatter.set_title("Each completed review")
    ax_scatter.legend(fontsize=8, loc="best")

    qualities = [q for q, _, _ in points]
    q_min, q_max = min(qualities), max(qualities)
    n_bins = 8
    if math.isclose(q_min, q_max):
        edges = [q_min - 0.05, q_max + 0.05]
        n_bins = 1
    else:
        width = (q_max - q_min) / n_bins
        edges = [q_min + i * width for i in range(n_bins + 1)]

    good_counts = [0] * n_bins
    bad_counts = [0] * n_bins
    for quality, _, kind in points:
        idx = n_bins - 1 if quality >= edges[-1] else min(
            n_bins - 1, max(0, int((quality - edges[0]) / (edges[-1] - edges[0]) * n_bins))
        )
        if kind == GOOD_FAITH_REVIEW:
            good_counts[idx] += 1
        elif kind == BAD_FAITH_REVIEW:
            bad_counts[idx] += 1

    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]
    width = (edges[-1] - edges[0]) / max(n_bins, 1) * 0.85
    ax_bins.bar(
        centers, good_counts, width=width, color=good_color,
        label="good faith", edgecolor="#14532d",
    )
    ax_bins.bar(
        centers, bad_counts, width=width, bottom=good_counts, color=bad_color,
        label="bad faith", edgecolor="#991b1b",
    )
    ax_bins.set_xlabel("Paper quality (bin center)")
    ax_bins.set_ylabel("Completed reviews")
    ax_bins.set_title("Review counts by paper-quality bin")
    ax_bins.legend(fontsize=8, loc="upper left")

    if good and bad:
        mean_good_q = sum(q for q, _ in good) / len(good)
        mean_bad_q = sum(q for q, _ in bad) / len(bad)
        ax_bins.text(
            0.98, 0.97,
            f"mean quality: good={mean_good_q:.2f}, bad={mean_bad_q:.2f}",
            transform=ax_bins.transAxes, ha="right", va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
    return True


def plot_paper_quality_vs_review_faith(
    history: "History",
    path: str | None = None,
    show: bool = False,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
):
    """Do higher-quality papers attract good-faith or bad-faith reviews?"""
    fig, (ax_scatter, ax_bins) = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_paper_quality_vs_review_faith(ax_scatter, ax_bins, history, threshold=threshold)
    fig.suptitle(
        "Paper quality vs review faith",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_writing_effort_vs_rate(ax, history: "History") -> None:
    """How much authors invested in each paper vs the base accrual rate it earned.

    Continuous mode: a paper's base rate approaches its quality ceiling as
    writing effort grows, so the cloud should trace a rising, saturating band.
    """
    efforts = getattr(history, "paper_writing_effort", {})
    rates = getattr(history, "paper_accrual_rate", {})
    points = [
        (efforts[label], rates[label])
        for label in efforts
        if label in rates
    ]
    if not points:
        ax.text(
            0.5, 0.5, "No writing-effort data\n(continuous mode only)",
            ha="center", va="center",
        )
        ax.set_axis_off()
        return
    xs, ys = zip(*points)
    ax.scatter(xs, ys, s=18, alpha=0.5, color="#0ea5e9", edgecolors="#075985")
    ax.set_xlabel("Writing effort invested (timesteps)")
    ax.set_ylabel("Base accrual rate at finish")


def plot_writing_effort_vs_rate(
    history: "History", path: str | None = None, show: bool = False
):
    """Per-paper writing effort vs the base accrual rate it locked in."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_writing_effort_vs_rate(ax, history)
    ax.set_title("Writing effort vs accrual rate (asymptote)")
    fig.tight_layout()
    return _finish(fig, path, show)


def _paper_writing_effort_over_time_points(
    history: "History",
) -> list[tuple[int, float, str]]:
    """Return (first-seen timestep, writing effort, author group) per paper."""
    efforts = getattr(history, "paper_writing_effort", {})
    first_seen = getattr(history, "paper_first_seen_timestep", {})
    authors = getattr(history, "paper_authors", {})
    groups = getattr(history, "agent_groups", {})
    paper_ac = getattr(history, "paper_ac", {})
    days = getattr(history, "days", [])

    points: list[tuple[int, float, str]] = []
    for paper, effort in efforts.items():
        timestep = first_seen.get(paper)
        if timestep is None:
            series = paper_ac.get(paper, [])
            for i, value in enumerate(series):
                if value > 0:
                    timestep = days[i] if i < len(days) else i
                    break
        if timestep is None:
            continue
        author = authors.get(paper)
        group = groups.get(author, "Agent")
        points.append((int(timestep), float(effort), group))
    return points


def _draw_paper_writing_effort_over_time(ax, history: "History") -> None:
    """Show whether papers are being published with more/less effort over time."""
    points = _paper_writing_effort_over_time_points(history)
    if not points:
        ax.text(
            0.5, 0.5, "No per-paper writing effort over time recorded",
            ha="center", va="center",
        )
        ax.set_axis_off()
        return

    points_by_group: dict[str, list[tuple[int, float]]] = {}
    for timestep, effort, group in points:
        points_by_group.setdefault(group, []).append((timestep, effort))

    cmap = plt.get_cmap("tab10")
    for i, group in enumerate(sorted(points_by_group)):
        xs, ys = zip(*points_by_group[group])
        ax.scatter(
            xs,
            ys,
            s=22,
            alpha=0.65,
            color=cmap(i % 10),
            edgecolors="#1e293b",
            linewidths=0.3,
            label=group,
        )
    ax.set_xlabel("Timestep paper appeared")
    ax.set_ylabel("Writing effort at publication")
    ax.legend(fontsize=8)


def plot_paper_writing_effort_over_time(
    history: "History", path: str | None = None, show: bool = False
):
    """Per-paper writing effort at publication, colored by author type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_paper_writing_effort_over_time(ax, history)
    ax.set_title("Paper-writing effort over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_review_reputation(ax, history: "History") -> None:
    """Per-agent peer-review reputation over time, with the mean highlighted."""
    series_map = getattr(history, "agent_review_history", {})
    drew = False
    for series in series_map.values():
        if any(series):
            ax.plot(history.days, series, linewidth=0.8, alpha=0.4, color="#38bdf8")
            drew = True
    mean_series = history.scalars.get("mean_peer_review_history")
    if mean_series is not None:
        ax.plot(
            history.days, mean_series, linewidth=2.0, color="#0369a1",
            label="mean reputation",
        )
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "No reviews completed", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Peer-review reputation\n(share-weighted accrual rate per review)")
    ax.legend(fontsize=8, loc="upper left")


def plot_review_reputation(
    history: "History", path: str | None = None, show: bool = False
):
    """Reviewer reputation (share-weighted accrual rate per completed review) over time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_review_reputation(ax, history)
    ax.set_title("Reviewer reputation over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _reputation_vs_ac_points(
    history: "History",
) -> list[tuple[float, float, str]]:
    """Return (reputation, final_capital, group) for each agent."""
    cached = getattr(history, "_agent_outcome_summary", None)
    if cached:
        return [
            (
                float(row["peer_review_history"]),
                float(row["final_capital"]),
                row.get("group", "Agent"),
            )
            for row in cached
        ]

    if history.agent_review_history:
        return [
            (
                float(row["peer_review_history"]),
                float(row["final_capital"]),
                row.get("group", "Agent"),
            )
            for row in history.agent_outcome_summary()
        ]

    final_rep = getattr(history, "_agent_final_reputation", None) or {}
    points: list[tuple[float, float, str]] = []
    for label, cap_series in history.agent_capital.items():
        if not cap_series:
            continue
        points.append(
            (
                float(final_rep.get(label, 0.0)),
                float(cap_series[-1]),
                history.agent_groups.get(label, "Agent"),
            )
        )
    return points


def _reputation_vs_review_ac_points(
    history: "History",
) -> list[tuple[float, float, str]]:
    """Return (reputation, ac_from_reviewing, group) for each agent."""
    cached = getattr(history, "_agent_outcome_summary", None)
    if cached:
        return [
            (
                float(row["peer_review_history"]),
                float(row.get("ac_from_reviewing", 0.0)),
                row.get("group", "Agent"),
            )
            for row in cached
        ]

    if history.agent_review_history or history.paper_reviewer:
        return [
            (
                float(row["peer_review_history"]),
                float(row["ac_from_reviewing"]),
                row.get("group", "Agent"),
            )
            for row in history.agent_outcome_summary()
        ]

    return []


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def _draw_reputation_vs_ac(ax, history: "History") -> bool:
    """Scatter final peer-review reputation vs total academic capital by agent."""
    points = _reputation_vs_ac_points(history)
    if not points:
        ax.text(0.5, 0.5, "No reputation data recorded", ha="center", va="center")
        ax.set_axis_off()
        return False

    by_group: dict[str, list[tuple[float, float]]] = {}
    for reputation, capital, group in points:
        by_group.setdefault(group, []).append((reputation, capital))

    cmap = plt.get_cmap("tab10")
    ordered_groups = sorted(by_group)
    color_for = {group: cmap(i % 10) for i, group in enumerate(ordered_groups)}
    for group in ordered_groups:
        xs, ys = zip(*by_group[group])
        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.75,
            color=color_for[group],
            edgecolors="#1e293b",
            linewidths=0.4,
            label=group,
        )

    reputations = [p[0] for p in points]
    capitals = [p[1] for p in points]
    corr = _pearson(reputations, capitals)
    if not math.isnan(corr):
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {corr:.3f}  (n = {len(points)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_xlabel("Peer-review reputation (share-weighted accrual rate per review)")
    ax.set_ylabel("Final academic capital")
    if len(ordered_groups) > 1:
        ax.legend(fontsize=8, loc="lower right")
    return True


def _draw_reputation_vs_review_ac(ax, history: "History") -> bool:
    """Scatter peer-review reputation vs AC held via reviewer shares."""
    points = _reputation_vs_review_ac_points(history)
    if not points:
        ax.text(0.5, 0.5, "No review-capital data recorded", ha="center", va="center")
        ax.set_axis_off()
        return False

    by_group: dict[str, list[tuple[float, float]]] = {}
    for reputation, review_ac, group in points:
        by_group.setdefault(group, []).append((reputation, review_ac))

    cmap = plt.get_cmap("tab10")
    ordered_groups = sorted(by_group)
    color_for = {group: cmap(i % 10) for i, group in enumerate(ordered_groups)}
    for group in ordered_groups:
        xs, ys = zip(*by_group[group])
        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.75,
            color=color_for[group],
            edgecolors="#1e293b",
            linewidths=0.4,
            label=group,
        )

    reputations = [p[0] for p in points]
    review_acs = [p[1] for p in points]
    corr = _pearson(reputations, review_acs)
    if not math.isnan(corr):
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {corr:.3f}  (n = {len(points)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_xlabel("Peer-review reputation (share-weighted accrual rate per review)")
    ax.set_ylabel("Academic capital from peer review")
    if len(ordered_groups) > 1:
        ax.legend(fontsize=8, loc="lower right")
    return True


def plot_reputation_vs_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Final academic capital vs peer-review reputation, colored by agent type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_reputation_vs_ac(ax, history)
    ax.set_title("Reviewer reputation vs final academic capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_reputation_vs_review_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """AC earned via reviewer shares vs peer-review reputation, by agent type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_reputation_vs_review_ac(ax, history)
    ax.set_title("Reviewer reputation vs peer-review academic capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def _talent_vs_final_ac_points(
    history: "History",
) -> list[tuple[float, float, str]]:
    """Return (talent, final_capital, group) for each agent."""
    cached = getattr(history, "_agent_outcome_summary", None)
    if cached:
        points: list[tuple[float, float, str]] = []
        for row in cached:
            agent = row.get("agent", "")
            talent = row.get("talent")
            if talent is None:
                talent = history.agent_talent.get(agent)
            if talent is None:
                continue
            points.append(
                (
                    float(talent),
                    float(row["final_capital"]),
                    row.get("group", "Agent"),
                )
            )
        if points:
            return points

    if history.agent_talent:
        return [
            (
                float(row["talent"]),
                float(row["final_capital"]),
                row.get("group", "Agent"),
            )
            for row in history.agent_outcome_summary()
        ]

    points = []
    for label, cap_series in history.agent_capital.items():
        if not cap_series:
            continue
        points.append(
            (
                float(history.agent_talent.get(label, 0.0)),
                float(cap_series[-1]),
                history.agent_groups.get(label, "Agent"),
            )
        )
    return points


def _talent_vs_review_ac_points(
    history: "History",
) -> list[tuple[float, float, str]]:
    """Return (talent, ac_from_reviewing, group) for each agent."""
    cached = getattr(history, "_agent_outcome_summary", None)
    if cached:
        points: list[tuple[float, float, str]] = []
        for row in cached:
            agent = row.get("agent", "")
            talent = row.get("talent")
            if talent is None:
                talent = history.agent_talent.get(agent)
            if talent is None:
                continue
            points.append(
                (
                    float(talent),
                    float(row.get("ac_from_reviewing", 0.0)),
                    row.get("group", "Agent"),
                )
            )
        if points:
            return points

    if history.agent_talent or history.agent_review_history or history.paper_reviewer:
        return [
            (
                float(row["talent"]),
                float(row["ac_from_reviewing"]),
                row.get("group", "Agent"),
            )
            for row in history.agent_outcome_summary()
        ]

    return []


def _draw_talent_scatter(
    ax,
    points: list[tuple[float, float, str]],
    *,
    xlabel: str,
    ylabel: str,
    empty_message: str,
) -> bool:
    if not points:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center")
        ax.set_axis_off()
        return False

    by_group: dict[str, list[tuple[float, float]]] = {}
    for x_val, y_val, group in points:
        by_group.setdefault(group, []).append((x_val, y_val))

    cmap = plt.get_cmap("tab10")
    ordered_groups = sorted(by_group)
    color_for = {group: cmap(i % 10) for i, group in enumerate(ordered_groups)}
    for group in ordered_groups:
        xs, ys = zip(*by_group[group])
        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.75,
            color=color_for[group],
            edgecolors="#1e293b",
            linewidths=0.4,
            label=group,
        )

    xs_all = [p[0] for p in points]
    ys_all = [p[1] for p in points]
    corr = _pearson(xs_all, ys_all)
    if not math.isnan(corr):
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {corr:.3f}  (n = {len(points)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(ordered_groups) > 1:
        ax.legend(fontsize=8, loc="lower right")
    return True


def _draw_talent_vs_final_ac(ax, history: "History") -> bool:
    """Scatter intrinsic talent vs total academic capital by agent."""
    return _draw_talent_scatter(
        ax,
        _talent_vs_final_ac_points(history),
        xlabel="Intrinsic talent",
        ylabel="Final academic capital",
        empty_message="No talent data recorded",
    )


def _draw_talent_vs_review_ac(ax, history: "History") -> bool:
    """Scatter intrinsic talent vs AC held via reviewer shares."""
    return _draw_talent_scatter(
        ax,
        _talent_vs_review_ac_points(history),
        xlabel="Intrinsic talent",
        ylabel="Academic capital from peer review",
        empty_message="No review-capital data recorded",
    )


def plot_talent_vs_final_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Final academic capital vs intrinsic talent, colored by agent type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_talent_vs_final_ac(ax, history)
    ax.set_title("Intrinsic talent vs final academic capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_talent_vs_review_ac(
    history: "History", path: str | None = None, show: bool = False
):
    """Peer-review AC held vs intrinsic talent, colored by agent type."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_talent_vs_review_ac(ax, history)
    ax.set_title("Intrinsic talent vs peer-review academic capital")
    fig.tight_layout()
    return _finish(fig, path, show)


def _daily_action_counts(history: "History") -> dict[str, list[int]]:
    """Per-day counts for each raw action kind."""
    kinds = sorted({kind for _, _, kind, _ in history.actions})
    counts = {kind: [0] * len(history.days) for kind in kinds}
    day_index = {day: i for i, day in enumerate(history.days)}

    for day, _, kind, _ in history.actions:
        idx = day_index.get(day)
        if idx is not None:
            counts[kind][idx] += 1
    return counts


def _completed_reviews(history: "History") -> list[tuple[int, str, float]]:
    """Return (day, reviewer label, effort) for each completed peer review."""
    return [(row[0], row[1], row[3]) for row in history.completed_reviews]


def _writing_effort_by_agent(history: "History") -> dict[str, float]:
    """Total writing effort accumulated by each agent."""
    totals: dict[str, float] = {}
    for _, agent, effort, _ in getattr(history, "writing_efforts", []):
        totals[agent] = totals.get(agent, 0.0) + effort
    return totals


def _draw_effort_histogram(
    ax, history: "History", threshold: float = MIN_REVIEW_EFFORT_THRESHOLD
) -> None:
    reviews = _completed_reviews(history)
    efforts = [effort for _, _, effort in reviews]
    if not efforts:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    lo = math.floor(min(efforts))
    hi = math.ceil(max(efforts))
    efforts_by_group: dict[str, list[float]] = {}
    for _, agent, effort in reviews:
        group = history.agent_groups.get(agent, "Agent")
        efforts_by_group.setdefault(group, []).append(effort)

    bins = range(lo, hi + 2)
    if len(efforts_by_group) > 1:
        groups = sorted(efforts_by_group)
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i, _ in enumerate(groups)]
        ax.hist(
            [efforts_by_group[group] for group in groups],
            bins=bins,
            align="left",
            rwidth=0.9,
            stacked=True,
            color=colors,
            edgecolor="#1e3a5f",
            label=groups,
        )
    else:
        ax.hist(
            efforts,
            bins=bins,
            align="left",
            rwidth=0.9,
            color="#60a5fa",
            edgecolor="#1e3a5f",
        )
    # One tick per integer effort when the range is small; otherwise let
    # matplotlib thin them so wide-effort runs don't overlap their labels.
    if hi - lo <= 30:
        ax.set_xticks(range(lo, hi + 1))
    ax.axvline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Review effort")
    ax.set_ylabel("Completed peer reviews")
    ax.legend(fontsize=8)


def _draw_effort_scatter(
    ax, history: "History", threshold: float = MIN_REVIEW_EFFORT_THRESHOLD
) -> None:
    points = _completed_reviews(history)
    if not points:
        ax.text(0.5, 0.5, "No completed reviews", ha="center", va="center")
        ax.set_axis_off()
        return
    timesteps = [day for day, _, _ in points]
    efforts = [effort for _, _, effort in points]
    ax.scatter(timesteps, efforts, alpha=0.65, s=28, color="#a855f7", edgecolors="#4c1d95")
    ax.axhline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="reward threshold",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Review effort")
    ax.legend(fontsize=8)


def plot_review_effort_histogram(
    history: "History",
    path: str | None = None,
    show: bool = False,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
):
    """Histogram of completed peer reviews by effort level."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_histogram(ax, history, threshold)
    ax.set_title("Completed peer reviews by effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_review_effort_scatter(
    history: "History",
    path: str | None = None,
    show: bool = False,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
):
    """Scatter of each completed review: day vs effort invested."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_effort_scatter(ax, history, threshold)
    ax.set_title("Completed peer reviews over time")
    fig.tight_layout()
    return _finish(fig, path, show)


def _reward_curve_max_effort(sim=SIM) -> float:
    """Upper x-axis bound for the review reward curve (matches gallery JS)."""
    threshold = float(sim.min_review_effort_threshold)
    saturation = float(sim.review_sigmoid_saturation_effort)
    good_ts = float(sim.good_review_timesteps)
    curve = str(sim.review_effort_curve).strip().lower()
    if curve == "log":
        return max(good_ts * 4, threshold + 15, 20)
    if curve == "jump":
        return max(good_ts * 2.5, float(sim.review_jump_threshold) + 5)
    return max(saturation * 1.15, threshold + 8)


def _reward_curve_label(sim=SIM) -> str:
    curve = str(sim.review_effort_curve).strip().lower()
    if curve == "log":
        return "log: base + first_extra × log₂(1 + extra effort)"
    if curve == "jump":
        return "jump: sigmoid baseline + high-effort threshold bonus"
    return "sigmoid: min + (max − min) × normalized sigmoid"


def _draw_review_reward_curve(ax, *, sim=SIM) -> None:
    """Plot review accrual bump E = F(T) using the active Paper reward function."""
    threshold = float(paper_mod.MIN_REVIEW_EFFORT_THRESHOLD)
    xmax = _reward_curve_max_effort(sim)
    steps = 80
    efforts = [xmax * i / steps for i in range(steps + 1)]
    bumps = [paper_mod.review_accrual_bump(effort, quality=1.0) for effort in efforts]

    ax.plot(efforts, bumps, linewidth=2.0, color="#16a34a", label="accrual bump")
    ax.fill_between(efforts, bumps, alpha=0.2, color="#16a34a")
    ax.axvline(
        threshold,
        color="#dc2626",
        linestyle="--",
        linewidth=1.2,
        label=f"reward threshold ({threshold:g})",
    )

    if str(sim.review_paradigm).strip().lower() == "discrete":
        for effort, label in (
            (float(sim.bad_review_timesteps), "bad"),
            (float(sim.good_review_timesteps), "good"),
        ):
            bump = paper_mod.review_accrual_bump(effort, quality=1.0)
            ax.scatter(
                [effort],
                [bump],
                s=80,
                marker="D",
                color="#fbbf24",
                edgecolors="#1e293b",
                linewidths=0.8,
                zorder=5,
                label=f"discrete {label} ({effort:g})",
            )

    ax.set_xlim(0, xmax)
    ymax = max(bumps) if bumps else 1.0
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.set_xlabel("Review effort (timesteps)")
    ax.set_ylabel("Accrual bump (ε)")
    ax.legend(fontsize=8, loc="best")


def plot_review_reward_curve(
    history: "History | None" = None,
    path: str | None = None,
    show: bool = False,
    *,
    sim=SIM,
):
    """Review accrual bump E = F(T) for the active curve and threshold."""
    del history  # config-driven chart; history unused
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_review_reward_curve(ax, sim=sim)
    ax.set_title(
        f"Review reward curve — {_reward_curve_label(sim)}\n"
        "new rate = base × (1 + bump), quality 1.0",
    )
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_writing_effort_distribution(
    history: "History", path: str | None = None, show: bool = False
):
    """Histogram of total paper-writing effort accumulated by each agent."""
    totals = _writing_effort_by_agent(history)
    fig, ax = plt.subplots(figsize=(11, 6))
    if not totals:
        ax.text(0.5, 0.5, "No writing effort recorded", ha="center", va="center")
        ax.set_axis_off()
    else:
        values = list(totals.values())
        bins = max(1, min(20, len(values)))
        ax.hist(values, bins=bins, color="#60a5fa", edgecolor="#1e3a5f")
        ax.set_xlabel("Total writing effort per agent")
        ax.set_ylabel("Agents")
        ax.set_title("Distribution of paper-writing effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_paper_writing_effort_distribution(
    history: "History", path: str | None = None, show: bool = False
):
    """Histogram of writing effort across all papers (effort vs paper count)."""
    values = list(getattr(history, "paper_writing_effort", {}).values())
    fig, ax = plt.subplots(figsize=(11, 6))
    if not values:
        ax.text(
            0.5, 0.5, "No per-paper writing effort recorded\n(continuous mode only)",
            ha="center", va="center",
        )
        ax.set_axis_off()
    else:
        bins = max(1, min(20, len(values)))
        ax.hist(values, bins=bins, color="#60a5fa", edgecolor="#1e3a5f")
        ax.set_xlabel("Writing effort per paper")
        ax.set_ylabel("Papers")
        ax.set_title("Distribution of per-paper writing effort")
    fig.tight_layout()
    return _finish(fig, path, show)


def _choice_tallies(history: "History") -> dict[str, int]:
    """Count top-level decisions across all agents."""
    tallies: dict[str, int] = {}
    for _, _, kind, _ in history.actions:
        decision = DECISION_LABELS.get(kind)
        if decision is None:
            continue
        tallies[decision] = tallies.get(decision, 0) + 1
    return tallies


def _stacked_action_bars(ax, history: "History") -> None:
    counts = _daily_action_counts(history)
    if not counts:
        ax.text(0.5, 0.5, "No actions recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    bottom = [0] * len(history.days)
    for kind in sorted(
        counts,
        key=lambda name: list(ACTION_COLORS).index(name)
        if name in ACTION_COLORS
        else len(ACTION_COLORS),
    ):
        values = counts[kind]
        ax.bar(
            history.days,
            values,
            bottom=bottom,
            label=kind.replace("_", " "),
            color=ACTION_COLORS.get(kind, "#94a3b8"),
            width=0.9,
        )
        bottom = [b + v for b, v in zip(bottom, values)]
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Actions per timestep")
    ax.legend(fontsize=8, ncol=2, loc="upper left")


def plot_action_mix_over_time(
    history: "History", path: str | None = None, show: bool = False
):
    """Stacked daily counts of every action kind (what agents did each day)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _stacked_action_bars(ax, history)
    ax.set_title("What agents did each timestep (stacked action counts)")
    fig.tight_layout()
    return _finish(fig, path, show)


_DECISION_ORDER = (
    "write_paper",
    "bad_faith_review",
    "good_faith_review",
    "start_review",
    "continue_review",
    "finish_and_write",
    "finish_and_review",
    "idle",
)


def _draw_choice_breakdown(ax, history: "History", fontsize: int = 9) -> None:
    tallies = _choice_tallies(history)
    labels = [d for d in _DECISION_ORDER if tallies.get(d, 0)]
    if not labels:
        ax.text(0.5, 0.5, "No decisions recorded", ha="center", va="center")
        ax.set_axis_off()
        return
    values = [tallies[d] for d in labels]
    total = sum(values)
    bars = ax.bar(
        range(len(labels)),
        values,
        color=[DECISION_COLORS.get(label, "#94a3b8") for label in labels],
    )
    for rect, value in zip(bars, values):
        ax.annotate(
            f"{value / total:.0%}",
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            ha="center", va="bottom", fontsize=fontsize - 1,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [label.replace("_", " ") for label in labels],
        rotation=20, ha="right", fontsize=fontsize,
    )
    ax.set_ylabel("Decision count")


def plot_choice_breakdown(
    history: "History", path: str | None = None, show: bool = False
):
    """Bar chart of top-level agent decisions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_choice_breakdown(ax, history)
    ax.set_title("Agent choices (write / review / finish)")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_run_summary(history: "History", path: str | None = None, show: bool = False):
    """Single-page dashboard encapsulating the single-review marketplace run.

    Top row tells the outcome story (mean review effort over time, effort
    distribution, and how the review market behaves); the bottom row explains
    the mechanics (quality payoff, reviewer reputation, and how agents spend
    their timesteps).
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    n_papers = len(history.paper_ac)
    horizon = history.timesteps[-1] if history.timesteps else 0
    fig.suptitle(
        f"Simulation summary — {len(history.agent_capital)} agents, "
        f"{n_papers} papers, {horizon} timesteps",
        fontsize=15, fontweight="bold",
    )

    ax = axes[0, 0]
    _draw_mean_review_effort(ax, history)
    ax.set_title("Average peer review effort over time")

    ax = axes[0, 1]
    _draw_effort_histogram(ax, history)
    ax.set_title("Completed peer reviews by effort")

    ax = axes[0, 2]
    _draw_marketplace(ax, history)
    ax.set_title("Review marketplace")

    ax = axes[1, 0]
    if getattr(history, "paper_writing_effort", {}):
        _draw_writing_effort_vs_rate(ax, history)
        ax.set_title("Writing effort vs accrual rate")
    else:
        _draw_quality_vs_ac(ax, history)
        ax.set_title("Paper quality vs accrued capital")

    ax = axes[1, 1]
    _draw_review_reputation(ax, history)
    ax.set_title("Reviewer reputation over time")

    ax = axes[1, 2]
    _draw_choice_breakdown(ax, history, fontsize=8)
    ax.set_title("How agents spent their timesteps")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, path, show)



def _draw_episode_return(
    ax,
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None = None,
) -> None:
    del epsilon  # kept for call-site compatibility; chart is raw returns only
    if not episodes or not returns:
        ax.text(0.5, 0.5, "No training episodes recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(episodes, returns, linewidth=1.5, color="#60a5fa")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean episode return (final AC)")


def plot_episode_return(
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None = None,
    path: str | None = None,
    show: bool = False,
):
    """Episode return curve for RL training (mean final AC per episode)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_episode_return(ax, episodes, returns, epsilon)
    ax.set_title("Episode return")
    fig.tight_layout()
    return _finish(fig, path, show)


def _draw_avg_peer_review_time(
    ax,
    episodes: list[int],
    avg_review_times: list[float],
) -> None:
    if not episodes or not avg_review_times:
        ax.text(0.5, 0.5, "No peer review data recorded", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.plot(episodes, avg_review_times, linewidth=1.5, color="#34d399")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean peer review time (timesteps)")


def plot_avg_peer_review_time(
    episodes: list[int],
    avg_review_times: list[float],
    path: str | None = None,
    show: bool = False,
):
    """Average completed peer-review duration per training episode."""
    fig, ax = plt.subplots(figsize=(11, 6))
    _draw_avg_peer_review_time(ax, episodes, avg_review_times)
    ax.set_title("Average peer review time")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_avg_peer_review_time_from_log(
    log_path: str,
    png_path: str | None = None,
    *,
    show: bool = False,
) -> str | None:
    """Render ``avg_peer_review_time.png`` from a ``training_log.json`` file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as fh:
        log: list[dict[str, Any]] = json.load(fh)
    if not log or "mean_review_effort" not in log[0]:
        return None
    episodes = [int(row["episode"]) for row in log]
    avg_times = [float(row["mean_review_effort"]) for row in log]
    if png_path is None:
        png_path = os.path.join(os.path.dirname(log_path), "avg_peer_review_time.png")
    plot_avg_peer_review_time(episodes, avg_times, png_path, show=show)
    return png_path if os.path.exists(png_path) else None


def plot_episode_return_from_log(
    log_path: str,
    png_path: str | None = None,
    *,
    show: bool = False,
) -> str | None:
    """Render ``episode_return.png`` from a ``training_log.json`` file."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as fh:
        log: list[dict[str, Any]] = json.load(fh)
    if not log:
        return None
    episodes = [int(row["episode"]) for row in log]
    returns = [float(row["mean_return"]) for row in log]
    epsilon = [float(row["epsilon"]) for row in log]
    if png_path is None:
        png_path = os.path.join(os.path.dirname(log_path), "episode_return.png")
    plot_episode_return(episodes, returns, epsilon, png_path, show=show)
    return png_path if os.path.exists(png_path) else None


# ---- static gallery charts (dark-themed, transparent PNGs) ----------------
#
# The committed gallery (docs/) shows the *static* charts as images instead of
# rebuilding them in the browser. They are rendered transparent with light text
# so they sit cleanly on the dark panel background (#161d24) and match the live
# Chart.js look closely. Only charts that have data for a given run are emitted,
# mirroring the front-end's per-panel ``setPanel`` gating, so simpler runs get
# fewer PNGs.
_DARK_RC = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "text.color": "#c4cdd5",
    "axes.labelcolor": "#c4cdd5",
    "axes.titlecolor": "#e6edf3",
    "axes.edgecolor": "#3a4450",
    "xtick.color": "#8b98a5",
    "ytick.color": "#8b98a5",
    "grid.color": "#1f272f",
    "legend.facecolor": "#161d24",
    "legend.edgecolor": "#30363d",
    "legend.framealpha": 0.85,
}


def render_episode_return_gallery(
    episodes: list[int],
    returns: list[float],
    epsilon: list[float] | None,
    outdir: str,
) -> str | None:
    """Render a dark-themed episode-return PNG for the static gallery."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "episode_return.png")
    with plt.rc_context(_DARK_RC):
        plot_episode_return(episodes, returns, epsilon, path)
    return path if os.path.exists(path) else None


def render_avg_peer_review_time_gallery(
    episodes: list[int],
    avg_review_times: list[float],
    outdir: str,
) -> str | None:
    """Render a dark-themed average peer-review-time PNG for the gallery."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "avg_peer_review_time.png")
    with plt.rc_context(_DARK_RC):
        plot_avg_peer_review_time(episodes, avg_review_times, path)
    return path if os.path.exists(path) else None


def _has_review_behavior(history: "History") -> bool:
    s = history.scalars
    return bool(s.get("num_papers") or s.get("completed_peer_reviews"))


def _has_marketplace(history: "History") -> bool:
    s = history.scalars
    return bool(s.get("papers_on_market") or s.get("completed_peer_reviews"))


def _has_reputation(history: "History") -> bool:
    series = getattr(history, "agent_review_history", {})
    if any(any(values) for values in series.values()):
        return True
    return bool(history.scalars.get("mean_peer_review_history"))


def _has_reputation_vs_ac(history: "History") -> bool:
    return bool(_reputation_vs_ac_points(history))


def _has_reputation_vs_review_ac(history: "History") -> bool:
    points = _reputation_vs_review_ac_points(history)
    return bool(points) and any(review_ac > 0.0 for _, review_ac, _ in points)


def _has_talent_vs_final_ac(history: "History") -> bool:
    points = _talent_vs_final_ac_points(history)
    return bool(points) and len({p[0] for p in points}) > 1


def _has_talent_vs_review_ac(history: "History") -> bool:
    points = _talent_vs_review_ac_points(history)
    return bool(points) and any(review_ac > 0.0 for _, review_ac, _ in points)


def _has_groups(history: "History") -> bool:
    return bool(history.agent_group_summary())


def _has_agent_capital_by_group(history: "History") -> bool:
    return bool(history.agent_capital)


def _has_talent_vs_ac(history: "History") -> bool:
    rates = getattr(history, "agent_accrual_rate", {})
    return any(any(values) for values in rates.values())


def _has_mean_review_effort_vs_ac(history: "History") -> bool:
    return bool(history.agent_capital) and bool(history.completed_reviews)


def _has_accepted_review_price(history: "History") -> bool:
    return bool(getattr(history, "accepted_review_claims", None))


def _has_review_faith_by_share_price(history: "History") -> bool:
    return bool(_review_faith_share_price_points(history))


def _has_market_pricing_scalars(history: "History") -> bool:
    return bool(history.scalars.get("fair_market_price"))


# Static charts shown in the gallery: (png name, data gate, plot function).
_GALLERY_CHARTS = (
    ("agent_capital_by_group", _has_agent_capital_by_group, plot_agent_capital_by_group),
    ("talent_vs_ac", _has_talent_vs_ac, plot_talent_vs_ac),
    ("mean_review_effort_vs_ac", _has_mean_review_effort_vs_ac, plot_mean_review_effort_vs_ac),
    ("review_behavior", _has_review_behavior, plot_review_behavior),
    ("marketplace_activity", _has_marketplace, plot_marketplace_activity),
    ("review_reputation", _has_reputation, plot_review_reputation),
    ("reputation_vs_ac", _has_reputation_vs_ac, plot_reputation_vs_ac),
    ("reputation_vs_review_ac", _has_reputation_vs_review_ac, plot_reputation_vs_review_ac),
    ("talent_vs_final_ac", _has_talent_vs_final_ac, plot_talent_vs_final_ac),
    ("talent_vs_review_ac", _has_talent_vs_review_ac, plot_talent_vs_review_ac),
    ("paper_quality_vs_ac",
     lambda h: bool(_paper_quality_ac_points(h)), plot_paper_quality_vs_ac),
    ("paper_quality_vs_citations",
     lambda h: bool(_paper_quality_citation_points(h)),
     plot_paper_quality_vs_citations),
    ("paper_quality_vs_review_faith",
     lambda h: bool(_paper_quality_review_faith_points(h)),
     plot_paper_quality_vs_review_faith),
    ("writing_effort_vs_rate",
     lambda h: any(l in h.paper_accrual_rate for l in h.paper_writing_effort),
     plot_writing_effort_vs_rate),
    ("paper_writing_effort_over_time",
     lambda h: bool(_paper_writing_effort_over_time_points(h)),
     plot_paper_writing_effort_over_time),
    ("paper_ac", lambda h: bool(h.paper_ac), plot_paper_ac),
    ("choice_breakdown", lambda h: bool(_choice_tallies(h)), plot_choice_breakdown),
    ("review_effort_histogram",
     lambda h: bool(h.completed_reviews), plot_review_effort_histogram),
    ("review_effort_scatter",
     lambda h: bool(h.completed_reviews), plot_review_effort_scatter),
    ("writing_effort_distribution",
     lambda h: bool(_writing_effort_by_agent(h)), plot_writing_effort_distribution),
    ("paper_writing_effort_distribution",
     lambda h: bool(getattr(h, "paper_writing_effort", {})),
     plot_paper_writing_effort_distribution),
    ("agent_group_comparison", _has_groups, plot_agent_group_comparison),
    ("overview", _has_groups, plot_overview),
    ("ac_source",
     lambda h: bool(h.scalars.get("writing_held_ac") or h.scalars.get("review_held_ac")),
     plot_ac_source),
    ("review_benefit", _has_review_benefit, plot_review_benefit),
    ("review_surplus_aggregate", _has_review_benefit, plot_review_surplus_aggregate),
    ("review_surplus_by_period", _has_review_surplus_by_period, plot_review_surplus_by_period),
    ("accepted_review_price_binned", _has_accepted_review_price, plot_accepted_review_price_binned),
    ("review_faith_by_share_price", _has_review_faith_by_share_price, plot_review_faith_by_share_price),
    ("market_pricing_dynamics", _has_market_pricing_scalars, plot_market_pricing_dynamics),
    ("review_reward_curve", lambda h: True, plot_review_reward_curve),
)


# Charts whose reward-threshold line must use the run's own configured value
# rather than the current global default.
_THRESHOLD_CHARTS = frozenset({
    "review_effort_histogram",
    "review_effort_scatter",
    "paper_quality_vs_review_faith",
})


def render_gallery_charts(
    history: "History",
    outdir: str,
    *,
    threshold: float = MIN_REVIEW_EFFORT_THRESHOLD,
) -> list[str]:
    """Render the dark-themed static charts a run has data for into ``outdir``.

    ``threshold`` is the run's ``min_review_effort_threshold`` so the review
    effort charts draw the reward-threshold line at the value that run used.

    Returns the list of chart names produced (the front-end shows a panel only
    when its chart name is present), so runs with fewer recorded fields get
    fewer images.
    """
    os.makedirs(outdir, exist_ok=True)
    produced: list[str] = []
    with plt.rc_context(_DARK_RC):
        for name, has_data, plot_fn in _GALLERY_CHARTS:
            if not has_data(history):
                continue
            path = os.path.join(outdir, f"{name}.png")
            if name in _THRESHOLD_CHARTS:
                plot_fn(history, path, threshold=threshold)
            else:
                plot_fn(history, path)
            produced.append(name)
        if _has_marketplace(history) and history.timesteps:
            for start, end, name in _MARKETPLACE_ZOOM_WINDOWS:
                path = os.path.join(outdir, f"{name}.png")
                plot_marketplace_zoom(history, start, end, path)
                produced.append(name)
    return produced


def plot_all(
    history: "History", outdir: str = "runs", *, show: bool = False
) -> dict[str, str]:
    """Render every chart into ``outdir`` and return {name: path}."""
    os.makedirs(outdir, exist_ok=True)
    paths: dict[str, str] = {
        "summary": plot_run_summary(
            history, os.path.join(outdir, "summary.png"), show=show
        ),
        "agent_capital": plot_agent_capital(
            history, os.path.join(outdir, "agent_capital.png"), show=show
        ),
        "agent_capital_by_group": plot_agent_capital_by_group(
            history, os.path.join(outdir, "agent_capital_by_group.png"), show=show
        ),
        "talent_vs_ac": plot_talent_vs_ac(
            history, os.path.join(outdir, "talent_vs_ac.png"), show=show
        ),
        "mean_review_effort_vs_ac": plot_mean_review_effort_vs_ac(
            history,
            os.path.join(outdir, "mean_review_effort_vs_ac.png"),
            show=show,
        ),
        "mean_review_effort": plot_mean_review_effort(
            history, os.path.join(outdir, "mean_review_effort.png"), show=show
        ),
        "agent_group_comparison": plot_agent_group_comparison(
            history, os.path.join(outdir, "agent_group_comparison.png"), show=show
        ),
        "overview": plot_overview(
            history, os.path.join(outdir, "overview.png"), show=show
        ),
        "review_benefit": plot_review_benefit(
            history, os.path.join(outdir, "review_benefit.png"), show=show
        ),
        "review_surplus_aggregate": plot_review_surplus_aggregate(
            history, os.path.join(outdir, "review_surplus_aggregate.png"), show=show
        ),
        "review_surplus_by_period": plot_review_surplus_by_period(
            history, os.path.join(outdir, "review_surplus_by_period.png"), show=show
        ),
        "ac_source": plot_ac_source(
            history, os.path.join(outdir, "ac_source.png"), show=show
        ),
        "system_aggregates": plot_system_aggregates(
            history, os.path.join(outdir, "system_aggregates.png"), show=show
        ),
        "review_behavior": plot_review_behavior(
            history, os.path.join(outdir, "review_behavior.png"), show=show
        ),
        "accepted_review_price_binned": plot_accepted_review_price_binned(
            history,
            os.path.join(outdir, "accepted_review_price_binned.png"),
            show=show,
        ),
        "review_faith_by_share_price": plot_review_faith_by_share_price(
            history,
            os.path.join(outdir, "review_faith_by_share_price.png"),
            show=show,
        ),
        "market_pricing_dynamics": plot_market_pricing_dynamics(
            history,
            os.path.join(outdir, "market_pricing_dynamics.png"),
            show=show,
        ),
        "marketplace_activity": plot_marketplace_activity(
            history, os.path.join(outdir, "marketplace_activity.png"), show=show
        ),
        "paper_quality_vs_ac": plot_paper_quality_vs_ac(
            history, os.path.join(outdir, "paper_quality_vs_ac.png"), show=show
        ),
        "paper_quality_vs_citations": plot_paper_quality_vs_citations(
            history,
            os.path.join(outdir, "paper_quality_vs_citations.png"),
            show=show,
        ),
        "paper_quality_vs_review_faith": plot_paper_quality_vs_review_faith(
            history,
            os.path.join(outdir, "paper_quality_vs_review_faith.png"),
            show=show,
        ),
        "writing_effort_vs_rate": plot_writing_effort_vs_rate(
            history, os.path.join(outdir, "writing_effort_vs_rate.png"), show=show
        ),
        "paper_writing_effort_over_time": plot_paper_writing_effort_over_time(
            history,
            os.path.join(outdir, "paper_writing_effort_over_time.png"),
            show=show,
        ),
        "review_reputation": plot_review_reputation(
            history, os.path.join(outdir, "review_reputation.png"), show=show
        ),
        "reputation_vs_ac": plot_reputation_vs_ac(
            history, os.path.join(outdir, "reputation_vs_ac.png"), show=show
        ),
        "reputation_vs_review_ac": plot_reputation_vs_review_ac(
            history, os.path.join(outdir, "reputation_vs_review_ac.png"), show=show
        ),
        "talent_vs_final_ac": plot_talent_vs_final_ac(
            history, os.path.join(outdir, "talent_vs_final_ac.png"), show=show
        ),
        "talent_vs_review_ac": plot_talent_vs_review_ac(
            history, os.path.join(outdir, "talent_vs_review_ac.png"), show=show
        ),
        "action_mix": plot_action_mix_over_time(
            history, os.path.join(outdir, "action_mix.png"), show=show
        ),
        "choice_breakdown": plot_choice_breakdown(
            history, os.path.join(outdir, "choice_breakdown.png"), show=show
        ),
        "review_effort_histogram": plot_review_effort_histogram(
            history, os.path.join(outdir, "review_effort_histogram.png"), show=show
        ),
        "review_effort_scatter": plot_review_effort_scatter(
            history, os.path.join(outdir, "review_effort_scatter.png"), show=show
        ),
        "review_reward_curve": plot_review_reward_curve(
            history, os.path.join(outdir, "review_reward_curve.png"), show=show
        ),
        "writing_effort_distribution": plot_writing_effort_distribution(
            history,
            os.path.join(outdir, "writing_effort_distribution.png"),
            show=show,
        ),
        "paper_writing_effort_distribution": plot_paper_writing_effort_distribution(
            history,
            os.path.join(outdir, "paper_writing_effort_distribution.png"),
            show=show,
        ),
        "paper_ac": plot_paper_ac(
            history, os.path.join(outdir, "paper_ac.png"), show=show
        ),
    }
    paths.update(plot_marketplace_zoom_charts(history, outdir, show=show))
    return paths
