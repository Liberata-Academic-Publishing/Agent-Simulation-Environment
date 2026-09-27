from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, replace

from Agent import Agent
from config import (
    SIM,
    TRAIN,
    TRAIN_DQN,
    default_discrete_policy_path,
    default_dqn_policy_path,
    default_low_talent_discrete_policy_path,
    default_low_talent_policy_path,
    default_policy_path,
)
import config as config_module
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper
from DQNAgent import DQNAgent, make_dqn_backend
from DiscreteQLearningAgent import (
    DiscreteQLearningAgent,
    LowTalentDiscreteQLearningAgent,
    make_discrete_backend,
)
from QLearningAgent import QLearningAgent, LowTalentQLearningAgent, make_backend
from RandomAgent import ProbabilisticDiscreteAgent, RandomAgent

# Defaults come from config.py; CLI flags override them at runtime.
NUM_AGENTS = SIM.num_heuristic_agents
NUM_TIMESTEPS = SIM.num_timesteps
PROGRESS_INTERVAL = 100
NUM_RL_AGENTS = SIM.num_rl_agents
NUM_DQN_AGENTS = SIM.num_dqn_agents
NUM_RANDOM_AGENTS = SIM.num_random_agents
NUM_PROBABILISTIC_AGENTS = SIM.num_probabilistic_agents
NUM_LOW_TALENT_RL_AGENTS = SIM.num_low_talent_rl_agents
OUTPUT_DIR = SIM.output_dir


def _optional_timesteps(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in ("none", "null"):
        return None
    return float(value)


def _apply_review_bump_config(args) -> None:
    """Patch runtime SIM settings from CLI flags."""
    import Paper as paper_module

    config_module.SIM = replace(
        config_module.SIM,
        review_bump_duration=args.review_bump_duration,
        review_bump_decay_rate=args.review_bump_decay_rate,
        review_bump_decay_cap_timesteps=args.review_bump_decay_cap_timesteps,
        reviewer_surplus_share=args.reviewer_surplus_share,
    )
    paper_module.SIM = config_module.SIM


def _parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return []
    seeds: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        seeds.append(int(text))
    return seeds


def _talent_for(
    index: int,
    count: int,
    *,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> float:
    """Spread talents across a group of agents for differentiation experiments."""
    if count <= 1:
        return talent_min
    frac = index / (count - 1)
    return talent_min + frac * (talent_max - talent_min)


# Map raw action kinds to the decision an agent actively made on its turn.
DECISION_LABELS = {
    "write_paper": "write_paper",
    "bad_faith_review": "bad_faith_review",
    "good_faith_review": "good_faith_review",
    "review_started": "start_review",
    "review_continued": "continue_review",
    "review_finished_write": "finish_and_write",
    "review_finished_peer_review": "finish_and_review",
    "idle": "idle",
}


def seed_initial_papers(agents: list[Agent]):
    """Seed starting papers per SimConfig (listed on the market from timestep 1)."""
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                quality=agent.intrinsic_talent,
                current_ac=random.uniform(SIM.init_ac_min, SIM.init_ac_max),
                market_listed=True,
            )
            paper.title = f"Paper {index}"
            paper.list_on_market(0)
            Agent.all_papers.append(paper)


def print_summary(env: Environment, history: History):
    print(f"\nSimulation finished after {env.timestep} timesteps")
    print(f"Agents: {len(env.agents)}")
    print(f"Papers: {len(env.papers)}")
    print(f"Review paradigm: {env.review_paradigm}")
    if env.review_paradigm == "continuous":
        print(f"Continuous publishing: {env.continuous_publishing}")
        if env.continuous_publishing == "threshold":
            print(
                f"- auto-publish after {env.continuous_paper_timesteps:g} "
                "writing-effort timesteps"
            )

    reviewed = sum(1 for p in env.papers if p.reviewed)
    on_market = sum(1 for p in env.papers if getattr(p, "review_available", False))
    in_review = sum(1 for p in env.papers if p.review_in_progress_by is not None)
    qualities = [p.quality for p in env.papers]
    capitals = [a.academic_capital for a in env.agents]
    from History import gini

    print("\nMarketplace overview")
    print(f"- papers reviewed: {reviewed}/{len(env.papers)}")
    print(
        f"- good-faith reviews: "
        f"{int(history.scalars.get('good_faith_reviews', [0])[-1]) if history.timesteps else 0}"
    )
    print(
        f"- bad-faith reviews: "
        f"{int(history.scalars.get('bad_faith_reviews', [0])[-1]) if history.timesteps else 0}"
    )
    print(f"- papers on market (unclaimed): {on_market}")
    print(f"- papers in review right now: {in_review}")
    if history.timesteps:
        fair_market = history.scalars.get("fair_market_price", [0.0])[-1]
        mean_epsilon = history.scalars.get("mean_peer_review_epsilon", [0.0])[-1]
        listed_now = history.scalars.get("papers_listed_this_timestep", [0.0])[-1]
        claimed_now = history.scalars.get("papers_claimed_this_timestep", [0.0])[-1]
        instant_rate = history.scalars.get("instant_claim_rate", [0.0])[-1]
        mean_wait = history.scalars.get("mean_time_on_market_claimed", [0.0])[-1]
        price_multiplier = history.scalars.get(
            "mean_author_price_multiplier",
            [1.0],
        )[-1]
        print(f"- fair-market review price: {fair_market:.2%}")
        print(f"- mean reviewer epsilon: {mean_epsilon:.3f}")
        print(
            f"- market diagnostics: listed this step={listed_now:.0f}, "
            f"claimed this step={claimed_now:.0f}, instant-claim rate="
            f"{instant_rate:.1%}, mean time on market={mean_wait:.2f} ts"
        )
        print(f"- mean author offer multiplier: {price_multiplier:.3f}")
    if qualities:
        print(
            f"- paper quality: mean={sum(qualities) / len(qualities):.2f}, "
            f"min={min(qualities):.2f}, max={max(qualities):.2f}"
        )
    if capitals:
        print(f"- capital Gini (inequality): {gini(capitals):.3f}")

    reviewers = sorted(
        (a for a in env.agents if a.completed_review_count > 0),
        key=lambda a: a.peer_review_history,
        reverse=True,
    )
    if reviewers:
        print("\nTop reviewers (by reputation = share-weighted accrual rate per review)")
        for agent in reviewers[:5]:
            print(
                f"- {agent.name}: reputation={agent.peer_review_history:.2f} "
                f"epsilon={agent.peer_review_epsilon_history:.3f} "
                f"over {agent.completed_review_count} reviews"
            )

    print("\nAction counts")
    for action, count in history.action_counts.most_common():
        print(f"- {action}: {count}")

    print_agent_group_summary(history)
    print_agent_extremes(history)

    print("\nFinal agent capital")
    for agent in sorted(env.agents, key=lambda item: item.academic_capital, reverse=True):
        print(f"- {agent.name}: AC={agent.academic_capital:.2f}")

    authored_counts = Counter(paper.author for paper in env.papers)
    print("\nPapers produced by agent")
    for agent in env.agents:
        print(f"- {agent.name}: {authored_counts[agent]}")

    print("\nPapers")
    for index, paper in enumerate(env.papers, start=1):
        title = getattr(paper, "title", f"Paper {index}")
        author_name = getattr(paper.author, "name", "Unknown")
        reviewers = [
            f"{getattr(agent, 'name', 'Unknown')}={share:.2%}"
            for agent, share in paper.share_distribution.items()
            if agent != paper.author
        ]
        reviewer_text = ", ".join(reviewers) if reviewers else "none"
        if paper.review_available:
            status = "on market"
        elif paper.reviewed:
            status = "reviewed"
        elif paper.review_in_progress_by is not None:
            status = "in review"
        else:
            status = "unlisted"
        print(
            f"- {title}: author={author_name}, quality={paper.quality:.2f}, "
            f"AC={paper.current_ac:.2f}, rate={paper.accrual_rate:.2f}, "
            f"status={status}, reviewer shares={reviewer_text}"
        )

    print("\nRecent agent actions")
    for agent in env.agents:
        recent = history.agent_actions.get(agent.name, [])
        if not recent:
            continue
        print(f"\n{agent.name}")
        for line in recent[-10:]:
            print(f"- {line}")


def print_agent_group_summary(history: History):
    """Display outcome comparisons across heuristic/RL/random/probability agents."""
    summary = history.agent_group_summary()
    print("\nAgent type comparison")
    if not summary:
        print("- no grouped agent data recorded")
        return

    ordered = sorted(
        summary.items(),
        key=lambda item: item[1].get("mean_final_capital", 0.0),
        reverse=True,
    )
    for group, stats in ordered:
        count = int(stats.get("agent_count", 0))
        reviews = int(stats.get("completed_reviews", 0))
        good = int(stats.get("good_faith_reviews", 0))
        bad = int(stats.get("bad_faith_reviews", 0))
        papers = int(stats.get("papers_authored", 0))
        mean_capital = float(stats.get("mean_final_capital", 0.0))
        mean_reputation = float(stats.get("mean_peer_review_history", 0.0))
        mean_epsilon = float(stats.get("mean_peer_review_epsilon", 0.0))
        avg_effort = float(stats.get("average_review_effort", 0.0))
        good_rate = float(stats.get("good_faith_review_rate", 0.0))
        mean_good = float(stats.get("mean_good_faith_reviews", 0.0))
        mean_bad = float(stats.get("mean_bad_faith_reviews", 0.0))
        print(
            f"- {group}: agents={count}, mean AC={mean_capital:.2f}, "
            f"papers={papers}, reviews={reviews} "
            f"(good={good}, bad={bad}, good%={100.0 * good_rate:.1f}), "
            f"avg effort={avg_effort:.2f}, good/agent={mean_good:.1f}, "
            f"bad/agent={mean_bad:.1f}, "
            f"mean reputation={mean_reputation:.2f}, mean epsilon={mean_epsilon:.3f}"
        )


def print_agent_extremes(history: History, limit: int = 3):
    """Display top/bottom agents with concise policy/outcome diagnostics."""
    rows = history.agent_outcome_summary()
    print("\nTop and bottom agents")
    if not rows:
        print("- no per-agent outcome data recorded")
        return

    top_rows = rows[:limit]
    bottom_rows = list(reversed(rows[-limit:]))

    def action_text(row: dict) -> str:
        actions = row.get("most_common_actions") or []
        if not actions:
            return "none"
        return ", ".join(
            f"{item.get('kind', 'unknown')}={int(item.get('count', 0))}"
            for item in actions
        )

    def print_row(bucket: str, row: dict) -> None:
        print(
            f"- {bucket}: {row.get('agent')} ({row.get('group')}), "
            f"AC={float(row.get('final_capital', 0.0)):.2f}, "
            f"papers={int(row.get('papers_authored', 0))}, "
            f"reviews={int(row.get('completed_reviews', 0))} "
            f"(good={int(row.get('good_faith_reviews', 0))}, "
            f"bad={int(row.get('bad_faith_reviews', 0))}), "
            f"avg review effort={float(row.get('average_review_effort', 0.0)):.2f}, "
            f"top actions: {action_text(row)}"
        )

    for row in top_rows:
        print_row("top", row)
    for row in bottom_rows:
        print_row("bottom", row)


def print_choice_breakdown(history: History):
    """Show the share of top-level agent decisions (see ``DECISION_LABELS``)."""
    tallies: Counter[str] = Counter()

    for _, _, kind, _ in history.actions:
        decision = DECISION_LABELS.get(kind)
        if decision is not None:
            tallies[decision] += 1

    print("\nChoice breakdown (share of decisions)")
    total = sum(tallies.values())
    if total == 0:
        print("- no decisions recorded")
        return
    for decision in (
        "write_paper",
        "bad_faith_review",
        "good_faith_review",
        "start_review",
        "continue_review",
        "finish_and_write",
        "finish_and_review",
        "idle",
    ):
        count = tallies.get(decision, 0)
        if count:
            print(f"- {decision}: {count / total:.1%} ({count})")


CHART_DESCRIPTIONS = {
    "summary": "Overview dashboard (review effort, effort histogram, market, quality, reputation)",
    "agent_capital": "Academic capital per agent over time",
    "agent_capital_by_group": "Academic capital over time, colored by agent type",
    "talent_vs_ac": "Portfolio AC accrual rate over time (legend: agent number and talent)",
    "mean_review_effort_vs_ac": "Running mean peer review effort vs total academic capital (labeled by agent number)",
    "mean_review_effort": "Running mean effort of completed peer reviews over time",
    "agent_group_comparison": "Mean capital and review outcomes by agent type",
    "review_benefit": "Reviewer vs author benefit (good vs bad faith); author_net<0 = exploited",
    "review_surplus_aggregate": "System-wide review surplus and split between authors and reviewers",
    "review_surplus_by_period": "Mean per-review author/reviewer surplus by time bin (not cumulative; shows if bad-faith trades turn profitable as prices fall)",
    "ac_source": "Academic capital generated by writing vs peer review, overall and by agent type",
    "system_aggregates": "Total/mean/max capital with the inequality (Gini) index",
    "review_behavior": "Cumulative good- vs bad-faith reviews and paper count over time",
    "accepted_review_price_binned": "Mean agreed review share per 10-timestep bin (one dot per bin)",
    "review_faith_by_share_price": "Good vs bad faith review percentage and counts by agreed share price",
    "market_pricing_dynamics": "Review share pricing over timesteps: fair-market baseline, scarcity, pressure, and claim prices",
    "marketplace_activity": "Papers on market, listings/claims, and cumulative reviews completed",
    "marketplace_0_100": "Marketplace supply zoom: timesteps 0–100",
    "marketplace_200_300": "Marketplace supply zoom: timesteps 200–300",
    "marketplace_600_700": "Marketplace supply zoom: timesteps 600–700",
    "marketplace_1800_1900": "Marketplace supply zoom: timesteps 1800–1900",
    "paper_quality_vs_ac": "Paper quality vs accrued capital (reviewed or not)",
    "paper_quality_vs_review_faith": "Paper quality vs good/bad-faith review effort and counts",
    "writing_effort_vs_rate": "Writing effort invested vs base accrual rate (asymptote)",
    "paper_writing_effort_over_time": "Paper-writing effort at publication over time",
    "review_reputation": "Reviewer reputation (share-weighted accrual rate per review) over time",
    "reputation_vs_ac": "Final academic capital vs peer-review reputation (by agent type)",
    "reputation_vs_review_ac": "Peer-review AC held vs reputation (by agent type)",
    "talent_vs_final_ac": "Final academic capital vs intrinsic talent (by agent type)",
    "talent_vs_review_ac": "Peer-review AC held vs intrinsic talent (by agent type)",
    "action_mix": "What every agent did each timestep (stacked bars)",
    "choice_breakdown": "Agent decisions (write / review / finish)",
    "review_effort_histogram": "Completed peer reviews by effort level",
    "review_effort_scatter": "Each completed review: timestep vs effort invested",
    "review_reward_curve": "Review accrual bump E = F(T) for the active curve (sigmoid/log/jump) and threshold",
    "writing_effort_distribution": "Total paper-writing effort by agent",
    "paper_writing_effort_distribution": "Writing effort per paper (frequency)",
    "paper_ac": "Accrued capital per paper over time",
    "episode_return": "Episode return from RL training (if training_log.json exists)",
    "avg_peer_review_time": "Average peer review time per training episode",
}


def open_chart(path: str) -> None:
    """Open a saved chart with the OS default image viewer."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def save_outputs(history: History, *, show: bool = False, open_charts: bool = False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = history.to_csv(os.path.join(OUTPUT_DIR, "history.csv"))
    json_path = history.to_json(os.path.join(OUTPUT_DIR, "history.json"))
    print(f"\nWrote time-series to {csv_path} and {json_path}")

    try:
        import visualize
    except ImportError as exc:
        print(f"Skipping charts ({exc}).")
        print("Install matplotlib with: python -m pip install matplotlib")
        print(f"Open {csv_path} in a spreadsheet to plot it instead.")
        return

    paths = visualize.plot_all(history, OUTPUT_DIR, show=show)
    training_log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    training_chart = visualize.plot_episode_return_from_log(
        training_log_path,
        os.path.join(OUTPUT_DIR, "episode_return.png"),
        show=show,
    )
    if training_chart:
        paths["episode_return"] = training_chart
    review_time_chart = visualize.plot_avg_peer_review_time_from_log(
        training_log_path,
        os.path.join(OUTPUT_DIR, "avg_peer_review_time.png"),
        show=show,
    )
    if review_time_chart:
        paths["avg_peer_review_time"] = review_time_chart

    print("\nWrote charts to the runs/ folder:")
    for name, path in paths.items():
        description = CHART_DESCRIPTIONS.get(name, name)
        print(f"- {path}  ({description})")

    summary_path = paths.get("summary")
    if open_charts and summary_path and os.path.exists(summary_path):
        print(f"\nOpening summary chart: {summary_path}")
        open_chart(summary_path)
    elif summary_path:
        print(f"\nView the summary chart at: {os.path.abspath(summary_path)}")


def build_discrete_rl_agents(
    count: int,
    *,
    backend_kind: str,
    policy_path: str | None,
    freeze: bool,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> list[DiscreteQLearningAgent]:
    """Create independent discrete RL agents (3-action marketplace policy)."""
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"Discrete RL: policy {policy_path} not found; starting from scratch.")

    agents: list[DiscreteQLearningAgent] = []
    for i in range(count):
        backend = make_discrete_backend(backend_kind)
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                if i == 0:
                    print(
                        f"Discrete RL: policy {policy_path} is incompatible; "
                        "using scratch."
                    )
                loaded = False
                backend = make_discrete_backend(backend_kind)
        agents.append(
            DiscreteQLearningAgent(
                intrinsic_talent=_talent_for(
                    i,
                    count,
                    talent_min=talent_min,
                    talent_max=talent_max,
                ),
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"RL Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.rl_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(
        f"Discrete RL: {count} independent {backend_kind} agents, "
        f"{source}, {mode}."
    )
    return agents


def build_rl_agents(
    count: int,
    *,
    backend_kind: str,
    policy_path: str | None,
    freeze: bool,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> list[QLearningAgent]:
    """Create independent RL agents (one private backend each).

    Each starts blank or, if ``policy_path`` points at an existing file, from
    that saved baseline (then they diverge via their own online learning).
    ``run_simulation.py`` never writes policies back.
    """
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"RL: policy {policy_path} not found; starting from scratch.")

    agents: list[QLearningAgent] = []
    for i in range(count):
        backend = make_backend(backend_kind)
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                # Policies from before the single-review overhaul are incompatible.
                if i == 0:
                    print(f"RL: policy {policy_path} is incompatible; using scratch.")
                loaded = False
                backend = make_backend(backend_kind)
        agents.append(
            QLearningAgent(
                intrinsic_talent=_talent_for(
                    i,
                    count,
                    talent_min=talent_min,
                    talent_max=talent_max,
                ),
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"RL Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.rl_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(f"RL: {count} independent {backend_kind} agents, {source}, {mode}.")
    return agents


def build_low_talent_rl_agents(
    count: int,
    *,
    backend_kind: str,
    policy_path: str | None,
    freeze: bool,
    low_talent_value: float = SIM.low_talent_value,
) -> list[LowTalentQLearningAgent]:
    """Create low-talent RL agents with a fixed intrinsic talent and separate policy."""
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"Low-talent RL: policy {policy_path} not found; starting from scratch.")

    agents: list[LowTalentQLearningAgent] = []
    for i in range(count):
        backend = make_backend(backend_kind)
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                if i == 0:
                    print(
                        f"Low-talent RL: policy {policy_path} is incompatible; "
                        "using scratch."
                    )
                loaded = False
                backend = make_backend(backend_kind)
        agents.append(
            LowTalentQLearningAgent(
                intrinsic_talent=low_talent_value,
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"Low-talent RL Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.rl_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(
        f"Low-talent RL: {count} independent {backend_kind} agents "
        f"(talent={low_talent_value:g}), {source}, {mode}."
    )
    return agents


def build_low_talent_discrete_rl_agents(
    count: int,
    *,
    backend_kind: str,
    policy_path: str | None,
    freeze: bool,
    low_talent_value: float = SIM.low_talent_value,
) -> list[LowTalentDiscreteQLearningAgent]:
    """Create low-talent discrete RL agents with fixed talent and separate policy."""
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"Low-talent discrete RL: policy {policy_path} not found; starting from scratch.")

    agents: list[LowTalentDiscreteQLearningAgent] = []
    for i in range(count):
        backend = make_discrete_backend(backend_kind)
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                if i == 0:
                    print(
                        f"Low-talent discrete RL: policy {policy_path} is incompatible; "
                        "using scratch."
                    )
                loaded = False
                backend = make_discrete_backend(backend_kind)
        agents.append(
            LowTalentDiscreteQLearningAgent(
                intrinsic_talent=low_talent_value,
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"Low-talent RL Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.rl_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(
        f"Low-talent discrete RL: {count} independent {backend_kind} agents "
        f"(talent={low_talent_value:g}), {source}, {mode}."
    )
    return agents


def build_dqn_agents(
    count: int,
    *,
    policy_path: str | None,
    freeze: bool,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> list[DQNAgent]:
    """Create independent DQN agents (one private backend each)."""
    if count <= 0:
        return []

    loaded = bool(policy_path) and os.path.exists(policy_path)
    if policy_path and not loaded:
        print(f"DQN: policy {policy_path} not found; starting from scratch.")

    agents: list[DQNAgent] = []
    for i in range(count):
        backend = make_dqn_backend()
        if loaded:
            try:
                backend.load(policy_path)
            except (ValueError, EOFError, OSError, KeyError):
                if i == 0:
                    print(f"DQN: policy {policy_path} is incompatible; using scratch.")
                loaded = False
                backend = make_dqn_backend()
        agents.append(
            DQNAgent(
                intrinsic_talent=_talent_for(
                    i,
                    count,
                    talent_min=talent_min,
                    talent_max=talent_max,
                ),
                forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
                name=f"DQN Agent {i + 1}",
                backend=backend,
                epsilon=0.0 if freeze else SIM.dqn_epsilon,
                learning=not freeze,
            )
        )

    source = (
        f"loaded baseline {policy_path}" if loaded else "starting from scratch"
    )
    mode = "frozen (greedy)" if freeze else "learning online"
    print(f"DQN: {count} independent agents, {source}, {mode}.")
    return agents


def build_random_agents(
    count: int,
    *,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> list[RandomAgent]:
    if count <= 0:
        return []
    return [
        RandomAgent(
            intrinsic_talent=_talent_for(
                i,
                count,
                talent_min=talent_min,
                talent_max=talent_max,
            ),
            name=f"Random Agent {i + 1}",
        )
        for i in range(count)
    ]


def build_probabilistic_agents(
    count: int,
    *,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
) -> list[ProbabilisticDiscreteAgent]:
    if count <= 0:
        return []
    return [
        ProbabilisticDiscreteAgent(
            intrinsic_talent=_talent_for(
                i,
                count,
                talent_min=talent_min,
                talent_max=talent_max,
            ),
            name=f"Probabilistic Agent {i + 1}",
        )
        for i in range(count)
    ]


def build_simulation(
    history: History,
    *,
    num_agents: int = NUM_AGENTS,
    seed: int = SIM.seed,
    rl_agents: int = NUM_RL_AGENTS,
    dqn_agents: int = NUM_DQN_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    low_talent_rl_agents: int = NUM_LOW_TALENT_RL_AGENTS,
    rl_backend: str = SIM.rl_backend,
    rl_policy_path: str | None = None,
    rl_freeze: bool = False,
    low_talent_rl_policy_path: str | None = None,
    dqn_policy_path: str | None = None,
    dqn_freeze: bool = False,
    review_paradigm: str = SIM.review_paradigm,
    continuous_publishing: str = SIM.continuous_publishing,
    continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
    discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
    paper_effort_mode: str = SIM.paper_effort_mode,
    paper_effort_min: float = SIM.paper_effort_min,
    paper_effort_max: float = SIM.paper_effort_max,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
    low_talent_value: float = SIM.low_talent_value,
    pricing_policy: str = SIM.pricing_policy,
    use_competition_adjusted_forecast: bool = SIM.use_competition_adjusted_forecast,
    use_scarcity_pricing: bool = SIM.use_scarcity_pricing,
    reviewer_pressure_exponent: float = SIM.reviewer_pressure_exponent,
    use_merit_market_clearing: bool = SIM.use_merit_market_clearing,
    target_market_wait_timesteps: float = SIM.target_market_wait_timesteps,
    adaptive_pricing_learning_rate: float = SIM.adaptive_pricing_learning_rate,
    adaptive_pricing_mode: str = SIM.adaptive_pricing_mode,
    min_author_price_multiplier: float = SIM.min_author_price_multiplier,
    max_author_price_multiplier: float = SIM.max_author_price_multiplier,
) -> Environment:
    """Construct a simulation of heuristics plus independent RL agents."""
    random.seed(seed)
    Agent.all_papers = []

    agents: list[Agent] = [
        HeuristicAgent(
            intrinsic_talent=_talent_for(
                i - 1,
                num_agents,
                talent_min=talent_min,
                talent_max=talent_max,
            ),
            forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
            name=f"Agent {i}",
        )
        for i in range(1, num_agents + 1)
    ]
    agents.extend(
        build_random_agents(
            random_agents,
            talent_min=talent_min,
            talent_max=talent_max,
        )
    )
    agents.extend(
        build_probabilistic_agents(
            probabilistic_agents,
            talent_min=talent_min,
            talent_max=talent_max,
        )
    )
    if review_paradigm == "discrete":
        agents.extend(
            build_discrete_rl_agents(
                rl_agents,
                backend_kind=rl_backend,
                policy_path=rl_policy_path,
                freeze=rl_freeze,
                talent_min=talent_min,
                talent_max=talent_max,
            )
        )
    else:
        agents.extend(
            build_rl_agents(
                rl_agents,
                backend_kind=rl_backend,
                policy_path=rl_policy_path,
                freeze=rl_freeze,
                talent_min=talent_min,
                talent_max=talent_max,
            )
        )
    if review_paradigm == "discrete":
        agents.extend(
            build_low_talent_discrete_rl_agents(
                low_talent_rl_agents,
                backend_kind=rl_backend,
                policy_path=low_talent_rl_policy_path,
                freeze=rl_freeze,
                low_talent_value=low_talent_value,
            )
        )
    else:
        agents.extend(
            build_low_talent_rl_agents(
                low_talent_rl_agents,
                backend_kind=rl_backend,
                policy_path=low_talent_rl_policy_path,
                freeze=rl_freeze,
                low_talent_value=low_talent_value,
            )
        )
    agents.extend(
        build_dqn_agents(
            dqn_agents,
            policy_path=dqn_policy_path,
            freeze=dqn_freeze,
            talent_min=talent_min,
            talent_max=talent_max,
        )
    )

    seed_initial_papers(agents)

    return Environment(
        agents=agents,
        papers=Agent.all_papers,
        forecast_horizon_timesteps=SIM.forecast_horizon_timesteps,
        review_paradigm=review_paradigm,
        continuous_publishing=continuous_publishing,
        continuous_paper_timesteps=continuous_paper_timesteps,
        discrete_paper_timesteps=discrete_paper_timesteps,
        paper_effort_mode=paper_effort_mode,
        paper_effort_min=paper_effort_min,
        paper_effort_max=paper_effort_max,
        pricing_policy=pricing_policy,
        use_competition_adjusted_forecast=use_competition_adjusted_forecast,
        use_scarcity_pricing=use_scarcity_pricing,
        reviewer_pressure_exponent=reviewer_pressure_exponent,
        use_merit_market_clearing=use_merit_market_clearing,
        target_market_wait_timesteps=target_market_wait_timesteps,
        adaptive_pricing_learning_rate=adaptive_pricing_learning_rate,
        adaptive_pricing_mode=adaptive_pricing_mode,
        min_author_price_multiplier=min_author_price_multiplier,
        max_author_price_multiplier=max_author_price_multiplier,
        history=history,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Liberata peer-review simulation. By default, after "
        "the run it asks whether to save it to the docs/ gallery and what to call it."
    )
    parser.add_argument(
        "--name",
        metavar="TITLE",
        help="Save with this title and skip the prompt (for scripting).",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip the prompt and do not save the run.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip the end-of-run text summary (charts and CSV/JSON are still saved).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Pop up matplotlib chart windows after the run (in addition to saving PNGs).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the summary chart in your default image viewer after the run.",
    )
    parser.add_argument(
        "--timesteps", dest="timesteps", type=int, default=NUM_TIMESTEPS,
        metavar="N", help="Number of simulation timesteps for this run.",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=SIM.seed,
        metavar="N", help="Random seed for a single run.",
    )
    parser.add_argument(
        "--seeds", dest="seeds", default=None, metavar="CSV",
        help="Comma-separated seeds for a batch of otherwise identical runs.",
    )
    parser.add_argument(
        "--batch-summary-csv", dest="batch_summary_csv", default=None,
        metavar="PATH", help="Optional CSV summary path for --seeds batches.",
    )
    parser.add_argument(
        "--heuristic-agents", dest="heuristic_agents", type=int,
        default=NUM_AGENTS, metavar="N",
        help="Number of heuristic agents.",
    )
    parser.add_argument(
        "--rl-agents", dest="rl_agents", type=int, default=NUM_RL_AGENTS,
        metavar="N", help="Number of tabular/linear RL agents (0 disables).",
    )
    parser.add_argument(
        "--dqn-agents", dest="dqn_agents", type=int, default=NUM_DQN_AGENTS,
        metavar="N", help="Number of deep Q-network agents (0 disables).",
    )
    parser.add_argument(
        "--random-agents", dest="random_agents", type=int,
        default=NUM_RANDOM_AGENTS, metavar="N",
        help="Number of random control agents.",
    )
    parser.add_argument(
        "--probabilistic-agents", dest="probabilistic_agents", type=int,
        default=NUM_PROBABILISTIC_AGENTS, metavar="N",
        help="Number of discrete-only probability agents.",
    )
    parser.add_argument(
        "--low-talent-rl-agents", dest="low_talent_rl_agents", type=int,
        default=NUM_LOW_TALENT_RL_AGENTS, metavar="N",
        help="Number of low-talent RL agents (separate trained policy).",
    )
    parser.add_argument(
        "--review-paradigm", dest="review_paradigm",
        choices=["continuous", "discrete"], default=SIM.review_paradigm,
        help="Review action paradigm for the whole simulation run.",
    )
    parser.add_argument(
        "--continuous-publishing", dest="continuous_publishing",
        choices=["choice", "threshold"], default=SIM.continuous_publishing,
        help="Continuous writing: agent-chosen finish (choice) or fixed effort "
        "before auto-publish (threshold).",
    )
    parser.add_argument(
        "--continuous-paper-timesteps", dest="continuous_paper_timesteps",
        type=float, default=SIM.continuous_paper_timesteps, metavar="N",
        help="Writing effort required to auto-publish in continuous threshold mode.",
    )
    parser.add_argument(
        "--discrete-paper-timesteps", dest="discrete_paper_timesteps",
        type=float, default=SIM.discrete_paper_timesteps, metavar="N",
        help="Writing effort required to publish in discrete mode.",
    )
    parser.add_argument(
        "--paper-effort-mode", dest="paper_effort_mode",
        choices=["fixed", "uniform", "quality_scaled"],
        default=SIM.paper_effort_mode,
        help="Per-paper writing target: fixed, uniform, or quality_scaled.",
    )
    parser.add_argument(
        "--paper-effort-min", dest="paper_effort_min",
        type=float, default=SIM.paper_effort_min, metavar="N",
        help="Minimum sampled paper effort for uniform/quality_scaled modes.",
    )
    parser.add_argument(
        "--paper-effort-max", dest="paper_effort_max",
        type=float, default=SIM.paper_effort_max, metavar="N",
        help="Maximum sampled paper effort for uniform/quality_scaled modes.",
    )
    parser.add_argument(
        "--talent-min", dest="talent_min",
        type=float, default=SIM.talent_min, metavar="X",
        help="Minimum intrinsic talent assigned within each agent type.",
    )
    parser.add_argument(
        "--talent-max", dest="talent_max",
        type=float, default=SIM.talent_max, metavar="X",
        help="Maximum intrinsic talent assigned within each agent type.",
    )
    parser.add_argument(
        "--low-talent-value", dest="low_talent_value",
        type=float, default=SIM.low_talent_value, metavar="X",
        help="Fixed intrinsic talent for low-talent RL agents.",
    )
    parser.add_argument(
        "--pricing-policy", dest="pricing_policy",
        choices=["static_fair_market", "adaptive_multiplier"],
        default=SIM.pricing_policy,
        help="Author pricing policy: current static fair-market formula or an "
        "optional adaptive multiplier experiment.",
    )
    parser.add_argument(
        "--target-market-wait", dest="target_market_wait_timesteps",
        type=float, default=SIM.target_market_wait_timesteps, metavar="N",
        help="Adaptive pricing target wait before a paper is claimed.",
    )
    parser.add_argument(
        "--adaptive-pricing-learning-rate", dest="adaptive_pricing_learning_rate",
        type=float, default=SIM.adaptive_pricing_learning_rate, metavar="X",
        help="Adaptive pricing multiplier update rate.",
    )
    parser.add_argument(
        "--adaptive-pricing-mode", dest="adaptive_pricing_mode",
        choices=["global", "binned"],
        default=SIM.adaptive_pricing_mode,
        help="Adaptive pricing scope: one global author multiplier or per reputation bin.",
    )
    parser.add_argument(
        "--min-author-price-multiplier", dest="min_author_price_multiplier",
        type=float, default=SIM.min_author_price_multiplier, metavar="X",
        help="Lower bound for adaptive author offer multiplier.",
    )
    parser.add_argument(
        "--max-author-price-multiplier", dest="max_author_price_multiplier",
        type=float, default=SIM.max_author_price_multiplier, metavar="X",
        help="Upper bound for adaptive author offer multiplier.",
    )
    parser.add_argument(
        "--review-bump-duration", dest="review_bump_duration",
        choices=["permanent", "decay"],
        default=SIM.review_bump_duration,
        help="Review accrual bump lifetime: permanent multiplier or exponential decay.",
    )
    parser.add_argument(
        "--review-bump-decay-rate", dest="review_bump_decay_rate",
        type=float, default=SIM.review_bump_decay_rate, metavar="K",
        help="Exponential decay rate k in exp(-k * t) for decay-mode review bumps.",
    )
    parser.add_argument(
        "--review-bump-decay-cap", dest="review_bump_decay_cap_timesteps",
        type=_optional_timesteps,
        default=SIM.review_bump_decay_cap_timesteps,
        metavar="N",
        help="Hard cap (timesteps after review) before bump reaches zero; use 'none'.",
    )
    parser.add_argument(
        "--use-competition-adjusted-forecast",
        action=argparse.BooleanOptionalAction,
        default=SIM.use_competition_adjusted_forecast,
        help="Discount claim value by estimated win probability.",
    )
    parser.add_argument(
        "--use-scarcity-pricing",
        action=argparse.BooleanOptionalAction,
        default=SIM.use_scarcity_pricing,
        help="Scale review offers down when reviewer pressure is high.",
    )
    parser.add_argument(
        "--reviewer-pressure-exponent", dest="reviewer_pressure_exponent",
        type=float, default=SIM.reviewer_pressure_exponent, metavar="X",
        help="Exponent for scarcity-based offer scaling.",
    )
    parser.add_argument(
        "--use-merit-market-clearing",
        action=argparse.BooleanOptionalAction,
        default=SIM.use_merit_market_clearing,
        help="Assign listed papers to highest-scoring eligible reviewers.",
    )
    parser.add_argument(
        "--reviewer-surplus-share", dest="reviewer_surplus_share",
        type=float, default=SIM.reviewer_surplus_share, metavar="X",
        help="Reviewer's fraction of incremental review surplus in fair-market offers.",
    )
    parser.add_argument(
        "--gallery-action-limit", dest="gallery_action_limit",
        type=int, default=SIM.gallery_action_limit, metavar="N",
        help="Max action rows committed to docs/data history.json; full data stays local.",
    )
    parser.add_argument(
        "--rl-backend", dest="rl_backend", choices=["tabular", "linear"],
        default=SIM.rl_backend, help="Q backend for the RL agents.",
    )
    parser.add_argument(
        "--rl-from-scratch", dest="rl_from_scratch", action="store_true",
        help="Start RL agents from a blank policy instead of the saved baseline.",
    )
    parser.add_argument(
        "--rl-policy", dest="rl_policy", metavar="PATH", default=None,
        help="Explicit baseline policy path (overrides the default baseline).",
    )
    parser.add_argument(
        "--rl-freeze", dest="rl_freeze", action="store_true",
        help="Run RL agents greedily with no online learning.",
    )
    parser.add_argument(
        "--low-talent-rl-from-scratch", dest="low_talent_rl_from_scratch",
        action="store_true",
        help="Start low-talent RL agents from a blank policy.",
    )
    parser.add_argument(
        "--low-talent-rl-policy", dest="low_talent_rl_policy", metavar="PATH",
        default=None,
        help="Explicit low-talent RL policy path (overrides default baseline).",
    )
    parser.add_argument(
        "--dqn-from-scratch", dest="dqn_from_scratch", action="store_true",
        help="Start DQN agents from a blank network instead of the saved baseline.",
    )
    parser.add_argument(
        "--dqn-policy", dest="dqn_policy", metavar="PATH", default=None,
        help="Explicit DQN policy path (overrides the default baseline).",
    )
    parser.add_argument(
        "--dqn-freeze", dest="dqn_freeze", action="store_true",
        help="Run DQN agents greedily with no online learning.",
    )
    return parser.parse_args(argv)


def build_run_config(
    *,
    heuristic_agents: int = NUM_AGENTS,
    timesteps: int = NUM_TIMESTEPS,
    seed: int = SIM.seed,
    rl_agents: int = NUM_RL_AGENTS,
    dqn_agents: int = NUM_DQN_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    low_talent_rl_agents: int = NUM_LOW_TALENT_RL_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
    continuous_publishing: str = SIM.continuous_publishing,
    continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
    discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
    paper_effort_mode: str = SIM.paper_effort_mode,
    paper_effort_min: float = SIM.paper_effort_min,
    paper_effort_max: float = SIM.paper_effort_max,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
    low_talent_value: float = SIM.low_talent_value,
    pricing_policy: str = SIM.pricing_policy,
    use_competition_adjusted_forecast: bool = SIM.use_competition_adjusted_forecast,
    use_scarcity_pricing: bool = SIM.use_scarcity_pricing,
    reviewer_pressure_exponent: float = SIM.reviewer_pressure_exponent,
    use_merit_market_clearing: bool = SIM.use_merit_market_clearing,
    target_market_wait_timesteps: float = SIM.target_market_wait_timesteps,
    adaptive_pricing_learning_rate: float = SIM.adaptive_pricing_learning_rate,
    min_author_price_multiplier: float = SIM.min_author_price_multiplier,
    max_author_price_multiplier: float = SIM.max_author_price_multiplier,
    gallery_action_limit: int = SIM.gallery_action_limit,
) -> dict:
    """Full SimConfig snapshot for this run, with runtime overrides applied.

    Dumps every SimConfig field so the gallery can display the complete set of
    simulation variables, then patches in the values that CLI flags may have
    changed for this run. ``num_agents`` is kept as an alias of
    ``num_heuristic_agents`` for backward compatibility with older gallery data.

    Uses ``config_module.SIM`` (not the import-time ``SIM`` binding) so runtime
    ``replace()`` patches from CLI flags or ``sweep_surplus.apply_sim_overrides``
    are archived correctly.
    """
    config = asdict(config_module.SIM)
    config["num_heuristic_agents"] = heuristic_agents
    config["num_timesteps"] = timesteps
    config["seed"] = seed
    config["num_rl_agents"] = rl_agents
    config["num_dqn_agents"] = dqn_agents
    config["num_random_agents"] = random_agents
    config["num_probabilistic_agents"] = probabilistic_agents
    config["num_low_talent_rl_agents"] = low_talent_rl_agents
    config["rl_backend"] = rl_backend
    config["review_paradigm"] = review_paradigm
    config["continuous_publishing"] = continuous_publishing
    config["continuous_paper_timesteps"] = continuous_paper_timesteps
    config["discrete_paper_timesteps"] = discrete_paper_timesteps
    config["paper_effort_mode"] = paper_effort_mode
    config["paper_effort_min"] = paper_effort_min
    config["paper_effort_max"] = paper_effort_max
    config["talent_min"] = talent_min
    config["talent_max"] = talent_max
    config["low_talent_value"] = low_talent_value
    config["pricing_policy"] = pricing_policy
    config["use_competition_adjusted_forecast"] = use_competition_adjusted_forecast
    config["use_scarcity_pricing"] = use_scarcity_pricing
    config["reviewer_pressure_exponent"] = reviewer_pressure_exponent
    config["use_merit_market_clearing"] = use_merit_market_clearing
    config["target_market_wait_timesteps"] = target_market_wait_timesteps
    config["adaptive_pricing_learning_rate"] = adaptive_pricing_learning_rate
    config["min_author_price_multiplier"] = min_author_price_multiplier
    config["max_author_price_multiplier"] = max_author_price_multiplier
    # Bump lifetime fields already come from ``config_module.SIM`` (live after
    # ``_apply_review_bump_config`` / sweep overrides). Do not overwrite with
    # import-time default parameters from this function's signature.
    config["gallery_action_limit"] = gallery_action_limit
    config["num_agents"] = heuristic_agents
    # Aliases so the static gallery (which also reads pre-overhaul runs) keeps
    # rendering the time-unit config fields under their old names.
    config["num_days"] = config["num_timesteps"]
    config["forecast_horizon_days"] = config["forecast_horizon_timesteps"]
    config["review_effort_per_day"] = config["review_effort_per_timestep"]
    # Training-harness settings used to produce the RL policy these agents load,
    # surfaced under their own keys so the gallery can show a Training panel.
    config["train_episodes"] = TRAIN.episodes
    config["train_timesteps"] = TRAIN.timesteps
    config["train_num_rl"] = TRAIN.num_rl
    config["train_num_heuristic"] = TRAIN.num_heuristic
    config["train_eps_start"] = TRAIN.eps_start
    config["train_eps_end"] = TRAIN.eps_end
    config["train_low_talent"] = TRAIN.low_talent
    config["train_low_talent_value"] = TRAIN.low_talent_value
    config["train_dqn_episodes"] = TRAIN_DQN.episodes
    config["train_dqn_timesteps"] = TRAIN_DQN.timesteps
    config["train_dqn_num_dqn"] = TRAIN_DQN.num_dqn
    config["train_dqn_num_heuristic"] = TRAIN_DQN.num_heuristic
    config["train_dqn_eps_start"] = TRAIN_DQN.eps_start
    config["train_dqn_eps_end"] = TRAIN_DQN.eps_end
    return config


def archive_run(
    history: History,
    title: str | None,
    *,
    heuristic_agents: int = NUM_AGENTS,
    timesteps: int = NUM_TIMESTEPS,
    seed: int = SIM.seed,
    rl_agents: int = NUM_RL_AGENTS,
    dqn_agents: int = NUM_DQN_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    low_talent_rl_agents: int = NUM_LOW_TALENT_RL_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
    continuous_publishing: str = SIM.continuous_publishing,
    continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
    discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
    paper_effort_mode: str = SIM.paper_effort_mode,
    paper_effort_min: float = SIM.paper_effort_min,
    paper_effort_max: float = SIM.paper_effort_max,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
    low_talent_value: float = SIM.low_talent_value,
    pricing_policy: str = SIM.pricing_policy,
    use_competition_adjusted_forecast: bool = SIM.use_competition_adjusted_forecast,
    use_scarcity_pricing: bool = SIM.use_scarcity_pricing,
    reviewer_pressure_exponent: float = SIM.reviewer_pressure_exponent,
    use_merit_market_clearing: bool = SIM.use_merit_market_clearing,
    target_market_wait_timesteps: float = SIM.target_market_wait_timesteps,
    adaptive_pricing_learning_rate: float = SIM.adaptive_pricing_learning_rate,
    min_author_price_multiplier: float = SIM.min_author_price_multiplier,
    max_author_price_multiplier: float = SIM.max_author_price_multiplier,
    gallery_action_limit: int = SIM.gallery_action_limit,
) -> None:
    from export_run import export_run

    run_id = export_run(
        history,
        config=build_run_config(
            heuristic_agents=heuristic_agents,
            timesteps=timesteps,
            seed=seed,
            rl_agents=rl_agents,
            dqn_agents=dqn_agents,
            random_agents=random_agents,
            probabilistic_agents=probabilistic_agents,
            low_talent_rl_agents=low_talent_rl_agents,
            rl_backend=rl_backend,
            review_paradigm=review_paradigm,
            continuous_publishing=continuous_publishing,
            continuous_paper_timesteps=continuous_paper_timesteps,
            discrete_paper_timesteps=discrete_paper_timesteps,
            paper_effort_mode=paper_effort_mode,
            paper_effort_min=paper_effort_min,
            paper_effort_max=paper_effort_max,
            talent_min=talent_min,
            talent_max=talent_max,
            low_talent_value=low_talent_value,
            pricing_policy=pricing_policy,
            use_competition_adjusted_forecast=use_competition_adjusted_forecast,
            use_scarcity_pricing=use_scarcity_pricing,
            reviewer_pressure_exponent=reviewer_pressure_exponent,
            use_merit_market_clearing=use_merit_market_clearing,
            target_market_wait_timesteps=target_market_wait_timesteps,
            adaptive_pricing_learning_rate=adaptive_pricing_learning_rate,
            min_author_price_multiplier=min_author_price_multiplier,
            max_author_price_multiplier=max_author_price_multiplier,
            gallery_action_limit=gallery_action_limit,
        ),
        title=title,
    )
    print(f"\nArchived run to docs/data/{run_id}/ (visible in the gallery).")
    print("Publish it with: "
          "git add docs/data && git commit -m 'Add run' && git push")


def prompt_and_archive(
    history: History,
    *,
    heuristic_agents: int = NUM_AGENTS,
    timesteps: int = NUM_TIMESTEPS,
    seed: int = SIM.seed,
    rl_agents: int = NUM_RL_AGENTS,
    dqn_agents: int = NUM_DQN_AGENTS,
    random_agents: int = NUM_RANDOM_AGENTS,
    probabilistic_agents: int = NUM_PROBABILISTIC_AGENTS,
    low_talent_rl_agents: int = NUM_LOW_TALENT_RL_AGENTS,
    rl_backend: str = SIM.rl_backend,
    review_paradigm: str = SIM.review_paradigm,
    continuous_publishing: str = SIM.continuous_publishing,
    continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
    discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
    paper_effort_mode: str = SIM.paper_effort_mode,
    paper_effort_min: float = SIM.paper_effort_min,
    paper_effort_max: float = SIM.paper_effort_max,
    talent_min: float = SIM.talent_min,
    talent_max: float = SIM.talent_max,
    low_talent_value: float = SIM.low_talent_value,
    pricing_policy: str = SIM.pricing_policy,
    use_competition_adjusted_forecast: bool = SIM.use_competition_adjusted_forecast,
    use_scarcity_pricing: bool = SIM.use_scarcity_pricing,
    reviewer_pressure_exponent: float = SIM.reviewer_pressure_exponent,
    use_merit_market_clearing: bool = SIM.use_merit_market_clearing,
    target_market_wait_timesteps: float = SIM.target_market_wait_timesteps,
    adaptive_pricing_learning_rate: float = SIM.adaptive_pricing_learning_rate,
    min_author_price_multiplier: float = SIM.min_author_price_multiplier,
    max_author_price_multiplier: float = SIM.max_author_price_multiplier,
    gallery_action_limit: int = SIM.gallery_action_limit,
) -> None:
    """Ask whether to save this run and, if so, what to title it."""
    try:
        answer = input("\nSave this run to the gallery? [y/N]: ").strip().lower()
    except EOFError:  # non-interactive (piped/no TTY): default to not saving
        print("Not archived (no interactive input).")
        return

    if answer not in ("y", "yes"):
        print("Not archived.")
        return

    try:
        name = input("Name this run (leave blank for an auto name): ").strip()
    except EOFError:
        name = ""
    archive_run(
        history,
        name or None,
        heuristic_agents=heuristic_agents,
        timesteps=timesteps,
        seed=seed,
        rl_agents=rl_agents,
        dqn_agents=dqn_agents,
        random_agents=random_agents,
        probabilistic_agents=probabilistic_agents,
        low_talent_rl_agents=low_talent_rl_agents,
        rl_backend=rl_backend,
        review_paradigm=review_paradigm,
        continuous_publishing=continuous_publishing,
        continuous_paper_timesteps=continuous_paper_timesteps,
        discrete_paper_timesteps=discrete_paper_timesteps,
        paper_effort_mode=paper_effort_mode,
        paper_effort_min=paper_effort_min,
        paper_effort_max=paper_effort_max,
        talent_min=talent_min,
        talent_max=talent_max,
        low_talent_value=low_talent_value,
        pricing_policy=pricing_policy,
        use_competition_adjusted_forecast=use_competition_adjusted_forecast,
        use_scarcity_pricing=use_scarcity_pricing,
        reviewer_pressure_exponent=reviewer_pressure_exponent,
        use_merit_market_clearing=use_merit_market_clearing,
        target_market_wait_timesteps=target_market_wait_timesteps,
        adaptive_pricing_learning_rate=adaptive_pricing_learning_rate,
        min_author_price_multiplier=min_author_price_multiplier,
        max_author_price_multiplier=max_author_price_multiplier,
        gallery_action_limit=gallery_action_limit,
    )


def _policy_paths(args) -> tuple[str | None, str | None, str | None]:
    rl_policy_path = args.rl_policy
    if rl_policy_path is None and not args.rl_from_scratch and SIM.rl_autoload_policy:
        if args.review_paradigm == "discrete":
            rl_policy_path = default_discrete_policy_path(args.rl_backend)
        else:
            rl_policy_path = default_policy_path(args.rl_backend)

    low_talent_rl_policy_path = args.low_talent_rl_policy
    if (
        low_talent_rl_policy_path is None
        and not args.low_talent_rl_from_scratch
        and args.low_talent_rl_agents > 0
        and SIM.rl_low_talent_autoload_policy
    ):
        if args.review_paradigm == "discrete":
            low_talent_rl_policy_path = default_low_talent_discrete_policy_path(
                args.rl_backend
            )
        else:
            low_talent_rl_policy_path = default_low_talent_policy_path(args.rl_backend)

    dqn_policy_path = args.dqn_policy
    if (dqn_policy_path is None and not args.dqn_from_scratch
            and SIM.dqn_autoload_policy):
        dqn_policy_path = default_dqn_policy_path()
    return rl_policy_path, low_talent_rl_policy_path, dqn_policy_path


def _run_once(args, *, seed: int, title: str | None = None) -> dict:
    _apply_review_bump_config(args)
    rl_policy_path, low_talent_rl_policy_path, dqn_policy_path = _policy_paths(args)

    history = History()
    env = build_simulation(
        history,
        num_agents=args.heuristic_agents,
        seed=seed,
        rl_agents=args.rl_agents,
        dqn_agents=args.dqn_agents,
        random_agents=args.random_agents,
        probabilistic_agents=args.probabilistic_agents,
        low_talent_rl_agents=args.low_talent_rl_agents,
        rl_backend=args.rl_backend,
        rl_policy_path=rl_policy_path,
        rl_freeze=args.rl_freeze,
        low_talent_rl_policy_path=low_talent_rl_policy_path,
        dqn_policy_path=dqn_policy_path,
        dqn_freeze=args.dqn_freeze,
        review_paradigm=args.review_paradigm,
        continuous_publishing=args.continuous_publishing,
        continuous_paper_timesteps=args.continuous_paper_timesteps,
        discrete_paper_timesteps=args.discrete_paper_timesteps,
        paper_effort_mode=args.paper_effort_mode,
        paper_effort_min=args.paper_effort_min,
        paper_effort_max=args.paper_effort_max,
        talent_min=args.talent_min,
        talent_max=args.talent_max,
        low_talent_value=args.low_talent_value,
        pricing_policy=args.pricing_policy,
        use_competition_adjusted_forecast=args.use_competition_adjusted_forecast,
        use_scarcity_pricing=args.use_scarcity_pricing,
        reviewer_pressure_exponent=args.reviewer_pressure_exponent,
        use_merit_market_clearing=args.use_merit_market_clearing,
        target_market_wait_timesteps=args.target_market_wait_timesteps,
        adaptive_pricing_learning_rate=args.adaptive_pricing_learning_rate,
        adaptive_pricing_mode=args.adaptive_pricing_mode,
        min_author_price_multiplier=args.min_author_price_multiplier,
        max_author_price_multiplier=args.max_author_price_multiplier,
    )
    for _ in range(args.timesteps):
        env.run_timestep()
        if env.timestep % PROGRESS_INTERVAL == 0:
            print(
                f"Timestep {env.timestep}/{args.timesteps} "
                f"({len(env.papers)} papers)",
                flush=True,
            )

    if not args.quiet:
        print_summary(env, history)
        print_choice_breakdown(history)
    save_outputs(history, show=args.show, open_charts=args.open)

    if args.no_archive:
        print("\nNot archived to the gallery (--no-archive).")
    elif args.name is not None or title is not None:
        archive_run(
            history,
            title or args.name,
            heuristic_agents=args.heuristic_agents,
            timesteps=args.timesteps,
            seed=seed,
            rl_agents=args.rl_agents,
            dqn_agents=args.dqn_agents,
            random_agents=args.random_agents,
            probabilistic_agents=args.probabilistic_agents,
            low_talent_rl_agents=args.low_talent_rl_agents,
            rl_backend=args.rl_backend,
            review_paradigm=args.review_paradigm,
            continuous_publishing=args.continuous_publishing,
            continuous_paper_timesteps=args.continuous_paper_timesteps,
            discrete_paper_timesteps=args.discrete_paper_timesteps,
            paper_effort_mode=args.paper_effort_mode,
            paper_effort_min=args.paper_effort_min,
            paper_effort_max=args.paper_effort_max,
            talent_min=args.talent_min,
            talent_max=args.talent_max,
            low_talent_value=args.low_talent_value,
            pricing_policy=args.pricing_policy,
            use_competition_adjusted_forecast=args.use_competition_adjusted_forecast,
            use_scarcity_pricing=args.use_scarcity_pricing,
            reviewer_pressure_exponent=args.reviewer_pressure_exponent,
            use_merit_market_clearing=args.use_merit_market_clearing,
            target_market_wait_timesteps=args.target_market_wait_timesteps,
            adaptive_pricing_learning_rate=args.adaptive_pricing_learning_rate,
            min_author_price_multiplier=args.min_author_price_multiplier,
            max_author_price_multiplier=args.max_author_price_multiplier,
            gallery_action_limit=args.gallery_action_limit,
        )
    else:
        prompt_and_archive(
            history,
            heuristic_agents=args.heuristic_agents,
            timesteps=args.timesteps,
            seed=seed,
            rl_agents=args.rl_agents,
            dqn_agents=args.dqn_agents,
            random_agents=args.random_agents,
            probabilistic_agents=args.probabilistic_agents,
            low_talent_rl_agents=args.low_talent_rl_agents,
            rl_backend=args.rl_backend,
            review_paradigm=args.review_paradigm,
            continuous_publishing=args.continuous_publishing,
            continuous_paper_timesteps=args.continuous_paper_timesteps,
            discrete_paper_timesteps=args.discrete_paper_timesteps,
            paper_effort_mode=args.paper_effort_mode,
            paper_effort_min=args.paper_effort_min,
            paper_effort_max=args.paper_effort_max,
            talent_min=args.talent_min,
            talent_max=args.talent_max,
            low_talent_value=args.low_talent_value,
            pricing_policy=args.pricing_policy,
            use_competition_adjusted_forecast=args.use_competition_adjusted_forecast,
            use_scarcity_pricing=args.use_scarcity_pricing,
            reviewer_pressure_exponent=args.reviewer_pressure_exponent,
            use_merit_market_clearing=args.use_merit_market_clearing,
            target_market_wait_timesteps=args.target_market_wait_timesteps,
            adaptive_pricing_learning_rate=args.adaptive_pricing_learning_rate,
            min_author_price_multiplier=args.min_author_price_multiplier,
            max_author_price_multiplier=args.max_author_price_multiplier,
            gallery_action_limit=args.gallery_action_limit,
        )
    return _summary_row(history, seed=seed, title=title or args.name or "")


def _summary_row(history: History, *, seed: int, title: str) -> dict:
    scalars = history.scalars
    group_summary = history.agent_group_summary()
    def last_scalar(name: str) -> float:
        values = scalars.get(name) or []
        return float(values[-1]) if values else 0.0

    row = {
        "title": title,
        "seed": seed,
        "timesteps": len(history.timesteps),
        "total_capital": last_scalar("total_capital"),
        "mean_capital": last_scalar("mean_capital"),
        "capital_gini": last_scalar("capital_gini"),
        "num_papers": last_scalar("num_papers"),
        "completed_peer_reviews": last_scalar("completed_peer_reviews"),
        "good_faith_reviews": last_scalar("good_faith_reviews"),
        "bad_faith_reviews": last_scalar("bad_faith_reviews"),
        "mean_completed_review_effort": last_scalar("mean_completed_review_effort"),
        "instant_claim_rate": last_scalar("instant_claim_rate"),
        "mean_time_on_market_claimed": last_scalar("mean_time_on_market_claimed"),
        "mean_author_price_multiplier": last_scalar("mean_author_price_multiplier"),
    }
    for group, stats in sorted(group_summary.items()):
        prefix = group.replace("Agent", "").replace("QLearning", "RL").lower() or "agent"
        row[f"{prefix}_mean_capital"] = float(stats.get("mean_final_capital", 0.0))
        row[f"{prefix}_reviews"] = int(stats.get("completed_reviews", 0))
        row[f"{prefix}_good_reviews"] = int(stats.get("good_faith_reviews", 0))
        row[f"{prefix}_bad_reviews"] = int(stats.get("bad_faith_reviews", 0))
    return row


def _write_batch_summary(rows: list[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote batch summary CSV to {path}")


def _print_batch_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\nBatch summary")
    for row in rows:
        print(
            f"- seed={row['seed']}: mean AC={row['mean_capital']:.2f}, "
            f"papers={row['num_papers']:.0f}, reviews={row['completed_peer_reviews']:.0f}, "
            f"good={row['good_faith_reviews']:.0f}, bad={row['bad_faith_reviews']:.0f}, "
            f"mean effort={row['mean_completed_review_effort']:.2f}, "
            f"instant claim={row['instant_claim_rate']:.1%}, "
            f"market wait={row['mean_time_on_market_claimed']:.2f}"
        )


def main(argv=None):
    args = parse_args(argv)
    seeds = _parse_seed_list(args.seeds)
    if not seeds:
        _run_once(args, seed=args.seed)
        return

    rows: list[dict] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"\n=== Batch run {index}/{len(seeds)} (seed={seed}) ===")
        if args.name is not None:
            title = f"{args.name} seed {seed}"
        else:
            title = f"batch seed {seed}"
        rows.append(_run_once(args, seed=seed, title=title))
    _print_batch_summary(rows)
    if args.batch_summary_csv:
        _write_batch_summary(rows, args.batch_summary_csv)


if __name__ == "__main__":
    main()
