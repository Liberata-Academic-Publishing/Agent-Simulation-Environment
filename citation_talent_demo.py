"""Ad-hoc experiment: how does author intrinsic_talent affect citation_count?

Builds groups of agents at different ``intrinsic_talent`` levels, runs a
simulation with the citation network enabled (see ``config.py``'s
``citations_enabled`` / ``citation_*`` knobs and ``Paper.generate_citations``),
and plots how ``citation_count`` / ``citation_accrued`` play out across talent
groups and over time. Not part of the test suite (see
``test_citation_network.py`` for that) — this is exploratory.

Usage:
    python citation_talent_demo.py

Outputs (to runs/, matching the project's chart convention):
    runs/citation_vs_talent.png
    runs/citation_mean_by_talent.png
    runs/citation_over_time_by_talent.png
"""

from __future__ import annotations

import random
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from Paper import Paper

TALENT_LEVELS = [0.3, 0.6, 1.0, 1.4, 1.8]
AGENTS_PER_LEVEL = 4
INIT_PAPERS_PER_AGENT = 0
TIMESTEPS = 1500
# Average references per new paper. With CITATION_REFERENCE_COUNT_DISTRIBUTION
# = "poisson", each paper's actual reference count is Poisson-distributed
# around this mean instead of every paper citing exactly this many.
CITATION_REFERENCE_COUNT = 7.5
CITATION_REFERENCE_COUNT_DISTRIBUTION = "poisson"  # "fixed" | "poisson"
SEED = 7

# Each agent gets its own random number of writing timesteps needed to publish
# a paper (instead of every agent sharing one fixed threshold), so papers stop
# publishing in lockstep across the corpus.
PUBLISH_TIMESTEPS_MIN = 60.0
PUBLISH_TIMESTEPS_MAX = 90.0


def build_agents() -> list[HeuristicAgent]:
    return [
        HeuristicAgent(intrinsic_talent=talent, name=f"talent{talent:.1f}_agent{i}")
        for talent in TALENT_LEVELS
        for i in range(AGENTS_PER_LEVEL)
    ]


def seed_papers(agents: list[HeuristicAgent]) -> None:
    for agent in agents:
        for _ in range(INIT_PAPERS_PER_AGENT):
            quality = random.gauss(agent.intrinsic_talent, 0.2)
            Agent.all_papers.append(Paper(author=agent, quality=quality, current_ac=1.0))


def stagger_publish_timing(env: Environment) -> None:
    """Give each agent its own writing-effort threshold before it publishes.

    ``Environment.__init__`` applies one shared ``continuous_paper_timesteps``
    to every agent, so this must run after construction to give each agent a
    distinct threshold via the same public ``configure_continuous_publishing``
    API the environment itself uses.
    """
    for agent in env.agents:
        agent.configure_continuous_publishing(
            "threshold", random.uniform(PUBLISH_TIMESTEPS_MIN, PUBLISH_TIMESTEPS_MAX)
        )


def run() -> tuple[Environment, list[int], dict[float, list[int]]]:
    random.seed(SEED)
    Agent.all_papers = []
    agents = build_agents()
    seed_papers(agents)

    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        citations_enabled=True,
        citation_reference_count=CITATION_REFERENCE_COUNT,
        citation_reference_count_distribution=CITATION_REFERENCE_COUNT_DISTRIBUTION,
    )
    stagger_publish_timing(env)

    # Cumulative citation_count summed across each talent group's papers,
    # snapshotted every timestep, to see how the gap opens up over time.
    history_by_talent: dict[float, list[int]] = {talent: [] for talent in TALENT_LEVELS}
    timesteps: list[int] = []

    for step in range(TIMESTEPS):
        env.run_timestep()
        timesteps.append(step + 1)
        totals: dict[float, int] = defaultdict(int)
        for paper in env.papers:
            totals[paper.author.intrinsic_talent] += paper.citation_count
        for talent in TALENT_LEVELS:
            history_by_talent[talent].append(totals.get(talent, 0))

    return env, timesteps, history_by_talent


def plot_citation_boxplot(env: Environment) -> None:
    """Boxplot of each paper's citation_count, grouped by author talent."""
    grouped: dict[float, list[int]] = defaultdict(list)
    for paper in env.papers:
        grouped[paper.author.intrinsic_talent].append(paper.citation_count)

    talents = sorted(grouped)
    data = [grouped[t] for t in talents]

    # One flat fill for every box: these are all the same measure
    # (citation_count), just faceted by talent, not distinct series.
    box_color = "#2a78d6"

    surface = "#fcfcfb"
    primary_ink = "#0b0b0b"
    secondary_ink = "#52514e"
    muted_ink = "#898781"
    gridline = "#e1e0d9"
    axis_line = "#c3c2b7"

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    box = ax.boxplot(
        data,
        tick_labels=[f"{t:.1f}" for t in talents],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": primary_ink, "linewidth": 2},
        whiskerprops={"color": muted_ink, "linewidth": 1.4},
        capprops={"color": muted_ink, "linewidth": 1.4},
        flierprops={
            "markerfacecolor": muted_ink,
            "markeredgecolor": "none",
            "markersize": 4,
            "alpha": 0.6,
        },
        boxprops={"linewidth": 1.2},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor(primary_ink)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.95)

    ax.set_xlabel("Author intrinsic talent", color=secondary_ink)
    ax.set_ylabel("Paper citation_count", color=secondary_ink)
    ax.set_title("Citations received by author talent", color=primary_ink)

    ax.yaxis.grid(True, color=gridline, linewidth=1)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(axis_line)
    ax.tick_params(colors=muted_ink)

    fig.tight_layout()
    fig.savefig("runs/citation_vs_talent.png", dpi=150, facecolor=surface)
    plt.close(fig)


def plot_mean_bar(env: Environment) -> None:
    grouped: dict[float, list[int]] = defaultdict(list)
    for paper in env.papers:
        grouped[paper.author.intrinsic_talent].append(paper.citation_count)

    talents = sorted(grouped)
    means = [sum(grouped[t]) / len(grouped[t]) for t in talents]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([f"{t:.1f}" for t in talents], means, color="#4C72B0")
    ax.set_xlabel("Author intrinsic talent")
    ax.set_ylabel("Mean citation_count per paper")
    ax.set_title("Mean citations per paper by author talent")
    fig.tight_layout()
    fig.savefig("runs/citation_mean_by_talent.png", dpi=150)
    plt.close(fig)


def plot_over_time(timesteps: list[int], history_by_talent: dict[float, list[int]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for talent, series in history_by_talent.items():
        ax.plot(timesteps, series, label=f"talent={talent:.1f}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative citation_count (sum of talent group's papers)")
    ax.set_title("Citation accumulation over time by author talent")
    ax.legend()
    fig.tight_layout()
    fig.savefig("runs/citation_over_time_by_talent.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    env, timesteps, history_by_talent = run()
    plot_citation_boxplot(env)
    plot_mean_bar(env)
    plot_over_time(timesteps, history_by_talent)
    print(
        "Saved: runs/citation_vs_talent.png, "
        "runs/citation_mean_by_talent.png, "
        "runs/citation_over_time_by_talent.png"
    )
