"""Single-agent RL check of the f(A, PR) and f(A, GFPR, BFPR) phase space plots.

One agent picks an action every step and earns that action's fixed reward per
step of work. A run holds the rates fixed; each grid varies them over the
matching plot's axes. Nothing depends on progress, so the agent has one state
and one Q-value per action, updated by Q(x) += alpha * (r - Q(x)) (the sim's
tabular update with gamma = 0). The prediction under test: the highest rate wins.

Exercise 1 actions: write (+a), review (+p).
Exercise 2 actions: write (+1), good-faith review (+g), bad-faith review (+b).

Run: python phase_space/single_agent_rl.py
Writes phase_space/figures/ex1_rl_grid.png, phase_space/figures/ex2_rl_grid.png
and phase_space/single_agent_results.json.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Callable

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Importing phase_plots also puts the repo root on sys.path for config.
from phase_plots import (
    COLORS,
    NAMES,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    boundary,
    draw_regions,
    new_fig,
    save,
    style_axes,
)
from config import SIM, TRAIN  # noqa: E402

RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "single_agent_results.json")

ALPHA = 0.1                                  # tabular default in train_rl.py
STEPS = TRAIN.episodes * TRAIN.timesteps
SEEDS = 20
GRID_SIZE = 12
A_MAX = 4.0                                  # Exercise 1 axis limits
P_MAX = 11.0
EX2_A = 1.0                                  # Exercise 2 fixed authoring rate
EX2_MAX = 3.0                                # Exercise 2 axis limit (g and b)
MIXED_GRAY = "#8a8985"                       # neutral midpoint: seeds split
DOT = dict(s=64, edgecolor=SURFACE, linewidth=2, zorder=5, clip_on=False)


def train(rates: tuple[float, ...], seed: int, steps: int = STEPS) -> int:
    """Train one agent at fixed rates; return the index of its greedy action."""
    draw = random.Random(seed).random
    num_actions = len(rates)
    q = [0.0] * num_actions
    epsilon = TRAIN.eps_start
    decay = (TRAIN.eps_start - TRAIN.eps_end) / steps
    for _ in range(steps):
        action = q.index(max(q))
        if draw() < epsilon:
            action = int(draw() * num_actions)
        q[action] += ALPHA * (rates[action] - q[action])
        epsilon -= decay
    return q.index(max(q))


def run_point(rates: dict[str, float]) -> dict[str, Any]:
    """Train every seed at one point and compare the choices to the prediction."""
    values = tuple(rates.values())
    picks = [train(values, SIM.seed + i) for i in range(SEEDS)]
    fractions = {key: picks.count(i) / SEEDS for i, key in enumerate(rates)}
    predicted = max(rates, key=rates.get)
    return {
        "rates": rates,
        "fractions": fractions,
        "predicted": predicted,
        "on_predicted_side": fractions[predicted] > 0.5,
    }


def grid_axis(limit: float) -> np.ndarray:
    return np.linspace(limit / GRID_SIZE, limit, GRID_SIZE)


def run_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    make_rates: Callable[[float, float], dict[str, float]],
) -> list[dict[str, Any]]:
    points = []
    for x in xs:
        for y in ys:
            rates = make_rates(float(x), float(y))
            if np.isclose(*sorted(rates.values())[-2:]):  # a tie has no predicted winner
                continue
            points.append(run_point(rates))
    return points


def plot_ex1(points: list[dict[str, Any]]) -> str:
    a = np.linspace(0, A_MAX, 401)
    p = np.linspace(0, P_MAX, 401)
    A, P = np.meshgrid(a, p)

    fig, (ax,) = new_fig()
    draw_regions(ax, a, p, {"A": A, "PR": P})
    boundary(ax, [0, A_MAX], [0, A_MAX], "a = p")

    cmap = LinearSegmentedColormap.from_list(
        "review_fraction", [COLORS["A"], MIXED_GRAY, COLORS["PR"]])
    dots = ax.scatter([pt["rates"]["A"] for pt in points],
                      [pt["rates"]["PR"] for pt in points],
                      c=[pt["fractions"]["PR"] for pt in points], cmap=cmap,
                      vmin=0, vmax=1, **DOT)
    style_axes(ax, "Authoring payoff rate, a  (AC per unit effort)",
               "Peer-review payoff rate, p  (AC per unit effort)",
               "f(A, PR): single RL agent")

    # Wide pad: the "a = p" label sits just past the right edge of the axes.
    colorbar = fig.colorbar(dots, ax=ax, ticks=[0, 0.5, 1], fraction=0.046, pad=0.14)
    colorbar.ax.set_yticklabels(["0% (write)", "50%", "100% (review)"])
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    colorbar.outline.set_visible(False)
    colorbar.set_label(f"Seeds choosing review  ({SEEDS} per point)",
                       color=TEXT_SECONDARY)
    return save(fig, "ex1_rl_grid.png")


def plot_ex2(points: list[dict[str, Any]]) -> str:
    g = np.linspace(0, EX2_MAX, 401)
    b = np.linspace(0, EX2_MAX, 401)
    G, B = np.meshgrid(g, b)

    fig, (ax,) = new_fig()
    draw_regions(ax, g, b, {"A": np.full_like(G, EX2_A), "GF": G, "BF": B})
    boundary(ax, [EX2_A, EX2_A], [0, EX2_A])              # g = 1 (A | GF)
    boundary(ax, [0, EX2_A], [EX2_A, EX2_A])              # b = 1 (A | BF)
    boundary(ax, [EX2_A, EX2_MAX], [EX2_A, EX2_MAX], "g = b")  # GF | BF
    ax.annotate("g = 1", xy=(EX2_A, 0.08), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=TEXT_PRIMARY)
    ax.annotate("b = 1", xy=(0.08, EX2_A), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=TEXT_PRIMARY)

    # A dot takes the color of the action all seeds chose; gray means they split.
    def scatter(group: list[dict[str, Any]], color: str, label: str) -> None:
        ax.scatter([pt["rates"]["GF"] for pt in group],
                   [pt["rates"]["BF"] for pt in group],
                   color=color, label=label, **DOT)

    for key in ("A", "GF", "BF"):
        scatter([pt for pt in points if pt["fractions"][key] == 1.0], COLORS[key], NAMES[key])
    mixed = [pt for pt in points if max(pt["fractions"].values()) < 1.0]
    if mixed:
        scatter(mixed, MIXED_GRAY, "Seeds split")

    style_axes(ax, "Good-faith payoff rate, g  (authoring rate a = 1)",
               "Bad-faith payoff rate, b", "f(A, GFPR, BFPR): single RL agent")
    ax.legend(title=f"Choice of all {SEEDS} seeds", loc="center left",
              bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=9,
              title_fontsize=9, labelcolor=TEXT_SECONDARY)
    return save(fig, "ex2_rl_grid.png")


def report(name: str, points: list[dict[str, Any]]) -> None:
    on_side = sum(pt["on_predicted_side"] for pt in points)
    mixed = [pt for pt in points if max(pt["fractions"].values()) < 1.0]
    print(f"{name}: {on_side}/{len(points)} grid points on the predicted side"
          f" ({100 * on_side / len(points):.1f}%)")
    print(f"{name}: {len(mixed)} points where seeds disagree")
    for pt in mixed:
        print(f"  rates {pt['rates']}: choices {pt['fractions']}")


def sanity_checks() -> None:
    assert train((10.0, 1.0), SIM.seed, steps=2000) == 0
    assert train((1.0, 10.0), SIM.seed, steps=2000) == 1
    assert train((1.0, 3.0, 0.5), SIM.seed, steps=2000) == 1
    assert train((1.0, 0.5, 3.0), SIM.seed, steps=2000) == 2
    rates = (1.0, 1.1, 0.9)
    assert train(rates, SIM.seed, steps=2000) == train(rates, SIM.seed, steps=2000)


def main() -> None:
    sanity_checks()
    ex1 = run_grid(grid_axis(A_MAX), grid_axis(P_MAX),
                   lambda a, p: {"A": a, "PR": p})
    ex2 = run_grid(grid_axis(EX2_MAX), grid_axis(EX2_MAX),
                   lambda g, b: {"A": EX2_A, "GF": g, "BF": b})

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "settings": {"alpha": ALPHA, "steps": STEPS, "seeds": SEEDS,
                         "base_seed": SIM.seed, "eps_start": TRAIN.eps_start,
                         "eps_end": TRAIN.eps_end, "grid_size": GRID_SIZE,
                         "a_max": A_MAX, "p_max": P_MAX,
                         "ex2_a": EX2_A, "ex2_max": EX2_MAX},
            "ex1": ex1,
            "ex2": ex2,
        }, fh, indent=2)
    paths = [plot_ex1(ex1), plot_ex2(ex2)]

    report("Exercise 1", ex1)
    report("Exercise 2", ex2)
    for path in [*paths, RESULTS_PATH]:
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
