"""Phase-space plots for the simple A / GFPR / BFPR linear-programming exercise.

One decision-maker splits a unit time budget across actions that each pay a
fixed rate (payoff per unit time). The objective is linear, so the optimum is a
corner: all time goes to the highest-rate action. Each plot varies two payoff
parameters, colors every grid cell by its winning action, and overlays the
analytic boundaries (where two rates are equal). Exercises 4 to 6 add paper
progress, active review effort and the best offered share, making four, five
and six axes, and show them two at a time. Exercise 7 keeps only features the
agent observes on the axes, with payoffs taken from config.py.

Run: python phase_space/phase_plots.py
Figures are written to phase_space/figures/.
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from config import SIM, discrete_good_review_timesteps  # noqa: E402
from Paper import (  # noqa: E402
    accrual_rate_from_quality,
    fair_market_offer_share,
    projected_paper_ac,
    review_accrual_bump,
)

T_B = float(SIM.bad_review_timesteps)          # bad-faith review time (sim default 1)
T_G = float(discrete_good_review_timesteps())  # good-faith review time (sim default 3)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Reference palette (dataviz skill): first three categorical slots, which
# validate all-pairs. Region fills are tints of these over the chart surface.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"A": "#2a78d6", "GF": "#1baf7a", "BF": "#eb6834", "PR": "#eb6834"}
NAMES = {
    "A": "Authoring",
    "PR": "Peer review",
    "GF": "Good-faith review",
    "BF": "Bad-faith review",
}


def best_action(rates: dict[str, np.ndarray]) -> np.ndarray:
    """Index (into ``rates`` order) of the highest-rate action per cell.

    Ties go to the earlier action, so the boundary lines carry the rule.
    """
    stack = np.stack([np.asarray(r, dtype=float) for r in rates.values()])
    return np.argmax(stack, axis=0)


def tint(hex_color: str, strength: float = 0.32) -> tuple[float, float, float]:
    """Blend a series color into the surface for a large region fill."""
    c = np.array(to_rgb(hex_color))
    s = np.array(to_rgb(SURFACE))
    return tuple(s + strength * (c - s))


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=12, pad=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def draw_regions(ax, x, y, rates: dict[str, np.ndarray]) -> None:
    """Color each cell by the action with the highest rate."""
    keys = list(rates)
    winner = best_action(rates)
    cmap = ListedColormap([tint(COLORS[k]) for k in keys])
    ax.imshow(
        winner,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=len(keys) - 0.5,
        interpolation="nearest",
    )


def boundary(ax, xs, ys, label: str | None = None, **kw) -> None:
    style = {"color": TEXT_PRIMARY, "linewidth": 2, "solid_capstyle": "round"}
    style.update(kw)
    ax.plot(xs, ys, **style)
    if label:
        ax.annotate(
            label,
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            annotation_clip=False,
            fontsize=9,
            color=TEXT_PRIMARY,
        )


def region_label(ax, x, y, key: str, sub: str | None = None) -> None:
    text = NAMES[key] if sub is None else f"{NAMES[key]}\n{sub}"
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color=TEXT_PRIMARY,
            fontweight="semibold", linespacing=1.4)


def new_fig(ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 5.4), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def save(fig, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Exercise 1: f(A, PR) ---------------------------------------------------

TALENT_POINTS = {
    "Low-talent RL": SIM.low_talent_value,
    "Min talent": SIM.talent_min,
    "Mid talent": (SIM.talent_min + SIM.talent_max) / 2,
    "Max talent": SIM.talent_max,
}


def mean_paper_timesteps() -> float:
    if SIM.paper_effort_mode == "fixed":
        return SIM.continuous_paper_timesteps
    return (SIM.paper_effort_min + SIM.paper_effort_max) / 2


def settings_payoff_rates() -> dict[str, tuple[float, float, float]]:
    """(talent, a, p) per talent level, computed from config.py defaults (AC per effort).

    a: an author of talent tau writes a paper of quality tau over the mean
       effort target, the paper earns its accrual rate (with a typical review
       bump) over the forecast horizon, and the author keeps what the reviewer
       is not paid.
    p: a talent-tau reviewer does one good-faith review (T_G effort) of a fresh
       quality-1.0 paper and earns the fair-market share of that paper's
       projected value. Multipliers (adaptive, scarcity, quality, history) = 1.
    """
    horizon = SIM.forecast_horizon_timesteps
    t_m = mean_paper_timesteps()

    def review(reviewer_talent: float) -> tuple[float, float, float]:
        eps = review_accrual_bump(T_G, quality=reviewer_talent)
        value = projected_paper_ac(0.0, accrual_rate_from_quality(1.0), eps, horizon)
        share = fair_market_offer_share(eps, 0.0, value, SIM.reviewer_surplus_share)
        return eps, share, value

    typical_eps, typical_share, _ = review(1.0)
    out = {}
    for label, tau in TALENT_POINTS.items():
        own_value = projected_paper_ac(0.0, accrual_rate_from_quality(tau), typical_eps, horizon)
        a = own_value * (1.0 - typical_share) / t_m
        _, share, value = review(tau)
        p = share * value / T_G
        out[label] = (tau, a, p)
    return out


def exercise_1() -> str:
    """Rates a (authoring) and p (peer review); best is A iff a > p."""
    points = settings_payoff_rates()
    x_lim = float(np.ceil(1.2 * max(a for _, a, _ in points.values())))
    y_lim = float(np.ceil(1.15 * max(p for _, _, p in points.values())))
    n = 401
    a = np.linspace(0, x_lim, n)
    p = np.linspace(0, y_lim, n)
    A, P = np.meshgrid(a, p)

    fig, (ax,) = new_fig()
    draw_regions(ax, a, p, {"A": A, "PR": P})
    boundary(ax, [0, x_lim], [0, x_lim], "a = p")
    region_label(ax, 0.78 * x_lim, 0.09 * y_lim, "A", "a > p")
    region_label(ax, 0.22 * x_lim, 0.88 * y_lim, "PR", "p > a")

    # Settings-implied rates: one dot per talent level, direct-labeled. The
    # top dot sits near the right edge, so its label goes on the left; the
    # bottom dot sits near the boundary, so its label goes above it.
    top = max(points, key=lambda k: points[k][2])
    bottom = min(points, key=lambda k: points[k][2])
    placement = {top: ((-9, 0), "right", "center"), bottom: ((0, 10), "center", "bottom")}
    for label, (tau, set_a, set_p) in points.items():
        ax.scatter([set_a], [set_p], s=70, color=TEXT_PRIMARY, edgecolor=SURFACE,
                   linewidth=2, zorder=5)
        offset, ha, va = placement.get(label, ((9, 0), "left", "center"))
        ax.annotate(f"{label} (talent {tau:g})\na = {set_a:.2f}, p = {set_p:.2f}",
                    xy=(set_a, set_p), xytext=offset, textcoords="offset points",
                    fontsize=9, color=TEXT_PRIMARY, ha=ha, va=va, linespacing=1.3)

    style_axes(ax, "Authoring payoff rate, a  (AC per unit effort)",
               "Peer-review payoff rate, p  (AC per unit effort)", "f(A, PR)")
    return save(fig, "ex1_A_vs_PR.png")


# --- Exercise 2: f(A, GFPR, BFPR) -------------------------------------------

def exercise_2_rates() -> str:
    """a = 1 fixed; vary g and b. Largest rate wins."""
    n = 401
    g = np.linspace(0, 3, n)
    b = np.linspace(0, 3, n)
    G, B = np.meshgrid(g, b)

    fig, (ax,) = new_fig()
    draw_regions(ax, g, b, {"A": np.ones_like(G), "GF": G, "BF": B})
    boundary(ax, [1, 1], [0, 1])           # g = 1 (A | GF)
    boundary(ax, [0, 1], [1, 1])           # b = 1 (A | BF)
    boundary(ax, [1, 3], [1, 3], "g = b")  # GF | BF
    ax.annotate("g = 1", xy=(1, 0.08), xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=TEXT_PRIMARY)
    ax.annotate("b = 1", xy=(0.08, 1), xytext=(0, 5), textcoords="offset points",
                fontsize=9, color=TEXT_PRIMARY)

    # Sim: same pay per review, so g = pay / T_G and b = pay / T_B -> g = b * T_B / T_G.
    ratio = T_B / T_G
    bb = np.array([0, 3])
    ax.plot(bb * ratio, bb, color=TEXT_SECONDARY, linewidth=2, linestyle=(0, (4, 3)))
    ax.annotate(f"sim: equal pay per review\n(g = b / {T_G / T_B:g})",
                xy=(2.2 * ratio, 2.2), xytext=(10, 0), textcoords="offset points",
                fontsize=9, color=TEXT_SECONDARY, va="center")

    region_label(ax, 0.5, 0.45, "A")
    region_label(ax, 2.25, 1.0, "GF", "g > b, g > 1")
    region_label(ax, 1.75, 2.6, "BF", "b > g, b > 1")
    style_axes(ax, "Good-faith payoff rate, g  (authoring rate a = 1)",
               "Bad-faith payoff rate, b", "f(A, GFPR, BFPR)")
    return save(fig, "ex2_rates.png")


def exercise_2_ratio() -> str:
    """GF beats BF iff pay_G / pay_B > T_G / T_B."""
    n = 401
    t_ratio = np.linspace(0.5, 6, n)
    pay_ratio = np.linspace(0.5, 6, n)
    TR, PR = np.meshgrid(t_ratio, pay_ratio)

    fig, (ax,) = new_fig()
    # g / b = pay_ratio / t_ratio; compare against 1. BF listed first so ties go to BF.
    draw_regions(ax, t_ratio, pay_ratio, {"BF": TR, "GF": PR})
    boundary(ax, [0.5, 6], [0.5, 6], "pay ratio = time ratio")

    sim_t, sim_p = T_G / T_B, 1.0
    ax.scatter([sim_t], [sim_p], s=70, color=TEXT_PRIMARY, edgecolor=SURFACE,
               linewidth=2, zorder=5)
    ax.annotate("sim today\n(equal pay, 3x time)", xy=(sim_t, sim_p), xytext=(10, 6),
                textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)

    region_label(ax, 1.9, 4.6, "GF", "wins")
    region_label(ax, 4.6, 1.9, "BF", "wins")
    style_axes(ax, "Time ratio, T_G / T_B", "Pay-per-review ratio, pay_G / pay_B",
               "When does good faith beat bad faith?")
    return save(fig, "ex2_pay_vs_time_ratio.png")


# --- Exercise 3: f(A, GFPR, BFPR, talent) -----------------------------------

def exercise_3() -> str:
    """Equal pay P per review; talent tau scales authoring and good-faith review.

    a = tau, b = P / T_B, and g = tau * P / T_G.
    """
    n = 501
    tau = np.linspace(0, 5, n)
    pay = np.linspace(0, 6, n)
    TAU, P = np.meshgrid(tau, pay)

    fig, (ax,) = new_fig()
    draw_regions(ax, tau, pay, {"A": TAU, "GF": TAU * P / T_G, "BF": P / T_B})
    style_axes(ax, "Talent, τ  (authoring rate a = τ)",
               "Pay per review, P  (same for GF and BF)", "f(A, GFPR, BFPR, talent)")
    ax.set_title("f(A, GFPR, BFPR, talent)", color=TEXT_PRIMARY, loc="left",
                 fontsize=12, pad=24)
    ax.text(0, 1.015, "Talent scales authoring and good-faith review:  "
            f"a = τ,  b = {'P' if T_B == 1 else f'P / {T_B:g}'},  g = τ·P / {T_G:g}",
            transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, va="bottom")

    # GF beats BF iff tau > T_G / T_B. Triple point at (T_G/T_B, T_G).
    tp_tau = T_G / T_B
    boundary(ax, [0, tp_tau], [0, tp_tau * T_B])        # A | BF: P = tau * T_B
    boundary(ax, [tp_tau, tp_tau], [tp_tau * T_B, 6])   # BF | GF: tau = T_G / T_B
    boundary(ax, [tp_tau, 5], [T_G, T_G])               # A | GF: P = T_G
    ax.annotate(f"τ = {tp_tau:g}", xy=(tp_tau, 5.75), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color=TEXT_PRIMARY)
    ax.annotate(f"P = {T_G:g}", xy=(5, T_G), xytext=(-4, 5), textcoords="offset points",
                ha="right", fontsize=9, color=TEXT_PRIMARY)
    region_label(ax, 2.6, 1.0, "A", f"τ > P, P < {T_G:g}")
    region_label(ax, 1.4, 4.4, "BF", f"P > τ, τ < {tp_tau:g}")
    region_label(ax, 4.0, 4.6, "GF", f"τ > {tp_tau:g}, P > {T_G:g}")
    return save(fig, "ex3_talent.png")


# --- Exercises 4 to 6: paper progress, review effort, offered share -----------

PAIR_TAU = 1.0     # talent, fixed for every pair figure
SHARE_REF = 0.03   # median accepted offer (see config.py); g and b are rates at this share

# Each panel holds the remaining axes at these values. At a = 1, pi = 0, e = 0,
# s = SHARE_REF the rates are Exercise 2's. The rest show all three regions in
# most panels.
PAIR_BASE = {"a": 1.0, "g": 2.0, "b": 1.5, "pi": 0.0, "e": 0.0, "s": SHARE_REF}
# A is undefined at pi = 1 and GF at e = T_G.
PAIR_LIMITS = {"a": 3.0, "g": 3.0, "b": 3.0, "pi": 0.99, "e": 0.99 * T_G, "s": 0.10}
PAIR_SYMBOLS = {"a": "a", "g": "g", "b": "b", "pi": "π", "e": "e", "s": "s"}
PAIR_LABELS = {
    "a": "Authoring rate, a",
    "g": "Good-faith rate, g",
    "b": "Bad-faith rate, b",
    "pi": "Paper progress, π",
    "e": "Active review effort, e  (steps)",
    "s": "Best offered share, s",
}


def pair_rates(a, g, b, pi, e=0.0, s=SHARE_REF, tau=PAIR_TAU) -> dict[str, np.ndarray]:
    """Payoff per step of the work left, for one agent at one state.

    A paper pays on completion, so its rate on the work left is a / (1 - pi).
    Review effort counts toward the T_G good-faith target the same way. The
    offered share scales the pay of both kinds of review.

    Talent follows Exercise 3: it scales authoring and good faith, not bad faith.
    """
    a, g, b, pi, e, s = np.broadcast_arrays(a, g, b, pi, e, s)
    pay = s / SHARE_REF
    return {"A": tau * a / (1 - pi), "GF": pay * tau * g / (1 - e / T_G), "BF": pay * b}


def draw_boundaries(ax, x, y, rates: dict[str, np.ndarray]) -> None:
    """Trace each pairwise tie where those two actions beat every other action."""
    for first, second in combinations(rates, 2):
        top = np.maximum(rates[first], rates[second])
        others = [rate for key, rate in rates.items() if key not in (first, second)]
        visible = np.all([top >= rate for rate in others], axis=0)
        gap = np.ma.masked_where(~visible, rates[first] - rates[second])
        ax.contour(x, y, gap, levels=[0], colors=TEXT_PRIMARY, linewidths=2)


def pair_figure(keys: tuple[str, ...], ncols: int, name: str, title: str, rates_text: str) -> str:
    """Show ``keys`` two at a time, holding the rest of them at PAIR_BASE."""
    n = 401
    # Grouped by the y axis, so each new axis adds one run of panels at the end.
    panels = sorted(combinations(keys, 2), key=lambda pair: keys.index(pair[1]))
    nrows = len(panels) // ncols
    height = 4.7 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.17 * ncols, height), facecolor=SURFACE)
    for ax, (x_key, y_key) in zip(axes.flat, panels):
        x = np.linspace(0, PAIR_LIMITS[x_key], n)
        y = np.linspace(0, PAIR_LIMITS[y_key], n)
        X, Y = np.meshgrid(x, y)
        rates = pair_rates(**{**PAIR_BASE, x_key: X, y_key: Y})
        draw_regions(ax, x, y, rates)
        draw_boundaries(ax, x, y, rates)
        held = ",  ".join(f"{PAIR_SYMBOLS[key]} = {PAIR_BASE[key]:g}" for key in keys
                          if key not in (x_key, y_key))
        style_axes(ax, PAIR_LABELS[x_key], PAIR_LABELS[y_key], held)

    add_header(fig, height, title,
               f"Highest rate wins:  {rates_text},  with talent fixed at τ = {PAIR_TAU:g}")
    return save(fig, name)


def add_header(fig, height: float, title: str, subtitle: str) -> None:
    """Title, subtitle and region legend above the axes; save() crops out to include them."""
    fig.suptitle(title, x=0.012, y=1 + 0.52 / height, ha="left", fontsize=14,
                 color=TEXT_PRIMARY)
    fig.text(0.012, 1 + 0.11 / height, subtitle, fontsize=10, color=TEXT_SECONDARY)
    handles = [Patch(facecolor=tint(COLORS[key]), edgecolor=GRID, label=NAMES[key])
               for key in ("A", "GF", "BF")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1 + 0.56 / height),
               ncols=3, frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)


# --- Exercise 7: only features the agent observes ---------------------------

# The axes are RL features; payoffs come from config.py, as in Exercise 1. The
# share axis is log-scaled over the RL feature's range, as the agent sees it.
OBS_LIMITS = {
    "pi": (0.0, 0.99),
    "s": (np.log10(SIM.rl_share_feature_min), np.log10(SIM.rl_share_feature_max)),
}
OBS_LABELS = {"pi": "Paper progress, π", "s": "Best offered share, s"}
SHARE_TICKS = [1e-4, 1e-3, 1e-2, 0.1, 0.5]


def review_values(tau: float) -> tuple[float, float]:
    """Projected value of a fresh quality-1.0 paper after a bad- and a good-faith review."""
    horizon = SIM.forecast_horizon_timesteps
    rate = accrual_rate_from_quality(1.0)
    bad = projected_paper_ac(0.0, rate, review_accrual_bump(T_B, quality=tau), horizon)
    good = projected_paper_ac(0.0, rate, review_accrual_bump(T_G, quality=tau), horizon)
    return bad, good


def observed_rates(pi, e, s, a: float, values: tuple[float, float]) -> dict[str, np.ndarray]:
    """Payoff per step of the work left for each choice, at one agent state.

    A review pays share s of the paper's value whether it is good or bad faith;
    good faith adds that review's bump to the value. Before T_B steps of effort
    both kinds of review still have work left. From T_B on, the bad-faith pay can
    be taken at once, so continuing earns only the extra value good faith adds,
    and the alternatives are taking it and then authoring or starting another
    bad-faith review. A fresh good-faith review never beats a fresh bad-faith one
    (V_G / T_G < V_B / T_B), so it is left out.
    """
    pi, e, s = np.broadcast_arrays(pi, e, s)
    bad, good = values
    early = e < T_B
    gf = np.where(early, s * good / (T_G - e), s * (good - bad) / (T_G - e))
    bf = np.where(early, s * bad / np.maximum(T_B - e, 1e-9), s * bad / T_B)
    return {"A": a / (1 - pi), "GF": gf, "BF": bf}


def share_label(share: float) -> str:
    return f"{100 * share:g}%"


def exercise_7() -> str:
    """Paper progress and offered share for a free agent, one panel per talent level."""
    n = 401
    talents = {label: point for label, point in settings_payoff_rates().items()
               if label in ("Min talent", "Max talent")}
    height = 4.7
    fig, axes = plt.subplots(1, len(talents), figsize=(5.17 * len(talents), height),
                             facecolor=SURFACE)
    pi = np.linspace(*OBS_LIMITS["pi"], n)
    log_s = np.linspace(*OBS_LIMITS["s"], n)
    PI, LOG_S = np.meshgrid(pi, log_s)
    for ax, (label, (tau, a, _)) in zip(axes, talents.items()):
        rates = observed_rates(PI, 0.0, 10 ** LOG_S, a, review_values(tau))
        draw_regions(ax, pi, log_s, rates)
        draw_boundaries(ax, pi, log_s, rates)
        style_axes(ax, OBS_LABELS["pi"], OBS_LABELS["s"], f"{label} (τ = {tau:g})")
        ax.set_yticks(np.log10(SHARE_TICKS), [share_label(t) for t in SHARE_TICKS])

    add_header(fig, height, "Paper progress and offered share",
               "Highest payoff per remaining step wins.  A review pays share s of a fresh"
               " paper's value; good faith adds its bump.")
    return save(fig, "ex7_observed_features.png")


# --- Exercise 8: paper progress, offered share and reputation ---------------

# Reputation is the agent's epsilon history: the mean epsilon of its completed
# reviews. A new review moves it 1 / (n + 1) of the way toward that review's
# epsilon. The sim's n grows with every review; here it is held fixed.
REPUTATION_BASE = SIM.prior_review_epsilon   # a new agent's epsilon history
REPUTATION_REVIEWS = 10
REPUTATION_WEIGHT = 1 / (REPUTATION_REVIEWS + 1)
REP_GRID = np.linspace(0.0, SIM.rl_reputation_feature_max, 129)
REP_BASE = {"pi": 0.0, "s": SHARE_REF, "r": REPUTATION_BASE}
# Share and reputation are log-scaled over the RL features' ranges.
REP_LIMITS = {
    "pi": (0.0, 0.99),
    "s": OBS_LIMITS["s"],
    "r": (np.log10(SIM.rl_reputation_feature_min), np.log10(SIM.rl_reputation_feature_max)),
}
REP_LABELS = {
    "pi": "Paper progress, π",
    "s": "Best offered share, s",
    "r": "Reputation, r  (mean review ε)",
}
REP_PANELS = [("pi", "s"), ("pi", "r"), ("s", "r")]
REPUTATION_TICKS = [0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


def review_epsilons(tau: float) -> tuple[float, float]:
    """Review effect of a bad- and a good-faith review by a reviewer of talent tau."""
    return review_accrual_bump(T_B, quality=tau), review_accrual_bump(T_G, quality=tau)


def reputation_offer(r) -> np.ndarray:
    """The sim's fair-market offer to a reviewer with reputation r, on a fresh paper.

    Other multipliers are 1, and the market's mean reputation is taken as the prior.
    """
    r = np.asarray(r, dtype=float)
    fair = SIM.reviewer_surplus_share * r / (1 + r)
    history = np.maximum(0.0, 1 + SIM.history_price_scale * (r - SIM.prior_review_epsilon))
    return np.clip(fair * history, SIM.min_offer_share, 1.0)


def lookup(W_ext: np.ndarray, x, r) -> np.ndarray:
    """W_ext at fractional paper step x and reputation r, interpolated linearly."""
    j = np.interp(r, REP_GRID, np.arange(REP_GRID.size))
    j0 = np.minimum(np.floor(j).astype(int), REP_GRID.size - 2)
    i0 = np.minimum(np.floor(x).astype(int), W_ext.shape[0] - 2)
    fj, fx = j - j0, x - i0

    def row(i):
        return W_ext[i, j0] * (1 - fj) + W_ext[i, j0 + 1] * fj

    return row(i0) * (1 - fx) + row(i0 + 1) * fx


def reputation_q(x, s, r, W, a: float, values, epsilons,
                 weight: float = REPUTATION_WEIGHT) -> dict[str, np.ndarray]:
    """Value of writing for a step, or doing a good- or bad-faith review, from one state.

    x is paper steps done. W is the agent's value at each paper step and reputation
    when it acts optimally afterwards, with later offers set by reputation. A paper
    pays a * steps on completion; a review pays s * value when it finishes, keeps
    paper progress where it is, and moves reputation toward its epsilon.
    """
    x, s, r = np.broadcast_arrays(x, s, r)
    steps = W.shape[0]
    W_ext = np.vstack([W, a * steps + W[:1]])   # reaching the last step publishes
    gamma = SIM.rl_gamma
    (bad, good), (eps_bad, eps_good) = values, epsilons
    return {
        "A": gamma * lookup(W_ext, x + 1, r),
        "GF": gamma ** T_G * (s * good + lookup(W_ext, x, r + weight * (eps_good - r))),
        "BF": gamma ** T_B * (s * bad + lookup(W_ext, x, r + weight * (eps_bad - r))),
    }


def solve_reputation(a: float, values, epsilons, weight: float = REPUTATION_WEIGHT) -> np.ndarray:
    """Value iteration for a free agent over (paper step, reputation)."""
    steps = int(round(mean_paper_timesteps()))
    x = np.arange(steps, dtype=float)[:, None]
    r = REP_GRID[None, :]
    W = np.zeros((steps, REP_GRID.size))
    for _ in range(100_000):
        q = reputation_q(x, reputation_offer(r), r, W, a, values, epsilons, weight)
        new = np.max(np.stack(list(q.values())), axis=0)
        if np.abs(new - W).max() < 1e-9 * np.abs(new).max():
            return new
        W = new
    raise RuntimeError("value iteration did not converge")


def exercise_8() -> str:
    """Paper progress, offered share and reputation, one row per talent level."""
    n = 401
    talents = {label: point for label, point in settings_payoff_rates().items()
               if label in ("Min talent", "Max talent")}
    height = 4.7 * len(talents)
    fig, axes = plt.subplots(len(talents), len(REP_PANELS),
                             figsize=(5.17 * len(REP_PANELS), height), facecolor=SURFACE)
    for row, (label, (tau, a, _)) in zip(axes, talents.items()):
        values, epsilons = review_values(tau), review_epsilons(tau)
        W = solve_reputation(a, values, epsilons)
        for ax, (x_key, y_key) in zip(row, REP_PANELS):
            x = np.linspace(*REP_LIMITS[x_key], n)
            y = np.linspace(*REP_LIMITS[y_key], n)
            grids = dict(zip((x_key, y_key), np.meshgrid(x, y)))
            state = {**REP_BASE, **{key: grid if key == "pi" else 10 ** grid
                                    for key, grid in grids.items()}}
            steps = W.shape[0]
            rates = reputation_q(state["pi"] * steps, state["s"], state["r"], W, a,
                                 values, epsilons)
            draw_regions(ax, x, y, rates)
            draw_boundaries(ax, x, y, rates)

            (held,) = [key for key in REP_BASE if key not in (x_key, y_key)]
            held_text = {"pi": "π = 0", "s": f"s = {share_label(REP_BASE['s'])}",
                         "r": f"r = {REP_BASE['r']:g}"}[held]
            style_axes(ax, REP_LABELS[x_key], REP_LABELS[y_key],
                       f"{label} (τ = {tau:g}),  {held_text}")
            if x_key == "s":
                ax.set_xticks(np.log10(SHARE_TICKS), [share_label(t) for t in SHARE_TICKS])
            if y_key == "s":
                ax.set_yticks(np.log10(SHARE_TICKS), [share_label(t) for t in SHARE_TICKS])
            if y_key == "r":
                ax.set_yticks(np.log10(REPUTATION_TICKS), [f"{t:g}" for t in REPUTATION_TICKS])

    add_header(fig, height, "What the agent sees, with reputation: paper progress, offered share, reputation",
               f"Best long-run value wins (γ = {SIM.rl_gamma:g}).  A review pays share s now; later"
               f" offers follow reputation, which each review moves 1/{REPUTATION_REVIEWS + 1}"
               " of the way toward its ε.")
    return save(fig, "ex8_reputation.png")


def exercise_4() -> str:
    return pair_figure(("a", "g", "b", "pi"), 3, "ex4_progress_pairs.png",
                       "f(A, GFPR, BFPR, paper progress)",
                       "A = τ · a / (1 − π),  GF = τ · g,  BF = b")


def exercise_5() -> str:
    return pair_figure(("a", "g", "b", "pi", "e"), 5, "ex5_review_effort_pairs.png",
                       "f(A, GFPR, BFPR, paper progress, review effort)",
                       f"A = τ · a / (1 − π),  GF = τ · g / (1 − e / {T_G:g}),  BF = b")


def exercise_6() -> str:
    return pair_figure(("a", "g", "b", "pi", "e", "s"), 5, "ex6_offered_share_pairs.png",
                       "f(A, GFPR, BFPR, paper progress, review effort, offered share)",
                       f"A = τ · a / (1 − π),  GF = (s / {SHARE_REF:g}) · τ · g / (1 − e / {T_G:g}),"
                       f"  BF = (s / {SHARE_REF:g}) · b")


def sanity_checks() -> None:
    zero = np.zeros(5)
    a = np.linspace(0.1, 2, 5)
    # Review pay 0 -> always author.
    assert (best_action({"A": a, "GF": zero, "BF": zero}) == 0).all()
    # Equal pay per review with T_G > T_B -> BF beats GF everywhere.
    pay = np.linspace(0.1, 10, 5)
    assert (best_action({"GF": pay / T_G, "BF": pay / T_B}) == 1).all()
    # The pair rates at a = 1, tau = 1, pi = 0, e = 0, s = SHARE_REF are Exercise 2's.
    g, b = np.meshgrid(np.linspace(0, 3, 31), np.linspace(0, 3, 31))
    ex2 = {"A": np.ones_like(g), "GF": g, "BF": b}
    assert (best_action(pair_rates(1.0, g, b, 0.0, tau=1.0)) == best_action(ex2)).all()
    # Authoring wins near completion, good faith near the end of a review.
    assert best_action(pair_rates(1.0, 3.0, 3.0, 0.99)) == 0
    assert best_action(pair_rates(1.0, 1.0, 3.0, 0.0, e=0.99 * T_G)) == 1
    # The offered share never changes good faith vs bad faith, and no pay means no review.
    low, high = pair_rates(1.0, g, b, 0.0, s=0.01), pair_rates(1.0, g, b, 0.0, s=0.10)
    assert (np.sign(low["GF"] - low["BF"]) == np.sign(high["GF"] - high["BF"])).all()
    assert best_action(pair_rates(1.0, 3.0, 3.0, 0.0, s=0.0)) == 0
    # Exercise 7: a fresh review never favors good faith, a tiny offer means
    # authoring, and good faith wins only with a review nearly finished.
    for tau, a, _ in settings_payoff_rates().values():
        values = review_values(tau)
        pi, s = np.meshgrid(np.linspace(0, 0.99, 41), np.logspace(-4, np.log10(0.5), 41))
        assert (best_action(observed_rates(pi, 0.0, s, a, values)) != 1).all()
        assert best_action(observed_rates(0.0, 0.0, 1e-4, a, values)) == 0
        assert best_action(observed_rates(0.0, 0.99 * T_G, 0.1, a, values)) == 1
    # Exercise 8: with reputation frozen, good faith never pays; with it moving,
    # more reputation is never worth less.
    _, a, _ = settings_payoff_rates()["Mid talent"]
    values, epsilons = review_values(1.0), review_epsilons(1.0)
    x = np.arange(int(round(mean_paper_timesteps())), dtype=float)[:, None]
    r = REP_GRID[None, :]
    frozen = solve_reputation(a, values, epsilons, weight=0.0)
    q = reputation_q(x, reputation_offer(r), r, frozen, a, values, epsilons, weight=0.0)
    assert (best_action(q) != 1).all()
    W = solve_reputation(a, values, epsilons)
    assert (np.diff(W, axis=1) >= -1e-9 * np.abs(W).max()).all()


def main() -> None:
    sanity_checks()
    paths = [exercise_1(), exercise_2_rates(), exercise_2_ratio(), exercise_3(),
             exercise_4(), exercise_5(), exercise_6(), exercise_7(), exercise_8()]
    print(f"T_B = {T_B:g}, T_G = {T_G:g}  (from config.py)")
    print("Boundaries:")
    print("  Ex1: a = p")
    print("  Ex2: g = a, b = a, g = b;  GF beats BF iff pay_G/pay_B > T_G/T_B"
          f" = {T_G / T_B:g}")
    print(f"  Ex3: GF beats BF iff tau > {T_G / T_B:g};"
          f" GF beats A iff P > {T_G:g}")
    print("  Ex4: A | GF g = a / (1 - pi);  A | BF b = tau * a / (1 - pi);  GF | BF b = tau * g")
    print(f"  Ex5: as Ex4, with g replaced by g / (1 - e / {T_G:g})")
    print(f"  Ex6: as Ex5, with g and b both scaled by s / {SHARE_REF:g}")
    print("  Ex7: a fresh review favors bad faith; good faith wins only with fewer than"
          " eps_G * T_B review steps left")
    print(f"  Ex8: value iteration with gamma = {SIM.rl_gamma:g}; each review moves reputation"
          f" 1/{REPUTATION_REVIEWS + 1} toward its epsilon")
    for path in paths:
        print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    main()
