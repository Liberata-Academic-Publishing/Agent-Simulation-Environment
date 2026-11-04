"""Payoff-rate phase diagram for continuous RL agents.

Question: when a timestep of peer review pays a different academic-capital rate
than a timestep of writing, does the learned policy take the higher-rate action
on turns where both are available?

Rates are not free parameters. A balanced probe (half the agents only write,
half take a good-faith review whenever a paper is listed) measures them under
each parameter cell:

* writing payoff rate = author-held AC gained per writing timestep
* review payoff rate = reviewer-held AC gained per review timestep

Both are taken from the second half of the probe, after the market has papers
in circulation. The phase boundary is the diagonal where the two rates are equal.
Two cells are chosen on each side of that diagonal. A shared linear Q policy is
trained in each cell, then scored only on free turns where a listed paper could
be claimed. Claiming is the review route; writing is the writing route.

Gamma is raised to 0.99 for this experiment so a manuscript's publication
return can be credited back along the writing steps that produced it. The
project default of 0.5 drops that credit to nearly zero within about ten
steps, which would make long writing look worthless regardless of its
academic-capital rate. Talent noise, experience growth, adaptive pricing, and
scarcity pricing are turned off so the cell parameters are what move the rates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass

import Agent as agent_module
import Paper as paper_module
import config as config_module
import train_rl
from Agent import (
    CONTINUOUS_CLAIM,
    CONTINUOUS_RESEARCH,
    Agent,
)
from Environment import Environment
from History import History
from Paper import Paper
from QLearningAgent import QAction, QLearningAgent
from config import SIM
from train_rl import parse_args as parse_train_args
from train_rl import train as run_training

OUTPUT_DIR = os.path.join("experiments", "payoff_phase")
FORECAST_HORIZON = 200
# A good-faith review is the review route whose rate is measured. Effort at or
# above this threshold is how continuous mode classifies a finished review.
REVIEW_TARGET_EFFORT = 3.0
EXPERIMENT_GAMMA = 0.99
OPTIMAL_BAR = 0.70
SUBOPTIMAL_BAR = 0.30
CLEAR_GAP = 0.15

_WRITING_TIME = frozenset({"write_paper", "review_finished_write"})
_REVIEW_TIME = frozenset({
    "review_started",
    "review_continued",
    "bad_faith_review",
    "good_faith_review",
})


@dataclass(frozen=True)
class Cell:
    """One parameter setting and the payoff rates measured in it."""

    paper_timesteps: float
    reviewer_surplus_share: float
    max_review_accrual_bump: float = 0.60
    writing_effort_per_timestep: float = 1.0
    review_effort_per_timestep: float = 1.0
    review_target_effort: float = REVIEW_TARGET_EFFORT

    def key(self) -> tuple:
        return (
            self.paper_timesteps,
            self.reviewer_surplus_share,
            self.max_review_accrual_bump,
            self.writing_effort_per_timestep,
            self.review_effort_per_timestep,
            self.review_target_effort,
        )


def payoff_rate(end_ac: float, start_ac: float, steps: int) -> float | None:
    """AC gained per timestep. Undefined when the activity was not used."""
    if steps <= 0:
        return None
    return (float(end_ac) - float(start_ac)) / float(steps)


def gap_fraction(writing_rate: float, review_rate: float) -> float:
    """Signed gap in units of the larger absolute rate. Positive: review pays more."""
    scale = max(abs(writing_rate), abs(review_rate), 1e-12)
    return (review_rate - writing_rate) / scale


def optimal_activity(writing_rate: float, review_rate: float) -> str:
    if review_rate > writing_rate:
        return "review"
    if writing_rate > review_rate:
        return "write"
    return "tie"


def choice_is_optimal(activity: str, claimed: bool) -> bool:
    """True when a free agent with a listed paper took the higher-rate route."""
    if activity == "review":
        return claimed
    if activity == "write":
        return not claimed
    return True


def verdict(optimal_choice_rate: float | None, opportunities: int) -> str:
    if opportunities < 20 or optimal_choice_rate is None:
        return "unobserved"
    if optimal_choice_rate >= OPTIMAL_BAR:
        return "optimal"
    if optimal_choice_rate <= SUBOPTIMAL_BAR:
        return "suboptimal"
    return "mixed"


def _set_sim(name: str, value) -> None:
    object.__setattr__(SIM, name, value)


def apply_world(cell: Cell) -> None:
    """Point the live config at this cell. Snapshot constants are copied too."""
    _set_sim("reviewer_surplus_share", float(cell.reviewer_surplus_share))
    _set_sim("max_review_accrual_bump", float(cell.max_review_accrual_bump))
    _set_sim("writing_effort_per_timestep", float(cell.writing_effort_per_timestep))
    _set_sim("review_effort_per_timestep", float(cell.review_effort_per_timestep))
    _set_sim("continuous_paper_timesteps", float(cell.paper_timesteps))
    _set_sim("paper_effort_min", float(cell.paper_timesteps))
    _set_sim("paper_effort_max", float(cell.paper_timesteps))
    _set_sim("paper_effort_mode", "fixed")
    _set_sim("continuous_publishing", "threshold")
    _set_sim("pricing_policy", "static_fair_market")
    _set_sim("use_scarcity_pricing", False)
    _set_sim("use_merit_market_clearing", False)
    _set_sim("use_competition_adjusted_forecast", False)
    _set_sim("use_rate_talents", False)
    _set_sim("use_experience_growth", False)
    _set_sim("use_review_quality_draws", False)
    _set_sim("randomize_initial_paper_progress", False)
    _set_sim("quality_sigma", 0.0)
    _set_sim("init_ac_min", 0.0)
    _set_sim("init_ac_max", 0.0)
    _set_sim("rl_gamma", EXPERIMENT_GAMMA)
    _set_sim("forecast_horizon_timesteps", FORECAST_HORIZON)

    paper_module.QUALITY_SIGMA = 0.0
    agent_module.QUALITY_SIGMA = 0.0
    paper_module.MAX_REVIEW_ACCRUAL_BUMP = float(cell.max_review_accrual_bump)
    paper_module.REVIEWER_SURPLUS_SHARE = float(cell.reviewer_surplus_share)
    paper_module.REVIEW_EFFORT_PER_TIMESTEP = float(cell.review_effort_per_timestep)
    agent_module.REVIEW_EFFORT_PER_TIMESTEP = float(cell.review_effort_per_timestep)
    agent_module.WRITING_EFFORT_PER_TIMESTEP = float(cell.writing_effort_per_timestep)
    # Keep the module binding used by scripts that did ``from config import SIM``
    # on the same object. Replacing config.SIM would not reach them.
    config_module.SIM = SIM


def environment_kwargs(cell: Cell) -> dict:
    length = float(cell.paper_timesteps)
    return dict(
        forecast_horizon_timesteps=FORECAST_HORIZON,
        review_paradigm="continuous",
        continuous_publishing="threshold",
        continuous_paper_timesteps=length,
        paper_effort_mode="fixed",
        paper_effort_min=length,
        paper_effort_max=length,
        discrete_paper_timesteps=length,
        randomize_initial_paper_progress=False,
        pricing_policy="static_fair_market",
        use_scarcity_pricing=False,
        use_merit_market_clearing=False,
        use_competition_adjusted_forecast=False,
        citations_enabled=True,
    )


def _capital_by_source(agents, papers) -> tuple[float, float]:
    roster = set(agents)
    writing = 0.0
    review = 0.0
    for paper in papers:
        author = paper.author
        current = float(paper.current_ac)
        for agent, share in paper.share_distribution.items():
            if agent not in roster:
                continue
            value = float(share) * current
            if agent is author:
                writing += value
            else:
                review += value
    return writing, review


def _seed_papers(agents: list[Agent], rng: random.Random) -> None:
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                quality=agent.intrinsic_talent,
                current_ac=0.0,
                market_listed=False,
            )
            stagger = max(1, int(SIM.initial_listing_stagger_timesteps))
            paper.scheduled_listing_timestep = 1 + ((index - 1) % stagger)
            paper.title = f"Seed {index}"
            Agent.all_papers.append(paper)


class _Writer(Agent):
    """Never claims. Every turn advances the manuscript."""

    def choose_marketplace_action(self):
        return None

    def choose_work_action(self):
        return "write_paper", None

    def choose_continuous_action(self):
        return (CONTINUOUS_RESEARCH, None)


class _Reviewer(Agent):
    """Claims whenever a paper is listed and holds it for a good-faith review.

    Continuous mode can leave a review only by spending a turn on research, so
    the turn that closes the review is writing time, matching the action log.
    """

    def __init__(self, *args, review_target_effort: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_target_effort = float(review_target_effort)

    def choose_marketplace_action(self):
        return None

    def choose_work_action(self):
        if self.active_review_paper is not None:
            return "peer_review", self.active_review_paper
        return "write_paper", None

    def choose_continuous_action(self):
        if self.active_review_paper is not None:
            if self.active_review_effort + 1e-9 < self.review_target_effort:
                return ("review", None)
            return (CONTINUOUS_RESEARCH, None)
        listed = [
            paper
            for paper in Agent.all_papers
            if paper.can_start_review(self)
        ]
        if listed:
            return (CONTINUOUS_CLAIM, random.choice(listed))
        return (CONTINUOUS_RESEARCH, None)


def _attach_time_counter(agent: Agent, counts: dict[str, int]) -> None:
    original = agent.act_continuous

    def wrapped():
        records = original()
        for record in records or []:
            kind = record.kind
            if kind in _WRITING_TIME:
                counts["writing"] += 1
            elif kind in _REVIEW_TIME:
                counts["review"] += 1
        return records

    agent.act_continuous = wrapped


def measure_rates(
    cell: Cell,
    *,
    steps: int,
    writers: int,
    reviewers: int,
    seed: int,
) -> dict:
    """Second-half AC per timestep for writing and for good-faith review."""
    apply_world(cell)
    random.seed(seed)
    Agent.all_papers = []
    agents: list[Agent] = []
    for index in range(writers):
        agents.append(_Writer(intrinsic_talent=1.0, name=f"Writer {index}"))
    for index in range(reviewers):
        agents.append(
            _Reviewer(
                intrinsic_talent=1.0,
                name=f"Reviewer {index}",
                review_target_effort=cell.review_target_effort,
            )
        )
    _seed_papers(agents, random.Random(seed))
    counts = {"writing": 0, "review": 0}
    for agent in agents:
        _attach_time_counter(agent, counts)
    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        history=None,
        **environment_kwargs(cell),
    )
    half = max(1, steps // 2)
    started = time.perf_counter()
    env.run(half)
    writing_start, review_start = _capital_by_source(env.agents, env.papers)
    counts["writing"] = 0
    counts["review"] = 0
    env.run(steps - half)
    writing_end, review_end = _capital_by_source(env.agents, env.papers)
    writing_rate = payoff_rate(writing_end, writing_start, counts["writing"])
    review_rate = payoff_rate(review_end, review_start, counts["review"])
    return {
        "paper_timesteps": cell.paper_timesteps,
        "reviewer_surplus_share": cell.reviewer_surplus_share,
        "max_review_accrual_bump": cell.max_review_accrual_bump,
        "writing_effort_per_timestep": cell.writing_effort_per_timestep,
        "review_effort_per_timestep": cell.review_effort_per_timestep,
        "review_target_effort": cell.review_target_effort,
        "steps": steps,
        "writing_steps": counts["writing"],
        "review_steps": counts["review"],
        "writing_ac_gain": writing_end - writing_start,
        "review_ac_gain": review_end - review_start,
        "writing_rate": writing_rate,
        "review_rate": review_rate,
        "gap_fraction": (
            None
            if writing_rate is None or review_rate is None
            else gap_fraction(writing_rate, review_rate)
        ),
        "elapsed_sec": time.perf_counter() - started,
    }


def _usable(row: dict, min_steps: int) -> bool:
    writing = row.get("writing_rate")
    review = row.get("review_rate")
    return (
        row.get("writing_steps", 0) >= min_steps
        and row.get("review_steps", 0) >= min_steps
        and writing is not None
        and review is not None
        and math.isfinite(writing)
        and math.isfinite(review)
    )


def select_cases(rows: list[dict], min_steps: int = 30) -> list[dict] | None:
    """Two cells above the diagonal and two below, mild and strong on each side.

    Returns None when either side of the boundary has no clear cell.
    """
    usable = [row for row in rows if _usable(row, min_steps)]
    above = sorted(
        [row for row in usable if row["gap_fraction"] > CLEAR_GAP],
        key=lambda row: row["gap_fraction"],
    )
    below = sorted(
        [row for row in usable if row["gap_fraction"] < -CLEAR_GAP],
        key=lambda row: row["gap_fraction"],
    )
    if not above or not below:
        return None

    def distinct_pair(side: list[dict]) -> tuple[dict, dict]:
        """Closest-to-boundary and farthest, preferring different parameters."""
        mild = side[0] if side[0]["gap_fraction"] > 0 else side[-1]
        strong = side[-1] if side[0]["gap_fraction"] > 0 else side[0]
        if _param_key(mild) == _param_key(strong):
            for candidate in side:
                if _param_key(candidate) != _param_key(strong):
                    mild = candidate
                    break
        return mild, strong

    review_mild, review_strong = distinct_pair(above)
    write_mild, write_strong = distinct_pair(below)
    chosen = [
        ("review-higher-mild", "Review pays more (mild)", review_mild),
        ("review-higher-strong", "Review pays more (strong)", review_strong),
        ("write-higher-mild", "Writing pays more (mild)", write_mild),
        ("write-higher-strong", "Writing pays more (strong)", write_strong),
    ]
    # Four different rate regimes. Duplicate parameter cells cannot show both.
    if len({_param_key(row) for _, _, row in chosen}) < 4:
        return None
    cases = []
    for name, label, row in chosen:
        case = dict(row)
        case["name"] = name
        case["label"] = label
        case["optimal_activity"] = optimal_activity(row["writing_rate"], row["review_rate"])
        cases.append(case)
    return cases


def _param_key(row: dict) -> tuple:
    return (
        row["paper_timesteps"],
        row["reviewer_surplus_share"],
        row["max_review_accrual_bump"],
        row["writing_effort_per_timestep"],
        row["review_effort_per_timestep"],
        row["review_target_effort"],
    )


def calibration_grid() -> list[Cell]:
    """Manuscript length × reviewer surplus. These two move the rates apart."""
    cells = []
    for paper in (12.0, 30.0, 80.0, 180.0):
        for surplus in (0.02, 0.20, 0.50, 0.90):
            cells.append(Cell(paper_timesteps=paper, reviewer_surplus_share=surplus))
    return cells


def extended_grid() -> list[Cell]:
    """Wider knobs used only if the first grid stays on one side of the diagonal."""
    cells = []
    for paper in (8.0, 240.0):
        for surplus in (0.0, 1.0):
            for bump in (0.20, 1.20):
                cells.append(
                    Cell(
                        paper_timesteps=paper,
                        reviewer_surplus_share=surplus,
                        max_review_accrual_bump=bump,
                    )
                )
    for writing_speed, review_speed in ((2.0, 0.5), (0.5, 2.0)):
        for paper in (20.0, 100.0):
            for surplus in (0.10, 0.70):
                cells.append(
                    Cell(
                        paper_timesteps=paper,
                        reviewer_surplus_share=surplus,
                        writing_effort_per_timestep=writing_speed,
                        review_effort_per_timestep=review_speed,
                    )
                )
    return cells


def _format_rate(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.5f}"


def run_calibration(cells: list[Cell], args, seen: set[tuple]) -> list[dict]:
    rows = []
    for index, cell in enumerate(cells, start=1):
        if cell.key() in seen:
            continue
        steps = max(args.calib_steps, int(4 * cell.paper_timesteps))
        print(
            f"[calibrate {index}/{len(cells)}] paper={cell.paper_timesteps:g} "
            f"surplus={cell.reviewer_surplus_share:g} "
            f"bump={cell.max_review_accrual_bump:g} "
            f"write_speed={cell.writing_effort_per_timestep:g} "
            f"review_speed={cell.review_effort_per_timestep:g} "
            f"steps={steps}",
            flush=True,
        )
        row = measure_rates(
            cell,
            steps=steps,
            writers=args.calib_writers,
            reviewers=args.calib_reviewers,
            seed=args.seed,
        )
        seen.add(cell.key())
        rows.append(row)
        print(
            f"    write {_format_rate(row['writing_rate'])} / step "
            f"({row['writing_steps']} steps)   "
            f"review {_format_rate(row['review_rate'])} / step "
            f"({row['review_steps']} steps)   "
            f"gap={_format_rate(row['gap_fraction'])}   "
            f"{row['elapsed_sec']:.1f}s",
            flush=True,
        )
    return rows


def cell_from_row(row: dict) -> Cell:
    return Cell(
        paper_timesteps=float(row["paper_timesteps"]),
        reviewer_surplus_share=float(row["reviewer_surplus_share"]),
        max_review_accrual_bump=float(row["max_review_accrual_bump"]),
        writing_effort_per_timestep=float(row["writing_effort_per_timestep"]),
        review_effort_per_timestep=float(row["review_effort_per_timestep"]),
        review_target_effort=float(row["review_target_effort"]),
    )


def install_training_env(cell: Cell) -> None:
    """train_rl.build_env reads Environment defaults captured at import.

    Those defaults ignore later config edits, so training goes through a
    builder that passes this cell's publishing and pricing settings explicitly.
    """

    def build_env(
        *,
        backend=None,
        backends=None,
        backends_b=None,
        epsilon: float,
        learning: bool,
        num_rl: int,
        num_heuristic: int,
        horizon: int,
        seed: int,
        review_paradigm: str = "continuous",
        gamma: float = EXPERIMENT_GAMMA,
        history: History | None = None,
        intrinsic_talent: float = 1.0,
        shared_talents: bool = False,
        frozen_backends=None,
    ):
        if shared_talents:
            raise RuntimeError("payoff-phase training does not use shared talents")
        if review_paradigm != "continuous":
            raise RuntimeError("payoff-phase training is continuous-mode only")
        apply_world(cell)
        if backends is None:
            if backend is None:
                raise ValueError("build_env requires backend or backends")
            backends = [backend] * num_rl
        random.seed(seed)
        Agent.all_papers = []
        rng = random.Random(seed)
        rl_agents = [
            QLearningAgent(
                intrinsic_talent=intrinsic_talent,
                forecast_horizon_timesteps=horizon,
                name=f"RL {i}",
                backend=backends[i],
                backend_b=None if backends_b is None else backends_b[i],
                epsilon=epsilon,
                learning=learning,
                gamma=gamma,
            )
            for i in range(num_rl)
        ]
        frozen = [
            QLearningAgent(
                intrinsic_talent=intrinsic_talent,
                forecast_horizon_timesteps=horizon,
                name=f"Frozen {i}",
                backend=frozen_backend,
                epsilon=0.0,
                learning=False,
                gamma=gamma,
            )
            for i, frozen_backend in enumerate(frozen_backends or [])
        ]
        agents: list[Agent] = [*rl_agents, *frozen]
        _seed_papers(agents, rng)
        # num_heuristic is accepted to match train_rl's call but left empty:
        # the policy is scored against itself, not against a heuristic opponent.
        del num_heuristic
        env = Environment(
            agents=agents,
            papers=Agent.all_papers,
            history=history,
            **environment_kwargs(cell),
        )
        return env, rl_agents, []

    train_rl.build_env = build_env


def train_case(case: dict, args) -> str:
    cell = cell_from_row(case)
    apply_world(cell)
    install_training_env(cell)
    policy_dir = os.path.join(OUTPUT_DIR, "policies")
    os.makedirs(policy_dir, exist_ok=True)
    policy_path = os.path.join(policy_dir, f"{case['name']}.npy")
    log_dir = os.path.join(OUTPUT_DIR, "training", case["name"])
    os.makedirs(log_dir, exist_ok=True)
    train_rl.OUTPUT_DIR = log_dir
    argv = [
        "--backend", "linear",
        "--shared-policy",
        "--episodes", str(args.episodes),
        "--timesteps", str(args.timesteps),
        "--num-rl", str(args.num_rl),
        "--num-heuristic", "0",
        "--horizon", str(FORECAST_HORIZON),
        "--gamma", str(EXPERIMENT_GAMMA),
        "--eps-start", "1.0",
        "--eps-end", "0.05",
        "--seed", str(args.seed),
        "--no-archive",
        "--save", policy_path,
    ]
    print(
        f"\n[train] {case['label']}: paper={cell.paper_timesteps:g} "
        f"surplus={cell.reviewer_surplus_share:g} "
        f"optimal={case['optimal_activity']}",
        flush=True,
    )
    run_training(parse_train_args(argv))
    return policy_path


def _score_one_seed(cell: Cell, backend, *, steps: int, num_rl: int, seed: int, activity: str) -> dict:
    apply_world(cell)
    random.seed(seed)
    Agent.all_papers = []
    agents = [
        QLearningAgent(
            intrinsic_talent=1.0,
            forecast_horizon_timesteps=FORECAST_HORIZON,
            name=f"RL {i}",
            backend=backend,
            epsilon=0.0,
            learning=False,
            gamma=EXPERIMENT_GAMMA,
        )
        for i in range(num_rl)
    ]
    _seed_papers(agents, random.Random(seed))
    stats = {
        "opportunities": 0,
        "claims": 0,
        "optimal_hits": 0,
        "early_opportunities": 0,
        "early_claims": 0,
        "early_hits": 0,
        "margins": [],
        "early_margins": [],
    }

    def wrap(agent: QLearningAgent):
        original = agent.choose_continuous_action

        def wrapped():
            in_review = agent.active_review_paper is not None
            best, share, ac = agent._best_reviewable(exclude=agent.active_review_paper)
            features = agent._features(share, ac)
            q_values = agent.backend.q_values(features)
            kind, _paper = original()
            if not in_review and best is not None:
                claimed = kind == CONTINUOUS_CLAIM
                hit = int(choice_is_optimal(activity, claimed))
                margin = float(q_values[QAction.CLAIM] - q_values[QAction.WRITE])
                stats["opportunities"] += 1
                stats["claims"] += int(claimed)
                stats["optimal_hits"] += hit
                stats["margins"].append(margin)
                # A nearly finished manuscript is worth finishing even when review
                # has the higher average rate. The route test uses the first half
                # of the manuscript, where that option value has not built up.
                threshold = max(agent.paper_completion_threshold(), 1e-9)
                if agent.paper_progress / threshold < 0.5:
                    stats["early_opportunities"] += 1
                    stats["early_claims"] += int(claimed)
                    stats["early_hits"] += hit
                    stats["early_margins"].append(margin)
            return kind, _paper

        agent.choose_continuous_action = wrapped

    for agent in agents:
        wrap(agent)
    history = History()
    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        history=history,
        **environment_kwargs(cell),
    )
    env.run(steps)
    writing_steps = 0
    review_steps = 0
    for _timestep, _agent, kind, _paper in history.actions:
        if kind in _WRITING_TIME:
            writing_steps += 1
        elif kind in _REVIEW_TIME:
            review_steps += 1
    opportunities = stats["opportunities"]
    early = stats["early_opportunities"]
    return {
        "seed": seed,
        "opportunities": opportunities,
        "claims": stats["claims"],
        "optimal_hits": stats["optimal_hits"],
        "optimal_choice_rate": (
            stats["optimal_hits"] / opportunities if opportunities else None
        ),
        "early_opportunities": early,
        "early_hits": stats["early_hits"],
        "early_optimal_choice_rate": (
            stats["early_hits"] / early if early else None
        ),
        "early_claim_rate": stats["early_claims"] / early if early else None,
        "claim_rate": stats["claims"] / opportunities if opportunities else None,
        "mean_q_margin": (
            sum(stats["margins"]) / len(stats["margins"]) if stats["margins"] else None
        ),
        "early_mean_q_margin": (
            sum(stats["early_margins"]) / len(stats["early_margins"])
            if stats["early_margins"] else None
        ),
        "writing_steps": writing_steps,
        "review_steps": review_steps,
        "writing_time_share": writing_steps / max(1, writing_steps + review_steps),
        "review_time_share": review_steps / max(1, writing_steps + review_steps),
        "mean_final_ac": sum(agent.academic_capital for agent in agents) / len(agents),
    }


def evaluate_case(case: dict, policy_path: str, args) -> dict:
    from QLearningAgent import LinearQ

    backend = LinearQ()
    backend.load(policy_path)
    cell = cell_from_row(case)
    per_seed = [
        _score_one_seed(
            cell,
            backend,
            steps=args.timesteps,
            num_rl=args.num_rl,
            seed=args.seed + 1000 + offset,
            activity=case["optimal_activity"],
        )
        for offset in range(args.eval_seeds)
    ]
    opportunities = sum(row["opportunities"] for row in per_seed)
    hits = sum(row["optimal_hits"] for row in per_seed)
    claims = sum(row["claims"] for row in per_seed)
    early = sum(row["early_opportunities"] for row in per_seed)
    early_hits = sum(row["early_hits"] for row in per_seed)
    margins = [row["mean_q_margin"] for row in per_seed if row["mean_q_margin"] is not None]
    early_margins = [
        row["early_mean_q_margin"] for row in per_seed if row["early_mean_q_margin"] is not None
    ]
    overall_rate = hits / opportunities if opportunities else None
    early_rate = early_hits / early if early else None
    # Prefer the early-manuscript score. Fall back when that slice is thin.
    if early >= 20 and early_rate is not None:
        rate = early_rate
        scored = "early manuscript"
    else:
        rate = overall_rate
        scored = "all free turns"
    summary = {
        "policy_path": policy_path,
        "opportunities": opportunities,
        "optimal_hits": hits,
        "optimal_choice_rate": rate,
        "scored_on": scored,
        "overall_optimal_choice_rate": overall_rate,
        "early_opportunities": early,
        "early_optimal_choice_rate": early_rate,
        "claim_rate": claims / opportunities if opportunities else None,
        "mean_q_margin": sum(margins) / len(margins) if margins else None,
        "early_mean_q_margin": (
            sum(early_margins) / len(early_margins) if early_margins else None
        ),
        "writing_time_share": (
            sum(row["writing_steps"] for row in per_seed)
            / max(1, sum(row["writing_steps"] + row["review_steps"] for row in per_seed))
        ),
        "review_time_share": (
            sum(row["review_steps"] for row in per_seed)
            / max(1, sum(row["writing_steps"] + row["review_steps"] for row in per_seed))
        ),
        "verdict": verdict(rate, early if scored.startswith("early") else opportunities),
        "seeds": per_seed,
    }
    print(
        f"[eval] {case['label']}: optimal route={case['optimal_activity']}  "
        f"choice={_format_rate(rate)} ({scored}, n={early if scored.startswith('early') else opportunities})  "
        f"overall={_format_rate(overall_rate)}  "
        f"claim_rate={_format_rate(summary['claim_rate'])}  "
        f"Q(claim)-Q(write)={_format_rate(summary['early_mean_q_margin'])}  "
        f"verdict={summary['verdict']}",
        flush=True,
    )
    return summary


def plot_phase(cases: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bars) = plt.subplots(1, 2, figsize=(12.2, 5.6))
    writing = [case["writing_rate"] for case in cases]
    review = [case["review_rate"] for case in cases]
    lo = min(writing + review)
    hi = max(writing + review)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    xs = [lo, hi]
    ax.fill_between(xs, xs, [hi, hi], color="#d6e6f5", alpha=0.85, label="Review pays more", zorder=0)
    ax.fill_between(xs, [lo, lo], xs, color="#f8e3c4", alpha=0.9, label="Writing pays more", zorder=0)
    ax.plot(xs, xs, linestyle="--", color="#555555", linewidth=1.0, label="Equal payoff rate", zorder=1)

    verdict_color = {
        "optimal": "#1b7f4e",
        "suboptimal": "#c0392b",
        "mixed": "#7f8c8d",
        "unobserved": "#7f8c8d",
    }
    # Labels sit in open parts of each region and point at the marker.
    placements = {
        "review-higher-strong": (0.014, 0.022),
        "review-higher-mild": (0.040, 0.076),
        "write-higher-mild": (0.048, 0.028),
        "write-higher-strong": (0.036, 0.008),
    }
    for case in cases:
        color = verdict_color.get(case["verdict"], "#7f8c8d")
        marker = "X" if case["verdict"] == "suboptimal" else "o"
        ax.scatter(
            case["writing_rate"],
            case["review_rate"],
            s=110 if marker == "X" else 80,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        note = (
            f"{case['label']}\n"
            f"{case['paper_timesteps']:g}-step papers, "
            f"{case['reviewer_surplus_share']:.0%} reviewer share"
        )
        ax.annotate(
            note,
            (case["writing_rate"], case["review_rate"]),
            textcoords="data",
            xytext=placements.get(case["name"], (case["writing_rate"], case["review_rate"])),
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "color": "#444444", "lw": 0.6},
            zorder=4,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Writing payoff rate (AC per writing timestep)")
    ax.set_ylabel("Review payoff rate (AC per review timestep)")
    ax.set_title("Phase diagram")
    ax.legend(loc="upper left", fontsize=8, frameon=True)

    labels = [case["label"] for case in cases]
    rates = [
        0.0 if case["optimal_choice_rate"] is None else 100.0 * case["optimal_choice_rate"]
        for case in cases
    ]
    colors = [verdict_color.get(case["verdict"], "#7f8c8d") for case in cases]
    positions = list(range(len(cases)))
    bars.barh(positions, rates, color=colors, edgecolor="black", linewidth=0.4)
    bars.axvline(50, color="#555555", linestyle=":", linewidth=1.0)
    bars.axvline(100 * OPTIMAL_BAR, color="#1b7f4e", linestyle="--", linewidth=0.8)
    bars.set_yticks(positions)
    bars.set_yticklabels(labels, fontsize=8)
    bars.set_xlim(0, 100)
    bars.set_xlabel("Share of choice turns on the higher-rate route (%)")
    bars.set_title("Does the policy take it?")
    for pos, case, rate in zip(positions, cases, rates):
        bars.text(min(rate + 1.5, 86), pos, case["verdict"], va="center", fontsize=8)
    fig.suptitle(
        "Learned review-vs-writing policy against the payoff-rate boundary",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.028,
        "Green circle: the greedy policy took the higher-rate action on at least 70% of early choice turns.",
        fontsize=8,
        color="#333333",
    )
    fig.text(
        0.01,
        0.008,
        "Red cross: it did so on at most 30%. Rates are AC per timestep from a balanced probe, not from the learned population.",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Payoff-rate phase diagram for RL agents.")
    parser.add_argument("--seed", type=int, default=SIM.seed)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=600)
    parser.add_argument("--num-rl", dest="num_rl", type=int, default=8)
    parser.add_argument("--eval-seeds", dest="eval_seeds", type=int, default=3)
    parser.add_argument("--calib-steps", dest="calib_steps", type=int, default=400)
    parser.add_argument("--calib-writers", dest="calib_writers", type=int, default=6)
    parser.add_argument("--calib-reviewers", dest="calib_reviewers", type=int, default=6)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument(
        "--from-grid",
        default=None,
        help="Skip calibration and select cases from this grid JSON.",
    )
    args = parser.parse_args(argv)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grid_path = os.path.join(OUTPUT_DIR, "calibration_grid.json")
    if args.from_grid:
        with open(args.from_grid, encoding="utf-8") as fh:
            rows = json.load(fh)
    else:
        print("Measuring writing and review payoff rates...", flush=True)
        seen: set[tuple] = set()
        rows = run_calibration(calibration_grid(), args, seen)
        cases = select_cases(rows)
        if cases is None:
            print("First grid did not put two distinct cells on each side. Extending.", flush=True)
            rows.extend(run_calibration(extended_grid(), args, seen))
        with open(grid_path, "w", encoding="utf-8") as fh:
            json.dump(_json_ready(rows), fh, indent=2)
        print(f"Wrote {grid_path}", flush=True)

    cases = select_cases(rows)
    if cases is None:
        raise SystemExit(
            "Could not find two distinct parameter cells on each side of the "
            "equal-payoff diagonal. See experiments/payoff_phase/calibration_grid.json."
        )
    for case in cases:
        print(
            f"Selected {case['label']}: write={case['writing_rate']:.5f} "
            f"review={case['review_rate']:.5f} "
            f"paper={case['paper_timesteps']:g} surplus={case['reviewer_surplus_share']:g} "
            f"optimal={case['optimal_activity']}",
            flush=True,
        )
    if args.calibrate_only:
        return

    results = []
    for case in cases:
        policy_path = train_case(case, args)
        evaluation = evaluate_case(case, policy_path, args)
        merged = dict(case)
        merged.update(evaluation)
        results.append(merged)

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(
            _json_ready({
                "gamma": EXPERIMENT_GAMMA,
                "optimal_bar": OPTIMAL_BAR,
                "definition": (
                    "Payoff rate is second-half academic capital gained from that "
                    "activity, per timestep spent on it, in a balanced probe."
                ),
                "cases": results,
            }),
            fh,
            indent=2,
        )
    figure_path = os.path.join(OUTPUT_DIR, "phase_diagram.png")
    plot_phase(results, figure_path)
    print(f"Wrote {results_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
