"""Central configuration for the Liberata simulation and RL training.

Single source of truth for defaults. Scripts (``run_simulation.py``,
``train_rl.py``) read these defaults and expose CLI flags that *override* them at
runtime — the standard config-first / CLI-override pattern. Edit the dataclass
defaults here to change behavior everywhere; pass flags for one-off runs.

``SimConfig`` (``SIM``) holds *every* parameter that defines a single simulation
run — world size, initial papers, paper economics, the effort/reward model, the
publishing threshold, heuristic forecasting weights, and the RL agents'
settings (the RL agents are part of the simulation). ``TrainConfig`` (``TRAIN``)
and ``TrainDQNConfig`` (``TRAIN_DQN``) hold training-loop knobs for
``train_rl.py`` and ``train_dqn.py`` respectively.

Stdlib only (``dataclasses``) — no extra dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Every parameter that defines a single simulation run."""

    # --- World -----------------------------------------------------------
    num_heuristic_agents: int = 0
    num_rl_agents: int = 20
    num_dqn_agents: int = 0
    num_random_agents: int = 0
    num_probabilistic_agents: int = 0
    num_low_talent_rl_agents: int = 0
    num_timesteps: int = 5000
    seed: int = 11
    # Must cover manuscript completion plus a post-publication earning window;
    # otherwise writing has zero value in the agents' forward-looking choices.
    forecast_horizon_timesteps: int = 200
    output_dir: str = "runs"

    # --- Initial papers --------------------------------------------------
    # Papers seeded before timestep 1 (bootstraps review material). Set
    # init_papers_per_agent=0 for no starting papers, or init_ac_min=init_ac_max=0
    # to start every agent at zero capital.
    init_papers_per_agent: int = 2
    initial_listing_stagger_timesteps: int = 50
    init_ac_min: float = 0
    init_ac_max: float = 0
    init_accrual_min: float = 0.8
    init_accrual_max: float = 1.5

    # --- Paper economics -------------------------------------------------
    default_accrual_rate: float = 1.0       # base AC gained per timestep, before bumps
    default_review_share: float = 0.005      # legacy fallback when fair-market pricing is off
    default_max_reviewer_share: float = 1.0        # max share per review (100% of author stake)
    min_offer_share: float = 0.0001          # floor on review offers (0%)
    use_fair_market_pricing: bool = True
    prior_review_epsilon: float = 0.05      # prior reviewer effect before review history exists
    # Author-side pricing policy. ``static_fair_market`` preserves the current
    # fair-market formula. ``adaptive_multiplier`` keeps that formula as the
    # base, then lets authors raise/lower future offers based on how quickly
    # their papers are claimed from the marketplace.
    pricing_policy: str = "adaptive_multiplier"  # "static_fair_market" | "adaptive_multiplier"
    # ``global``: one author multiplier (legacy). ``binned``: per reputation tier.
    adaptive_pricing_mode: str = "binned"  # "global" | "binned"
    target_market_wait_timesteps: float = 1.0
    adaptive_pricing_learning_rate: float = 0.1
    min_author_price_multiplier: float = 0.0005
    max_author_price_multiplier: float = 10.0
    # Reputation bins for binned adaptive pricing (len(names) == len(edges) + 1).
    reputation_bin_edges: tuple[float, ...] = (0.08, 0.15)
    reputation_bin_names: tuple[str, ...] = ("low", "mid", "high")
    # Legacy tier labels (kept for run metadata / gallery). Binned adaptive
    # feedback is symmetric per claimer bin: wait <= target lowers that bin's
    # multiplier; wait > target raises it (same rule as global mode).
    adaptive_raise_bins: tuple[str, ...] = ("high",)
    adaptive_lower_bins: tuple[str, ...] = ("low", "mid")
    adaptive_slow_raise_bins: tuple[str, ...] = ("low", "mid", "high")
    fast_claim_max_wait: float | None = 0.0
    # Reviewer's fraction of incremental review surplus in fair-market offers
    # (author keeps ``1 - reviewer_surplus_share`` of the bump-created value).
    reviewer_surplus_share: float = 0.5

    # --- Market economics (optional; all off preserves legacy behavior) ----
    use_competition_adjusted_forecast: bool = True
    # Fast supply/demand scaler; disabled automatically when ``pricing_policy``
    # is ``adaptive_multiplier`` (adaptive author pricing already responds to demand).
    use_scarcity_pricing: bool = True
    reviewer_pressure_exponent: float = 0.5
    use_merit_market_clearing: bool = True

    # --- Paper quality ---------------------------------------------------
    # Each paper's quality is drawn from N(author talent, quality_sigma) and is
    # known to the author before they start writing it. Quality scales the
    # paper's base accrual rate and the accrual bump a review can earn, and it
    # drives the per-reviewer share the author is willing to offer.
    quality_sigma: float = 0.20
    min_paper_quality: float = 0.10
    quality_price_scale: float = 1.5    # higher quality -> smaller offered share
    history_price_scale: float = 0.5    # better reviewer history -> larger offered share

    # --- Effort & reward model -------------------------------------------
    review_paradigm: str = "continuous"       # "continuous" | "discrete"
    review_effort_per_timestep: float = 1.0     # effort added per review timestep
    writing_effort_per_timestep: float = 1.0    # continuous writing effort per timestep
    min_review_effort_threshold: float = 3.0    # minimum valid review/share effort
    good_faith_review_threshold: float = 5.0    # continuous-mode classification
    bad_review_timesteps: float = 1.0           # discrete bad-faith duration (T_B)
    # Fallback when continuous publishing is ``choice``; otherwise derived from
    # ``good_faith_review_threshold`` (see ``discrete_good_review_timesteps``).
    good_review_timesteps: float = 5.0
    review_effort_curve: str = "sigmoid"        # "sigmoid" | "log" | "jump"
    min_review_accrual_bump: float = 0.0       # sigmoid bump at one timestep
    max_review_accrual_bump: float = 0.60       # sigmoid saturation near a long review
    review_sigmoid_midpoint: float = 3.0        # review length where impact accelerates
    review_sigmoid_steepness: float = 0.5        # slope: lower = gentler (0.35–0.6 stretched)
    review_sigmoid_saturation_effort: float = 25.0  # effort where bump ~saturates
    review_jump_threshold: float = 15.0          #f optional jump-mode high-effort cutoff
    review_jump_bump: float = 0.50               # optional extra bump after the cutoff
    review_jump_width: float = 0.75              # softness of the jump-mode transition
    base_review_accrual_bump: float = 0.20      # log-mode rate bump at exactly the threshold
    first_extra_day_bump: float = 0.10          # log-mode first extra timestep bump
    # Review bump lifetime. ``permanent`` keeps the current multiplier forever;
    # ``decay`` applies epsilon at review time then decays it exponentially to
    # zero (optionally hard-capped after ``review_bump_decay_cap_timesteps``).
    review_bump_duration: str = "permanent"     # "permanent" | "decay"
    review_bump_decay_rate: float = 0.05        # k in exp(-k * t) since review
    review_bump_decay_cap_timesteps: float | None = 800  # force bump=0 after this

    # --- Publishing ------------------------------------------------------
    paper_threshold: float = 10.0   # legacy forecast normalizer (choice-mode continuous)
    # Continuous writing: "choice" = agent picks when to finish/list; "threshold"
    # = auto-publish after a fixed amount of writing effort (no early finish).
    continuous_publishing: str = "threshold"       # "choice" | "threshold"
    continuous_paper_timesteps: float = 50.0    # writing effort to auto-publish
    # Fallback when continuous publishing is ``choice``; otherwise derived from
    # ``continuous_paper_timesteps`` (see ``discrete_manuscript_timesteps``).
    discrete_paper_timesteps: float = 200.0
    discrete_writing_effort_per_timestep: float = 1.0
    # Paper effort target for each manuscript. ``fixed`` preserves the existing
    # thresholds above; ``uniform`` samples once per paper from the 50-150 band
    # discussed in sync; ``quality_scaled`` maps higher sampled paper quality to
    # a larger target inside that same band.
    paper_effort_mode: str = "quality_scaled"             # "fixed" | "uniform" | "quality_scaled"
    paper_effort_min: float = 130.0
    paper_effort_max: float = 170.0
    # Continuous-mode asymptotic writing model: a paper's accrual rate approaches
    # its quality-defined ceiling as accumulated writing effort grows. Higher k
    # reaches the ceiling faster (k=0.2 -> ~63% at 5 ts, ~86% at 10 ts).
    writing_saturation: float = 0.2

    # --- Control / probability agents ------------------------------------
    random_claim_probability: float = 0.5
    random_good_faith_probability: float = 0.5
    probabilistic_claim_probability: float = 0.5
    probabilistic_good_faith_probability: float = 0.5

    # --- Heuristic forecasting -------------------------------------------
    expected_write_progress: float = 0.5
    max_forecast_effort: int = 25
    continue_marginal_weight: float = 0.15
    preferred_extra_review_timesteps: float = 4.0

    # --- RL agents (part of the simulation) ------------------------------
    rl_backend: str = "tabular"     # "tabular" | "linear"
    rl_epsilon: float = 0.1         # exploration when learning online
    rl_gamma: float = 0.99          # TD discount; preserves delayed paper rewards
    rl_reward_ac_weight: float = 1.0      # weight on Δ academic capital
    rl_reward_rank_weight: float = 10.0   # avoid a dominant short-term rank race
    rl_reward_accrual_weight: float = 1.0  # weight on Δ portfolio accrual rate
    rl_autoload_policy: bool = True  # auto-load the saved baseline for RL agents
    rl_low_talent_autoload_policy: bool = True  # auto-load low-talent RL policy
    talent_min: float = 0.6         # default talent spread; CLI can widen/narrow it
    talent_max: float = 1.4
    low_talent_value: float = 0.3   # fixed intrinsic talent for low-talent RL agents
    policies_dir: str = "policies"

    # --- Export / gallery limits -----------------------------------------
    # Full histories remain in local_data/. The committed static gallery keeps
    # only the most recent action rows for feed/replay while retaining compact
    # per-timestep action counts for full action-mix plots.
    gallery_action_limit: int = 5000

    # --- DQN agents (separate from tabular/linear RL) ------------------
    dqn_hidden_size: int = 64
    dqn_hidden_layers: int = 2
    dqn_lr: float = 0.001
    dqn_gamma: float = 0.95
    dqn_epsilon: float = 0.1
    dqn_replay_capacity: int = 10000
    dqn_batch_size: int = 32
    dqn_target_sync: int = 200
    dqn_autoload_policy: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Defaults for the training harness (train_rl.py) — kept separate from the
    simulation parameters above."""

    episodes: int = 200
    timesteps: int = 1000
    num_rl: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05
    low_talent: bool = True
    low_talent_value: float = 0.3


@dataclass(frozen=True)
class TrainDQNConfig:
    """Defaults for the DQN training harness (train_dqn.py)."""

    episodes: int = 200
    timesteps: int = 1000
    num_dqn: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05


@dataclass(frozen=True)
class TrainDiscreteConfig:
    """Defaults for the discrete RL harness (train_discrete_rl.py)."""

    episodes: int = 200
    timesteps: int = 1000
    num_rl: int = 10
    num_heuristic: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.05
    low_talent: bool = False
    low_talent_value: float = 0.3


SIM = SimConfig()
TRAIN = TrainConfig()
TRAIN_DQN = TrainDQNConfig()
TRAIN_DISCRETE = TrainDiscreteConfig()


def discrete_good_review_timesteps() -> float:
    """Good-faith review length in discrete mode.

    Tracks ``good_faith_review_threshold`` so discrete and continuous share the
    same good/bad-faith cutoff without duplicating tunables.
    """
    return SIM.good_faith_review_threshold


def discrete_manuscript_timesteps() -> float:
    """Manuscript effort to publish in discrete mode.

    When continuous mode uses threshold publishing, discrete writing matches
    ``continuous_paper_timesteps``. Otherwise falls back to ``discrete_paper_timesteps``.
    """
    if SIM.continuous_publishing == "threshold":
        return SIM.continuous_paper_timesteps
    return SIM.discrete_paper_timesteps


def default_policy_path(backend_kind: str) -> str:
    """Canonical on-disk path for a trained policy of the given backend.

    Tabular pickles (``.pkl``); linear uses ``np.save`` (``.npy``). This is the
    one implementation both train_rl.py (save) and run_simulation.py (load) use.
    """
    ext = ".pkl" if backend_kind == "tabular" else ".npy"
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}{ext}")


def default_dqn_policy_path() -> str:
    """Canonical on-disk path for a trained DQN policy."""
    return os.path.join(SIM.policies_dir, "policy_dqn.pkl")


def default_discrete_policy_path(backend_kind: str) -> str:
    """Canonical path for a discrete 3-action Q policy (``train_discrete_rl.py``)."""
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}_discrete.pkl")


def default_low_talent_policy_path(backend_kind: str) -> str:
    """Canonical path for a low-talent continuous RL policy."""
    ext = ".pkl" if backend_kind == "tabular" else ".npy"
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}_low_talent{ext}")


def default_low_talent_discrete_policy_path(backend_kind: str) -> str:
    """Canonical path for a low-talent discrete RL policy."""
    return os.path.join(SIM.policies_dir, f"policy_{backend_kind}_discrete_low_talent.pkl")
