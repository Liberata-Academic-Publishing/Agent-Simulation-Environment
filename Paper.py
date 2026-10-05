from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Agent import Agent

from config import SIM, discrete_good_review_timesteps

# Bound from SIM (config.py is the single source of truth); names kept for the
# many internal references and default-argument signatures below.
DEFAULT_ACCRUAL_RATE = SIM.default_accrual_rate
DEFAULT_REVIEW_SHARE = SIM.default_review_share
USE_FAIR_MARKET_PRICING = SIM.use_fair_market_pricing
PRIOR_REVIEW_EPSILON = SIM.prior_review_epsilon
MIN_REVIEW_EFFORT_THRESHOLD = SIM.min_review_effort_threshold
GOOD_FAITH_REVIEW_THRESHOLD = SIM.good_faith_review_threshold
REVIEW_EFFORT_PER_TIMESTEP = SIM.review_effort_per_timestep
BAD_REVIEW_TIMESTEPS = SIM.bad_review_timesteps
# Prefer ``discrete_good_review_timesteps()`` at runtime; this alias is import-time.
GOOD_REVIEW_TIMESTEPS = discrete_good_review_timesteps()
DISCRETE_PAPER_TIMESTEPS = SIM.discrete_paper_timesteps
DISCRETE_WRITING_EFFORT_PER_TIMESTEP = SIM.discrete_writing_effort_per_timestep
REVIEW_EFFORT_CURVE = SIM.review_effort_curve
MIN_REVIEW_ACCRUAL_BUMP = SIM.min_review_accrual_bump
MAX_REVIEW_ACCRUAL_BUMP = SIM.max_review_accrual_bump
REVIEW_SIGMOID_MIDPOINT = SIM.review_sigmoid_midpoint
REVIEW_SIGMOID_STEEPNESS = SIM.review_sigmoid_steepness
REVIEW_SIGMOID_SATURATION_EFFORT = SIM.review_sigmoid_saturation_effort
REVIEW_JUMP_THRESHOLD = SIM.review_jump_threshold
REVIEW_JUMP_BUMP = SIM.review_jump_bump
REVIEW_JUMP_WIDTH = SIM.review_jump_width
BASE_REVIEW_ACCRUAL_BUMP = SIM.base_review_accrual_bump
FIRST_EXTRA_DAY_BUMP = SIM.first_extra_day_bump
DEFAULT_MAX_REVIEWER_SHARE = SIM.default_max_reviewer_share
MIN_OFFER_SHARE = SIM.min_offer_share
QUALITY_SIGMA = SIM.quality_sigma
MIN_PAPER_QUALITY = SIM.min_paper_quality
QUALITY_PRICE_SCALE = SIM.quality_price_scale
HISTORY_PRICE_SCALE = SIM.history_price_scale
WRITING_SATURATION = SIM.writing_saturation
PRICING_POLICY = SIM.pricing_policy
FORECAST_HORIZON_TIMESTEPS = SIM.forecast_horizon_timesteps
REVIEWER_SURPLUS_SHARE = SIM.reviewer_surplus_share
CITATION_REFERENCE_COUNT = SIM.citation_reference_count
CITATION_REFERENCE_COUNT_DISTRIBUTION = SIM.citation_reference_count_distribution
CITATION_QUALITY_WEIGHT = SIM.citation_quality_weight
CITATION_AGE_DECAY_RATIO = SIM.citation_age_decay_ratio
CITATION_AC_PER_NEW_PAPER = SIM.citation_ac_per_new_paper

CITATION_REFERENCE_COUNT_FIXED = "fixed"
CITATION_REFERENCE_COUNT_POISSON = "poisson"
VALID_CITATION_REFERENCE_COUNT_DISTRIBUTIONS = frozenset({
    CITATION_REFERENCE_COUNT_FIXED,
    CITATION_REFERENCE_COUNT_POISSON,
})

REVIEW_BUMP_PERMANENT = "permanent"
REVIEW_BUMP_DECAY = "decay"
VALID_REVIEW_BUMP_DURATIONS = frozenset({
    REVIEW_BUMP_PERMANENT,
    REVIEW_BUMP_DECAY,
})

REVIEW_PARADIGM_CONTINUOUS = "continuous"
REVIEW_PARADIGM_DISCRETE = "discrete"
VALID_REVIEW_PARADIGMS = frozenset({
    REVIEW_PARADIGM_CONTINUOUS,
    REVIEW_PARADIGM_DISCRETE,
})

BAD_FAITH_REVIEW = "bad_faith"
GOOD_FAITH_REVIEW = "good_faith"
VALID_REVIEW_KINDS = frozenset({BAD_FAITH_REVIEW, GOOD_FAITH_REVIEW})

PRICING_POLICY_STATIC = "static_fair_market"
PRICING_POLICY_ADAPTIVE = "adaptive_multiplier"
VALID_PRICING_POLICIES = frozenset({
    PRICING_POLICY_STATIC,
    PRICING_POLICY_ADAPTIVE,
})

ADAPTIVE_PRICING_GLOBAL = "global"
ADAPTIVE_PRICING_BINNED = "binned"
VALID_ADAPTIVE_PRICING_MODES = frozenset({
    ADAPTIVE_PRICING_GLOBAL,
    ADAPTIVE_PRICING_BINNED,
})


def validate_adaptive_pricing_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in VALID_ADAPTIVE_PRICING_MODES:
        allowed = ", ".join(sorted(VALID_ADAPTIVE_PRICING_MODES))
        raise ValueError(f"adaptive_pricing_mode must be one of: {allowed}")
    return value


def validate_reputation_bin_config(
    edges: tuple[float, ...] | list[float],
    names: tuple[str, ...] | list[str],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return sorted edges and names with one name per bin (``len(edges) + 1``)."""
    edge_values = tuple(sorted(max(0.0, float(edge)) for edge in edges))
    name_values = tuple(str(name).strip().lower() for name in names)
    if len(name_values) != len(edge_values) + 1:
        raise ValueError(
            "reputation_bin_names must contain exactly one more entry than "
            "reputation_bin_edges"
        )
    if not name_values or any(not name for name in name_values):
        raise ValueError("reputation_bin_names must be non-empty strings")
    if len(set(name_values)) != len(name_values):
        raise ValueError("reputation_bin_names must be unique")
    return edge_values, name_values


def reputation_bin_name(
    epsilon: float,
    edges: tuple[float, ...] | list[float],
    names: tuple[str, ...] | list[str],
) -> str:
    """Map a reviewer's epsilon history into a configured reputation bin."""
    _, name_values = validate_reputation_bin_config(edges, names)
    value = max(0.0, float(epsilon))
    for index, edge in enumerate(sorted(float(edge) for edge in edges)):
        if value < edge:
            return name_values[index]
    return name_values[-1]

def validate_citation_reference_count_distribution(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in VALID_CITATION_REFERENCE_COUNT_DISTRIBUTIONS:
        allowed = ", ".join(sorted(VALID_CITATION_REFERENCE_COUNT_DISTRIBUTIONS))
        raise ValueError(
            f"citation_reference_count_distribution must be one of: {allowed}"
        )
    return value


def _sample_poisson(mean: float, rng: random.Random) -> int:
    """Poisson-distributed sample with the given mean (Knuth's algorithm, stdlib-only)."""
    threshold = math.exp(-max(0.0, float(mean)))
    count = 0
    product = 1.0
    while True:
        product *= rng.random()
        if product <= threshold:
            return count
        count += 1


def sample_citation_reference_count(
    mean: float,
    distribution: str = CITATION_REFERENCE_COUNT_DISTRIBUTION,
    rng: random.Random | None = None,
) -> int:
    """Resolve one paper's reference count from the configured distribution.

    ``"fixed"`` always returns ``round(mean)`` (today's behavior — every paper
    cites the same number of prior papers). ``"poisson"`` draws from
    Poisson(mean) independently per paper, so individual papers vary while the
    *average* reference count across papers still equals ``mean``.
    """
    mode = validate_citation_reference_count_distribution(distribution)
    chooser = rng if rng is not None else random
    if mode == CITATION_REFERENCE_COUNT_POISSON:
        return _sample_poisson(mean, chooser)
    return max(0, round(float(mean)))


def validate_review_paradigm(paradigm: str) -> str:
    value = str(paradigm).strip().lower()
    if value not in VALID_REVIEW_PARADIGMS:
        allowed = ", ".join(sorted(VALID_REVIEW_PARADIGMS))
        raise ValueError(f"review_paradigm must be one of: {allowed}")
    return value


def validate_pricing_policy(policy: str) -> str:
    value = str(policy).strip().lower()
    if value not in VALID_PRICING_POLICIES:
        allowed = ", ".join(sorted(VALID_PRICING_POLICIES))
        raise ValueError(f"pricing_policy must be one of: {allowed}")
    return value


def validate_review_bump_duration(duration: str) -> str:
    value = str(duration).strip().lower()
    if value not in VALID_REVIEW_BUMP_DURATIONS:
        allowed = ", ".join(sorted(VALID_REVIEW_BUMP_DURATIONS))
        raise ValueError(f"review_bump_duration must be one of: {allowed}")
    return value


def normalize_review_kind(review_kind: str | None) -> str:
    value = str(review_kind or "").strip().lower()
    if value not in VALID_REVIEW_KINDS:
        allowed = ", ".join(sorted(VALID_REVIEW_KINDS))
        raise ValueError(f"review_kind must be one of: {allowed}")
    return value


def review_kind_from_effort(effort: float) -> str:
    """Classify a completed review by the continuous-mode threshold."""
    return (
        GOOD_FAITH_REVIEW
        if float(effort) >= GOOD_FAITH_REVIEW_THRESHOLD
        else BAD_FAITH_REVIEW
    )


def fixed_review_effort(review_kind: str) -> float:
    """Discrete-mode duration for a chosen review kind."""
    kind = normalize_review_kind(review_kind)
    if kind == GOOD_FAITH_REVIEW:
        return discrete_good_review_timesteps()
    return BAD_REVIEW_TIMESTEPS


def review_action_kind(review_kind: str) -> str:
    """Action-log label for a completed good/bad-faith review."""
    return f"{normalize_review_kind(review_kind)}_review"


def quality_multiplier(quality: float) -> float:
    """Clamp paper quality to a strictly positive multiplier."""
    return max(MIN_PAPER_QUALITY, float(quality))


def accrual_rate_from_quality(quality: float) -> float:
    """Expected citation AC per timestep for a fresh, unreviewed paper of ``quality``.

    AC is only earned through citations, so this is a forecast, not a source of
    AC: the paper's citation weight ``exp(quality_weight * quality)`` times the
    market-wide citation income per unit weight (``Paper.citation_income_per_weight``,
    refreshed by the environment each timestep).
    """
    return Paper.citation_income_per_weight * math.exp(
        Paper.citation_quality_weight * quality_multiplier(quality)
    )


def accrual_rate_from_effort(quality: float, writing_effort: float) -> float:
    """Continuous-mode base accrual rate after ``writing_effort`` timesteps.

    The rate approaches the quality-defined ceiling
    ``accrual_rate_from_quality(quality)`` asymptotically:
    ``ceiling * (1 - exp(-WRITING_SATURATION * effort))``. Zero effort yields a
    zero rate; more effort moves the paper closer to its ceiling with
    diminishing returns. There is no minimum threshold to finish a paper.
    """
    effort = max(0.0, float(writing_effort))
    ceiling = accrual_rate_from_quality(quality)
    return ceiling * (1.0 - math.exp(-WRITING_SATURATION * effort))


def fair_market_component(epsilon: float) -> float:
    """Incremental surplus fraction from a review bump: epsilon / (1 + epsilon)."""
    value = max(0.0, float(epsilon))
    return value / (1.0 + value)


def incremental_ac_fraction(current_ac: float, projected_final_ac: float) -> float:
    """Share of projected paper value still to be created: (F - A0) / F."""
    projected = max(0.0, float(projected_final_ac))
    current = max(0.0, float(current_ac))
    if projected <= 0.0:
        return 0.0
    return max(0.0, (projected - current) / projected)


def projected_paper_ac(
    current_ac: float,
    accrual_rate: float,
    expected_epsilon: float,
    horizon_timesteps: float = FORECAST_HORIZON_TIMESTEPS,
) -> float:
    """Forecast total paper AC at review settlement using agent forecast horizon."""
    current = max(0.0, float(current_ac))
    rate = max(0.0, float(accrual_rate))
    epsilon = max(0.0, float(expected_epsilon))
    horizon = max(0.0, float(horizon_timesteps))
    future = rate * (1.0 + epsilon) * horizon
    return current + future


def fair_market_offer_share(
    epsilon: float,
    current_ac: float,
    projected_final_ac: float,
    reviewer_surplus_share: float = REVIEWER_SURPLUS_SHARE,
) -> float:
    """Ownership share offer that splits incremental surplus between author and reviewer.

    Settlement still grants ``share * paper.current_ac`` at review finish; this
    only computes the posted offer so authors do not pay reviewers for AC they
    already built (A0).
    """
    return (
        fair_market_component(epsilon)
        * incremental_ac_fraction(current_ac, projected_final_ac)
        * max(0.0, float(reviewer_surplus_share))
    )


def fair_market_price_from_epsilons(
    epsilons,
    prior_epsilon: float = PRIOR_REVIEW_EPSILON,
) -> float:
    """Expected fair-market price from the empirical reviewer epsilon distribution.

    This implements the schematic's formula
    ``sum(epsilon / (1 + epsilon) * Probability(epsilon))`` by treating the
    current reviewer epsilon list as an equally weighted empirical distribution.
    """
    cleaned = []
    for epsilon in epsilons:
        try:
            value = float(epsilon)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value) or value < 0.0:
            continue
        cleaned.append(value)

    if not cleaned:
        cleaned = [max(0.0, float(prior_epsilon))]

    return sum(fair_market_component(value) for value in cleaned) / len(cleaned)


def _normalized_sigmoid(effort: float) -> float:
    """Map effort into [0, 1] between min effort and long-review saturation."""
    lower = MIN_REVIEW_EFFORT_THRESHOLD
    upper = max(lower + 1e-9, REVIEW_SIGMOID_SATURATION_EFFORT)
    steepness = max(1e-9, REVIEW_SIGMOID_STEEPNESS)
    midpoint = REVIEW_SIGMOID_MIDPOINT

    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))

    lo = sig(lower)
    hi = sig(upper)
    if abs(hi - lo) < 1e-12:
        return 1.0 if effort >= upper else 0.0
    scaled = (sig(effort) - lo) / (hi - lo)
    return min(1.0, max(0.0, scaled))


def review_accrual_bump(effort: float, quality: float = 1.0) -> float:
    """Accrual-rate bump fraction for a single review of ``effort`` and ``quality``.

    Effort below ``MIN_REVIEW_EFFORT_THRESHOLD`` yields 0. The default sigmoid
    mode approximates the review-length evidence discussed by the team: short
    reviews have limited effect, impact rises through the good-faith region, and
    long reviews saturate. The previous logarithmic curve remains available by
    setting ``SIM.review_effort_curve = "log"``; ``"jump"`` is an optional
    experiment for a high-effort threshold bonus discussed in sync.
    """
    if effort < MIN_REVIEW_EFFORT_THRESHOLD:
        return 0.0

    curve = str(REVIEW_EFFORT_CURVE).strip().lower()
    if curve == "log":
        base = BASE_REVIEW_ACCRUAL_BUMP * quality_multiplier(quality)
        extra = effort - MIN_REVIEW_EFFORT_THRESHOLD
        return base + FIRST_EXTRA_DAY_BUMP * math.log2(1 + extra)

    span = max(0.0, MAX_REVIEW_ACCRUAL_BUMP - MIN_REVIEW_ACCRUAL_BUMP)
    bump = MIN_REVIEW_ACCRUAL_BUMP + span * _normalized_sigmoid(float(effort))
    if curve == "jump":
        width = max(1e-9, float(REVIEW_JUMP_WIDTH))
        jump = 1.0 / (1.0 + math.exp(-(float(effort) - REVIEW_JUMP_THRESHOLD) / width))
        bump += max(0.0, REVIEW_JUMP_BUMP) * jump
    return bump * quality_multiplier(quality)


def review_epsilon_from_effort(effort: float, quality: float = 1.0) -> float:
    """Public helper for the review-induced proportional accrual improvement."""
    return review_accrual_bump(effort, quality)


def review_bump_factor_at_age(epsilon: float, age: float) -> float:
    """Remaining bump fraction ``epsilon * exp(-k * age)``, capped to zero."""
    if epsilon <= 0.0:
        return 0.0
    if validate_review_bump_duration(SIM.review_bump_duration) == REVIEW_BUMP_PERMANENT:
        return float(epsilon)
    age_value = max(0.0, float(age))
    cap = SIM.review_bump_decay_cap_timesteps
    if cap is not None and age_value >= float(cap):
        return 0.0
    decay_rate = max(0.0, float(SIM.review_bump_decay_rate))
    return float(epsilon) * math.exp(-decay_rate * age_value)


def decayed_accrual_rate(base_rate: float, epsilon: float, age: float) -> float:
    """Accrual rate from a frozen base rate and a decaying bump fraction."""
    bump = review_bump_factor_at_age(epsilon, age)
    return max(0.0, float(base_rate)) * (1.0 + bump)


def forecast_decayed_accrual_gain(
    base_rate: float,
    epsilon: float,
    horizon: float,
) -> float:
    """Integrated AC gain over ``horizon`` timesteps from a fresh review bump."""
    horizon_value = max(0.0, float(horizon))
    if horizon_value <= 0.0:
        return 0.0
    base = max(0.0, float(base_rate))
    if validate_review_bump_duration(SIM.review_bump_duration) == REVIEW_BUMP_PERMANENT:
        return base * (1.0 + max(0.0, float(epsilon))) * horizon_value

    epsilon_value = max(0.0, float(epsilon))
    decay_rate = max(0.0, float(SIM.review_bump_decay_rate))
    cap = SIM.review_bump_decay_cap_timesteps
    cap_value = None if cap is None else max(0.0, float(cap))

    def bump_integral(length: float) -> float:
        if length <= 0.0 or epsilon_value <= 0.0:
            return 0.0
        if decay_rate <= 1e-12:
            return epsilon_value * length
        return (epsilon_value / decay_rate) * (1.0 - math.exp(-decay_rate * length))

    total = base * horizon_value
    if epsilon_value <= 0.0:
        return total

    if cap_value is None:
        return total + base * bump_integral(horizon_value)

    if horizon_value <= cap_value:
        return total + base * bump_integral(horizon_value)
    return total + base * bump_integral(cap_value)


def _weighted_sample_without_replacement(
    items: list["Paper"],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list["Paper"]:
    """Draw ``k`` distinct items without replacement, weighted by ``weights``.

    Sequentially draws one item at a time and renormalizes the remaining
    weights on each draw. This is the standard substitute for
    ``numpy``'s weighted choice without replacement.
    """
    pool = list(zip(items, weights))
    selected: list["Paper"] = []
    for _ in range(k):
        total = sum(weight for _, weight in pool) # add up all weights
        if total <= 0.0:
            index = rng.randrange(len(pool))
            print("All 0 weights")
        else:
            target = rng.uniform(0.0, total) # choose a random number from 0 to that total
            running = 0.0
            index = len(pool) - 1
            for i, (_, weight) in enumerate(pool): #walk through the papers and add each weight until target
                running += weight
                if running >= target:
                    index = i
                    break
        selected.append(pool.pop(index)[0]) # add paper and remove from pool
    return selected


def generate_citations(
    new_paper: "Paper",
    prior_papers: list["Paper"],
    current_timestep: int,
    reference_count: float = CITATION_REFERENCE_COUNT,
    quality_weight: float = CITATION_QUALITY_WEIGHT,
    age_decay_ratio: float = CITATION_AGE_DECAY_RATIO,
    ac_per_new_paper: float = CITATION_AC_PER_NEW_PAPER,
    reference_count_distribution: str = CITATION_REFERENCE_COUNT_DISTRIBUTION,
    rng: random.Random | None = None,
) -> list["Paper"]:
    """Sample outgoing citations for a newly published paper.

    Citation targets are drawn without replacement from ``prior_papers``
    (self-citation and citing ``new_paper`` itself are excluded by construction
    — callers pass only strictly-earlier papers). Selection uses numerically
    stable softmax probabilities over
    ``quality_weight * quality + ln(age_decay_ratio) * (current_timestep - publish_timestep)``,
    i.e. weights proportional to ``exp(quality_weight * quality) * r**age`` with
    ``r = age_decay_ratio`` in ``(0, 1]``. Each timestep of age multiplies a
    paper's weight by ``r`` (geometric decay, no plateau), so higher-quality and
    more recent papers are favored. A completed peer review multiplies the
    reviewed paper's weight by ``1 + epsilon`` (see ``Paper.citation_log_weight``).

    ``reference_count`` is this paper's reference count under
    ``"fixed"`` distribution, or the *average* reference count under
    ``"poisson"`` (see ``sample_citation_reference_count``) — either way the
    sampled count is capped at ``min(count, len(prior_papers))``, which keeps
    citations per paper bounded regardless of corpus size.

    Each selected paper's ``citation_count`` is incremented and receives an
    equal split of ``ac_per_new_paper`` AC. Citations are the only source of
    AC: the split is added to ``current_ac`` (which shareholders own) and also
    logged in ``citation_accrued``. Marks
    ``new_paper.citations_generated`` so it is only ever processed once.
    """
    if not 0.0 < age_decay_ratio <= 1.0:
        raise ValueError("age_decay_ratio must be in (0, 1]")
    chooser = rng if rng is not None else random
    sampled_count = sample_citation_reference_count(
        reference_count, reference_count_distribution, chooser
    )
    k = min(max(0, sampled_count), len(prior_papers))
    new_paper.citations_generated = True
    if k <= 0:
        new_paper.references = []
        return [] 
    
    log_ratio = math.log(age_decay_ratio) # log space: r**age never underflows

    scores = [ #calculate every older paper's citation score
        paper.citation_log_weight(current_timestep, quality_weight, log_ratio)
        for paper in prior_papers
    ]
    max_score = max(scores)
    weights = [math.exp(value - max_score) for value in scores] # preserves same softmax probabilities

    selected = _weighted_sample_without_replacement( #randomly select k distinct older papers
        list(prior_papers), weights, k, chooser
    )

    ac_each = ac_per_new_paper / k 
    for cited in selected:
        cited.citation_count += 1
        cited.citation_accrued += ac_each # distribute ac
        cited.current_ac += ac_each
    new_paper.references = list(selected)
    return selected


class Paper:
    """A paper in the single-review marketplace.

    A paper is created with a known ``quality`` and is published to the market
    one timestep after its author finishes writing it. While listed, its author
    offers each potential reviewer a distinct share price (``price_table``). The
    first agent to claim it takes it permanently off the market; the paper can be
    reviewed exactly once.

    AC is earned only through citations. ``accrual_rate`` is a forecast of the
    citation AC this paper is expected to earn per timestep, refreshed by the
    environment from the citation network; it is used for pricing and agent
    decisions but never adds AC by itself.
    """

    # Citation market state shared by all papers; ``Environment`` refreshes it
    # each timestep. The default income per weight calibrates a quality-1.0
    # paper to ``DEFAULT_ACCRUAL_RATE`` before any environment has run.
    citation_quality_weight: float = CITATION_QUALITY_WEIGHT
    citation_log_decay_ratio: float = math.log(CITATION_AGE_DECAY_RATIO)
    citation_income_per_weight: float = DEFAULT_ACCRUAL_RATE * math.exp(
        -CITATION_QUALITY_WEIGHT
    )
    # Mean ``accrual_rate`` across the corpus; reputation is measured relative
    # to it so reputation bins keep their scale as the corpus grows.
    mean_accrual_rate: float = DEFAULT_ACCRUAL_RATE

    @classmethod
    def reset_citation_market(cls) -> None:
        """Restore the default citation market state (e.g. between tests)."""
        cls.citation_quality_weight = CITATION_QUALITY_WEIGHT
        cls.citation_log_decay_ratio = math.log(CITATION_AGE_DECAY_RATIO)
        cls.citation_income_per_weight = DEFAULT_ACCRUAL_RATE * math.exp(
            -CITATION_QUALITY_WEIGHT
        )
        cls.mean_accrual_rate = DEFAULT_ACCRUAL_RATE

    def __init__(
        self,
        author: Agent,
        quality: float = 1.0,
        accrual_rate: float | None = None,
        current_ac: float = 0.0,
        share_distribution: dict[Agent, float] | None = None,
        completion_progress: float = 1.0,
        market_listed: bool = False,
        max_reviewer_share: float | None = None,
        writing_effort: float | None = None,
        required_writing_effort: float | None = None,
        publish_timestep: int | None = None,
    ):
        if author is None:
            raise ValueError("author cannot be None")

        self.author = author
        self.paper_quality = quality_multiplier(quality)
        self.writing_effort = (
            None if writing_effort is None else max(0.0, float(writing_effort))
        )
        self.required_writing_effort = (
            None
            if required_writing_effort is None
            else max(0.0, float(required_writing_effort))
        )
        # Timestep this paper was published at; unset (e.g. seeded initial
        # papers) is treated as pre-existing corpus at t=0.
        self.publish_timestep = (
            0 if publish_timestep is None else int(publish_timestep)
        )
        # Continuous mode: writing effort sets how close the paper gets to its
        # quality-defined citation weight (1.0 = full weight).
        self.writing_completeness = (
            1.0
            if self.writing_effort is None
            else 1.0 - math.exp(-WRITING_SATURATION * self.writing_effort)
        )
        if accrual_rate is not None:
            rate = accrual_rate
        elif self.writing_effort is not None:
            rate = accrual_rate_from_effort(self.quality, self.writing_effort)
        else:
            # Discrete / back-compat: full quality-defined rate.
            rate = accrual_rate_from_quality(self.quality)
        self.accrual_rate = self._nonnegative_float(rate, "accrual_rate")
        self.current_ac = self._nonnegative_float(current_ac, "current_ac")
        self.share_distribution = (
            {author: 1.0} if share_distribution is None else dict(share_distribution)
        )
        self._validate_share_distribution()
        self.completion_progress = self._nonnegative_float(
            completion_progress,
            "completion_progress",
        )
        self.max_reviewer_share = self._validate_share_value(
            DEFAULT_MAX_REVIEWER_SHARE
            if max_reviewer_share is None
            else max_reviewer_share,
            "max_reviewer_share",
        )

        # Marketplace / single-review lifecycle.
        self.market_listed = bool(market_listed)
        self.scheduled_listing_timestep: int | None = None
        self.review_claimed = False
        self.reviewed = False
        self.reviewer: Agent | None = None
        self.review_in_progress_by: Agent | None = None
        self.agreed_review_share = 0.0
        self.price_table: dict[Agent, float] = {}
        self.review_records: list[dict[str, object]] = []
        self.listed_timestep: int | None = None
        self.claimed_timestep: int | None = None
        self.time_on_market_timesteps: int | None = None

        # Citation network (see ``generate_citations``). ``references`` lists
        # the papers this paper cites; ``citation_count``/``citation_accrued``
        # track incoming citations and the AC they granted (also added to
        # ``current_ac``, since citations are the only source of AC).
        self.references: list["Paper"] = []
        self.citation_count: int = 0
        self.citation_accrued: float = 0.0
        self.citations_generated: bool = False

        # Review bump on citation weight (decays when ``review_bump_duration == decay``).
        self.base_accrual_rate: float | None = None
        self.review_bump_epsilon: float = 0.0
        self.review_completed_timestep: int | None = None

    # ---- compatibility aliases ------------------------------------------
    @property
    def quality(self) -> float:
        """Legacy name for the single stored paper quality."""
        return self.paper_quality

    @quality.setter
    def quality(self, value: float) -> None:
        self.paper_quality = quality_multiplier(value)

    @property
    def ac_accrual_rate(self) -> float:
        return self.accrual_rate

    @ac_accrual_rate.setter
    def ac_accrual_rate(self, value: float):
        self.accrual_rate = self._nonnegative_float(value, "ac_accrual_rate")

    @property
    def completed_peer_reviews(self) -> int:
        """0 or 1 — a paper can be reviewed at most once."""
        return 1 if self.reviewed else 0

    @property
    def reviewed_by(self) -> set[Agent]:
        return {self.reviewer} if self.reviewer is not None else set()

    @property
    def review_available(self) -> bool:
        """True when the paper is listed and not yet claimed/reviewed."""
        return self.market_listed and not self.review_claimed and not self.reviewed

    # ---- pricing --------------------------------------------------------
    def update_price_table(
        self,
        reviewers,
        market_median_quality: float,
        mean_peer_review_epsilon: float,
        fair_market_price: float | None = None,
        pricing_policy: str = PRICING_POLICY,
        scarcity_multiplier: float = 1.0,
        forecast_horizon_timesteps: float = FORECAST_HORIZON_TIMESTEPS,
    ) -> None:
        """Recompute the per-reviewer share offer for this listed paper.

        Higher paper quality (relative to the market) lowers the offered share;
        a stronger reviewer epsilon history raises it. Fair-market offers use
        ``epsilon/(1+epsilon) * (F-A0)/F * reviewer_surplus_share`` so authors
        pay for incremental bump value, not AC already on the paper. The result
        is clamped to ``[min_offer_share, author share]`` (0--100% of the author's
        stake). ``fair_market_price`` is retained for logging only.
        """
        author_share = max(0.0, self.share_distribution.get(self.author, 0.0))
        ceiling = author_share
        if fair_market_price is None:
            fair_market_price = fair_market_price_from_epsilons([PRIOR_REVIEW_EPSILON])
        policy = validate_pricing_policy(pricing_policy)
        multiplier_fn = None
        if policy == PRICING_POLICY_ADAPTIVE:
            multiplier_fn = getattr(self.author, "review_offer_multiplier_for", None)
        try:
            market_scarcity = float(scarcity_multiplier)
        except (TypeError, ValueError):
            market_scarcity = 1.0
        if math.isnan(market_scarcity) or math.isinf(market_scarcity):
            market_scarcity = 1.0
        market_scarcity = max(0.0, market_scarcity)
        table: dict[Agent, float] = {}
        for agent in reviewers:
            if agent is self.author:
                continue
            author_multiplier = 1.0
            if multiplier_fn is not None:
                author_multiplier = multiplier_fn(agent)
            elif policy == PRICING_POLICY_ADAPTIVE:
                author_multiplier = getattr(
                    self.author,
                    "review_offer_multiplier",
                    1.0,
                )
            try:
                author_multiplier = float(author_multiplier)
            except (TypeError, ValueError):
                author_multiplier = 1.0
            if math.isnan(author_multiplier) or math.isinf(author_multiplier):
                author_multiplier = 1.0
            author_multiplier = max(0.0, author_multiplier)
            quality_factor = math.exp(
                -QUALITY_PRICE_SCALE * (self.quality - market_median_quality)
            )
            history = getattr(agent, "peer_review_epsilon_history", PRIOR_REVIEW_EPSILON)
            history_factor = max(
                0.0,
                1.0 + HISTORY_PRICE_SCALE * (history - mean_peer_review_epsilon),
            )
            if USE_FAIR_MARKET_PRICING:
                projected_final_ac = projected_paper_ac(
                    self.current_ac,
                    self.accrual_rate,
                    history,
                    forecast_horizon_timesteps,
                )
                base_offer = fair_market_offer_share(
                    history,
                    self.current_ac,
                    projected_final_ac,
                    SIM.reviewer_surplus_share,
                )
            else:
                base_offer = DEFAULT_REVIEW_SHARE
            offer = (
                base_offer
                * author_multiplier
                * market_scarcity
                * quality_factor
                * history_factor
            )
            offer = min(max(offer, MIN_OFFER_SHARE), ceiling)
            table[agent] = offer
        self.price_table = table

    def offered_share(self, agent: Agent) -> float:
        """The share currently offered to ``agent`` (0 if none/ineligible)."""
        return float(self.price_table.get(agent, 0.0))

    # ---- single-review lifecycle ----------------------------------------
    def can_start_review(self, agent: Agent) -> bool:
        if agent is self.author:
            return False
        if self.reviewed or self.review_claimed:
            return False
        return self.market_listed

    def list_on_market(self, timestep: int | None = None) -> None:
        """Put the paper on the review market and record when it became available."""
        self.market_listed = True
        if timestep is not None and self.listed_timestep is None:
            self.listed_timestep = int(timestep)

    def start_review(
        self,
        agent: Agent,
        current_timestep: int | None = None,
    ) -> bool:
        """Claim the paper for review: permanently delist it and lock the price."""
        if not self.can_start_review(agent):
            return False
        self.review_in_progress_by = agent
        self.review_claimed = True
        self.market_listed = False
        self.agreed_review_share = self.offered_share(agent)
        if current_timestep is not None:
            self.claimed_timestep = int(current_timestep)
            if self.listed_timestep is not None:
                self.time_on_market_timesteps = max(
                    0,
                    self.claimed_timestep - int(self.listed_timestep),
                )
            feedback = getattr(self.author, "record_review_claim_feedback", None)
            if feedback is not None:
                feedback(self.time_on_market_timesteps, claimer=agent)
        return True

    def finish_review(
        self,
        agent: Agent,
        effort: float,
        review_kind: str | None = None,
        current_timestep: int | None = None,
    ) -> float:
        """Finalize the in-progress review, granting the locked share.

        Returns the share actually transferred (0 below the effort threshold).
        Consumes the paper's single review either way.
        """
        if self.review_in_progress_by is not agent:
            return 0.0

        review_effort = self._nonnegative_float(effort, "review_effort")
        completed_review_kind = (
            normalize_review_kind(review_kind)
            if review_kind is not None
            else review_kind_from_effort(review_effort)
        )
        self.review_in_progress_by = None
        self.reviewed = True
        self.reviewer = agent

        share = 0.0
        review_quality = agent.sample_review_quality()
        epsilon = review_epsilon_from_effort(review_effort, self.quality) * review_quality
        quality_delta = 0.0
        if review_effort >= MIN_REVIEW_EFFORT_THRESHOLD:
            share = min(
                self.agreed_review_share,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
            if share > 0.0:
                self.share_distribution[self.author] = (
                    self.share_distribution.get(self.author, 0.0) - share
                )
                self.share_distribution[agent] = (
                    self.share_distribution.get(agent, 0.0) + share
                )
            self.review_bump_epsilon = epsilon
            self.review_completed_timestep = (
                int(current_timestep) if current_timestep is not None else None
            )
            if validate_review_bump_duration(SIM.review_bump_duration) == REVIEW_BUMP_DECAY:
                self.base_accrual_rate = self.accrual_rate
                self.refresh_accrual_rate(current_timestep)
            else:
                self.accrual_rate *= 1.0 + epsilon
            if agent.talent_sampling_enabled:
                quality_delta = self.paper_quality * epsilon
                self.paper_quality += quality_delta
        self.review_records.append(
            {
                "reviewer": agent,
                "share": share,
                "effort": review_effort,
                "epsilon": epsilon,
                "review_quality": review_quality,
                "review_quality_delta": quality_delta,
                "review_kind": completed_review_kind,
                "accrual_rate": self.accrual_rate,
                # Paper AC at the instant the review finished, before any
                # accrual at the bumped rate. This is the counterfactual
                # baseline A0 used by the reviewer-vs-author benefit metric.
                "current_ac_at_review": self.current_ac,
                "base_accrual_rate": self.base_accrual_rate,
                "review_completed_timestep": self.review_completed_timestep,
                "bump_duration": SIM.review_bump_duration,
            }
        )
        return share

    def add_review(
        self,
        agent: Agent,
        effort: float,
        share: float | None = None,
    ) -> float:
        """Claim and finish a review in one call (direct/testing convenience)."""
        if not self.start_review(agent):
            return 0.0
        if share is not None:
            self.agreed_review_share = self._validate_share_value(share, "share")
        elif self.agreed_review_share <= 0.0:
            self.agreed_review_share = min(
                DEFAULT_REVIEW_SHARE,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
        return self.finish_review(agent, effort)

    def estimate_review_share(self, agent: Agent) -> float:
        """Share the author would grant ``agent`` for completing a review now."""
        if self.review_in_progress_by is agent:
            return min(
                self.agreed_review_share,
                max(0.0, self.share_distribution.get(self.author, 0.0)),
            )
        if not self.can_start_review(agent):
            return 0.0
        return min(
            self.offered_share(agent),
            max(0.0, self.share_distribution.get(self.author, 0.0)),
        )

    def estimate_accrual_rate_after_review(self, effort: float) -> float:
        return self.accrual_rate * (1.0 + review_accrual_bump(effort, self.quality))

    # ---- citation weight ----------------------------------------------
    def review_bump_at(self, current_timestep: int | None = None) -> float:
        """Remaining review bump epsilon on this paper's citation weight."""
        if not self.reviewed or self.review_bump_epsilon <= 0.0:
            return 0.0
        if self.review_completed_timestep is None or current_timestep is None:
            age = 0
        else:
            age = max(0, int(current_timestep) - int(self.review_completed_timestep))
        return review_bump_factor_at_age(self.review_bump_epsilon, age)

    def citation_log_weight(
        self,
        current_timestep: int,
        quality_weight: float | None = None,
        log_decay_ratio: float | None = None,
    ) -> float:
        """Log citation weight: ``quality_weight * quality + ln(r) * age``,
        plus ``ln(writing_completeness)`` and ``ln(1 + review bump)``.

        A peer review therefore acts like a quality boost for citations: it
        multiplies the paper's chance of being cited by ``1 + epsilon``.
        """
        if quality_weight is None:
            quality_weight = Paper.citation_quality_weight
        if log_decay_ratio is None:
            log_decay_ratio = Paper.citation_log_decay_ratio
        age = max(0, int(current_timestep) - self.publish_timestep)
        value = quality_weight * self.quality + log_decay_ratio * age
        value += math.log(max(self.writing_completeness, 1e-12))
        return value + math.log1p(self.review_bump_at(current_timestep))

    # ---- shares ---------------------------------------------------------
    def set_share(self, agent: Agent, share: float):
        if agent is None:
            raise ValueError("agent cannot be None")

        share_value = self._validate_share_value(share, "share")
        other_total = sum(
            current_share
            for contributor, current_share in self.share_distribution.items()
            if contributor is not agent
        )
        if other_total + share_value > 1.0 + 1e-12:
            raise ValueError("total paper shares cannot exceed 1.0")

        self.share_distribution[agent] = share_value

    # ---- validation helpers --------------------------------------------
    def _validate_share_distribution(self):
        total = 0.0
        for contributor, share in self.share_distribution.items():
            if contributor is None:
                raise ValueError("share_distribution cannot contain None contributors")
            total += self._validate_share_value(share, "share_distribution share")
        if total > 1.0 + 1e-12:
            raise ValueError("initial share_distribution cannot exceed 1.0 total")

    @staticmethod
    def _nonnegative_float(value: float, name: str) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if math.isnan(number) or math.isinf(number) or number < 0.0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        return number

    @staticmethod
    def _validate_share_value(value: float, name: str) -> float:
        number = Paper._nonnegative_float(value, name)
        if number > 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")
        return number
