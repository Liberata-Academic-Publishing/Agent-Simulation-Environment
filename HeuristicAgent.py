from __future__ import annotations

from Agent import (
    Agent,
    CONTINUOUS_CLAIM,
    CONTINUOUS_RESEARCH,
    CONTINUOUS_RESEARCH_FINISH,
    CONTINUOUS_REVIEW,
)
from config import SIM
from Paper import (
    BAD_FAITH_REVIEW,
    BAD_REVIEW_TIMESTEPS,
    GOOD_FAITH_REVIEW,
    GOOD_REVIEW_TIMESTEPS,
    MIN_PAPER_QUALITY,
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_BUMP_DECAY,
    Paper,
    REVIEW_PARADIGM_DISCRETE,
    accrual_rate_from_effort,
    accrual_rate_from_quality,
    forecast_decayed_accrual_gain,
    review_accrual_bump,
    validate_review_bump_duration,
)

EXPECTED_WRITE_PROGRESS = SIM.expected_write_progress
MAX_FORECAST_EFFORT = SIM.max_forecast_effort


class HeuristicAgent(Agent):
    """Agent that claims and finishes reviews to maximize forecast academic capital.

    Each timestep it picks the action with the best projected return: in the
    marketplace phase it claims the most valuable listed paper if that beats its
    work-phase alternative, and in the work phase it either keeps investing in an
    active review (while the marginal accrual bump outweighs a timestep of
    writing) or finishes and writes.
    """

    def __init__(
        self,
        intrinsic_talent: float,
        academic_capital: float = 0.0,
        paper_progress: float = 0.0,
        review_progress: float = 0.0,
        forecast_horizon_timesteps: int = 30,
        name: str | None = None,
    ):
        super().__init__(
            intrinsic_talent=intrinsic_talent,
            academic_capital=academic_capital,
            paper_progress=paper_progress,
            review_progress=review_progress,
            name=name,
        )
        self.forecast_horizon_timesteps = forecast_horizon_timesteps

    # ---- marketplace phase ----------------------------------------------
    def choose_marketplace_action(self) -> Paper | None:
        assigned = self._assigned_claim_paper()
        if assigned is not None:
            return assigned

        listed = [
            paper
            for paper in Agent.all_papers
            if self._can_review(paper) and paper.offered_share(self) > 0.0
        ]
        if not listed:
            return None

        best = max(listed, key=self._score_claim)
        claim_value = self._expected_claim_value(best)
        if claim_value <= 0.0:
            return None

        if self.active_review_paper is None:
            return best if claim_value > self._write_value() else None

        # Grabbing finalizes the active review at its current effort, then starts
        # the new one. Compare against the best work-phase alternative.
        current = self.active_review_paper
        finish_now = self._review_value(
            current, self.active_review_effort, self.forecast_horizon_timesteps
        )
        not_grab = max(
            finish_now + self._write_value(),
            self._score_continue_review(current),
        )
        return best if finish_now + claim_value > not_grab else None

    # ---- work phase ------------------------------------------------------
    def choose_work_action(self) -> tuple[str, Paper | None]:
        paper = self.active_review_paper
        if paper is None:
            return "write_paper", None

        if self._score_continue_review(paper) > self._score_finish_and_write(paper):
            return "peer_review", paper
        return "finish_review_write_paper", None

    # ---- continuous merged phase ----------------------------------------
    def choose_continuous_action(self) -> tuple[str, Paper | None]:
        horizon = self.forecast_horizon_timesteps
        research_kind, research_total, build_len = self._best_research_plan(horizon)
        # Per-timestep value of researching: a paper only earns once finished, so
        # amortize its forecast value over the timesteps spent building it. This
        # puts research on equal footing with a (roughly one-timestep) review.
        research_step_value = research_total / max(1.0, build_len)

        assigned = self._assigned_claim_paper()
        if assigned is not None and self.active_review_paper is None:
            claim_value = self._expected_claim_value(assigned)
            if claim_value > research_step_value:
                return (CONTINUOUS_CLAIM, assigned)

        listed = [
            paper
            for paper in Agent.all_papers
            if self._can_review(paper) and paper.offered_share(self) > 0.0
        ]
        best = max(listed, key=self._score_claim) if listed else None
        claim_value = self._expected_claim_value(best) if best is not None else 0.0

        if self.active_review_paper is None:
            if best is not None and claim_value > research_step_value:
                return (CONTINUOUS_CLAIM, best)
            return (research_kind, None)

        # Holding a review: ending it now banks its forecast value; compare
        # continuing the review against ending it to claim or to research.
        current = self.active_review_paper
        finish_current = self._review_value(
            current, self.active_review_effort, horizon
        )
        review_continue = self._score_continue_review(current)
        claim_switch = (
            finish_current + claim_value if best is not None else float("-inf")
        )
        research_after = finish_current + research_step_value

        best_alt = max(review_continue, claim_switch, research_after)
        if review_continue >= best_alt:
            return (CONTINUOUS_REVIEW, None)
        if best is not None and claim_switch >= research_after:
            return (CONTINUOUS_CLAIM, best)
        return (CONTINUOUS_RESEARCH, None)

    def _best_research_plan(self, horizon: float) -> tuple[str, float, float]:
        """Best own-research move: finish after this step vs keep researching.

        Models the asymptotic-accrual tradeoff: more writing effort raises the
        paper's permanent accrual rate (toward its quality ceiling) but delays
        when it starts earning. Returns ``(action_kind, total_value,
        build_length)`` where ``total_value`` is the AC the paper is forecast to
        earn over the horizon (the author owns ~all of it) and ``build_length``
        is the number of writing timesteps the plan implies (for amortization).

        In threshold publishing mode the agent only researches until the fixed
        effort target is reached, so there is no finish-vs-continue choice.
        """
        quality = (
            self.next_paper_quality
            if self.next_paper_quality is not None
            else max(MIN_PAPER_QUALITY, self.intrinsic_talent)
        )
        effort = self.paper_progress
        span = max(1.0, float(horizon))

        if self.continuous_publish_by_threshold():
            threshold = self.paper_completion_threshold()
            build_len = max(1.0, threshold - effort)
            earning = max(0.0, span - build_len)
            value = accrual_rate_from_effort(quality, threshold) * earning
            return (CONTINUOUS_RESEARCH, value, build_len)

        # Invest this timestep, then finish (paper earns over the remaining span).
        finish_value = accrual_rate_from_effort(quality, effort + 1.0) * max(
            0.0, span - 1.0
        )

        # Invest this timestep and keep researching k more, then finish. The
        # build length counts from the paper's current progress so amortization
        # reflects the whole manuscript, not just the steps still to come.
        best_continue = float("-inf")
        best_build = 1.0
        kmax = int(min(MAX_FORECAST_EFFORT, span - 1.0))
        for k in range(1, kmax + 1):
            earning = span - 1.0 - k
            if earning < 0.0:
                break
            value = accrual_rate_from_effort(quality, effort + 1.0 + k) * earning
            if value > best_continue:
                best_continue = value
                best_build = effort + 1.0 + k

        if best_continue > finish_value:
            return (CONTINUOUS_RESEARCH, best_continue, best_build)
        return (CONTINUOUS_RESEARCH_FINISH, finish_value, effort + 1.0)

    def choose_review_kind(self, paper: Paper) -> str:
        """Discrete-mode choice between fixed bad- and good-faith review work."""
        bad = self._score_fixed_review(paper, BAD_REVIEW_TIMESTEPS)
        good = self._score_fixed_review(paper, GOOD_REVIEW_TIMESTEPS)
        return GOOD_FAITH_REVIEW if good > bad else BAD_FAITH_REVIEW

    # ---- market participation (for clearing + forecasts) -----------------
    def would_claim_paper(self, paper: Paper) -> bool:
        """True when this agent would try to claim ``paper`` if it could win."""
        if self.active_review_paper is not None:
            return False
        if not self._can_review(paper) or paper.offered_share(self) <= 0.0:
            return False
        claim_value = self._expected_claim_value(paper)
        return claim_value > 0.0 and claim_value > self._write_value()

    def claim_preference_score(self, paper: Paper) -> float:
        """Ranking score for merit-based marketplace clearing."""
        return self._expected_claim_value(paper)

    def _assigned_claim_paper(self) -> Paper | None:
        paper = getattr(self, "market_claim_assignment", None)
        if paper is None:
            return None
        if not paper.review_available or not self._can_review(paper):
            return None
        if paper.offered_share(self) <= 0.0:
            return None
        return paper

    def _listed_paper_count(self) -> int:
        return sum(
            1 for paper in Agent.all_papers if getattr(paper, "review_available", False)
        )

    def _eligible_reviewer_count(self) -> int:
        count = sum(
            1 for agent in Agent.all_agents if agent.active_review_paper is None
        )
        return max(1, count)

    def _claim_win_probability(self) -> float:
        listed = self._listed_paper_count()
        if listed <= 0:
            return 0.0
        return min(1.0, listed / self._eligible_reviewer_count())

    def _expected_claim_value(self, paper: Paper | None) -> float:
        if paper is None:
            return 0.0
        raw = self._score_claim(paper)
        if not self.use_competition_adjusted_forecast:
            return raw
        return self._claim_win_probability() * raw

    # ---- scoring ---------------------------------------------------------
    def _review_value(self, paper: Paper, effort: float, horizon: float) -> float:
        """Forecast capital from owning the agreed review share of ``paper``."""
        share = self._prospective_share(paper)
        if share <= 0.0:
            return 0.0
        epsilon = review_accrual_bump(effort, paper.quality)
        horizon_value = max(0.0, horizon)
        if validate_review_bump_duration(SIM.review_bump_duration) == REVIEW_BUMP_DECAY:
            future_gain = forecast_decayed_accrual_gain(
                paper.accrual_rate,
                epsilon,
                horizon_value,
            )
        else:
            future_gain = paper.accrual_rate * (1.0 + epsilon) * horizon_value
        return share * (paper.current_ac + future_gain)

    def _score_claim(self, paper: Paper) -> float:
        """Value of claiming ``paper`` and completing a minimum-effort review."""
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return max(
                self._score_fixed_review(paper, BAD_REVIEW_TIMESTEPS),
                self._score_fixed_review(paper, GOOD_REVIEW_TIMESTEPS),
            )
        return self._review_value(
            paper, MIN_REVIEW_EFFORT_THRESHOLD, self.forecast_horizon_timesteps
        )

    def _score_fixed_review(self, paper: Paper, effort: float) -> float:
        horizon = self.forecast_horizon_timesteps - max(0.0, effort - 1.0)
        return self._review_value(paper, effort, horizon)

    def _score_finish_and_write(self, paper: Paper) -> float:
        """Finish the active review now and spend this timestep writing."""
        return (
            self._review_value(
                paper, self.active_review_effort, self.forecast_horizon_timesteps
            )
            + self._write_value()
        )

    def _score_continue_review(self, paper: Paper) -> float:
        """Invest one more timestep, then finish at the higher effort (no writing)."""
        if self.active_review_effort >= MAX_FORECAST_EFFORT:
            return 0.0
        next_effort = self.active_review_effort + self.review_effort_delta()
        return self._review_value(
            paper, next_effort, self.forecast_horizon_timesteps - 1
        )

    def _write_value(self) -> float:
        """Forecast capital from one timestep of progress on the agent's next paper."""
        quality = (
            self.next_paper_quality
            if self.next_paper_quality is not None
            else max(MIN_PAPER_QUALITY, self.intrinsic_talent)
        )
        paper_rate = accrual_rate_from_quality(quality)
        full_value = paper_rate * self.forecast_horizon_timesteps
        timesteps_to_finish = max(
            1.0, self.paper_completion_threshold() / max(EXPECTED_WRITE_PROGRESS, 1e-9)
        )
        return full_value / timesteps_to_finish

    # ---- shares ----------------------------------------------------------
    def _prospective_share(self, paper: Paper) -> float:
        helper = getattr(paper, "estimate_review_share", None)
        if helper is not None:
            return max(0.0, float(helper(self)))
        return 0.0

    def _current_share(self, paper: Paper) -> float:
        return paper.share_distribution.get(self, 0.0)

    def _can_review(self, paper: Paper) -> bool:
        helper = getattr(paper, "can_start_review", None)
        if helper is not None:
            return bool(helper(self))
        return paper.author is not self
