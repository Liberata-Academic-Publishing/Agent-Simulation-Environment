from __future__ import annotations

import inspect
import math
import random
from collections.abc import Callable, Sequence
from statistics import median
from typing import TYPE_CHECKING

from Agent import Agent, validate_continuous_publishing
from config import SIM
from Paper import (
    REVIEW_PARADIGM_DISCRETE,
    PRICING_POLICY_ADAPTIVE,
    Paper,
    fair_market_price_from_epsilons,
    generate_citations,
    validate_citation_reference_count_distribution,
    validate_pricing_policy,
    validate_review_paradigm,
)

if TYPE_CHECKING:
    from History import History

AgentFactory = Callable[..., Agent]


class Environment:
    """Single-review marketplace environment.

    Each timestep reshuffles the agent order, then advances the world by one
    timestep. In the ``continuous`` paradigm this is one merged decision per
    agent (claim+review, continue a review, research, or finish research). In the
    ``discrete`` paradigm it is the two-phase flow (a marketplace claim phase
    then a work phase). Papers list one timestep after they are written and leave
    the market permanently the moment they are claimed.
    """

    def __init__(
        self,
        agents: Sequence[Agent] | None = None,
        papers: Sequence[Paper] | None = None,
        num_agents: int | None = None,
        agent_cls: type[Agent] | AgentFactory | None = None,
        forecast_horizon_timesteps: int = 30,
        review_paradigm: str = "continuous",
        continuous_publishing: str = SIM.continuous_publishing,
        continuous_paper_timesteps: float = SIM.continuous_paper_timesteps,
        paper_effort_mode: str = SIM.paper_effort_mode,
        paper_effort_min: float = SIM.paper_effort_min,
        paper_effort_max: float = SIM.paper_effort_max,
        discrete_paper_timesteps: float = SIM.discrete_paper_timesteps,
        randomize_initial_paper_progress: bool = SIM.randomize_initial_paper_progress,
        pricing_policy: str = SIM.pricing_policy,
        use_competition_adjusted_forecast: bool = SIM.use_competition_adjusted_forecast,
        use_scarcity_pricing: bool = SIM.use_scarcity_pricing,
        reviewer_pressure_exponent: float = SIM.reviewer_pressure_exponent,
        use_merit_market_clearing: bool = SIM.use_merit_market_clearing,
        target_market_wait_timesteps: float = SIM.target_market_wait_timesteps,
        adaptive_pricing_learning_rate: float = SIM.adaptive_pricing_learning_rate,
        min_author_price_multiplier: float = SIM.min_author_price_multiplier,
        max_author_price_multiplier: float = SIM.max_author_price_multiplier,
        adaptive_pricing_mode: str = SIM.adaptive_pricing_mode,
        reputation_bin_edges: tuple[float, ...] = SIM.reputation_bin_edges,
        reputation_bin_names: tuple[str, ...] = SIM.reputation_bin_names,
        adaptive_raise_bins: tuple[str, ...] = SIM.adaptive_raise_bins,
        adaptive_lower_bins: tuple[str, ...] = SIM.adaptive_lower_bins,
        adaptive_slow_raise_bins: tuple[str, ...] = SIM.adaptive_slow_raise_bins,
        fast_claim_max_wait: float | None = SIM.fast_claim_max_wait,
        citations_enabled: bool = SIM.citations_enabled,
        citation_reference_count: int = SIM.citation_reference_count,
        citation_reference_count_distribution: str = SIM.citation_reference_count_distribution,
        citation_quality_weight: float = SIM.citation_quality_weight,
        citation_age_decay_ratio: float = SIM.citation_age_decay_ratio,
        citation_ac_per_new_paper: float = SIM.citation_ac_per_new_paper,
        citation_income_smoothing: float = SIM.citation_income_smoothing,
        history: "History | None" = None,
    ):
        if agents is not None and num_agents is not None:
            raise ValueError("Pass either agents or num_agents, not both.")

        self.forecast_horizon_timesteps = forecast_horizon_timesteps
        self.review_paradigm = validate_review_paradigm(review_paradigm)
        self.continuous_publishing = validate_continuous_publishing(
            continuous_publishing
        )
        self.continuous_paper_timesteps = float(continuous_paper_timesteps)
        self.paper_effort_mode = paper_effort_mode
        self.paper_effort_min = float(paper_effort_min)
        self.paper_effort_max = float(paper_effort_max)
        self.discrete_paper_timesteps = float(discrete_paper_timesteps)
        self.randomize_initial_paper_progress = bool(randomize_initial_paper_progress)
        self.pricing_policy = validate_pricing_policy(pricing_policy)
        self.use_competition_adjusted_forecast = bool(use_competition_adjusted_forecast)
        self.use_scarcity_pricing = bool(use_scarcity_pricing)
        if self.pricing_policy == PRICING_POLICY_ADAPTIVE:
            self.use_scarcity_pricing = False
        self.reviewer_pressure_exponent = float(reviewer_pressure_exponent)
        self.use_merit_market_clearing = bool(use_merit_market_clearing)
        self.target_market_wait_timesteps = float(target_market_wait_timesteps)
        self.adaptive_pricing_learning_rate = float(adaptive_pricing_learning_rate)
        self.min_author_price_multiplier = float(min_author_price_multiplier)
        self.max_author_price_multiplier = float(max_author_price_multiplier)
        self.adaptive_pricing_mode = adaptive_pricing_mode
        self.reputation_bin_edges = reputation_bin_edges
        self.reputation_bin_names = reputation_bin_names
        self.adaptive_raise_bins = adaptive_raise_bins
        self.adaptive_lower_bins = adaptive_lower_bins
        self.adaptive_slow_raise_bins = adaptive_slow_raise_bins
        self.fast_claim_max_wait = fast_claim_max_wait
        self.citations_enabled = bool(citations_enabled)
        self.citation_reference_count = float(citation_reference_count)
        self.citation_reference_count_distribution = (
            validate_citation_reference_count_distribution(
                citation_reference_count_distribution
            )
        )
        self.citation_quality_weight = float(citation_quality_weight)
        self.citation_age_decay_ratio = float(citation_age_decay_ratio)
        self.citation_ac_per_new_paper = float(citation_ac_per_new_paper)
        self.citation_income_smoothing = float(citation_income_smoothing)
        self.history = history
        self.timestep = 0
        self.fair_market_price = fair_market_price_from_epsilons(
            [SIM.prior_review_epsilon]
        )
        self.reviewer_pressure = 0.0
        self.scarcity_multiplier = 1.0
        self.market_assignments = 0

        if agents is None:
            count = 0 if num_agents is None else num_agents
            self.agents = [
                self._create_agent(agent_cls, forecast_horizon_timesteps)
                for _ in range(count)
            ]
        else:
            self.agents = list(agents)

        self.papers = list(Agent.all_papers if papers is None else papers)
        Agent.all_papers = self.papers
        Agent.all_agents = self.agents
        self._configure_agents()
        # Smoothed citation AC handed out per timestep. Prior: every agent
        # publishes one paper per expected paper-effort target.
        self.citation_income_per_timestep = (
            len(self.agents)
            * self.citation_ac_per_new_paper
            / max(1.0, self._expected_paper_effort())
        )
        self._update_citation_forecasts()

    # ---- main loop -------------------------------------------------------
    def run_timestep(self):
        """Advance the simulation by one full timestep."""
        self.timestep += 1
        self._sync_papers()
        self._update_citation_forecasts()
        self._list_scheduled_papers()
        self._update_market_prices()
        self._clear_review_market()

        order = list(self.agents)
        random.shuffle(order)
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            claimers = self._marketplace_phase(order)
            self._work_phase(order, claimers)
        else:
            self._continuous_phase(order)

        self._sync_papers()
        distributed = self._process_citations()
        self.citation_income_per_timestep += self.citation_income_smoothing * (
            distributed - self.citation_income_per_timestep
        )
        self._schedule_new_papers()

        # AC comes only from citations (credited in ``_process_citations``).
        self.update_agent_capital()

        if self.history is not None:
            self.history.record_step(self)

    def run(self, timesteps: int):
        if timesteps < 0:
            raise ValueError("timesteps must be non-negative")
        for _ in range(timesteps):
            self.run_timestep()
        return self

    # ---- phases ----------------------------------------------------------
    def _continuous_phase(self, order: list[Agent]) -> None:
        """Single merged phase: each agent makes one decision and spends one
        timestep of effort on it (claim+review, continue review, research, or
        finish research). Agents act in the freshly shuffled order so claims
        race for whatever papers are still listed."""
        for agent in order:
            agent.current_timestep = self.timestep
            act = getattr(agent, "act_continuous", None)
            if act is None:
                continue
            records = act()
            if self.history is None or not records:
                continue
            for record in records:
                if record is not None:
                    self.history.record_action(self, agent, record)

    def _marketplace_phase(self, order: list[Agent]) -> set[Agent]:
        """Phase 1 (instantaneous): each agent may select one listed paper.

        Selecting is pure choice — no effort is applied here. Returns the set of
        agents who claimed a paper so the work phase can apply their first unit
        of review effort instead of re-deciding.
        """
        claimers: set[Agent] = set()
        if self.use_merit_market_clearing:
            for agent in self.agents:
                agent.current_timestep = self.timestep
                if agent.active_review_paper is not None:
                    continue
                paper = getattr(agent, "market_claim_assignment", None)
                if paper is None or not paper.can_start_review(agent):
                    continue
                review_kind = agent.choose_review_kind(paper)
                record = agent.claim_review(paper, review_kind=review_kind)
                claimers.add(agent)
                if self.history is not None and record is not None:
                    self.history.record_action(self, agent, record)
            return claimers

        for agent in order:
            agent.current_timestep = self.timestep
            if (
                self.review_paradigm == REVIEW_PARADIGM_DISCRETE
                and agent.active_review_paper is not None
            ):
                continue
            choose = getattr(agent, "choose_marketplace_action", None)
            if choose is None:
                continue
            paper = choose()
            if paper is None or not paper.can_start_review(agent):
                continue
            review_kind = None
            if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
                review_kind = agent.choose_review_kind(paper)
            record = agent.claim_review(paper, review_kind=review_kind)
            claimers.add(agent)
            if self.history is not None and record is not None:
                self.history.record_action(self, agent, record)
        return claimers

    def _work_phase(self, order: list[Agent], claimers: set[Agent]) -> None:
        """Phase 2 (effort application): every agent spends one timestep.

        Agents who claimed in the marketplace apply the first unit of effort to
        the new review; everyone else writes or advances/finishes their active
        review.
        """
        for agent in order:
            agent.current_timestep = self.timestep
            if agent in claimers:
                record = agent.apply_initial_review_effort()
            else:
                record = agent.work_turn()
            if self.history is not None and record is not None:
                self.history.record_action(self, agent, record)

    # ---- marketplace bookkeeping ----------------------------------------
    def market(self):
        """Backwards-compatible alias for the price-table refresh."""
        self._update_market_prices()

    def _update_market_prices(self) -> None:
        listed = [p for p in self.papers if p.review_available]
        eligible = sum(
            1 for agent in self.agents if agent.active_review_paper is None
        )
        eligible = max(1, eligible)
        if not listed:
            self.reviewer_pressure = 0.0
            self.scarcity_multiplier = 1.0
            return

        pressure = eligible / max(1, len(listed))
        self.reviewer_pressure = pressure
        if self.use_scarcity_pricing:
            exponent = max(0.0, self.reviewer_pressure_exponent)
            self.scarcity_multiplier = pressure ** (-exponent)
        else:
            self.scarcity_multiplier = 1.0

        median_quality = median(p.quality for p in listed)
        epsilons = [
            getattr(a, "peer_review_epsilon_history", SIM.prior_review_epsilon)
            for a in self.agents
        ]
        mean_epsilon = (
            sum(epsilons) / len(epsilons)
            if epsilons
            else SIM.prior_review_epsilon
        )
        self.fair_market_price = fair_market_price_from_epsilons(epsilons)
        for paper in listed:
            paper.update_price_table(
                self.agents,
                median_quality,
                mean_epsilon,
                fair_market_price=self.fair_market_price,
                pricing_policy=self.pricing_policy,
                scarcity_multiplier=self.scarcity_multiplier,
                forecast_horizon_timesteps=self.forecast_horizon_timesteps,
            )

    def _clear_review_market(self) -> None:
        for agent in self.agents:
            agent.market_claim_assignment = None

        if not self.use_merit_market_clearing:
            self.market_assignments = 0
            return

        listed = [paper for paper in self.papers if paper.review_available]
        if not listed:
            self.market_assignments = 0
            return

        candidates: list[tuple[float, Paper, Agent]] = []
        for paper in listed:
            for agent in self.agents:
                would_claim = getattr(agent, "would_claim_paper", None)
                score_fn = getattr(agent, "claim_preference_score", None)
                if would_claim is None or score_fn is None:
                    continue
                if not would_claim(paper):
                    continue
                candidates.append((score_fn(paper), paper, agent))

        candidates.sort(key=lambda row: row[0], reverse=True)
        assigned_agents: set[Agent] = set()
        assigned_papers: set[Paper] = set()
        assignments = 0
        for _score, paper, agent in candidates:
            if paper in assigned_papers or agent in assigned_agents:
                continue
            agent.market_claim_assignment = paper
            assigned_agents.add(agent)
            assigned_papers.add(paper)
            assignments += 1
        self.market_assignments = assignments

    def _list_scheduled_papers(self) -> None:
        for paper in self.papers:
            scheduled = getattr(paper, "scheduled_listing_timestep", None)
            if (
                scheduled is not None
                and scheduled <= self.timestep
                and not paper.review_claimed
                and not paper.reviewed
            ):
                paper.list_on_market(self.timestep)
                paper.scheduled_listing_timestep = None

    def _process_citations(self) -> None:
        """Generate outgoing citations for every not-yet-processed paper.

        Eligible targets are papers published strictly before the citing
        paper (``publish_timestep`` ordering), so this is independent of
        ``self.papers`` iteration order and never allows self-citation or
        citing the paper currently being processed.

        Returns the total AC handed out to cited papers this timestep.

        **** potentially wasteful because we loop through all the papers each time step
        """
        if not self.citations_enabled:
            return 0.0
        distributed = 0.0
        for paper in list(self.papers):
            if paper.citations_generated:
                continue
            eligible = [
                other
                for other in self.papers
                if other.publish_timestep < paper.publish_timestep
            ]
            cited = generate_citations(
                paper,
                eligible,
                paper.publish_timestep,
                reference_count=self.citation_reference_count,
                quality_weight=self.citation_quality_weight,
                age_decay_ratio=self.citation_age_decay_ratio,
                ac_per_new_paper=self.citation_ac_per_new_paper,
                reference_count_distribution=self.citation_reference_count_distribution,
            )
            if cited:
                distributed += self.citation_ac_per_new_paper
        return distributed

    def _update_citation_forecasts(self) -> None:
        """Refresh every paper's ``accrual_rate`` forecast from the citation network.

        A paper's expected citation AC per timestep is the smoothed AC handed
        out per timestep times its share of total citation weight. Also stores
        income per unit weight on ``Paper`` so forecasts for not-yet-written
        papers (``accrual_rate_from_quality``) use the same scale.
        """
        log_decay = math.log(self.citation_age_decay_ratio)
        Paper.citation_quality_weight = self.citation_quality_weight
        Paper.citation_log_decay_ratio = log_decay
        if not self.papers:
            return
        log_weights = [
            paper.citation_log_weight(self.timestep, self.citation_quality_weight, log_decay)
            for paper in self.papers
        ]
        top = max(log_weights)
        log_total = top + math.log(sum(math.exp(value - top) for value in log_weights))
        income = max(0.0, self.citation_income_per_timestep)
        Paper.citation_income_per_weight = income * math.exp(-log_total)
        Paper.mean_accrual_rate = income / len(self.papers)
        for paper, value in zip(self.papers, log_weights):
            paper.accrual_rate = income * math.exp(value - log_total)

    def _expected_paper_effort(self) -> float:
        if self.paper_effort_mode != "fixed":
            return (self.paper_effort_min + self.paper_effort_max) / 2.0
        if self.review_paradigm == REVIEW_PARADIGM_DISCRETE:
            return self.discrete_paper_timesteps
        return self.continuous_paper_timesteps

    def _schedule_new_papers(self) -> None:
        for paper in self.papers:
            already_scheduled = getattr(paper, "scheduled_listing_timestep", None)
            if (
                already_scheduled is None
                and not paper.market_listed
                and not paper.review_claimed
                and not paper.reviewed
            ):
                paper.scheduled_listing_timestep = self.timestep + 1

    # ---- capital ---------------------------------------------------------
    def update_agent_capital(self):
        agent_set = set(self.agents)
        for agent in self.agents:
            agent.academic_capital = 0.0

        for paper in self.papers:
            for agent, share in paper.share_distribution.items():
                if agent in agent_set:
                    agent.academic_capital += share * paper.current_ac

    # ---- helpers ---------------------------------------------------------
    def _sync_papers(self):
        if Agent.all_papers is not self.papers:
            self.papers = Agent.all_papers

    def _configure_agents(self):
        for agent in self.agents:
            required = getattr(agent, "requires_review_paradigm", None)
            if required is not None and required != self.review_paradigm:
                raise ValueError(
                    f"{type(agent).__name__} requires review_paradigm={required!r}"
                )
            if hasattr(agent, "configure_review_paradigm"):
                agent.configure_review_paradigm(self.review_paradigm)
            if hasattr(agent, "configure_continuous_publishing"):
                agent.configure_continuous_publishing(
                    self.continuous_publishing,
                    self.continuous_paper_timesteps,
                )
            if hasattr(agent, "configure_paper_effort"):
                agent.configure_paper_effort(
                    self.paper_effort_mode,
                    self.paper_effort_min,
                    self.paper_effort_max,
                    discrete_timesteps=self.discrete_paper_timesteps,
                )
            if hasattr(agent, "configure_pricing_policy"):
                agent.configure_pricing_policy(
                    self.pricing_policy,
                    adaptive_pricing_mode=self.adaptive_pricing_mode,
                    target_market_wait_timesteps=self.target_market_wait_timesteps,
                    learning_rate=self.adaptive_pricing_learning_rate,
                    min_multiplier=self.min_author_price_multiplier,
                    max_multiplier=self.max_author_price_multiplier,
                    reputation_bin_edges=self.reputation_bin_edges,
                    reputation_bin_names=self.reputation_bin_names,
                    adaptive_raise_bins=self.adaptive_raise_bins,
                    adaptive_lower_bins=self.adaptive_lower_bins,
                    adaptive_slow_raise_bins=self.adaptive_slow_raise_bins,
                    fast_claim_max_wait=self.fast_claim_max_wait,
                )
            if hasattr(agent, "configure_market_economics"):
                agent.configure_market_economics(
                    use_competition_adjusted_forecast=self.use_competition_adjusted_forecast,
                    use_scarcity_pricing=self.use_scarcity_pricing,
                    use_merit_market_clearing=self.use_merit_market_clearing,
                )
            if hasattr(agent, "forecast_horizon_timesteps"):
                agent.forecast_horizon_timesteps = self.forecast_horizon_timesteps
            if self.randomize_initial_paper_progress and agent.paper_progress == 0.0:
                agent.paper_progress = random.random() * agent.paper_completion_threshold()

    def _create_agent(
        self,
        agent_cls: type[Agent] | AgentFactory | None,
        forecast_horizon_timesteps: int,
    ) -> Agent:
        if agent_cls is None:
            from HeuristicAgent import HeuristicAgent

            agent_cls = HeuristicAgent

        kwargs = {"intrinsic_talent": 1.0}
        signature = inspect.signature(agent_cls)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if accepts_kwargs or "forecast_horizon_timesteps" in signature.parameters:
            kwargs["forecast_horizon_timesteps"] = forecast_horizon_timesteps

        return agent_cls(**kwargs)
