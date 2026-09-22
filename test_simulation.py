from __future__ import annotations

import csv
import os
import statistics
import tempfile
import unittest
from unittest import mock

from Agent import Agent, PAPER_THRESHOLD
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History, gini
from Paper import (
    ADAPTIVE_PRICING_BINNED,
    ADAPTIVE_PRICING_GLOBAL,
    BAD_FAITH_REVIEW,
    BAD_REVIEW_TIMESTEPS,
    GOOD_FAITH_REVIEW,
    GOOD_REVIEW_TIMESTEPS,
    MIN_REVIEW_EFFORT_THRESHOLD,
    REVIEW_BUMP_DECAY,
    REVIEW_EFFORT_PER_TIMESTEP,
    Paper,
    accrual_rate_from_effort,
    accrual_rate_from_quality,
    decayed_accrual_rate,
    fair_market_component,
    fair_market_offer_share,
    fair_market_price_from_epsilons,
    projected_paper_ac,
    reputation_bin_name,
    review_accrual_bump,
    review_bump_factor_at_age,
    validate_reputation_bin_config,
)
import config
from dataclasses import replace
from RandomAgent import ProbabilisticDiscreteAgent, RandomAgent

try:
    import matplotlib  # noqa: F401

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


class ScriptAgent(Agent):
    """Agent driven by scripted marketplace/work decisions for deterministic tests."""

    def __init__(self, name: str = "agent", marketplace=None, work=None, continuous=None):
        super().__init__(intrinsic_talent=1.0)
        self.name = name
        self.marketplace = list(marketplace or [])
        self.work = list(work or [])
        # Scripted (kind, paper) tuples for the merged continuous phase.
        self.continuous = list(continuous or [])

    def choose_marketplace_action(self):
        if self.marketplace:
            return self.marketplace.pop(0)
        return None

    def choose_work_action(self):
        if self.work:
            return self.work.pop(0)
        return "write_paper", None

    def choose_continuous_action(self):
        if self.continuous:
            return self.continuous.pop(0)
        return super().choose_continuous_action()

    def writing_effort_delta(self):
        return 0.5


class ReviewKindScriptAgent(ScriptAgent):
    def __init__(self, review_kind: str, **kwargs):
        super().__init__(**kwargs)
        self.review_kind = review_kind

    def choose_review_kind(self, paper):
        return self.review_kind


class RecordingAgent(ScriptAgent):
    def __init__(self, log: list[str], name: str):
        super().__init__(name=name)
        self.log = log

    def work_turn(self):
        self.log.append(self.name)
        return super().work_turn()

    def act_continuous(self):
        self.log.append(self.name)
        return super().act_continuous()


def _listed_paper(author, **kwargs) -> Paper:
    paper = Paper(author=author, market_listed=True, **kwargs)
    return paper


class MarketplaceLifecycleTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_published_paper_lists_one_timestep_later(self):
        # Continuous mode: the author finishes a paper by choice, not a threshold.
        author = ScriptAgent("author", continuous=[("research_finish", None)])
        env = Environment(agents=[author], continuous_publishing="choice")

        env.run_timestep()  # timestep 1: author researches and finishes a paper
        self.assertEqual(len(Agent.all_papers), 1)
        paper = Agent.all_papers[0]
        self.assertFalse(paper.market_listed)
        self.assertFalse(paper.review_available)

        env.run_timestep()  # timestep 2: scheduled listing happens at the start
        self.assertTrue(paper.market_listed)
        self.assertTrue(paper.review_available)

    def test_claimed_paper_leaves_market_for_everyone(self):
        author = ScriptAgent("author")
        first = ScriptAgent("first")
        second = ScriptAgent("second")
        paper = _listed_paper(author)

        self.assertTrue(paper.start_review(first))
        self.assertTrue(paper.review_claimed)
        self.assertFalse(paper.market_listed)
        self.assertFalse(paper.review_available)
        self.assertFalse(paper.can_start_review(second))

    def test_paper_can_only_be_reviewed_once(self):
        author = ScriptAgent("author")
        first = ScriptAgent("first")
        second = ScriptAgent("second")
        paper = _listed_paper(author, quality=1.0)

        paper.start_review(first)
        paper.finish_review(first, MIN_REVIEW_EFFORT_THRESHOLD)

        self.assertTrue(paper.reviewed)
        self.assertFalse(paper.review_available)
        self.assertFalse(paper.can_start_review(second))

    def test_min_effort_review_earns_reward(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0, current_ac=10.0)
        paper.update_price_table([author, reviewer], 1.0, 0.0)

        share = paper.add_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD)

        self.assertGreater(share, 0.0)
        self.assertGreater(paper.accrual_rate, 1.0)
        self.assertEqual(paper.completed_peer_reviews, 1)
        self.assertIn(reviewer, paper.share_distribution)

    def test_subthreshold_review_consumes_opportunity_without_reward(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)

        paper.start_review(reviewer)
        share = paper.finish_review(reviewer, MIN_REVIEW_EFFORT_THRESHOLD - 0.5)

        self.assertEqual(share, 0.0)
        self.assertEqual(paper.accrual_rate, 1.0)
        self.assertTrue(paper.reviewed)
        self.assertNotIn(reviewer, paper.share_distribution)


class EconomicsTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_review_bump_rises_and_saturates_with_sigmoid(self):
        import Paper as paper_mod

        self.assertEqual(review_accrual_bump(MIN_REVIEW_EFFORT_THRESHOLD - 0.5), 0.0)
        with mock.patch.object(paper_mod, "REVIEW_EFFORT_CURVE", "sigmoid"):
            at_threshold = MIN_REVIEW_EFFORT_THRESHOLD
            b1 = review_accrual_bump(at_threshold)
            b2 = review_accrual_bump(at_threshold + 1.0)
            b5 = review_accrual_bump(GOOD_REVIEW_TIMESTEPS)
            sat = review_accrual_bump(paper_mod.REVIEW_SIGMOID_SATURATION_EFFORT)
            past_sat = review_accrual_bump(
                paper_mod.REVIEW_SIGMOID_SATURATION_EFFORT + 5.0
            )
        self.assertGreater(b1, 0.0)
        self.assertGreater(b2, b1)
        self.assertGreater(b5, b2)
        self.assertAlmostEqual(past_sat, sat)

    def test_fair_market_price_uses_empirical_epsilon_distribution(self):
        epsilons = [0.0, 0.05, 0.20]
        expected = sum(fair_market_component(e) for e in epsilons) / len(epsilons)

        self.assertAlmostEqual(fair_market_price_from_epsilons(epsilons), expected)

    def test_higher_quality_raises_bump(self):
        effort = MIN_REVIEW_EFFORT_THRESHOLD + 1.0
        self.assertGreater(
            review_accrual_bump(effort, quality=1.5),
            review_accrual_bump(effort, quality=0.8),
        )

    def test_higher_effort_yields_higher_accrual_rate(self):
        author = ScriptAgent("author")
        low_reviewer = ScriptAgent("low")
        high_reviewer = ScriptAgent("high")
        low = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        high = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        low.update_price_table([low_reviewer], 1.0, 0.0)
        high.update_price_table([high_reviewer], 1.0, 0.0)

        low.add_review(low_reviewer, MIN_REVIEW_EFFORT_THRESHOLD)
        high.add_review(high_reviewer, MIN_REVIEW_EFFORT_THRESHOLD + 3)

        self.assertGreater(high.accrual_rate, low.accrual_rate)

    def test_price_drops_for_higher_quality_papers(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        low_q = _listed_paper(author, quality=0.8)
        high_q = _listed_paper(author, quality=1.5)
        median_q = statistics.median([0.8, 1.5])

        low_q.update_price_table([reviewer], median_q, 0.0)
        high_q.update_price_table([reviewer], median_q, 0.0)

        self.assertGreater(low_q.offered_share(reviewer), high_q.offered_share(reviewer))

    def test_price_table_uses_fair_market_offer_base(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        reviewer.peer_review_epsilon_history = 0.10
        paper = _listed_paper(author, quality=1.0, current_ac=0.0, accrual_rate=1.0)
        epsilon = 0.10
        projected = projected_paper_ac(0.0, 1.0, epsilon)
        expected = fair_market_offer_share(epsilon, 0.0, projected)

        paper.update_price_table(
            [author, reviewer],
            market_median_quality=1.0,
            mean_peer_review_epsilon=0.10,
            fair_market_price=fair_market_price_from_epsilons([0.10]),
        )

        self.assertAlmostEqual(paper.offered_share(reviewer), expected)

    def test_fair_market_offer_scales_down_with_existing_ac(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        reviewer.peer_review_epsilon_history = 0.10
        fresh = _listed_paper(author, quality=1.0, current_ac=0.0, accrual_rate=1.0)
        seasoned = _listed_paper(author, quality=1.0, current_ac=50.0, accrual_rate=1.0)

        fresh.update_price_table([reviewer], 1.0, 0.10)
        seasoned.update_price_table([reviewer], 1.0, 0.10)

        self.assertGreater(fresh.offered_share(reviewer), seasoned.offered_share(reviewer))

    def test_price_rises_for_stronger_reviewer_epsilon_history(self):
        author = ScriptAgent("author")
        rookie = ScriptAgent("rookie")
        veteran = ScriptAgent("veteran")
        rookie.peer_review_epsilon_history = 0.02
        veteran.peer_review_epsilon_history = 0.20
        paper = _listed_paper(author, quality=1.0)

        with mock.patch("Paper.HISTORY_PRICE_SCALE", 0.5):
            paper.update_price_table(
                [rookie, veteran],
                market_median_quality=1.0,
                mean_peer_review_epsilon=0.11,
                fair_market_price=fair_market_price_from_epsilons([0.02, 0.20]),
            )

            self.assertGreater(paper.offered_share(veteran), paper.offered_share(rookie))

    def test_scarcity_multiplier_lowers_offers(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = _listed_paper(author, quality=1.0)

        paper.update_price_table([reviewer], 1.0, 0.0, scarcity_multiplier=1.0)
        base = paper.offered_share(reviewer)
        paper.update_price_table([reviewer], 1.0, 0.0, scarcity_multiplier=0.5)

        self.assertAlmostEqual(paper.offered_share(reviewer), base * 0.5)


class ReviewBumpDecayTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []
        self.decay_sim = replace(
            config.SIM,
            review_bump_duration=REVIEW_BUMP_DECAY,
            review_bump_decay_rate=0.1,
            review_bump_decay_cap_timesteps=20.0,
        )

    def _decay_patches(self):
        return (
            mock.patch.object(config, "SIM", self.decay_sim),
            mock.patch("Paper.SIM", self.decay_sim),
            mock.patch("Environment.SIM", self.decay_sim),
        )

    def test_decay_peak_matches_permanent_bump_at_review(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        effort = MIN_REVIEW_EFFORT_THRESHOLD + 2.0
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)
        expected_peak = paper.accrual_rate * (1.0 + review_accrual_bump(effort, 1.0))

        p1, p2, p3 = self._decay_patches()
        with p1, p2, p3:
            paper.start_review(reviewer, current_timestep=5)
            paper.finish_review(reviewer, effort, current_timestep=5)

        self.assertAlmostEqual(paper.accrual_rate, expected_peak)
        self.assertAlmostEqual(paper.base_accrual_rate, 1.0)
        self.assertAlmostEqual(
            paper.review_bump_epsilon, review_accrual_bump(effort, 1.0)
        )

    def test_decay_reduces_rate_over_time(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        effort = MIN_REVIEW_EFFORT_THRESHOLD + 1.0
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)

        p1, p2, p3 = self._decay_patches()
        with p1, p2, p3:
            paper.start_review(reviewer, current_timestep=1)
            paper.finish_review(reviewer, effort, current_timestep=1)
            peak = paper.accrual_rate
            paper.refresh_accrual_rate(11)
            later = paper.accrual_rate

        self.assertGreater(peak, later)
        self.assertGreater(later, paper.base_accrual_rate)

    def test_decay_hard_cap_returns_to_base_rate(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        effort = MIN_REVIEW_EFFORT_THRESHOLD
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)

        p1, p2, p3 = self._decay_patches()
        with p1, p2, p3:
            paper.start_review(reviewer, current_timestep=0)
            paper.finish_review(reviewer, effort, current_timestep=0)
            paper.refresh_accrual_rate(25)

        self.assertAlmostEqual(paper.accrual_rate, paper.base_accrual_rate)

    def test_decay_accrual_uses_shrinking_rate(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        effort = MIN_REVIEW_EFFORT_THRESHOLD + 1.0
        paper = _listed_paper(author, quality=1.0, accrual_rate=1.0, current_ac=0.0)
        paper.update_price_table([reviewer], 1.0, 0.0)

        p1, p2, p3 = self._decay_patches()
        with p1, p2, p3:
            paper.start_review(reviewer, current_timestep=10)
            paper.finish_review(reviewer, effort, current_timestep=10)
            paper.refresh_accrual_rate(10)
            paper.accrue_ac()
            first_gain = paper.current_ac
            paper.refresh_accrual_rate(11)
            paper.accrue_ac()
            second_gain = paper.current_ac - first_gain

        self.assertGreater(first_gain, 0.0)
        self.assertGreaterEqual(first_gain, second_gain)

    def test_review_bump_factor_helpers(self):
        with mock.patch("Paper.SIM", self.decay_sim):
            self.assertAlmostEqual(review_bump_factor_at_age(0.2, 0.0), 0.2)
            self.assertLess(review_bump_factor_at_age(0.2, 10.0), 0.2)
            self.assertAlmostEqual(review_bump_factor_at_age(0.2, 25.0), 0.0)
            self.assertAlmostEqual(decayed_accrual_rate(1.0, 0.2, 0.0), 1.2)


class MarketEconomicsTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []
        Agent.all_agents = []

    def test_environment_scarcity_multiplier_when_crowded(self):
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=1.0)
        agents = [author] + [
            HeuristicAgent(intrinsic_talent=1.0, name=f"reviewer-{i}")
            for i in range(5)
        ]
        Agent.all_papers = [paper]
        Agent.all_agents = agents
        env = Environment(
            agents=agents,
            papers=[paper],
            use_scarcity_pricing=True,
            reviewer_pressure_exponent=0.5,
        )

        env._update_market_prices()

        self.assertGreater(env.reviewer_pressure, 1.0)
        self.assertLess(env.scarcity_multiplier, 1.0)

    def test_competition_forecast_reduces_claim_intent(self):
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=1.0, current_ac=200.0, accrual_rate=1.0)
        agents = [
            HeuristicAgent(intrinsic_talent=1.0, name=f"reviewer-{i}")
            for i in range(10)
        ]
        agent = agents[0]
        agent.configure_paper_effort("fixed", 10.0, 10.0, discrete_timesteps=10.0)
        Agent.all_papers = [paper]
        Agent.all_agents = agents
        paper.update_price_table(agents, 1.0, 0.05)

        agent.configure_market_economics(use_competition_adjusted_forecast=False)
        self.assertIs(agent.choose_marketplace_action(), paper)

        with mock.patch.object(agent, "_claim_win_probability", return_value=0.01):
            agent.configure_market_economics(use_competition_adjusted_forecast=True)
            self.assertIsNone(agent.choose_marketplace_action())

    def test_merit_clearing_assigns_highest_scoring_reviewer(self):
        author = ScriptAgent("author")
        rookie = HeuristicAgent(intrinsic_talent=1.0, name="rookie")
        veteran = HeuristicAgent(intrinsic_talent=1.0, name="veteran")
        rookie.peer_review_epsilon_history = 0.02
        veteran.peer_review_epsilon_history = 0.20
        paper = _listed_paper(author, quality=1.0, current_ac=500.0, accrual_rate=1.0)
        agents = [author, rookie, veteran]
        Agent.all_papers = [paper]
        Agent.all_agents = agents
        with mock.patch("Paper.HISTORY_PRICE_SCALE", 0.5):
            env = Environment(
                agents=agents,
                papers=[paper],
                review_paradigm="discrete",
                use_merit_market_clearing=True,
            )

            env._update_market_prices()
            env._clear_review_market()

        self.assertIs(veteran.market_claim_assignment, paper)
        self.assertIsNone(rookie.market_claim_assignment)
        self.assertEqual(env.market_assignments, 1)

    def test_adaptive_pricing_policy_disables_scarcity(self):
        author = ScriptAgent("author")
        env = Environment(
            agents=[author],
            pricing_policy="adaptive_multiplier",
            use_scarcity_pricing=True,
        )

        self.assertEqual(env.pricing_policy, "adaptive_multiplier")
        self.assertFalse(env.use_scarcity_pricing)

    def test_static_pricing_policy_honored(self):
        author = ScriptAgent("author")
        env = Environment(agents=[author], pricing_policy="static_fair_market")

        self.assertEqual(env.pricing_policy, "static_fair_market")

    def test_reputation_bin_name_mapping(self):
        edges, names = validate_reputation_bin_config((0.08, 0.15), ("low", "mid", "high"))
        self.assertEqual(reputation_bin_name(0.02, edges, names), "low")
        self.assertEqual(reputation_bin_name(0.10, edges, names), "mid")
        self.assertEqual(reputation_bin_name(0.20, edges, names), "high")

    def test_binned_fast_claim_lowers_claimer_bin_only(self):
        author = ScriptAgent("author")
        author.configure_pricing_policy(
            "adaptive_multiplier",
            adaptive_pricing_mode=ADAPTIVE_PRICING_BINNED,
            learning_rate=0.03,
            target_market_wait_timesteps=1.0,
        )
        veteran = ScriptAgent("veteran")
        veteran.peer_review_epsilon_history = 0.20

        author.record_review_claim_feedback(0, claimer=veteran)

        self.assertAlmostEqual(author.review_offer_multipliers["high"], 0.97)
        self.assertAlmostEqual(author.review_offer_multipliers["mid"], 1.0)
        self.assertAlmostEqual(author.review_offer_multipliers["low"], 1.0)

    def test_binned_fast_low_rep_lowers_low_bin_only(self):
        author = ScriptAgent("author")
        author.configure_pricing_policy(
            "adaptive_multiplier",
            adaptive_pricing_mode=ADAPTIVE_PRICING_BINNED,
            learning_rate=0.03,
            target_market_wait_timesteps=1.0,
        )
        rookie = ScriptAgent("rookie")
        rookie.peer_review_epsilon_history = 0.02

        author.record_review_claim_feedback(0, claimer=rookie)

        self.assertAlmostEqual(author.review_offer_multipliers["low"], 0.97)
        self.assertAlmostEqual(author.review_offer_multipliers["mid"], 1.0)
        self.assertAlmostEqual(author.review_offer_multipliers["high"], 1.0)

    def test_binned_slow_claim_raises_claimer_bin_only(self):
        author = ScriptAgent("author")
        author.configure_pricing_policy(
            "adaptive_multiplier",
            adaptive_pricing_mode=ADAPTIVE_PRICING_BINNED,
            learning_rate=0.1,
            target_market_wait_timesteps=1.0,
        )
        veteran = ScriptAgent("veteran")
        veteran.peer_review_epsilon_history = 0.20

        author.record_review_claim_feedback(3, claimer=veteran)

        self.assertAlmostEqual(author.review_offer_multipliers["high"], 1.2)
        self.assertAlmostEqual(author.review_offer_multipliers["mid"], 1.0)
        self.assertAlmostEqual(author.review_offer_multipliers["low"], 1.0)

    def test_global_adaptive_feedback_preserves_legacy_behavior(self):
        author = ScriptAgent("author")
        author.configure_pricing_policy(
            "adaptive_multiplier",
            adaptive_pricing_mode=ADAPTIVE_PRICING_GLOBAL,
            learning_rate=0.03,
        )

        author.record_review_claim_feedback(0)

        self.assertAlmostEqual(author.review_offer_multiplier, 0.97)

    def test_binned_multiplier_applies_per_reviewer_in_price_table(self):
        author = ScriptAgent("author")
        author.configure_pricing_policy(
            "adaptive_multiplier",
            adaptive_pricing_mode=ADAPTIVE_PRICING_BINNED,
        )
        author.review_offer_multipliers["high"] = 1.2
        author.review_offer_multipliers["low"] = 0.8
        rookie = ScriptAgent("rookie")
        veteran = ScriptAgent("veteran")
        rookie.peer_review_epsilon_history = 0.02
        veteran.peer_review_epsilon_history = 0.20
        paper = _listed_paper(author, quality=1.0)

        paper.update_price_table(
            [rookie, veteran],
            market_median_quality=1.0,
            mean_peer_review_epsilon=0.11,
            pricing_policy="adaptive_multiplier",
        )

        rookie_offer = paper.offered_share(rookie)
        veteran_offer = paper.offered_share(veteran)
        self.assertGreater(veteran_offer, rookie_offer)
        ratio = veteran_offer / rookie_offer
        self.assertGreater(ratio, 1.2)

    def test_legacy_heuristic_claim_unchanged_when_flags_off(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)
        author = ScriptAgent("author")
        low = _listed_paper(author, quality=1.0, current_ac=10.0)
        high = _listed_paper(author, quality=1.0, current_ac=200.0)
        Agent.all_papers = [low, high]
        Agent.all_agents = [agent, author]
        for paper in (low, high):
            paper.update_price_table([agent], 1.0, 0.0)

        agent.configure_market_economics(use_competition_adjusted_forecast=False)
        self.assertIs(agent.choose_marketplace_action(), high)


class ReviewerStateTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_peer_review_history_updates_on_completion(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent(
            "reviewer",
            work=[("peer_review", None), ("finish_review_write_paper", None)],
        )
        reviewer.configure_review_paradigm("continuous")
        paper = _listed_paper(author, quality=1.0, current_ac=100.0, accrual_rate=1.0)
        paper.update_price_table([reviewer], 1.0, 0.0)
        Agent.all_papers = [paper]

        claimed = reviewer.claim_review(paper)
        self.assertIsNone(claimed)  # phase 1 is pure selection, no record
        self.assertEqual(reviewer.active_review_effort, 0.0)

        started = reviewer.apply_initial_review_effort()
        self.assertEqual(started.kind, "review_started")
        self.assertEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_TIMESTEP)

        continued = reviewer.work_turn()
        self.assertEqual(continued.kind, "review_continued")
        self.assertEqual(reviewer.active_review_effort, 2 * REVIEW_EFFORT_PER_TIMESTEP)

        finished = reviewer.work_turn()
        self.assertEqual(finished.kind, "review_finished_write")
        self.assertEqual(reviewer.completed_review_count, 1)
        self.assertGreater(reviewer.peer_review_history, 0.0)
        self.assertGreater(reviewer.peer_review_epsilon_history, 0.0)
        self.assertIsNone(reviewer.active_review_paper)

    def test_peer_review_history_uses_accrual_rate_not_ac(self):
        reviewer = ScriptAgent("reviewer")
        high_ac = Paper(
            author=ScriptAgent("author_a"),
            current_ac=500.0,
            accrual_rate=1.2,
        )
        low_ac = Paper(
            author=ScriptAgent("author_b"),
            current_ac=10.0,
            accrual_rate=2.0,
        )
        share = 0.25

        reviewer._record_review_outcome(high_ac, share, 5.0, "good_faith")
        self.assertAlmostEqual(reviewer.peer_review_history, share * 1.2)

        reviewer._record_review_outcome(low_ac, share, 5.0, "good_faith")
        expected = (share * 1.2 + share * 2.0) / 2
        self.assertAlmostEqual(reviewer.peer_review_history, expected)
        self.assertNotAlmostEqual(reviewer.peer_review_history, (share * 500.0 + share * 10.0) / 2)

    def test_grabbing_new_paper_finalizes_active_review(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer", work=[("peer_review", None)])
        first = _listed_paper(author, quality=1.0, current_ac=50.0)
        second = _listed_paper(author, quality=1.0, current_ac=100.0)
        first.update_price_table([reviewer], 1.0, 0.0)
        second.update_price_table([reviewer], 1.0, 0.0)

        reviewer.claim_review(first)
        reviewer.apply_initial_review_effort()  # phase 2 effort on the first review
        reviewer.work_turn()  # reach the good-faith effort threshold
        record = reviewer.claim_review(second)

        self.assertEqual(record.kind, "review_finished_peer_review")
        self.assertIs(record.paper, first)
        self.assertIs(reviewer.active_review_paper, second)
        self.assertEqual(reviewer.active_review_effort, 0.0)  # second not worked yet
        self.assertTrue(first.reviewed)
        self.assertFalse(second.reviewed)
        self.assertIn(reviewer, first.share_distribution)


class ReviewParadigmTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_continuous_mode_classifies_finished_reviews_by_threshold(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent(
            "reviewer",
            marketplace=[],
            work=[("peer_review", None), ("finish_review_write_paper", None)],
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="continuous",
        )
        paper.update_price_table([reviewer], 1.0, 0.0)

        reviewer.claim_review(paper)
        reviewer.apply_initial_review_effort()
        reviewer.work_turn()
        record = reviewer.work_turn()

        self.assertEqual(record.review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(reviewer.last_review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["review_kind"], GOOD_FAITH_REVIEW)

    def test_discrete_bad_faith_review_finishes_after_one_timestep(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            BAD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
            use_merit_market_clearing=False,
        )

        env.run_timestep()

        self.assertTrue(paper.reviewed)
        self.assertIsNone(reviewer.active_review_paper)
        self.assertEqual(reviewer.last_review_kind, BAD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["effort"], BAD_REVIEW_TIMESTEPS)
        self.assertEqual(history.completed_reviews[-1][4], BAD_FAITH_REVIEW)
        self.assertEqual(history.action_counts["bad_faith_review"], 1)
        self.assertEqual(history.scalars["bad_faith_reviews"][-1], 1.0)
        self.assertEqual(history.scalars["good_faith_reviews"][-1], 0.0)
        self.assertGreater(paper.review_records[-1]["epsilon"], 0.0)
        self.assertIn(reviewer, paper.share_distribution)

    def test_discrete_good_faith_review_tracks_continuous_threshold(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            GOOD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
            use_merit_market_clearing=False,
        )

        env.run(int(GOOD_REVIEW_TIMESTEPS) - 1)

        self.assertFalse(paper.reviewed)
        self.assertIs(reviewer.active_review_paper, paper)
        self.assertEqual(reviewer.active_review_effort, GOOD_REVIEW_TIMESTEPS - 1)

        env.run_timestep()

        self.assertTrue(paper.reviewed)
        self.assertIsNone(reviewer.active_review_paper)
        self.assertEqual(reviewer.last_review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["effort"], GOOD_REVIEW_TIMESTEPS)
        self.assertEqual(history.completed_reviews[-1][4], GOOD_FAITH_REVIEW)
        self.assertEqual(history.action_counts["good_faith_review"], 1)
        self.assertEqual(history.scalars["good_faith_reviews"][-1], 1.0)

    def test_probabilistic_agents_are_discrete_only(self):
        agent = ProbabilisticDiscreteAgent(intrinsic_talent=1.0)

        with self.assertRaises(ValueError):
            Environment(agents=[agent], review_paradigm="continuous")

        env = Environment(agents=[agent], review_paradigm="discrete")
        self.assertEqual(env.review_paradigm, "discrete")

    def test_invalid_review_paradigm_is_rejected(self):
        with self.assertRaises(ValueError):
            Environment(agents=[], review_paradigm="mixed")

    def test_build_simulation_can_add_random_control_agents(self):
        from run_simulation import build_simulation

        history = History()
        env = build_simulation(
            history,
            num_agents=0,
            rl_agents=0,
            random_agents=2,
            probabilistic_agents=0,
            seed=3,
        )

        self.assertEqual(len(env.agents), 2)
        self.assertTrue(all(isinstance(agent, RandomAgent) for agent in env.agents))

    def test_quality_scaled_paper_effort_samples_stable_target(self):
        agent = ScriptAgent("author")
        agent.configure_paper_effort("quality_scaled", 50.0, 150.0)

        first = agent.paper_completion_threshold()
        second = agent.paper_completion_threshold()

        self.assertGreaterEqual(first, 50.0)
        self.assertLessEqual(first, 150.0)
        self.assertEqual(first, second)


class HeuristicPolicyTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_writes_when_nothing_is_reviewable(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)
        self.assertIsNone(agent.choose_marketplace_action())
        self.assertEqual(agent.choose_work_action(), ("write_paper", None))

    def test_claims_highest_value_listed_paper(self):
        agent = HeuristicAgent(intrinsic_talent=1.0)
        author = ScriptAgent("author")
        low = _listed_paper(author, quality=1.0, current_ac=10.0)
        high = _listed_paper(author, quality=1.0, current_ac=200.0)
        Agent.all_papers = [low, high]
        for paper in (low, high):
            paper.update_price_table([agent], 1.0, 0.0)

        self.assertIs(agent.choose_marketplace_action(), high)

    def test_work_phase_finishes_a_normal_review(self):
        agent = HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_timesteps=5)
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=1.0, current_ac=10.0)
        paper.update_price_table([agent], 1.0, 0.0)
        Agent.all_papers = [paper]

        agent.claim_review(paper)
        agent.apply_initial_review_effort()
        action, _ = agent.choose_work_action()

        self.assertEqual(action, "finish_review_write_paper")

    def test_work_phase_continues_when_marginal_effort_dominates(self):
        # A low-talent agent's own research is weak, so investing another
        # timestep in a valuable review beats finishing and writing.
        agent = HeuristicAgent(intrinsic_talent=0.1, forecast_horizon_timesteps=60)
        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=2.0, current_ac=10.0)
        paper.update_price_table([agent], 2.0, 0.0)
        Agent.all_papers = [paper]

        agent.claim_review(paper)
        agent.apply_initial_review_effort()
        action, target = agent.choose_work_action()

        self.assertEqual(action, "peer_review")
        self.assertIs(target, paper)


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_work_phase_runs_agents_each_timestep(self):
        log: list[str] = []
        agents = [RecordingAgent(log, "first"), RecordingAgent(log, "second")]
        env = Environment(agents=agents)

        env.run_timestep()

        self.assertEqual(sorted(log), ["first", "second"])
        self.assertEqual(env.timestep, 1)

    def test_accrual_and_capital_update_each_timestep(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = Paper(
            author=author,
            accrual_rate=2.0,
            current_ac=10.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        env = Environment(agents=[author, reviewer], papers=[paper])

        env.run_timestep()

        self.assertEqual(env.timestep, 1)
        self.assertEqual(paper.current_ac, 12.0)
        self.assertAlmostEqual(author.academic_capital, 9.0)
        self.assertAlmostEqual(reviewer.academic_capital, 3.0)

    def test_history_records_timesteps_and_actions(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent("reviewer")
        paper = Paper(
            author=author,
            accrual_rate=2.0,
            current_ac=10.0,
            share_distribution={author: 0.75, reviewer: 0.25},
        )
        history = History()
        env = Environment(agents=[author, reviewer], papers=[paper], history=history)

        env.run_timestep()

        self.assertEqual(history.timesteps, [1])
        self.assertEqual(history.days, [1])  # backwards-compatible alias
        self.assertEqual(len(history.actions), 2)
        self.assertAlmostEqual(history.agent_capital["author"][0], 9.0)
        self.assertEqual(history.scalars["num_papers"][0], 1.0)

    def test_history_records_publication_and_time_allocation_metrics(self):
        author = ScriptAgent("author", continuous=[("research_finish", None)])
        history = History()
        env = Environment(
            agents=[author],
            papers=[],
            history=history,
            continuous_publishing="choice",
        )

        env.run_timestep()

        self.assertEqual(history.scalars["papers_published_this_timestep"], [1.0])
        self.assertEqual(history.scalars["cumulative_papers_published"], [1.0])
        self.assertEqual(history.scalars["paper_publication_rate_trailing_100"], [1.0])
        self.assertEqual(history.scalars["writing_time_share_this_timestep"], [1.0])
        self.assertEqual(history.scalars["review_time_share_this_timestep"], [0.0])
        self.assertEqual(history.scalars["unallocated_time_share_this_timestep"], [0.0])

    def test_history_distinguishes_open_and_in_progress_review_backlog(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            GOOD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
            use_merit_market_clearing=False,
        )

        env.run_timestep()

        self.assertEqual(history.scalars["papers_on_market"][-1], 0.0)
        self.assertEqual(history.scalars["papers_in_review"][-1], 1.0)
        self.assertEqual(history.scalars["review_backlog_total"][-1], 1.0)
        self.assertEqual(history.scalars["writing_time_share_this_timestep"][-1], 0.5)
        self.assertEqual(history.scalars["review_time_share_this_timestep"][-1], 0.5)

    def test_history_exports_agent_group_summary(self):
        author = ScriptAgent("author")
        reviewer = ReviewKindScriptAgent(
            GOOD_FAITH_REVIEW, name="reviewer", marketplace=[]
        )
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        reviewer.marketplace = [paper]
        Agent.all_papers = [paper]
        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
            use_merit_market_clearing=False,
        )

        env.run(int(GOOD_REVIEW_TIMESTEPS))
        data = history.to_dict()
        summary = data["agent_group_summary"]

        self.assertEqual(data["agent_groups"]["author"], "ScriptAgent")
        self.assertEqual(
            data["agent_groups"]["reviewer"],
            "ReviewKindScriptAgent",
        )
        self.assertIn("agent_review_epsilon_history", data)
        self.assertIn("agent_talent", data)
        self.assertIn("paper_quality", data)
        self.assertIn("paper_required_writing_effort", data)
        self.assertIn("action_counts_by_timestep", data)
        outcome = data["agent_outcome_summary"][0]
        self.assertIn("talent", outcome)
        self.assertIn("average_paper_quality", outcome)
        self.assertGreaterEqual(
            summary["ReviewKindScriptAgent"]["completed_reviews"],
            1,
        )
        group_stats = summary["ReviewKindScriptAgent"]
        self.assertIn("average_review_effort", group_stats)
        self.assertIn("good_faith_review_rate", group_stats)
        self.assertIn("mean_good_faith_reviews", group_stats)
        self.assertIn("mean_bad_faith_reviews", group_stats)

    def test_full_run_produces_reviews(self):
        from run_simulation import build_simulation

        history = History()
        env = build_simulation(history, num_agents=12, rl_agents=0, seed=3)
        env.run(220)

        completed = sum(
            getattr(p, "completed_peer_reviews", 0) for p in env.papers
        )
        self.assertGreater(len(env.papers), 0)
        self.assertGreater(completed, 0)
        # Reviewed papers leave the market permanently.
        for paper in env.papers:
            if paper.reviewed:
                self.assertFalse(paper.review_available)

    def test_history_to_csv_has_header_and_one_row_per_timestep(self):
        author = ScriptAgent("author")
        paper = Paper(author=author, accrual_rate=1.0, current_ac=5.0)
        history = History()
        env = Environment(agents=[author], papers=[paper], history=history)
        env.run(3)

        path = os.path.join(tempfile.mkdtemp(), "history.csv")
        history.to_csv(path)
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))

        self.assertEqual(rows[0][0], "timestep")
        self.assertIn("total_capital", rows[0])
        self.assertEqual(len(rows), 1 + 3)
        self.assertEqual([row[0] for row in rows[1:]], ["1", "2", "3"])

    def test_history_records_accepted_review_price_on_claim(self):
        author = ScriptAgent("author")
        reviewer = ScriptAgent(
            "reviewer",
            marketplace=[],
            work=[("peer_review", None)],
        )
        paper = _listed_paper(author, quality=1.0, current_ac=50.0)
        reviewer.marketplace = [paper]

        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=[paper],
            history=history,
            review_paradigm="discrete",
            use_merit_market_clearing=False,
        )
        env._update_market_prices()
        expected = paper.offered_share(reviewer)
        env.run(2)

        self.assertGreater(expected, 0.0)
        self.assertGreaterEqual(len(history.accepted_review_claims), 1)
        self.assertAlmostEqual(history.accepted_review_claims[0][1], expected)


class ContinuousAccrualTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_accrual_rate_rises_and_saturates_with_effort(self):
        ceiling = accrual_rate_from_quality(1.0)
        r0 = accrual_rate_from_effort(1.0, 0.0)
        r5 = accrual_rate_from_effort(1.0, 5.0)
        r20 = accrual_rate_from_effort(1.0, 20.0)

        self.assertAlmostEqual(r0, 0.0)
        self.assertGreater(r5, r0)
        self.assertGreater(r20, r5)
        self.assertLess(r20, ceiling)
        # Diminishing returns: equal-width effort bands add less rate as effort grows.
        self.assertGreater(r5 - r0, r20 - r5)

    def test_more_writing_effort_raises_paper_base_rate(self):
        author = ScriptAgent("author")
        low = Paper(author=author, quality=1.0, writing_effort=2.0)
        high = Paper(author=author, quality=1.0, writing_effort=12.0)

        self.assertGreater(high.accrual_rate, low.accrual_rate)
        self.assertLess(high.accrual_rate, accrual_rate_from_quality(1.0))


class ContinuousMergedPhaseTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def _listed(self, author, **kwargs):
        paper = _listed_paper(author, **kwargs)
        Agent.all_papers.append(paper)
        return paper

    def test_nonreviewer_claim_starts_review_with_one_effort(self):
        author = ScriptAgent("author")
        paper = self._listed(author, quality=1.0, current_ac=10.0)
        reviewer = ScriptAgent("reviewer", continuous=[("claim", paper)])
        reviewer.configure_review_paradigm("continuous")
        paper.update_price_table([reviewer], 1.0, 0.0)

        records = reviewer.act_continuous()

        self.assertEqual([r.kind for r in records], ["review_started"])
        self.assertIs(reviewer.active_review_paper, paper)
        self.assertAlmostEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_TIMESTEP)

    def test_nonreviewer_research_adds_progress_without_publishing(self):
        agent = ScriptAgent("a", continuous=[("research", None)])
        agent.configure_review_paradigm("continuous")

        records = agent.act_continuous()

        self.assertEqual([r.kind for r in records], ["write_paper"])
        self.assertFalse(records[0].published)
        self.assertEqual(len(Agent.all_papers), 0)
        self.assertGreater(agent.paper_progress, 0.0)

    def test_nonreviewer_research_finish_publishes_and_resets(self):
        agent = ScriptAgent("a", continuous=[("research_finish", None)])
        agent.configure_review_paradigm("continuous")
        agent.configure_continuous_publishing("choice")

        records = agent.act_continuous()

        self.assertEqual([r.kind for r in records], ["write_paper"])
        self.assertTrue(records[0].published)
        self.assertEqual(len(Agent.all_papers), 1)
        self.assertEqual(agent.paper_progress, 0.0)
        self.assertAlmostEqual(Agent.all_papers[0].writing_effort, 0.5)

    def test_reviewer_review_continues_active_review(self):
        author = ScriptAgent("author")
        paper = self._listed(author, quality=1.0, current_ac=10.0)
        reviewer = ScriptAgent("reviewer")
        reviewer.configure_review_paradigm("continuous")
        paper.update_price_table([reviewer], 1.0, 0.0)
        reviewer.claim_review(paper)
        reviewer.apply_initial_review_effort()
        reviewer.continuous = [("review", None)]

        records = reviewer.act_continuous()

        self.assertEqual([r.kind for r in records], ["review_continued"])
        self.assertAlmostEqual(
            reviewer.active_review_effort, 2 * REVIEW_EFFORT_PER_TIMESTEP
        )

    def test_reviewer_claim_switch_finalizes_old_review(self):
        author = ScriptAgent("author")
        first = self._listed(author, quality=1.0, current_ac=50.0)
        second = self._listed(author, quality=1.0, current_ac=100.0)
        reviewer = ScriptAgent("reviewer")
        reviewer.configure_review_paradigm("continuous")
        for paper in (first, second):
            paper.update_price_table([reviewer], 1.0, 0.0)
        reviewer.claim_review(first)
        reviewer.apply_initial_review_effort()
        reviewer.continuous = [("claim", second)]

        records = reviewer.act_continuous()

        self.assertEqual(
            [r.kind for r in records],
            ["review_finished_peer_review", "review_started"],
        )
        self.assertTrue(first.reviewed)
        self.assertIs(reviewer.active_review_paper, second)
        self.assertAlmostEqual(reviewer.active_review_effort, REVIEW_EFFORT_PER_TIMESTEP)

    def test_reviewer_research_ends_review_and_writes(self):
        author = ScriptAgent("author")
        paper = self._listed(author, quality=1.0, current_ac=50.0)
        reviewer = ScriptAgent("reviewer")
        reviewer.configure_review_paradigm("continuous")
        paper.update_price_table([reviewer], 1.0, 0.0)
        reviewer.claim_review(paper)
        reviewer.apply_initial_review_effort()
        reviewer.continuous = [("research", None)]

        records = reviewer.act_continuous()

        self.assertEqual([r.kind for r in records], ["review_finished_write"])
        self.assertTrue(paper.reviewed)
        self.assertIsNone(reviewer.active_review_paper)
        self.assertFalse(records[0].published)
        self.assertGreater(reviewer.paper_progress, 0.0)


class ContinuousThresholdPublishingTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []

    def test_auto_publishes_when_writing_effort_reaches_threshold(self):
        agent = ScriptAgent("author")
        agent.configure_review_paradigm("continuous")
        agent.configure_continuous_publishing("threshold", 5.0)
        for _ in range(9):
            records = agent.act_continuous()
            self.assertFalse(records[0].published)
        self.assertEqual(len(Agent.all_papers), 0)
        self.assertAlmostEqual(agent.paper_progress, 4.5)

        records = agent.act_continuous()
        self.assertTrue(records[0].published)
        self.assertEqual(len(Agent.all_papers), 1)
        self.assertEqual(agent.paper_progress, 0.0)
        self.assertAlmostEqual(Agent.all_papers[0].writing_effort, 5.0)

    def test_research_finish_is_ignored_in_threshold_mode(self):
        agent = ScriptAgent("author", continuous=[("research_finish", None)])
        agent.configure_review_paradigm("continuous")
        agent.configure_continuous_publishing("threshold", 50.0)

        records = agent.act_continuous()

        self.assertFalse(records[0].published)
        self.assertEqual(len(Agent.all_papers), 0)
        self.assertGreater(agent.paper_progress, 0.0)


class DiscreteQLearningAgentTest(unittest.TestCase):
    def setUp(self):
        Agent.all_papers = []
        Agent.all_agents = []

    def test_choose_review_kind_uses_q_pending_kind(self):
        from DiscreteQLearningAgent import DiscreteQLearningAgent

        author = ScriptAgent("author")
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        agent = DiscreteQLearningAgent(intrinsic_talent=1.0, learning=False)
        agent._pending_review_kind = BAD_FAITH_REVIEW

        self.assertEqual(agent.choose_review_kind(paper), BAD_FAITH_REVIEW)
        self.assertIsNone(agent._pending_review_kind)

    def test_greedy_q_selects_bad_faith_claim(self):
        import numpy as np

        from DiscreteQLearningAgent import (
            DiscreteQLearningAgent,
            DiscreteTabularQ,
        )

        author = ScriptAgent("author")
        reviewer = DiscreteQLearningAgent(intrinsic_talent=1.0, learning=False)
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        paper.update_price_table([author, reviewer], 1.0, 0.0)
        Agent.all_papers = [paper]

        backend = DiscreteTabularQ()
        _, best_share, best_ac = reviewer._best_reviewable()
        features = reviewer._features(best_share, best_ac)
        key = backend._key(features)
        backend.table[key] = np.array([0.0, 10.0, 1.0], dtype=np.float64)
        reviewer.backend = backend

        claimed = reviewer.choose_marketplace_action()
        self.assertIs(claimed, paper)
        self.assertEqual(reviewer.choose_review_kind(paper), BAD_FAITH_REVIEW)

    def test_discrete_policy_rejects_continuous_q_table(self):
        import pickle
        import tempfile

        from DiscreteQLearningAgent import DiscreteTabularQ, NUM_DISCRETE_ACTIONS
        from QLearningAgent import NUM_ACTIONS

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            path = fh.name
            pickle.dump(
                {
                    "alpha": 0.1,
                    "buckets": 5,
                    "table": {(0,) * 9: [0.0] * NUM_ACTIONS},
                },
                fh,
            )

        backend = DiscreteTabularQ()
        try:
            with self.assertRaises(ValueError):
                backend.load(path)
        finally:
            os.remove(path)

        self.assertEqual(NUM_DISCRETE_ACTIONS, 3)

    def test_discrete_rl_good_faith_review_runs_five_timesteps(self):
        from DiscreteQLearningAgent import (
            DiscreteQLearningAgent,
            DiscreteTabularQ,
        )

        author = ScriptAgent("author")
        reviewer = DiscreteQLearningAgent(intrinsic_talent=1.0, learning=False)
        paper = _listed_paper(author, quality=1.0, current_ac=100.0)
        paper.update_price_table([author, reviewer], 1.0, 0.0)
        Agent.all_papers = [paper]

        import numpy as np

        backend = DiscreteTabularQ()
        _, best_share, best_ac = reviewer._best_reviewable()
        features = reviewer._features(best_share, best_ac)
        key = backend._key(features)
        backend.table[key] = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        reviewer.backend = backend

        history = History()
        env = Environment(
            agents=[author, reviewer],
            papers=Agent.all_papers,
            history=history,
            review_paradigm="discrete",
        )

        env.run(int(GOOD_REVIEW_TIMESTEPS))

        self.assertTrue(paper.reviewed)
        self.assertEqual(reviewer.last_review_kind, GOOD_FAITH_REVIEW)
        self.assertEqual(paper.review_records[-1]["review_kind"], GOOD_FAITH_REVIEW)
        self.assertEqual(history.action_counts["good_faith_review"], 1)


class QLearningRewardTest(unittest.TestCase):
    def test_ac_percentile_rank_handles_ties(self):
        from QLearningAgent import ac_percentile_rank

        self.assertAlmostEqual(ac_percentile_rank(5.0, [1.0, 5.0, 5.0, 9.0]), 0.5)
        self.assertAlmostEqual(ac_percentile_rank(9.0, [1.0, 5.0, 5.0, 9.0]), 0.875)
        self.assertAlmostEqual(ac_percentile_rank(1.0, [1.0, 5.0, 5.0, 9.0]), 0.125)

    def test_reward_includes_delta_rank(self):
        from config import SIM
        from QLearningAgent import QLearningAgent, ac_percentile_rank

        peers = [
            ScriptAgent("peer-a"),
            ScriptAgent("peer-b"),
        ]
        peers[0].academic_capital = 10.0
        peers[1].academic_capital = 30.0
        agent = QLearningAgent(intrinsic_talent=1.0, academic_capital=20.0, learning=False)
        Agent.all_agents = [agent, *peers]
        agent._last_capital = 15.0
        agent._last_rank = ac_percentile_rank(15.0, [15.0, 10.0, 30.0])

        reward = agent._compute_reward()
        expected_ac = 5.0
        expected_rank = ac_percentile_rank(20.0, [20.0, 10.0, 30.0]) - agent._last_rank

        self.assertAlmostEqual(
            reward,
            SIM.rl_reward_ac_weight * expected_ac
            + SIM.rl_reward_rank_weight * expected_rank,
        )

    def test_reward_includes_delta_accrual_rate(self):
        from config import SIM
        from QLearningAgent import QLearningAgent

        author = ScriptAgent("author")
        paper = Paper(author=author, accrual_rate=1.0, current_ac=10.0)
        Agent.all_papers = [paper]
        agent = QLearningAgent(intrinsic_talent=1.0, learning=False)
        agent._last_accrual_rate = 0.5

        reward = agent._compute_reward()
        expected_accrual = agent._portfolio_accrual_rate() - 0.5

        self.assertAlmostEqual(
            reward,
            SIM.rl_reward_accrual_weight * expected_accrual,
        )


class LowTalentRLAgentsTest(unittest.TestCase):
    def test_low_talent_rl_agents_grouped_separately(self):
        from QLearningAgent import LowTalentQLearningAgent
        from run_simulation import build_simulation

        history = History()
        env = build_simulation(
            history,
            num_agents=0,
            rl_agents=2,
            low_talent_rl_agents=2,
            random_agents=0,
            probabilistic_agents=0,
            rl_policy_path=None,
            low_talent_rl_policy_path=None,
        )
        env.run(5)

        groups = set(history.agent_groups.values())
        self.assertIn("QLearningAgent", groups)
        self.assertIn("LowTalentQLearningAgent", groups)

        summary = history.agent_group_summary()
        self.assertIn("QLearningAgent", summary)
        self.assertIn("LowTalentQLearningAgent", summary)
        self.assertEqual(summary["LowTalentQLearningAgent"]["agent_count"], 2)

        low_talent_agents = [
            a for a in env.agents if isinstance(a, LowTalentQLearningAgent)
        ]
        self.assertEqual(len(low_talent_agents), 2)
        for agent in low_talent_agents:
            self.assertAlmostEqual(agent.intrinsic_talent, 0.3)

    def test_build_run_config_includes_low_talent_fields(self):
        from run_simulation import build_run_config

        config = build_run_config(low_talent_rl_agents=5, low_talent_value=0.25)
        self.assertEqual(config["num_low_talent_rl_agents"], 5)
        self.assertEqual(config["low_talent_value"], 0.25)
        self.assertIn("train_low_talent", config)


class BuildRunConfigTest(unittest.TestCase):
    def test_reflects_runtime_sim_replace(self):
        """Gallery config must snapshot live config.SIM after runtime patches."""
        from dataclasses import replace

        import config as config_module
        from run_simulation import build_run_config

        old_sim = config_module.SIM
        config_module.SIM = replace(
            old_sim,
            max_review_accrual_bump=0.42,
            min_review_accrual_bump=0.11,
            review_bump_duration="permanent",
        )
        try:
            config = build_run_config()
            self.assertEqual(config["max_review_accrual_bump"], 0.42)
            self.assertEqual(config["min_review_accrual_bump"], 0.11)
            self.assertEqual(config["review_bump_duration"], "permanent")
        finally:
            config_module.SIM = old_sim


class DQNAgentTest(unittest.TestCase):
    def test_dqn_backend_q_values_shape(self):
        import numpy as np

        from DQNAgent import make_dqn_backend
        from QLearningAgent import NUM_ACTIONS, NUM_FEATURES

        backend = make_dqn_backend(hidden_size=8, hidden_layers=1)
        q = backend.q_values(np.zeros(NUM_FEATURES, dtype=np.float64))
        self.assertEqual(q.shape, (NUM_ACTIONS,))

    def test_dqn_remember_and_train(self):
        import numpy as np

        from DQNAgent import make_dqn_backend
        from QLearningAgent import NUM_ACTIONS, NUM_FEATURES

        backend = make_dqn_backend(batch_size=4, replay_capacity=32)
        state = np.random.rand(NUM_FEATURES)
        next_state = np.random.rand(NUM_FEATURES)
        mask = np.ones(NUM_ACTIONS)
        for _ in range(8):
            backend.remember(state, 0, 1.0, next_state, mask, done=False)
        loss = backend.train_step()
        self.assertIsNotNone(loss)

    def test_dqn_save_load_roundtrip(self):
        import tempfile

        import numpy as np

        from DQNAgent import make_dqn_backend
        from QLearningAgent import NUM_FEATURES

        backend = make_dqn_backend(hidden_size=16, hidden_layers=1)
        features = np.linspace(0.1, 0.9, NUM_FEATURES)
        before = backend.q_values(features).copy()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            path = fh.name
        try:
            backend.save(path)
            reloaded = make_dqn_backend(hidden_size=8, hidden_layers=1)
            reloaded.load(path)
            after = reloaded.q_values(features)
            np.testing.assert_allclose(before, after)
        finally:
            os.remove(path)


class UtilityTest(unittest.TestCase):
    def test_gini_ranges_from_equal_to_unequal(self):
        self.assertEqual(gini([]), 0.0)
        self.assertEqual(gini([5.0, 5.0, 5.0]), 0.0)
        self.assertAlmostEqual(gini([0.0, 0.0, 0.0, 10.0]), 0.75)

    @unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
    def test_market_pricing_series_helpers(self):
        import visualize

        history = History()
        history.timesteps = [1, 2, 3]
        history.scalars["fair_market_price"] = [0.01, 0.02, 0.03]
        history.scalars["scarcity_multiplier"] = [1.1, 1.0, 0.9]
        history.accepted_review_claims = [(2, 0.015), (3, 0.0)]

        days, values = visualize._aligned_scalar_plot(history, "fair_market_price")
        self.assertEqual(days, [1.0, 2.0, 3.0])
        self.assertEqual(values, [0.01, 0.02, 0.03])
        self.assertEqual(
            visualize._market_pricing_claim_points(history),
            [(2, 0.015)],
        )
        self.assertTrue(visualize._has_market_pricing_scalars(history))

    @unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
    def test_review_faith_share_price_helpers(self):
        import visualize

        history = History()
        history.actions = [
            (2, "Agent 1", "review_started", "Paper 1"),
            (3, "Agent 2", "review_started", "Paper 2"),
        ]
        history.accepted_review_claims = [(2, 0.02), (3, 0.04)]
        history.completed_reviews = [
            (5, "Agent 1", "Paper 1", 5.0, GOOD_FAITH_REVIEW),
            (6, "Agent 2", "Paper 2", 1.0, BAD_FAITH_REVIEW),
        ]

        prices = visualize._paper_claim_share_prices(history)
        self.assertAlmostEqual(prices["Paper 1"], 0.02)
        self.assertAlmostEqual(prices["Paper 2"], 0.04)
        points = visualize._review_faith_share_price_points(history)
        self.assertEqual(len(points), 2)
        self.assertTrue(visualize._has_review_faith_by_share_price(history))

    @unittest.skipUnless(_HAS_MPL, "matplotlib not installed")
    def test_visualize_writes_pngs(self):
        import visualize

        Agent.all_papers = []
        author = ScriptAgent("author")
        paper = Paper(author=author, current_ac=5.0)
        history = History()
        env = Environment(agents=[author], papers=[paper], history=history)
        env.run(2)

        outdir = tempfile.mkdtemp()
        paths = visualize.plot_all(history, outdir)
        for path in paths.values():
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
