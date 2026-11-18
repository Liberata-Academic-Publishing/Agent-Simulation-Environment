"""Contract, bounded search, and labour invariants for the AC/price market."""
from __future__ import annotations

from dataclasses import replace
import math
import random
import unittest
from unittest.mock import patch

from Agent import Agent
from config import SIM
from Environment import Environment
from History import History
from Paper import Paper
from RandomAgent import RandomAgent
from HeuristicAgent import HeuristicAgent
from ReviewMarketplace import ReviewMarketplace
from run_simulation import build_simulation, parse_args


class ReviewMarketTest(unittest.TestCase):
    def setUp(self):
        previous_papers = Agent.all_papers
        previous_agents = Agent.all_agents
        previous_random = random.getstate()
        self.addCleanup(setattr, Agent, "all_papers", previous_papers)
        self.addCleanup(setattr, Agent, "all_agents", previous_agents)
        self.addCleanup(random.setstate, previous_random)
        Agent.all_papers = []
        Agent.all_agents = []
        random.seed(11)

    def agent(self, ac=0, ask=0.005, talent=1):
        a = RandomAgent(talent, academic_capital=ac, claim_probability=1)
        a.review_ask = ask
        return a

    def env(self, agents, papers, **config):
        Agent.all_papers = list(papers)
        return Environment(agents=agents, papers=Agent.all_papers, history=History(),
            market_mechanism='sampled_bidding', market_seed=11,
            market_config=replace(SIM, market_candidate_limit=100, **config))

    def test_selects_ac_per_price_not_talent_price_or_ac_alone(self):
        author = self.agent()
        cheap = self.agent(80, .004, talent=100)
        winner = self.agent(120, .005, talent=.1)
        rich = self.agent(140, .008)
        paper = Paper(author, market_listed=True)
        env = self.env([author, cheap, winner, rich], [paper])
        env.run_timestep()
        self.assertIs(paper.review_in_progress_by, winner)
        self.assertEqual(paper.agreed_review_share, .005)
        self.assertEqual(winner.active_review_effort, winner.review_effort_delta())

    def test_market_alias_does_not_reenable_legacy_prices(self):
        agents = [self.agent(), self.agent(10)]
        env = self.env(agents, [Paper(agents[0], market_listed=True)])
        with patch.object(env, '_update_market_prices', side_effect=AssertionError('legacy prices')):
            env.market()

    def test_budget_minimum_self_and_withholding(self):
        author = self.agent(1000)
        a = self.agent(10, .06)
        b = self.agent(2, .001)
        author.review_min_ac = 5
        paper = Paper(author, market_listed=True)
        env = self.env([author, a, b], [paper])
        env.run_timestep()
        self.assertFalse(paper.review_claimed)
        author.request_reviews_enabled = False
        env.run_timestep()
        self.assertFalse(paper.market_listed)
        author.request_reviews_enabled = True
        author.review_min_ac = 0
        env.run_timestep()
        self.assertTrue(paper.review_claimed)

    def test_bounded_candidates_and_no_legacy_scans(self):
        agents = [self.agent(i) for i in range(30)]
        papers = [Paper(agents[0], market_listed=True) for _ in range(8)]
        Agent.all_papers = papers
        env = Environment(agents=agents, papers=papers, market_mechanism='sampled_bidding',
            market_config=replace(SIM, market_candidate_limit=3))
        with patch.object(env, '_update_market_prices', side_effect=AssertionError('legacy prices')), \
             patch.object(env, '_clear_review_market', side_effect=AssertionError('legacy match')), \
             patch.object(RandomAgent, 'choose_continuous_action', side_effect=AssertionError('global scan')):
            env.run_timestep()
        self.assertLessEqual(env.review_market.metrics['candidates'], 3 * len(papers))
        reviewers = [p.review_in_progress_by for p in papers if p.review_claimed]
        self.assertEqual(len(reviewers), len(set(reviewers)))

    def test_locked_share_and_conservation(self):
        author, reviewer = self.agent(), self.agent(10)
        p = Paper(author, market_listed=True)
        env = self.env([author, reviewer], [p])
        env.run_timestep()
        reviewer.review_ask = .04
        p.finish_review(reviewer, 5, current_timestep=2)
        self.assertEqual(p.share_distribution[reviewer], .005)
        self.assertAlmostEqual(sum(p.share_distribution.values()), 1)

    def test_heuristic_can_decline(self):
        author = self.agent()
        reviewer = HeuristicAgent(1)
        p = Paper(author, market_listed=True, current_ac=0, accrual_rate=.001)
        env = self.env([author, reviewer], [p])
        with patch.object(reviewer, '_score_claim', return_value=0):
            env.run_timestep()
        self.assertFalse(p.review_claimed)
        self.assertEqual(env.review_market.metrics['declined'], 1)

    def test_cold_start_and_seed_reproducibility(self):
        def run(seed):
            agents = [self.agent(0) for _ in range(8)]
            paper = Paper(agents[0], market_listed=True)
            env = self.env(agents, [paper])
            env.review_market.rng.seed(seed)
            env.run_timestep()
            return env.history.market_trades[0]['reviewer_id']
        self.assertEqual(run(11), run(11))
        self.assertGreater(len({run(i) for i in range(10)}), 1)

    def test_discrete_first_effort_once_and_busy_not_reassigned(self):
        agents = [self.agent(i) for i in range(3)]
        papers = [Paper(agents[0], market_listed=True) for _ in range(2)]
        env = self.env(agents, papers)
        env.review_paradigm = 'discrete'
        for a in agents:
            a.configure_review_paradigm('discrete')
            a.good_faith_probability = 1
        env.run_timestep()
        for a in agents[1:]:
            self.assertEqual(a.active_review_effort, a.review_effort_delta())
        env.run_timestep()
        self.assertEqual(env.review_market.metrics['offers'], 1)

    def test_history_roundtrip_and_metadata(self):
        agents = [self.agent(), self.agent(10)]
        env = self.env(agents, [Paper(agents[0], market_listed=True)])
        env.run(4)
        data = env.history.to_dict()
        restored = History.from_dict(data).to_dict()
        self.assertEqual(data['market_trades'], restored['market_trades'])
        self.assertEqual(data['market_metadata'], restored['market_metadata'])
        self.assertEqual(len(data['scalars']['market_matches']), 4)

    def test_tick_rounding_and_invalid_config(self):
        author, reviewer = self.agent(), self.agent(10, .00501)
        p = Paper(author, market_listed=True)
        env = self.env([author, reviewer], [p])
        env.run_timestep()
        self.assertAlmostEqual(p.agreed_review_share, .0051)
        for kwargs in ({'market_share_tick': 0}, {'market_ac_bin_edges': (2, 1)},
                       {'market_candidate_limit': 0}, {'market_min_reviewer_ac': math.nan}):
            with self.assertRaises(ValueError):
                ReviewMarketplace(replace(SIM, **kwargs), [], 1)

    def test_reject_invalid_contract_without_mutating_paper(self):
        author, reviewer = self.agent(), self.agent(10)
        p = Paper(author, market_listed=True, max_reviewer_share=.01)
        with self.assertRaises(ValueError):
            p.start_review(reviewer, agreed_share=.02)
        self.assertFalse(p.review_claimed)
        self.assertIsNone(p.review_in_progress_by)
        reviewer.claim_review(p, agreed_share=.005)
        other = Paper(author, market_listed=True)
        with self.assertRaises(ValueError):
            reviewer.claim_review(other, agreed_share=.005)
        self.assertIs(reviewer.active_review_paper, p)
        self.assertFalse(p.reviewed)
        self.assertFalse(other.review_claimed)

    def test_longer_mixed_strategies_keep_working(self):
        from compare_review_markets import run_one
        for mode in ('legacy', 'sampled'):
            result = run_one(mode, 11, 80, 8)
            self.assertEqual(result['invariant_checks'], 'passed')
            self.assertGreater(result['new_papers'], 0)
            self.assertGreater(result['completed_reviews'], 0)
            self.assertAlmostEqual(result['writing_fraction'] + result['review_fraction'], 1)

    def test_main_builder_and_cli(self):
        args = parse_args(['--market-mechanism', 'sampled_bidding', '--rl-agents', '0'])
        self.assertEqual(args.market_mechanism, 'sampled_bidding')
        env = build_simulation(History(), num_agents=2, rl_agents=0, dqn_agents=0,
            low_talent_rl_agents=0, random_agents=2, probabilistic_agents=0,
            market_mechanism='sampled_bidding', seed=19)
        env.run(5)
        self.assertEqual(env.history.market_metadata['mechanism'], 'sampled_bidding')


if __name__ == '__main__':
    unittest.main()
