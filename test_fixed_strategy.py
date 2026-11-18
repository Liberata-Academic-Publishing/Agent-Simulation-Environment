"""Integration checks for the standalone, non-training simulation."""
from __future__ import annotations

import math
import random
import unittest

from Agent import Agent
from Paper import Paper
from talent_agents import TalentAgent
from run_talent_comparison import Actions, mechanism_results, run


class FixedStrategyTests(unittest.TestCase):
    def setUp(self):
        self.papers, self.agents = Agent.all_papers, Agent.all_agents
        self.state = random.getstate()
        self.forecasts = (Paper.citation_income_per_weight, Paper.mean_accrual_rate,
                          Paper.citation_quality_weight, Paper.citation_log_decay_ratio)
        Agent.all_papers = []

    def tearDown(self):
        Agent.all_papers, Agent.all_agents = self.papers, self.agents
        random.setstate(self.state)
        (Paper.citation_income_per_weight, Paper.mean_accrual_rate,
         Paper.citation_quality_weight, Paper.citation_log_decay_ratio) = self.forecasts

    def test_completed_review_persists_in_citation_weight(self):
        author = TalentAgent(1, 1, 'author')
        reviewer = TalentAgent(1, 1, 'reviewer')
        paper = Paper(author, quality=1, market_listed=True)
        paper.price_table[reviewer] = 0.1
        paper.start_review(reviewer, current_timestep=1)
        paper.finish_review(reviewer, effort=10, current_timestep=10)
        epsilon = paper.review_records[-1]['epsilon']
        self.assertGreater(epsilon, 0)
        self.assertAlmostEqual(paper.review_bump_at(100), epsilon)
        self.assertAlmostEqual(paper.citation_log_weight(100, 1, 0),
                               paper.quality + math.log1p(epsilon))

    def test_live_run_connects_all_three_mechanisms(self):
        actions = Actions()
        run(11, 250, actions, count_per_group=2)
        result = mechanism_results(actions.environment)
        self.assertGreater(result['total_citations'], 0)
        self.assertAlmostEqual(result['total_agent_ac'], result['total_citation_ac'])
        self.assertGreater(result['market']['good_reviews'], 0)
        self.assertEqual(result['market']['bad_reviews'], 0)
        self.assertIsNotNone(result['market']['mean_wait'])
        for agent in result['agents']:
            self.assertGreater(agent['publications'], 0)
            self.assertGreater(agent['experience_multiplier'], 1)
            self.assertGreater(agent['effective_quality_talent'], agent['quality_talent'])
            self.assertGreater(agent['effective_rate_talent'], agent['rate_talent'])

    def test_short_run_without_publications(self):
        actions = Actions()
        rows = run(11, 1, actions, count_per_group=1)
        result = mechanism_results(actions.environment)
        self.assertEqual(len(rows), 4)
        self.assertEqual(result['total_papers'], 0)
        self.assertIsNone(result['market']['mean_wait'])

    def test_future_training_environment_without_training(self):
        from train_rl import build_env, parse_args
        from QLearningAgent import make_backend
        args = parse_args(['--shared-talents'])
        self.assertTrue(args.shared_talents)
        env, agents, _ = build_env(backend=make_backend('tabular'), epsilon=0,
            learning=False, num_rl=4, num_heuristic=0, horizon=30, seed=11,
            shared_talents=True)
        self.assertEqual(len({(a.quality_talent, a.rate_talent) for a in agents}), 4)
        self.assertTrue(all(a.talent_sampling_enabled for a in agents))
        self.assertEqual(env.papers, [])
        self.assertEqual(env.paper_effort_mode, 'fixed')
        self.assertTrue(env.citations_enabled)

    def test_standard_history_preserves_four_cohorts(self):
        from History import History
        from Environment import Environment
        from talent_agents import build_talent_cohorts
        from run_simulation import parse_args
        args = parse_args(['--fixed-strategy', '--agents-per-group', '1'])
        self.assertTrue(args.fixed_strategy)
        groups = build_talent_cohorts(1)
        history = History()
        env = Environment(agents=[a for cohort in groups.values() for a in cohort],
                          papers=[], history=history, use_merit_market_clearing=False)
        env.run(1)
        self.assertEqual(set(history.to_dict()['agent_group_summary']), set(groups))
        self.assertEqual(set(history.to_gallery_dict()['agent_groups'].values()), set(groups))
