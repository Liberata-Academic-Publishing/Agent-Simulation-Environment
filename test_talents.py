from __future__ import annotations

import random
import statistics
import unittest
from unittest.mock import patch

from Agent import Agent
from Environment import Environment
from Paper import Paper, GOOD_FAITH_REVIEW, fixed_review_effort
from config import SIM
from talent_agents import TalentAgent, build_talent_cohorts


class TalentTests(unittest.TestCase):
    def setUp(self):
        self.rng_state = random.getstate()
        self.papers = Agent.all_papers
        Agent.all_papers = []
        random.seed(7)

    def tearDown(self):
        random.setstate(self.rng_state)
        Agent.all_papers = self.papers

    def test_four_equal_groups(self):
        groups = build_talent_cohorts()
        self.assertEqual([len(g) for g in groups.values()], [20] * 4)
        self.assertEqual({(g[0].quality_talent, g[0].rate_talent) for g in groups.values()},
                         {(q, r) for q in (SIM.talent_low, SIM.talent_high) for r in (SIM.talent_low, SIM.talent_high)})

    def test_higher_talents_shift_both_distributions(self):
        low, high = TalentAgent(0.6, 0.6, "low"), TalentAgent(1.4, 1.4, "high")
        for method in ("_sample_quality", "writing_effort_delta", "review_effort_delta"):
            random.seed(7)
            lows = [getattr(low, method)() for _ in range(1000)]
            random.seed(7)
            highs = [getattr(high, method)() for _ in range(1000)]
            self.assertGreater(statistics.mean(highs), statistics.mean(lows))
            self.assertGreater(min(lows), 0)
            self.assertGreater(statistics.pstdev(highs), 0)

    def test_quality_draw_once_per_manuscript_rate_each_turn(self):
        agent = TalentAgent(1, 1, "author")
        with patch.object(agent, "_sample_quality", return_value=1.2) as quality:
            with patch.object(agent, "_sample_rate", return_value=1) as rate:
                agent.add_research_effort()
                agent.add_research_effort()
                paper = agent.finish_research_paper()
        self.assertEqual(quality.call_count, 1)
        self.assertEqual(rate.call_count, 2)
        self.assertEqual(paper.paper_quality, 1.2)
        paper.quality = 1.5
        self.assertEqual(paper.paper_quality, 1.5)

    def test_review_quality_changes_improvement_not_locked_share(self):
        author = TalentAgent(1, 1, "author")
        reviewer = TalentAgent(1, 1, "reviewer")
        improvements = []
        for sampled_quality in (0.6, 1.4):
            paper = Paper(author, quality=1, market_listed=True)
            paper.price_table[reviewer] = 0.1
            paper.start_review(reviewer)
            with patch.object(reviewer, "sample_review_quality", return_value=sampled_quality) as draw:
                share = paper.finish_review(reviewer, max(100, fixed_review_effort(GOOD_FAITH_REVIEW)), GOOD_FAITH_REVIEW)
                paper.finish_review(reviewer, 100)
            self.assertEqual(draw.call_count, 1)
            self.assertEqual(share, 0.1)
            self.assertGreater(paper.paper_quality, 1)
            improvements.append(paper.review_records[-1]["review_quality_delta"])
        self.assertGreater(improvements[1], improvements[0])

    def test_invalid_talent_is_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                TalentAgent(value, 1, "invalid")

    def test_cohorts_publish_and_review_in_both_paradigms(self):
        roster = Agent.all_agents
        try:
            for paradigm in ("continuous", "discrete"):
                with self.subTest(paradigm=paradigm):
                    Agent.all_papers = []
                    groups = build_talent_cohorts(2)
                    agents = [agent for group in groups.values() for agent in group]
                    env = Environment(agents=agents, papers=[], review_paradigm=paradigm,
                                      continuous_publishing="threshold", continuous_paper_timesteps=5,
                                      use_merit_market_clearing=False)
                    env.run(100)
                    self.assertTrue(Agent.all_papers)
                    self.assertGreater(sum(a.completed_review_count for a in agents), 0)
                    for paper in Agent.all_papers:
                        self.assertGreater(paper.paper_quality, 0)
                        self.assertAlmostEqual(sum(paper.share_distribution.values()), 1)
        finally:
            Agent.all_agents = roster


if __name__ == "__main__":
    unittest.main()
