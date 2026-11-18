import random
import unittest
from dataclasses import replace
from unittest.mock import patch

from Agent import Agent
from Paper import Paper
from config import SIM
from talent_agents import TalentAgent


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.papers = Agent.all_papers
        self.roster = Agent.all_agents.copy()
        self.rng = random.getstate()
        Agent.all_papers = []

    def tearDown(self):
        Agent.all_papers = self.papers
        Agent.all_agents = self.roster
        random.setstate(self.rng)

    def test_counts_and_sampling(self):
        agent = TalentAgent(0.6, 1.4, "author")
        other = TalentAgent(1, 1, "other")
        self.assertEqual(agent.experience_multiplier, 1)
        Agent.all_papers = [Paper(agent, quality=1) for _ in range(9)]
        reviewed = Paper(other, quality=1)
        reviewed.share_distribution[agent] = 0.1
        Agent.all_papers.append(reviewed)
        self.assertEqual(agent.publication_count, 9)
        self.assertAlmostEqual(agent.experience_multiplier, 1.15)
        with patch("Agent.random.gauss", return_value=1) as draw:
            agent.sample_review_quality()
            self.assertAlmostEqual(draw.call_args.args[0], 0.6 * 1.15)
            self.assertEqual(draw.call_args.args[1], SIM.quality_sigma)
            for sample in (agent.writing_effort_delta, agent.review_effort_delta):
                sample()
                self.assertAlmostEqual(draw.call_args.args[0], 1.4 * 1.15)
                self.assertEqual(draw.call_args.args[1], SIM.talent_rate_sigma)
        for publish in (agent.publish_paper, agent.finish_research_paper):
            before = agent.publication_count
            agent.next_paper_quality = 1.23
            paper = publish()
            self.assertEqual(agent.publication_count, before + 1)
            self.assertEqual(paper.paper_quality, 1.23)
        self.assertEqual((agent.quality_talent, agent.rate_talent), (0.6, 1.4))

    def test_disabled_and_validation(self):
        agent = TalentAgent(1, 1, "author")
        Agent.all_papers = [Paper(agent, quality=1) for _ in range(9)]
        with patch("Agent.SIM", replace(SIM, experience_alpha=0)):
            for sample in (agent._sample_quality, agent._sample_rate):
                state = random.getstate()
                expected = random.gauss(1, 0.2)
                random.setstate(state)
                self.assertEqual(sample(), max(0.1, expected))
            self.assertEqual(agent.experience_multiplier, 1)
        for alpha in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(SIM, experience_alpha=alpha)
        for h in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(SIM, experience_h=h)
