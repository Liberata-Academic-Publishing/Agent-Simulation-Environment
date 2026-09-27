"""Tests for the citation-network mechanism (``citation_count``/``citation_accrued``).

Covers the unit-level behavior of ``Paper.generate_citations`` (bounded,
distinct, quality/recency-weighted sampling; conserved citation AC) and its
integration into ``Environment`` (the only source of paper AC; bounded
per-paper reference count as the corpus grows; no self- or future-citation;
peer-reviewed papers are cited more; per-paper forecasts add up to the
citation AC handed out).

Run as a test module:
    python test_citation_network.py
    python -m unittest test_citation_network

Running it directly also prints a short demonstration summary from a live
simulated corpus, so the mechanism's effect on ``citation_count`` and
``citation_accrued`` is visible, not just asserted.
"""

from __future__ import annotations

import random
import unittest

from Agent import Agent
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper, generate_citations


def _author() -> HeuristicAgent:
    return HeuristicAgent(intrinsic_talent=1.0)


class PaperCitationFieldsTest(unittest.TestCase):
    """New Paper instances start with a clean, well-typed citation ledger."""

    def test_default_citation_fields(self):
        paper = Paper(author=_author())
        self.assertEqual(paper.citation_count, 0)
        self.assertEqual(paper.citation_accrued, 0.0)
        self.assertEqual(paper.references, [])
        self.assertFalse(paper.citations_generated)
        self.assertEqual(paper.publish_timestep, 0)  # unset -> pre-existing corpus at t=0

    def test_publish_timestep_passthrough(self):
        paper = Paper(author=_author(), publish_timestep=42)
        self.assertEqual(paper.publish_timestep, 42)


class GenerateCitationsTest(unittest.TestCase):
    """Unit tests for the pure ``generate_citations`` sampling function."""

    def setUp(self):
        self.author = _author()

    def _prior_papers(self, n: int, quality: float = 1.0, publish_timestep: int = 0):
        return [
            Paper(author=self.author, quality=quality, publish_timestep=publish_timestep)
            for _ in range(n)
        ]

    def test_no_eligible_papers_is_a_noop(self):
        new_paper = Paper(author=self.author, publish_timestep=5)
        selected = generate_citations(new_paper, [], current_timestep=5)
        self.assertEqual(selected, [])
        self.assertEqual(new_paper.references, [])
        self.assertTrue(new_paper.citations_generated)

    def test_reference_count_capped_by_small_corpus(self):
        new_paper = Paper(author=self.author, publish_timestep=10)
        prior = self._prior_papers(3)
        selected = generate_citations(
            new_paper, prior, current_timestep=10,
            reference_count=5, rng=random.Random(1),
        )
        self.assertEqual(len(selected), 3)  # capped at len(eligible), not reference_count

    def test_reference_count_does_not_grow_with_corpus_size(self):
        rng = random.Random(2)
        for corpus_size in (5, 50, 500):
            prior = self._prior_papers(corpus_size)
            new_paper = Paper(author=self.author, publish_timestep=1)
            selected = generate_citations(
                new_paper, prior, current_timestep=1,
                reference_count=5, rng=rng,
            )
            self.assertEqual(len(selected), 5)

    def test_selected_targets_are_distinct(self):
        prior = self._prior_papers(6)
        new_paper = Paper(author=self.author, publish_timestep=1)
        selected = generate_citations(
            new_paper, prior, current_timestep=1,
            reference_count=4, rng=random.Random(3),
        )
        self.assertEqual(len(selected), len(set(id(p) for p in selected)))

    def test_citation_ac_conserved_and_split_evenly(self):
        prior = self._prior_papers(4)
        new_paper = Paper(author=self.author, publish_timestep=10)
        selected = generate_citations(
            new_paper, prior, current_timestep=10,
            reference_count=3, ac_per_new_paper=1.0, rng=random.Random(4),
        )
        self.assertEqual(len(selected), 3)

        total_ac = sum(p.citation_accrued for p in prior)
        self.assertAlmostEqual(total_ac, 1.0, places=9)  # conserved: sums to 1.0

        for paper in selected:
            self.assertAlmostEqual(paper.citation_accrued, 1.0 / 3, places=9)
            self.assertEqual(paper.citation_count, 1)
        for paper in prior:
            if paper not in selected:
                self.assertEqual(paper.citation_count, 0)
                self.assertEqual(paper.citation_accrued, 0.0)

    def test_citation_selection_favors_higher_quality(self):
        high_quality = Paper(author=self.author, quality=2.0, publish_timestep=0)
        low_quality = Paper(author=self.author, quality=0.2, publish_timestep=0)
        rng = random.Random(1234)
        trials = 2000
        for _ in range(trials):
            new_paper = Paper(author=self.author, publish_timestep=50)
            generate_citations(
                new_paper, [high_quality, low_quality], current_timestep=50,
                reference_count=1, quality_weight=2.0, age_decay_ratio=1.0, rng=rng,
            )
        # softmax(2*2.0) vs softmax(2*0.2) predicts ~97% of picks go to high_quality
        self.assertGreater(high_quality.citation_count, low_quality.citation_count)
        self.assertGreater(high_quality.citation_count, trials * 0.8)

    def test_citation_selection_favors_recency(self):
        old_paper = Paper(author=self.author, quality=1.0, publish_timestep=0)
        recent_paper = Paper(author=self.author, quality=1.0, publish_timestep=90)
        rng = random.Random(99)
        trials = 2000
        for _ in range(trials):
            new_paper = Paper(author=self.author, publish_timestep=100)
            generate_citations(
                new_paper, [old_paper, recent_paper], current_timestep=100,
                reference_count=1, quality_weight=0.0, age_decay_ratio=0.95, rng=rng,
            )
        # ages are 100 vs 10; softmax skew (~99%) favors the more recent paper
        self.assertGreater(recent_paper.citation_count, old_paper.citation_count)
        self.assertGreater(recent_paper.citation_count, trials * 0.8)

    def test_peer_review_boosts_citation_weight(self):
        reviewed = Paper(author=self.author, quality=1.0, publish_timestep=0)
        unreviewed = Paper(author=self.author, quality=1.0, publish_timestep=0)
        reviewed.reviewed = True
        reviewed.review_bump_epsilon = 3.0  # weight x4 -> ~80% of picks
        rng = random.Random(5)
        trials = 2000
        for _ in range(trials):
            new_paper = Paper(author=self.author, publish_timestep=10)
            generate_citations(
                new_paper, [reviewed, unreviewed], current_timestep=10,
                reference_count=1, quality_weight=1.0, age_decay_ratio=1.0, rng=rng,
            )
        self.assertGreater(reviewed.citation_count, trials * 0.7)

    def test_citation_ac_is_added_to_current_ac(self):
        cited = Paper(author=self.author, quality=1.0, current_ac=2.0, publish_timestep=0)
        new_paper = Paper(author=self.author, publish_timestep=5)
        generate_citations(
            new_paper, [cited], current_timestep=5, reference_count=1,
            ac_per_new_paper=1.5, rng=random.Random(0),
        )
        self.assertAlmostEqual(cited.citation_accrued, 1.5)
        self.assertAlmostEqual(cited.current_ac, 3.5)


class CitationEnvironmentIntegrationTest(unittest.TestCase):
    """Integration through ``Environment.run_timestep`` / ``_process_citations``."""

    def _build_env(
        self,
        citations_enabled: bool,
        num_agents: int = 6,
        reference_count: int = 3,
        seed: int = 0,
        history: History | None = None,
    ) -> Environment:
        random.seed(seed)
        Agent.all_papers = []
        agents = [HeuristicAgent(intrinsic_talent=1.0, name=f"A{i}") for i in range(num_agents)]
        for agent in agents:
            for _ in range(2):
                Agent.all_papers.append(
                    Paper(author=agent, quality=random.uniform(0.5, 1.5), current_ac=1.0)
                )
        return Environment(
            agents=agents,
            papers=Agent.all_papers,
            citations_enabled=citations_enabled,
            citation_reference_count=reference_count,
            history=history,
        )

    def test_citations_disabled_creates_no_ac(self):
        env = self._build_env(citations_enabled=False)
        for _ in range(100):
            env.run_timestep()
        for paper in env.papers:
            self.assertEqual(paper.current_ac, 0.0 if paper.publish_timestep else 1.0)
            self.assertEqual(paper.citation_count, 0)
            self.assertEqual(paper.citation_accrued, 0.0)
            self.assertEqual(paper.references, [])
            self.assertFalse(paper.citations_generated)

    def test_citations_enabled_builds_a_bounded_conserved_network(self):
        env = self._build_env(citations_enabled=True, reference_count=3)
        for _ in range(150):
            env.run_timestep()

        papers = env.papers
        processed = [p for p in papers if p.citations_generated]
        self.assertTrue(processed, "expected at least one paper to be processed")

        for paper in papers:
            self.assertLessEqual(len(paper.references), 3)  # bounded reference count
            self.assertNotIn(paper, paper.references)        # no self-citation
            for ref in paper.references:
                self.assertLess(ref.publish_timestep, paper.publish_timestep)  # no future citation

        papers_with_refs = [p for p in papers if p.references]
        total_citation_ac = sum(p.citation_accrued for p in papers)
        expected_ac = len(papers_with_refs) * env.citation_ac_per_new_paper
        self.assertAlmostEqual(total_citation_ac, expected_ac, places=6)  # AC conserved in aggregate

        total_citations = sum(p.citation_count for p in papers)
        expected_citations = sum(len(p.references) for p in papers)
        self.assertEqual(total_citations, expected_citations)

    def test_paper_ac_comes_only_from_citations(self):
        env = self._build_env(citations_enabled=True, reference_count=3)
        for _ in range(150):
            env.run_timestep()
        for paper in env.papers:
            starting_ac = 0.0 if paper.publish_timestep else 1.0
            self.assertAlmostEqual(paper.current_ac, starting_ac + paper.citation_accrued)

    def test_forecast_rates_sum_to_citation_income(self):
        env = self._build_env(citations_enabled=True, reference_count=3)
        for _ in range(150):
            env.run_timestep()
        env._update_citation_forecasts()
        self.assertAlmostEqual(
            sum(p.accrual_rate for p in env.papers), env.citation_income_per_timestep
        )

    def test_reference_bound_holds_as_corpus_keeps_growing(self):
        env = self._build_env(citations_enabled=True, reference_count=4, num_agents=8)
        for _ in range(3):
            for _ in range(80):
                env.run_timestep()
            ref_lengths = [len(p.references) for p in env.papers if p.citations_generated]
            if ref_lengths:
                self.assertLessEqual(max(ref_lengths), 4)

    def test_history_reports_citation_metrics(self):
        history = History()
        env = self._build_env(citations_enabled=True, reference_count=3, history=history)
        for _ in range(100):
            env.run_timestep()

        expected_total_citations = sum(p.citation_count for p in env.papers)
        expected_total_ac = sum(p.citation_accrued for p in env.papers)
        self.assertEqual(history.scalars["total_citations"][-1], expected_total_citations)
        self.assertAlmostEqual(
            history.scalars["total_citation_ac"][-1], expected_total_ac, places=6
        )
        self.assertGreater(expected_total_citations, 0)


def _print_demo_summary() -> None:
    """Small standalone model: run a corpus with citations enabled and report."""
    random.seed(7)
    Agent.all_papers = []
    agents = [HeuristicAgent(intrinsic_talent=1.0, name=f"Agent {i}") for i in range(10)]
    for agent in agents:
        for _ in range(3):
            Agent.all_papers.append(
                Paper(author=agent, quality=random.uniform(0.5, 1.5), current_ac=1.0)
            )

    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        citations_enabled=True,
        citation_reference_count=5,
    )
    for _ in range(400):
        env.run_timestep()

    papers = env.papers
    cited = sorted(papers, key=lambda p: p.citation_count, reverse=True)
    total_citations = sum(p.citation_count for p in papers)
    total_ac = sum(p.citation_accrued for p in papers)

    print("\n--- Citation network demo ---")
    print(f"papers in corpus:        {len(papers)}")
    print(f"total citations:         {total_citations}")
    print(f"total citation AC:       {total_ac:.4f}")
    print("top 5 most-cited papers (quality, citation_count, citation_accrued):")
    for paper in cited[:5]:
        print(
            f"  quality={paper.quality:.2f}  "
            f"citation_count={paper.citation_count}  "
            f"citation_accrued={paper.citation_accrued:.4f}"
        )


if __name__ == "__main__":
    unittest.main(exit=False)
    _print_demo_summary()
