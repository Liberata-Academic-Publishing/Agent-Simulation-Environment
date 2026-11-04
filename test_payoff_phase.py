"""Tests for the payoff-rate phase diagram helpers and a short probe."""

from __future__ import annotations

import unittest

from run_payoff_phase import (
    Cell,
    choice_is_optimal,
    gap_fraction,
    measure_rates,
    optimal_activity,
    payoff_rate,
    select_cases,
    verdict,
)


class PayoffPhaseHelpersTest(unittest.TestCase):
    def test_payoff_rate_is_ac_gained_per_step(self):
        self.assertAlmostEqual(payoff_rate(10.0, 6.0, 4), 1.0)
        self.assertIsNone(payoff_rate(10.0, 6.0, 0))

    def test_optimal_activity_follows_the_higher_rate(self):
        self.assertEqual(optimal_activity(1.0, 2.0), "review")
        self.assertEqual(optimal_activity(2.0, 1.0), "write")
        self.assertEqual(optimal_activity(1.0, 1.0), "tie")

    def test_choice_matches_only_the_higher_rate_route(self):
        self.assertTrue(choice_is_optimal("review", claimed=True))
        self.assertFalse(choice_is_optimal("review", claimed=False))
        self.assertTrue(choice_is_optimal("write", claimed=False))
        self.assertFalse(choice_is_optimal("write", claimed=True))

    def test_gap_fraction_is_positive_when_review_pays_more(self):
        self.assertGreater(gap_fraction(1.0, 2.0), 0.0)
        self.assertLess(gap_fraction(2.0, 1.0), 0.0)

    def test_verdict_thresholds(self):
        self.assertEqual(verdict(0.85, 40), "optimal")
        self.assertEqual(verdict(0.10, 40), "suboptimal")
        self.assertEqual(verdict(0.50, 40), "mixed")
        self.assertEqual(verdict(1.0, 5), "unobserved")
        self.assertEqual(verdict(None, 100), "unobserved")

    def test_select_cases_takes_two_on_each_side(self):
        rows = []
        for index, gap in enumerate((0.8, 0.2, -0.2, -0.9)):
            writing = 1.0
            review = writing * (1.0 + gap) if gap > 0 else writing * (1.0 + gap)
            # Reconstruct rates whose gap_fraction matches the requested gap
            # when the larger absolute rate is the writing rate of 1.
            if gap >= 0:
                writing_rate, review_rate = 1.0, 1.0 + gap
            else:
                writing_rate, review_rate = 1.0, 1.0 + gap
            rows.append({
                "paper_timesteps": 10.0 + index,
                "reviewer_surplus_share": 0.1 * (index + 1),
                "max_review_accrual_bump": 0.6,
                "writing_effort_per_timestep": 1.0,
                "review_effort_per_timestep": 1.0,
                "review_target_effort": 3.0,
                "writing_steps": 100,
                "review_steps": 100,
                "writing_rate": writing_rate,
                "review_rate": review_rate,
                "gap_fraction": gap if abs(gap) <= 1 else gap / max(abs(writing_rate), abs(review_rate)),
            })
        # Recompute gap so the selector's threshold sees the real fraction.
        for row in rows:
            row["gap_fraction"] = gap_fraction(row["writing_rate"], row["review_rate"])
        cases = select_cases(rows, min_steps=30)
        self.assertIsNotNone(cases)
        self.assertEqual(len(cases), 4)
        sides = [case["optimal_activity"] for case in cases]
        self.assertEqual(sides.count("review"), 2)
        self.assertEqual(sides.count("write"), 2)

    def test_select_cases_rejects_a_one_sided_grid(self):
        row = {
            "paper_timesteps": 30.0,
            "reviewer_surplus_share": 0.5,
            "max_review_accrual_bump": 0.6,
            "writing_effort_per_timestep": 1.0,
            "review_effort_per_timestep": 1.0,
            "review_target_effort": 3.0,
            "writing_steps": 100,
            "review_steps": 100,
            "writing_rate": 1.0,
            "review_rate": 2.0,
            "gap_fraction": 0.5,
        }
        self.assertIsNone(select_cases([row]))


class PayoffProbeTest(unittest.TestCase):
    def test_short_probe_records_both_activities(self):
        row = measure_rates(
            Cell(paper_timesteps=8.0, reviewer_surplus_share=0.5),
            steps=40,
            writers=3,
            reviewers=3,
            seed=1,
        )
        self.assertGreater(row["writing_steps"], 0)
        self.assertGreater(row["review_steps"], 0)
        self.assertIsNotNone(row["writing_rate"])
        self.assertIsNotNone(row["review_rate"])


if __name__ == "__main__":
    unittest.main()
