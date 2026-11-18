"""Replay saved experience experiments to measure listing-to-claim waits."""
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch

from config import SIM
from run_talent_comparison import Actions, run


class MarketHistory(Actions):
    def record_step(self, env):
        self.env = env


def main():
    root = Path('experiments/experience_calibration')
    results = {}
    for label, alpha, folder in (
        ('baseline', 0, 'seed_11_steps_1000'),
        ('experience', 0.30, 'experience_a030_h9_seed11'),
    ):
        history = MarketHistory()
        with patch('Agent.SIM', replace(SIM, experience_alpha=alpha)):
            rows = run(11, 1000, actions=history)
        saved = json.loads((root / folder / 'baseline.json').read_text())
        assert rows == saved['groups'], 'Replay did not reproduce saved results'
        papers = history.env.papers
        claimed = [p for p in papers if p.claimed_timestep is not None]
        waits = [p.claimed_timestep - p.listed_timestep for p in claimed]
        pending = [p for p in papers if p.listed_timestep is not None
                   and p.claimed_timestep is None]
        good = sum(r['good_reviews'] for r in rows)
        bad = sum(r['bad_reviews'] for r in rows)
        results[label] = dict(
            reproduced_saved_results=True,
            claimed_papers=len(claimed), mean_wait=mean(waits),
            median_wait=median(waits), max_wait=max(waits),
            same_step_claims=sum(w == 0 for w in waits),
            unclaimed_listed_papers=len(pending),
            unclaimed_wait_ages=[1000-p.listed_timestep for p in pending],
            not_yet_listed=sum(p.listed_timestep is None for p in papers),
            good_reviews=good, bad_reviews=bad,
            paper_waits=[dict(listed=p.listed_timestep, claimed=p.claimed_timestep)
                         for p in papers],
        )
        print(label, {k:v for k,v in results[label].items() if k != 'paper_waits'}, flush=True)
    (root / 'market_wait_comparison.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
