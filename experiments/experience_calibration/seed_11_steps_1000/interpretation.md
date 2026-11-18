# Experience parameter calibration

Baseline branch: `codex/two-talent-agents`.
Commit: `ab8585039ceb2c406deb1817d8e1708fc0abd9ae` plus the working-tree
changes and source hashes recorded in `baseline.json`.

Command: `python calibrate_experience_baseline.py`.
This uses the existing `run_talent_comparison.run` without changing mechanics.
No experience modifier is enabled and no RL training is involved.

## Conditions

- One run, seed 11, 1000 timesteps, 80 agents, 20 per cohort.
- Quality and rate means independently cross 0.6 and 1.4; both sigmas 0.2.
- No initial papers; publication requires 50 units of writing effort.
- Continuous paradigm, fixed paper effort, no merit market clearing.
- Existing fixed strategy: publish, then seek one good-faith review; write
  when no review is available. Adaptive pricing uses current defaults.
- The runner's effective forecast horizon is 30, overriding SIM's 200.
- Publication counts are action publication events, checked against authored
  papers at each snapshot. This does not settle how future seeded papers count.

## Observed publications per agent

| Cohort | Mean | Median | Range |
| --- | ---: | ---: | ---: |
| Low quality, slow | 10.10 | 10 | 10-11 |
| Low quality, fast | 24.30 | 24 | 24-25 |
| High quality, slow | 10.15 | 10 | 10-11 |
| High quality, fast | 24.35 | 24 | 24-25 |

Total publications: 1378. Pooled publication medians at steps 250, 500, 750,
1000 are 4, 8.5, 13, 17.5. The pooled median falls between the two speed
cohorts; it is a balanced calibration anchor, not a typical observed agent.

## Recommendation

Use M(n) = 1 + alpha*n/(n+h), with h = 9 and alpha = 0.30.

Choose h by rounding the midpoint-run pooled median 8.5 to 9. This locates
half of the maximum experience gain near the middle of this baseline run.
It is a design choice anchored to the run length, not an estimated learning law.

Choose a 20% end-of-run mean increase at the pooled endpoint anchor n=17.5.
Solving alpha = 0.20*(17.5+9)/17.5 gives 0.30286, rounded to 0.30.
The 20% target is an explicit modeling choice: a no-experience baseline cannot
identify alpha or demonstrate an optimal value. Both talent means improve,
so this moderate initial target allows for their interacting effects.

At baseline publication counts, proposed mean multipliers would be:

- n=0: 1.0000
- n=9: 1.1500
- n=10: 1.1579
- n=17.5: 1.1981 (pooled calibration anchor)
- n=24: 1.2182
- Infinite n: approaches 1.3000

These are post-hoc evaluations, not experience-enabled simulation outcomes.
Increasing rate will change publication counts and hence experience. Parameters
need reevaluation for longer runs, different manuscript effort, or adaptive/RL
strategies. One seed does not establish robustness or review-strategy behavior.
The question of amplifying talent versus enabling catch-up remains open;
this recommendation follows the agreed proportional amplification model.

No simulation defaults or experience mechanics were changed.
