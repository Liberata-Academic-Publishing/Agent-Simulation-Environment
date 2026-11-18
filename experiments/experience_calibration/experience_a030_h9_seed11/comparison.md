# Experience comparison

Branch: codex/two-talent-agents. Alpha=0.30, h=9.
80 agents, 1000 steps, seed 11, same fixed strategy and conditions as baseline.

| Cohort | Baseline papers/agent | Experience papers/agent | Baseline capital | Experience capital |
| --- | ---: | ---: | ---: | ---: |
| Low quality, slow | 10.10 | 11.25 | 3190.93 | 3746.91 |
| Low quality, fast | 24.30 | 28.00 | 8218.11 | 10897.93 |
| High quality, slow | 10.15 | 11.30 | 10161.45 | 12758.30 |
| High quality, fast | 24.35 | 28.05 | 25262.29 | 35949.80 |

Total publications: 1378 -> 1572 (+14.1%). Final mean multipliers: 1.165-1.22895.
Both talent means improve; Gaussian sigmas stay unchanged. Counts include
seeded authored papers and exclude review shares. Shared talent sampling must
be enabled; legacy agents retain their previous behavior.

Eight targeted tests passed. The 81-test legacy suite has nine failures, with
the same nine failures reproduced with experience disabled (alpha=0).

This single fixed-strategy run does not establish robustness or emergent review
choices. A shared seed does not ensure matching per-agent draws once action
timing changes. Baseline data remains in ../seed_11_steps_1000/baseline.json.
