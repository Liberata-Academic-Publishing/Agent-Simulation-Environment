# AC-to-price reviewer market: single-run summary

Date: November 10, 2026. Branch: `codex/ac-review-market`.
This document describes only the single run in this directory. It does not combine
its results with the earlier multi-seed comparison. Chinese version: `summary_zh.md`.

## 1. Implementation scope and what this run actually tested

The market workflow is implemented: reviewers submit ownership-share quotes;
authors specify a minimum reviewer AC and a budget; the market samples a bounded
set of candidates from AC buckets and ranks eligible candidates by AC divided by
their quote. A reviewer confirms acceptance before the share is locked. Actual
reviewing uses the existing labour budget, effort decisions, paper-value model,
and ownership settlement rules.

However, this run used a fixed quoting policy: every reviewer always requested
0.5% of the whole paper. Consequently, AC-to-price ranking was equivalent to
ranking by AC. Having a quoting interface does not mean that adaptive quoting
behaviour has been implemented.

This run tested sampling, matching, acceptance, work and settlement at a fixed
price. It did not test price competition, autonomous quote adjustment, or a
continuous double auction. There were no best_ask minus 0.01 or best_bid plus
0.01 updates.

## 2. Fixed parameters

| Parameter | Setting in this run |
| --- | --- |
| Market mechanism | `sampled_bidding` |
| Population | 20 heuristic and 20 random agents |
| RL / DQN | None; no training |
| Duration and seed | 1,000 timesteps; seed 11 |
| Initial papers | 10 per agent, 400 total; zero initially accrued AC |
| Reviewer quote | Fixed at 0.5% of the whole paper for every reviewer |
| Author budget | At most 5% of the whole paper, also limited by available ownership and the paper's share cap |
| Minimum reviewer AC | 0 |
| Candidate cap | At most 5 per paper per timestep; subsequent eligibility filters can leave fewer |
| AC bucket boundaries | 1, 10, 100, 1,000; outer buckets extend beyond these boundaries |
| Price tick | 0.0001 of whole-paper ownership, or 0.01 percentage points |
| Review paradigm | Continuous |
| Good-faith classification threshold | At least 3 units of review effort; a simulation label, not verified real-world quality |
| Minimum valid review effort | 0; no additional requirement that a review be good-faith to receive its share |
| Effort per review / writing work turn | 1 / 1 |
| Review uplift curve | Sigmoid |
| Uplift duration | Permanent |
| Publication rule | Threshold: publication after 50 cumulative units of writing effort |
| Heuristic forecast horizon | 30 timesteps |
| Random-agent offer probability | 0.5 per timestep when idle |

The configuration also records `paper_effort_mode=quality_scaled` and a 130–170
target range. However, the active threshold publication path uses
`continuous_paper_timesteps=50` to determine publication. This run should not be
described as requiring 130–170 writing timesteps per paper.

AC is an observable selection signal, not cash spent to pay for a review. The
price is an ownership fraction, not a deduction from the reviewer's existing AC.
Existing talent still affects production and strategy estimates, but does not
enter the market's ranking rule.

## 3. What changed during the simulation

- Reviewer AC changed with paper earnings and ownership. Matching used a snapshot
  taken before work in each timestep.
- Candidate samples, assignments, effort, completion times and good/bad-faith
  classifications varied.
- Heuristic reviewers could reject a particular paper when its expected return
  was below their writing alternative.
- Author ownership, paper earnings, completed reviews and available supply varied.
- Default authors requested reviews for all eligible papers. A withholding
  interface exists, but this run did not use a learned or adaptive release policy.
  Budgets and minimum AC requirements did not adapt to supply or demand.

## 4. Results

| Metric | Result |
| --- | ---: |
| Newly published papers | 703 |
| Final papers, including the initial corpus | 1,103 |
| Matches / completed reviews | 1,103 / 1,101 |
| Good-faith / bad-faith reviews | 564 / 537 |
| Good-faith fraction | 51.23% |
| Writing / reviewing share of labour turns | 90.54% / 9.46% |
| Waiting / active reviews at the end | 0 / 2 |
| Assignment Gini / final capital Gini | 0.292 / 0.163 |
| Share of assignments received by the busiest 20% of agents | 36.63% |

Average waiting time across all matches was 8.61 timesteps, largely reflecting
the 400 papers entering together at startup. Initial papers waited 23.723
timesteps on average. The 703 new papers waited only 0.013 timesteps on average,
with a maximum of one timestep.

| Strategy | New papers | Completed reviews | Good-faith fraction | Mean review effort | Mean final AC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 340 | 428 | 89.02% | 5.62 | 23,229.14 |
| Random | 363 | 673 | 27.19% | 2.04 | 23,441.84 |

New-paper counts exclude the 200 initial papers in each group. Mean review effort
covers completed reviews only. Per-step checks passed for labour counts, review
state consistency, ownership conservation, capital accounting and candidate limits.

## 5. Conclusions supported by the data

1. Under this configuration, writing continued, the simulation ran consistently,
   and new papers did not create an end-of-run waiting backlog. This does not
   establish an optimal writing fraction or stability under other supply levels.
2. Heuristic agents had a higher good-faith fraction but approximately 0.91% lower
   mean final AC. This is a single-seed group comparison, not causal evidence that
   good-faith reviewing reduces earnings.
3. Every transaction paid the same 0.5% share. No price formation or quote
   adjustment was observed or tested.

## 6. Possible explanations for the second finding

A fixed 0.5% share gives the reviewer only a small portion of the additional value
created by reviewing, while the author retains 99.5%. More effort can increase
the value of the reviewer's holding, but does not automatically increase the
ownership percentage. Longer reviews may therefore displace other review jobs
and writing opportunities.

Random agents completed more short reviews and published more papers. Final AC
attributed to reviewer holdings was approximately 4,667.07, only 0.50% of total
AC of 933,419.52. Final AC therefore mostly reflected authored-paper holdings and
other capital sources, rather than being a pure measure of review contribution.

These explanations are consistent with the observations but were not isolated
by this run. Paper quality, timing of earnings, reviews received from others,
the 30-step forecast horizon and random trajectories may also affect the result.

## 7. Unsupported claims and next steps

This run does not establish that the new market outperforms the old market,
that random strategies are generally superior, that good-faith reviewing lacks
value, or that dynamic bidding works. Earnings come from local main's accrual
model, not measured real-world citation gains. There was no old-market control
with the same initialization in this particular run.

The next step is to specify and implement a quoting policy, then compare
AC-to-price selection against random selection under the same workflow and
initialization. Use multiple seeds and jointly evaluate publication throughput,
model-generated review value, author net benefit, waiting, labour allocation and
assignment concentration. Total AC or transaction counts alone are insufficient.

## 8. Data and reproduction

- `summary.json`: configuration snapshot and summary; `history.json`: full history.
- `run.log`: execution log; `summary.png`: overview;
  `agent_group_comparison.png`: group comparison.
- `conclusions.md`: additional preliminary analysis of this run.

```powershell
python experiments/ac_market_single_2026-11-10/run.py
```

The command overwrites simulation outputs in this directory. It does not train
RL, change global configuration defaults, or publish to the gallery.
