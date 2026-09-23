# Liberata Peer-Review Simulation

Academic research project: an agent-based simulation of incentive structures, market dynamics, and quality accrual on the Liberata academic publishing platform. Agents allocate each timestep between advancing their own research and participating in a single-review peer-review marketplace.

## Single-review marketplace

Each paper has a `quality` sampled from a Gaussian centered on its author's intrinsic talent, known to the author before they start writing. Quality sets the paper's base accrual rate and the accrual bump a review can earn. A paper is listed on the market one timestep after it is written, and it can be reviewed exactly once: the first agent to claim it takes it off the market permanently.

While a paper is listed, its author offers each potential reviewer a distinct share price (`Paper.price_table`). The default base offer splits incremental review surplus fairly: `ε/(1+ε) × (F−A₀)/F × reviewer_surplus_share` (default 50/50), using each reviewer's epsilon history and the same forecast horizon agents use for claim decisions. A higher-quality paper (relative to the market) offers a smaller share; scarcity and adaptive author multipliers can adjust offers further. The price table refreshes every timestep because it depends on which papers are currently on the market.

`peer_review_history` remains a public per-agent reputation metric: the mean share-weighted accrual rate on papers the agent has reviewed (i.e. reviewer share times each paper's accrual rate, averaged over completed reviews). `peer_review_epsilon_history` separately tracks the average proportional accrual improvement caused by that reviewer, and is the metric used for fair-market pricing.

## Timestep structure

Agent order is freshly shuffled each timestep. Continuous mode uses one merged
decision per agent: claim and start reviewing, continue reviewing, or research.
An agent already reviewing cannot switch directly to a new claim; it continues
the current review instead. Switching to research finalizes that review.

Discrete mode uses two phases:

1. Marketplace phase — each agent may claim at most one listed paper to review. Claiming a paper while already reviewing finalizes the current review (at its accumulated effort) and starts the new one.
2. Work phase — agents that did not claim either continue their own research or, if mid-review, choose between continuing the review and finishing it to write.

## Review effort model

The simulation supports two run-level review paradigms. A single run is either
`continuous` or `discrete`; both paradigms are not mixed within one environment.

In `continuous` mode, agents choose review time by continuing or finishing a
review. The environment classifies completed reviews as bad faith below
`good_faith_review_threshold` and good faith at or above it.

In `discrete` mode, agents choose fixed bad- or good-faith review actions. By
default, bad faith takes `T_B = 1` timestep, good faith takes `T_G = 5 * T_B`,
and manuscript work uses `T_M = 200 * T_B`.

The current minimum reward threshold is 3 units of effective review effort;
continuous good-faith classification begins at 5 units. Eligibility for a share
does not guarantee a positive bump: the configured sigmoid has zero bump at
its minimum threshold. The default review reward curve is a
sigmoid-like `E = F(T)` curve inspired by the team discussion of review length
and citation impact: very short reviews have limited effect, the bump rises
around the good-faith region, and long reviews saturate. The previous
logarithmic curve remains available by setting `review_effort_curve = "log"` in
`config.py`; `review_effort_curve = "jump"` enables the optional high-effort
threshold experiment discussed in sync.

## Writing Effort Model

Paper writing effort is tracked per manuscript. Each `write_paper` action
contributes a `writing_effort_delta` to the agent's current paper progress. By
default, the existing fixed thresholds are preserved (`continuous_paper_timesteps`
or `discrete_paper_timesteps`), but `paper_effort_mode` can be set to `uniform`
or `quality_scaled` to sample a stable per-paper target from the 50-150 timestep
range requested during sync.

## Features
Our environment stresses a few main features:
- Flexible interfaces for agent, environment, market, and paper classes. This allows for multiple implementations of various algorithms.
- Various methods for more complex simulation. This can be chosen to be turned on or off depending on the simulation we want to run.

Run a discrete CLI simulation with random controls:

```
python run_simulation.py --review-paradigm discrete --random-agents 5 --no-archive
python run_simulation.py --seeds 1,2,3,4,5 --timesteps 10000 --heuristic-agents 50 --rl-agents 50 --name "discrete sweep"
```

## Logging runs
You can save completed runs and browse them later in a static web page. This gets published to GitHub
Pages.

After running the simulation, the terminal will prompt you whether or not to save this run to the log and ask for a name.

Saved runs include an agent-type comparison report for heuristic, random,
probabilistic, and RL agents. The CLI summary prints the same comparison, and
the static gallery displays it as a table/chart next to the existing action and
review behavior plots.

## Reinforcement-learning agents

`train_rl.py` trains Q-learning agents. Note that the action space and feature vector changed with the single-review marketplace overhaul, so any policy saved before that change (in `policies/`) is incompatible and must be retrained. Agent counts come from `config.py`; the current defaults select 20 RL agents and no heuristic agents. CLI flags override these counts.

## Progress update: quality and speed talents (2026-09-23)

Branch: `codex/two-talent-agents`. This is the first fixed-strategy baseline for
the agent-talent task, not a completed study of strategic equilibria. The latest
agreed scope uses **two independent talents per agent**, shared between writing
and reviewing, rather than four independently configured abilities.

### Requirements and implementation

| Requirement | Implementation |
| --- | --- |
| Separate quality and speed | `Agent.py`: `quality_talent`, `rate_talent`, and `configure_talents()`; positive, finite values required before work starts. |
| Sample manuscript quality from talent | `_sample_quality()` draws a floored Gaussian centered on quality talent once when a manuscript starts. Publication reuses that draw. |
| Sample writing speed each timestep | `_sample_rate()` and `writing_effort_delta()` multiply base writing progress by a fresh positive rate draw. |
| Apply the same talents to reviewing | `review_effort_delta()` samples speed each review turn; `sample_review_quality()` draws quality once per completed review. |
| Store actual paper quality | `Paper.paper_quality` is the stored value; the legacy `quality` property reads/writes the same value. |
| Improve papers through reviews | `Paper.finish_review()` multiplies the existing effort-based epsilon by sampled reviewer quality. For enabled agents, quality increases by `old_quality * epsilon`; the existing AC bump uses that epsilon too. Locked review shares are not directly changed by the draw. |
| Four types of agents | `talent_agents.py`: `build_talent_cohorts()` creates quality low/high crossed with rate low/high, 20 agents per group. |
| Start with hard-coded strategies | `TalentAgent` publishes a paper, then attempts to claim another author's paper for a good-faith review. It keeps writing if no paper is available. |
| Compare outcomes | `run_talent_comparison.py` produces per-seed results and a summary chart; `plot_talent_diagnostics.py` records time series and generates four diagnostic figures. |
| Verify mechanics | `test_talents.py` has six passing tests covering cohorts, sampling, input validation, review improvement, and continuous/discrete integration with share conservation. |

All distribution parameters live in `config.py`. Low/high talent values are
0.6/1.4; quality and rate standard deviations are both 0.2. Quality is floored
at 0.1 and rate at 0.01. These are **floored Gaussians**, not truncated normals;
the floor can shift the observed mean. Log-normal alternatives have not been
implemented or compared, and these parameters have not been empirically calibrated.
Talents stay constant during a run; outputs fluctuate. Higher rate means less
time to completion, so rate and `timesteps_to_paper` are inversely related.

The new model is opt-in through `configure_talents()`; `TalentAgent` enables it
automatically. The standard `run_simulation.py` population is not automatically
replaced with these cohorts. See [TALENT_MODEL.md](TALENT_MODEL.md) for details.

### Experiment setup and results

- Continuous review mode; threshold-based publication at **50 units of work**,
  not 50 elapsed timesteps for every agent.
- 80 agents (20 per group), 1,000 timesteps, no initial papers.
- Fixed manuscript effort mode; merit-based market assignment disabled.
- Same fixed strategy in all groups, with no RL training or strategy learning.
- Seeds 1, 2, 3, 4, 5: five separate runs with the same settings but different
  random draws and action order. Reusing a seed reproduces a run. Plots use
  averages across runs; shaded bands show one standard deviation across seeds,
  not confidence intervals.
- Existing AC accrual, pricing, and ownership economics retained; no citation
  model was implemented as part of this task.

| Agent type | Papers/agent | Completed reviews/agent | Writing time | Final AC/agent |
| --- | ---: | ---: | ---: | ---: |
| Low quality / Slow | 10.16 | 10.04 | 91.036% | 3,446 |
| Low quality / Fast | 24.33 | 24.03 | 90.219% | 8,533 |
| High quality / Slow | 10.16 | 10.03 | 91.059% | 10,232 |
| High quality / Fast | 24.28 | 24.03 | 90.187% | 25,988 |

Values are means across five seeds. Faster agents produce about 2.4 times as
many papers/reviews. Under these settings, the high-quality slow group earns
about 20% more AC than the low-quality fast group. This ranking is specific to
the chosen parameters and retained accrual economics, not a general result.

### Reference figures and underlying data

These are Matplotlib plots of actual simulation outputs, not illustrative or
AI-generated images. Instrumented reruns reproduced all saved per-seed summary
values exactly. Output directory: `experiments/two_talent_comparison/`.

1. **AC trajectories:** high-quality fast agents accumulate the most AC.
2. **Paper supply and review activity:** new publications and review claims
   track closely; listed backlog remains small. The upper panel counts events
   in non-overlapping 50-step windows; the lower panel shows end-of-step stocks.
3. **Good/bad-faith ratio:** all 6,813 completed reviews across the five runs are
   classified as good faith. This follows the prescribed strategy and is not
   evidence that honesty is strategically superior. Incomplete reviews are excluded.
4. **Time allocation:** approximately 90% writing and 10% reviewing in all groups;
   every agent timestep is accounted for.

Published baseline data: [per-seed results and configuration](experiments/two_talent_comparison/results.json).
Additional diagnostic images and time-series CSVs were generated locally and
are not included in this README-only update. The local CSV filenames are
`agent_group_timeseries.csv` and `market_timeseries.csv`.

### Interpretation limits and follow-up work

**Why are published papers claimed almost immediately?** The fixed strategy
couples supply and demand: publish a paper, then immediately attempt to claim
another author's paper for review. Agents choose the first eligible paper
without comparing its offered reward against writing returns. New papers are
listed on the next timestep; claiming removes them from the available pool
before the review is finished. Near-zero listed backlog therefore does not mean
instant review completion or prove that the market is efficient. Average final
listed backlog is 0.2 papers, while 12.2 reviews remain in progress.

Other limitations and next steps:

- The baseline meets the instruction to hard-code strategies for the first
  experiment. It does not answer when honest or exploitative strategies win.
  Compare writing-only, writing plus good-faith reviewing, and writing plus
  bad-faith reviewing across each talent group before drawing strategy conclusions.
- Decouple review claims from publication and consider reward-based selection
  to investigate meaningful supply/demand imbalance.
- Continuous good/bad classification uses effective work, not elapsed days.
  Fast agents can finish a good-faith review in fewer timesteps. Confirm this
  interpretation with the project team.
- The proportional paper-quality increase is an implementation assumption,
  not a supplied empirical formula. Quality improvement persists even if the
  legacy AC review bump is configured to decay. Existing RL policies and
  economic forecasts have not been recalibrated for these distributions.
- The repository already has `History.py` and `visualize.py` with capital,
  marketplace, review-behavior, and action-mix plots. The new diagnostic script
  currently runs separately. Existing grouping uses agent class names, so all
  four cohorts would otherwise be grouped as `TalentAgent`. Integrate talent
  cohort labels and cross-seed aggregation into that pipeline rather than
  maintaining duplicate reporting systems.
- Nine of the 81 existing `test_simulation` tests fail; the same nine failures
  were reproduced against the pre-change HEAD in an isolated temporary directory.
  All six new talent tests pass, but the full suite is not green. Existing failures
  concern review thresholds, switching, heuristic decisions, and scarcity pricing.

### Reproduce this baseline

Run from the repository root with Python, NumPy, and Matplotlib available:

```bash
python -m unittest test_talents -v
python run_talent_comparison.py --steps 1000 --seeds 1,2,3,4,5
python plot_talent_diagnostics.py
```

The additional `plot_talent_diagnostics.py` script is currently local and is not
included in this README-only update. The baseline runner and talent tests are
already present on this branch.

The diagnostic script intentionally uses the same fixed 1,000-step, five-seed
setup and checks its results against `results.json`; rerun the baseline command
above first if that file was generated with different settings. Both scripts
write to the dedicated experiment directory rather than replacing `runs/`.
