# Liberata Peer-Review Simulation

Academic research project: an agent-based simulation of incentive structures, market dynamics, and quality accrual on the Liberata academic publishing platform. Agents allocate each timestep between advancing their own research and participating in a single-review peer-review marketplace.

## Single-review marketplace

Each paper has a `quality` sampled from a Gaussian centered on its author's intrinsic talent, known to the author before they start writing. Quality sets the paper's base accrual rate and the accrual bump a review can earn. A paper is listed on the market one timestep after it is published, and it can be reviewed exactly once. With value-based matching enabled, the market awards it to the eligible reviewer with the highest expected claim value. If disabled, the first eligible agent to claim it receives the review.

While a paper is listed, its author offers each potential reviewer a distinct share price (`Paper.price_table`). The default base offer splits incremental review surplus fairly: `ε/(1+ε) × (F−A₀)/F × reviewer_surplus_share` (default 50/50), using each reviewer's epsilon history and the same forecast horizon agents use for claim decisions. A higher-quality paper (relative to the market) offers a smaller share; scarcity and adaptive author multipliers can adjust offers further. The price table refreshes every timestep because it depends on which papers are currently on the market.

`peer_review_history` remains a public per-agent reputation metric: the mean share-weighted accrual rate on papers the agent has reviewed (i.e. reviewer share times each paper's accrual rate, averaged over completed reviews). `peer_review_epsilon_history` separately tracks the average proportional accrual improvement caused by that reviewer, and is the metric used for fair-market pricing.

## Timestep structure

Agent order is shuffled at the start of every timestep. The action sequence then
depends on the review paradigm, discussed below in **Review effort model**.

In continuous mode, each agent makes one merged decision and spends that
timestep on it. It may write, claim a listed paper and begin reviewing it,
continue an active review, or finish an active review and return to writing. An
agent holding a review may claim another listed paper; doing so finalizes the
current review at its accumulated effort and starts the new review.

In discrete mode, each timestep has two phases:

1. Marketplace phase: agents without an active review may claim at most one
   listed paper. Claiming selects the review type and duration, but does not yet
   add review effort.
2. Work phase: agents who claimed apply their first unit of review effort.
   Agents with active reviews automatically continue them until their chosen
   fixed duration is complete. All other agents spend the timestep writing.

In either mode, an agent that starts a review cannot also write during the same
timestep.

## Review effort model

The simulation supports two run-level review paradigms. A single simulation is either
`continuous` or `discrete`; both paradigms are not mixed within one environment.

In `continuous` mode, reviews have no fixed duration. After claiming a paper,
an agent spends one timestep on the review. On each later timestep, it chooses
either to continue reviewing (adding one more unit of effort) or to stop and
return to its own research or claim another paper. Stopping finalizes the review at its accumulated
effort. The environment labels a completed review **bad faith** when its effort
is below `good_faith_review_threshold` and **good faith** when it meets or
exceeds that threshold.



In `discrete` mode, an agent chooses a fixed **bad-faith** or **good-faith**
review when it claims a paper. That choice locks in the review's duration: the
environment automatically adds review effort each timestep until the duration is complete, so the agent cannot switch back to writing partway
through. A bad-faith review takes `T_B = 1` timestep. A good-faith review takes
the configured `good_faith_review_threshold`. 

Review effort determines the proportional increase (the review's `epsilon`) in
the paper's accrual rate.
Because a claimed review immediately receives one unit of effort, a review
finished after that first timestep receives the smallest realized bump. The
bump is multiplied by the paper's quality, so the same review effort has a
larger effect on a higher-quality paper.

By default, `review_effort_curve` is set to `"sigmoid"`: short reviews have little
effect, the gain rises most quickly around the good-faith threshold, and the
gain levels out for very long reviews. Set
`review_effort_curve = "log"` for a logarithmic, diminishing-returns curve, or
set it to `"jump"` to add an extra reward near a configured high-effort
threshold.

## Writing Effort Model
Each unfinished manuscript has its own writing-effort counter. Every writing
action adds one `writing_effort_delta` to that counter. Writing therefore does not need to be
consecutive.

With the `continuous_publishing = "threshold"` setting, a manuscript
publishes automatically when it reaches `continuous_paper_timesteps`, currently
`50` writing timesteps. 

With `continuous_publishing = "choice"`, publishing behavior differs by review paradigm. In continuous mode, the agent instead decides when to publish, trading earlier market entry against more time spent writing. The agent theoretically can choose to publish after its first writing timestep, so the minimum is 1 unit of writing effort. 
In discrete mode, publication remains automatic once the manuscript reaches its
assigned target. However, that target is determined by `paper_effort_mode`. It can be `fixed`, which uses `discrete_paper_timesteps`; `uniform`, which samples a per-paper target between `paper_effort_min` and `paper_effort_max` (currently 130-170), and `quality_scaled`, which similarly assigns a value between `paper_effort_min` and `paper_effort_max` but in accordance to the paper’s sampled quality. This quality is sampled from a Gaussian distribution centered on the author’s intrinsic talent. 


## Features
Our environment stresses a few main features:
- Flexible interfaces for agent, environment, market, and paper classes. This allows for multiple implementations of various algorithms and agent populations, including heuristic, random, probabilistic, tabular/linear RL, low-talent RL, and DQN. 
- `config.py` is the central place for tunable simulation settings, including review and writing rules, pricing, paper quality, and learning parameters. CLI flags can be used to override individual settings for a single experiment.
- Each run records action choices, review effort and classification, paper ownership and accrual, agent capital, and inequality metrics. The reports compare strategies and outcomes across agent groups in the same simulation.
- Optional market rules can be toggled for targeted experiments: fair-market offers, adaptive author pricing, reviewer supply-and-demand adjustments, and value-based matching of papers to reviewers.


### Example: discrete-mode runs
```bash
python run_simulation.py --review-paradigm discrete --random-agents 5 --no-archive

python run_simulation.py --review-paradigm discrete --seeds 1,2,3,4,5 --timesteps 10000 --heuristic-agents 50 --rl-agents 50 --name "discrete sweep"

```

## Logging runs
Every run writes working outputs to `runs/`, including `history.csv`,
`history.json`, and summary charts such as `summary.png`,
`agent_group_comparison.png`, `choice_breakdown.png`, and
`review_behavior.png`. These files are overwritten by the next local run.

At the end of an interactive run, the CLI asks whether to archive the results
to the static GitHub Pages gallery. Use `--name "<title>"` to archive without a
prompt, or `--no-archive` to skip archiving. An archived run stores gallery
data and charts in `docs/data/<run_id>/`; its full history is retained locally
in `local_data/<run_id>/`.


Both the terminal summary and the gallery compare the various agent groups and report action choices, good- and bad-faith reviews, review behavior, academic capital, and inequality metrics.


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

## Publication experience experiment (2026-09-28)

### Integrated fixed-strategy entry point (2026-10-05)

Use this entry point to run shared quality/rate talents, publication experience,
and citations together without training or loading any RL policy:

```bash
python run_talent_comparison.py --steps 1000 --seeds 11
```

For the existing visualization website, use the main simulation entry point:

```bash
python run_simulation.py --fixed-strategy --timesteps 1000 --seed 11 --name "Fixed strategy: talents + experience + citations"
```

This generates standard charts and history in `runs/`, then exports gallery
charts and history to `docs/data/<run_id>/` and updates `docs/data/index.json`.
Publish those gallery files through the normal GitHub Pages workflow to update
the online website. Local archival alone does not publish the live site.
Four talent cohorts are separate groups in the existing comparison charts.
Additional per-agent and market metrics are saved as `fixed_strategy_metrics.json`
in both `runs/` and the gallery run directory. Use `--no-archive` to skip export.
`--agents-per-group` defaults to 20. Fixed mode uses its own continuous,
50-unit (configurable in config.py), no-initial-paper conditions and ignores
legacy agent-count/policy/market-mode flags; it never loads RL policies.

Defaults: four equal cohorts (20 agents each), low/high talent means 0.6/1.4,
Gaussian sigmas 0.2, experience alpha=0.30 and h=9, no seeded papers, continuous
mode, fixed 50-unit manuscripts, no merit assignment, and forecast horizon 30.
The fixed strategy publishes, seeks one good-faith review, and writes when none
is available. Config defaults control experience and citation parameters.
Use `--agents-per-group`, `--seeds`, `--steps`, and `--output` for run overrides.
The default output is `experiments/fixed_strategy_integrated`; repeated runs
overwrite that output, so use a separate output directory to keep a run.

`results.json` includes per-agent base/effective talents, publication counts,
experience, citations and capital, plus listing-to-claim wait statistics,
unclaimed paper ages, good/bad counts, and citation AC totals. `comparison.png`
shows cohort outcomes. Empty review/wait samples are not evidence of zero wait:
market statistics use null when no claims exist. All-good reviews are prescribed
by the strategy, not an emergent finding.

Permanent reviews now persist their citation bonus, just as decay reviews do.
Existing shared-talent quality improvement is retained: citation weighting uses
both the improved quality and the separate review bonus. Author self-citation
remains allowed. This entry point starts with no papers and therefore does not
depend on the legacy initial-listing behavior.

Verified integrated run: `experiments/fixed_strategy_integrated/2026-10-05_seed11/`.
It produces 1577 papers, 31191 citations, 1576 total AC, a 0.231-step mean market
wait, and 1562 good / 0 bad completed reviews. This is not directly comparable
to the archived pre-citation economics.

Future training is separate and must be explicitly invoked (not run as part of
this experiment):

```bash
python train_rl.py --shared-talents --num-rl 80 --horizon 30 --no-archive
```

This option enables the same four ability combinations, experience, citations,
zero seed papers, fixed manuscript effort, and no merit assignment in training
and greedy evaluation. Use agent counts divisible by four for balanced groups.
Policies save separately as `policies/policy_<backend>_shared_talents.*`.
It does not redesign RL state/features or automatically configure the legacy
`run_simulation.py` entry point; policy quality still needs future evaluation.
No RL training was performed for the integrated fixed-strategy run.

### Archived calibration

Agents that enable shared talent sampling through `configure_talents()` now
multiply both base talent means by `1 + alpha * n / (n + h)`. Defaults in
`config.py` are `experience_alpha=0.30` and `experience_h=9.0`; alpha zero
disables the modifier. Gaussian standard deviations and outcome floors stay
unchanged. Base talents remain fixed, and previously sampled manuscript quality
is preserved. Publication count is authored papers in the current world,
including seeded papers, excluding review ownership shares.

Scope: this applies to the four `TalentAgent` cohorts. Legacy agents that do not
enable shared talent sampling retain their existing behavior; this is not an
automatic migration of all heuristic/RL agents or their forecasting models.

The fixed-strategy experiment uses 80 agents, 1000 steps, seed 11, zero initial
papers, and 50 units of work per manuscript. Publications increase from 1378 to
1572 (+14.1%). Mean listing-to-claim wait decreases from 0.703 to 0.613 steps.
Completed reviews are 1360 good / 0 bad without experience and 1563 good / 0 bad
with experience. Good faith is prescribed by the strategy, and review demand
is coupled to publication. These results do not establish improved autonomous
review choices or general market efficiency. Alpha is a modeling choice anchored
to a target gain; one seed does not establish robustness.

Results and parameter rationale: `experiments/experience_calibration/`.
Reproduce the experience run into a fresh directory:

```bash
python -m unittest test_experience test_talents -v
python calibrate_experience_baseline.py --output experiments/experience_calibration/new_experience_run
python measure_experience_market.py
```

The market replay checks exact agreement with the two saved group summaries
before recording wait statistics. Unclaimed listed papers are reported separately.

This change includes the experience implementation, its tests, calibration and
market replay scripts, saved results, and the optional recorder argument in
`run_talent_comparison.run`. The pre-existing local `plot_talent_diagnostics.py`
and `experiments/two_talent_comparison/diagnostics/` are outside this commit.
Eight focused tests pass. The nine known legacy failures are accepted as
non-blocking by the project owner; see `AGENTS.md` for their exact identities.
The saved calibration results describe the pre-citation-network experiment at
commit `b50d8d3`. Later remote citation-network changes were merged for publishing;
use that experiment commit to reproduce the archived figures exactly. Running
the replay on changed mechanics may intentionally fail its equality check.
Post-merge validation: 24 of 25 experience, talent, and citation tests pass.
`CitationEnvironmentIntegrationTest.test_history_reports_citation_metrics`
also fails with experience disabled; this separate citation test issue is not
part of the nine accepted legacy failures.

### RL Training

Reinforcement-learning (RL) agents learn a policy that maps the simulation state, such as writing progress, active review effort, available offers, and capital, to the most rewarding action. They train against heuristic opponents and then use the
saved policy while competing with other agent groups in a simulation.


All training scripts save their learned policies to `policies/` by default. Use the script that matches the desired review paradigm:

```bash
# Continuous review paradigm agents (tabular or linear Q-learning)
python train_rl.py --no-archive

# Discrete review paradigm agents (tabular or linear Q-learning)
python train_discrete_rl.py --no-archive

# Continuous review paradigm only: deep Q-network policy
python train_dqn.py --no-archive
```

When a simulation includes RL or DQN agents, it automatically loads the matching saved policy from `policies/` when autoloading is enabled and the file exists. Otherwise, the agents start with a blank policy. 

A saved policy can be used only when it was trained with the same agent architecture, available actions, state inputs, and review paradigm as the current simulation. If any of these change, the old policy is incompatible and must be retrained. For example, after adding an action or changing the feature vector, the previously trained policy becomes invalid. Additionally, a continuous policy cannot be used in a discrete review paradigm simulation, and
vice versa.


### Evaluating Policies
To evaluate the policies trained, adding `--rl-freeze` or `--dqn-freeze` makes the agents act greedily according to the policies without further online learning during the simulation.

```bash
# Continuous tabular/linear Q-learning policy
python run_simulation.py --review-paradigm continuous --rl-agents 20 \
  --heuristic-agents 20 --rl-freeze --name "continuous RL evaluation"

# Discrete tabular/linear Q-learning policy
python run_simulation.py --review-paradigm discrete --rl-agents 20 \
  --heuristic-agents 20 --rl-freeze --name "discrete RL evaluation"

# Continuous DQN policy
python run_simulation.py --review-paradigm continuous --dqn-agents 20 \
  --rl-agents 0 --heuristic-agents 20 --dqn-freeze --name "DQN evaluation"
```
