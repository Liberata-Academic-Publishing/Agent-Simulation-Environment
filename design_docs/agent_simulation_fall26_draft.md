# Agent Simulation Fall 26: Incentives, Shared Talents, and Publication Experience

### Approvals

| Username | Role | Status | Last Change |
| --- | --- | --- | --- |
| Haider Khan | Approver (from supplied template; confirm assignment) | Pending review | Not yet reviewed |
| TBD | Technical reviewer / team lead | Pending assignment | Not yet reviewed |

**Status:** Draft for author review; not yet published to Notion.

**Author(s):** Eva / Yijin Liu (confirm preferred display name).

**Contributor(s):** Existing Liberata simulation contributors; feature-specific attribution follows the linked commits.

**Reviewer(s):** TBD.

**Last Updated:** 2026-09-28.

**Intended destination:** [Agent Simulation Fall 26](https://app.notion.com/p/Agent-Simulation-Fall-26-3eaae90e9bb9805087c4dcb7bd435ef7).

**Scope and evidence:** This is a retrospective design document and current review proposal, covering the author's verified September work: the September 15 attempts committed September 21, the September 23 talent implementation, and the September 28 experience implementation. Earlier team infrastructure is background, not attributed to this work. Git history and saved experiment artifacts support the details below; discussions or experiments absent from those sources cannot be reconstructed with certainty. Implementation exists on `codex/two-talent-agents`; this document does not assert PR approval or merge to main. Test results below are recorded historical results, not newly executed tests for this document.

---

# Context

## Objective

Enable controlled study of how incentives, researcher ability, and accumulated publishing experience affect writing, peer review, and academic-capital outcomes. The simulation should support interpretable comparisons that help the team investigate when good-faith and bad-faith reviewing emerge.

## Background

Liberata's existing agent-based simulation gives researchers a choice between writing their own papers and reviewing other authors' papers. Publishing generates an academic-capital stream; reviewing grants ownership in another paper and may improve its future accrual. Existing infrastructure includes continuous and discrete review paradigms, heuristic and learning agents, market pricing, and run reporting.

Several issues motivated this work. Short forecast windows can hide the delayed return from writing. Comparing total review rewards against per-timestep writing rewards can bias choices toward review. Initial review supply can dominate early behavior, while an empty listed queue alone does not show whether reviews are still underway. A single quality parameter also cannot separately explain differences in output quality and work speed. Finally, fixed ability does not represent learning through publication.

These concerns were addressed in stages. Parameter and decision changes came first; controlled talent comparisons then separated ability from strategy; publication experience was added after measuring publication counts in that baseline. The fixed-strategy experiments support mechanism analysis, not conclusions about learned equilibria.

---

# Design

## Overview

The work has eight separately traceable modules. Each module below states its change, rationale, implementation boundary, and evidence.

| ID | Module | Verified milestone | Status |
| --- | --- | --- | --- |
| D1 | Forecast horizon | September 15 attempts, committed September 21 | Implemented default change |
| D2 | Bootstrap paper supply | Same milestone | Implemented scheduling changes; entry-point consistency needs verification |
| D3 | Review commitment and value comparison | Same milestone | Implemented; related legacy tests remain unresolved |
| D4 | RL discount and reward weights | Same milestone | Implemented defaults; causal improvement not established here |
| D5 | Supply and time-allocation diagnostics | Same milestone | Implemented metrics; some reporting remains separate |
| D6 | Shared quality and rate talents | September 23 | Implemented opt-in model and five-seed fixed-strategy baseline |
| D7 | Publication experience | September 28 | Implemented for agents enabling shared talent sampling |
| D8 | Calibration and experience comparison | September 28 | Saved single-seed comparison and market replay |

The model sequence is: initial ability -> stochastic writing/review output -> publication -> experience multiplier -> updated talent location parameters for subsequent draws. Paper accrual, ownership, and market rules translate output into academic capital.

## Infrastructure

Reuse the existing local Python simulation. No new hosted service or managed infrastructure is proposed.

| Component | Responsibility |
| --- | --- |
| `config.py` | Frozen configuration dataclasses and tunable defaults |
| `Agent.py` | Work actions, publication lifecycle, talent sampling, experience properties |
| `HeuristicAgent.py` | Forward-looking action valuation |
| `Paper.py` | Stored paper quality, review effects, ownership, accrual |
| `Environment.py` | Turn ordering, work execution, paper listing |
| `History.py` | Time series and action-derived diagnostics |
| `talent_agents.py` | Four cohorts and their shared prescribed strategy |
| `run_talent_comparison.py` | Dedicated cohort experiment |
| `calibrate_experience_baseline.py` | Per-agent publication counts, effective talents, snapshots, provenance |
| `measure_experience_market.py` | Market-wait replay and saved-summary agreement checks |
| `experiments/` | Versioned research artifacts |

The optional local `plot_talent_diagnostics.py` and its diagnostic output directory are not part of the experience commit. Existing `runs/`, the `docs/` gallery, and policy files remain separate parts of the repository workflow.

## Detailed Design

### D1. Forecast Horizon

**Problem and rationale.** A 30-step planning window can end before a manuscript is completed, making writing appear to have no return inside the forecast. The horizon needs to cover work completion and a subsequent earning window.

**Change.** Set `forecast_horizon_timesteps` from 30 to 200 in `config.py`.

**Boundary.** This changes a forecast parameter, not simulation duration or manuscript effort. It is not evidence that 200 is universally optimal. The later talent comparison runner uses an effective environment horizon of 30; its fixed strategy does not use economic forecasts to choose writing versus reviewing. Do not label that runner as a horizon-200 experiment.

**Evidence.** Implemented in [the September parameter and decision commit](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/commit/d67c106). No isolated horizon-only effect estimate is established by the reviewed artifacts.

### D2. Bootstrap Paper Supply

**Problem and rationale.** A large initial stock can create an artificial review rush and obscure whether agents generate enough new papers through writing.

**Change.** Reduce initial papers per agent from 10 to 2 and add a 50-step initial listing schedule. Scheduling changes appear in the simulation entry point and continuous/discrete RL training seed helpers.

**Boundary and verification need.** The simulation seed helper also calls `paper.list_on_market(0)` after assigning the schedule. The intended stagger should therefore be checked end to end across entry points before claiming uniformly delayed listing. This draft records the design and source changes, not a completed validation of that behavior.

**Experiment distinction.** Both the talent baseline and experience comparison start with zero papers. They do not test the bootstrap scheduling change. Seeded papers nevertheless count as publications for experience when a run includes them.

### D3. Review Commitment, Thresholds, and Comparable Value Units

**Problem and rationale.** Switching to another claim could finalize an existing review with too little work. Separately, a claim's whole-horizon ownership value was compared against writing value per unit of time.

**Changes.**

| Element | Before | After |
| --- | --- | --- |
| Minimum valid review effort | 0 | 3 |
| Continuous good-faith threshold | 3 | 5 |
| Fresh continuous claim while already reviewing | Could finalize/switch | Continue the active review instead |
| Heuristic claim score | Whole claim value | Divide by expected review duration before competition adjustment |

The expected duration uses the minimum credible review effort in continuous mode and the bad-faith duration in discrete mode, with a lower bound of one timestep. It is an opportunity-cost approximation, not a newly learned duration model. An explicit research action is used to finish the active continuous review.

**Boundary.** Minimum valid effort and good-faith classification are distinct concepts. Continuous classification is based on effective work; discrete classification retains the selected review kind. Faster agents can accumulate the same effective work in fewer elapsed turns.

**Validation state.** Changes and regression tests are present in the September commit. Some older tests encode incompatible switching/threshold expectations; known failures are listed in `AGENTS.md`. Their accepted status does not establish that all affected behavior is fully validated.

### D4. RL Discount and Reward Parameters

**Problem and rationale.** Delayed publishing returns can be weakened by discounting, while an overwhelmingly rank-based reward can emphasize short-term relative position.

| Parameter | Before | After | Intended purpose |
| --- | ---: | ---: | --- |
| `rl_gamma` | 0.95 | 0.99 | Retain more value from delayed rewards |
| `rl_reward_ac_weight` | 0 | 1 | Reward absolute capital changes |
| `rl_reward_rank_weight` | 100 | 10 | Reduce dominance of relative rank |
| `rl_reward_accrual_weight` | 1 | 1 | Retain portfolio-accrual reward |

**Boundary.** These are revised defaults, not a claim of demonstrated convergence or optimal behavior. The later talent/experience experiments use no RL training and cannot validate these reward changes. Policies used for future comparisons need provenance and training conditions consistent with the tested environment.

### D5. Supply and Time-Allocation Diagnostics

**Problem and rationale.** A nearly empty open market can coexist with many unfinished reviews. Multiple log records within one turn can also be mistaken for multiple units of work.

**Change.** Add metrics for papers currently in review, total review backlog, publications per step, cumulative publications, trailing publication rate, realized supply/claim coverage, market emptiness, writing effort, and writing/review/unallocated time shares. Completion-only events are distinguished from the action that consumes a turn.

**Interpretation.** Open listings measure unclaimed supply; in-review papers measure claimed unfinished work. Listing-to-claim wait is not review completion time. Realized publication/claim flow is not latent demand. Action-derived measures require validation against the underlying events, especially rolling-window metrics.

**Reporting boundary.** The dedicated talent recorder distinguishes four cohorts, while general history grouping by agent class would combine them as `TalentAgent`. Consolidating those views remains follow-up work.

### D6. Two Shared Talents

**Design decision.** Give each agent two independently configured base parameters, quality and rate, shared across writing and reviewing. The experiment crosses low/high settings for each; it does not introduce four unrelated writing/review abilities.

| Quantity | Sampling rule before experience | Draw timing |
| --- | --- | --- |
| Manuscript quality | `max(0.1, Normal(base_quality, 0.2))` | Once when a manuscript starts; retained at publication |
| Review quality | Same quality distribution, independent draw | Once when a review completes |
| Writing progress multiplier | `max(0.01, Normal(base_rate, 0.2))` | Every writing turn |
| Review progress multiplier | Same rate distribution, fresh draw | Every review turn |

These are floored Gaussians, not truncated normals. The post-floor expectation can differ from the Gaussian location parameter. Increasing rate increases expected work per turn, rather than directly setting completion time.

**Review integration.** Sampled reviewer quality scales the existing effort-based proportional improvement, epsilon. For enabled agents, paper quality increases by `old_quality * epsilon`; the existing accrual bump uses that epsilon as well. The draw does not directly change the locked ownership share. Stored quality improvement persists even if the separately configured accrual bump decays.

**Implementation.** `configure_talents()` enables this model before work begins. `TalentAgent` enables it automatically. `Paper.paper_quality` stores quality and the legacy `quality` property aliases it. Other agents keep legacy behavior unless explicitly configured.

**Baseline experiment.** Four groups of 20 agents cross base quality/rate 0.6 and 1.4. Each run lasts 1,000 steps, starts with no papers, uses continuous threshold publication at 50 units of fixed work, and disables merit-based market assignment. All groups publish, then seek one good-faith review; if none is available, they keep writing. Five runs use seeds 1-5, without experience or RL learning.

| Cohort | Papers/agent | Completed reviews/agent | Writing time | Final capital/agent |
| --- | ---: | ---: | ---: | ---: |
| Low quality / Slow | 10.16 | 10.04 | 91.036% | 3,446 |
| Low quality / Fast | 24.33 | 24.03 | 90.219% | 8,533 |
| High quality / Slow | 10.16 | 10.03 | 91.059% | 10,232 |
| High quality / Fast | 24.28 | 24.03 | 90.187% | 25,988 |

These are averages across seeds, with capital rounded as in the recorded report. All 6,813 completed reviews are good faith because the strategy prescribes it. Fast agents produce roughly 2.4 times as many papers/reviews; high-quality slow agents earn more than low-quality fast agents under these conditions. Neither ranking nor honesty is established as a general strategic result.

**Sources.** [Implementation](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/commit/ca19cae); [baseline data](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/blob/ab85850/experiments/two_talent_comparison/results.json); [recorded interpretation](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/blob/ab85850/README.md).

### D7. Publication-Based Experience Modifier

**Design decision.** Experience proportionally increases both talent location parameters using the same bounded multiplier:

```text
n_i = number of authored papers in the current simulation world
M_i(n_i) = 1 + alpha * n_i / (n_i + h)
effective_quality_i = base_quality_i * M_i(n_i)
effective_rate_i = base_rate_i * M_i(n_i)
alpha = 0.30; h = 9.0
```

**Count semantics.** Count authored papers in `Agent.all_papers`, including seeded authored papers. Do not count unfinished manuscripts, completed reviews, or ownership shares in other authors' papers. Deriving the count from authorship avoids a second mutable counter. This is a current-world count; a future paper-deletion feature would require revisiting its equivalence to lifetime publications.

**Sampling and timing.** Use the effective quality/rate as Gaussian locations, retaining existing sigmas and floors. Base talents stay fixed, so gains do not compound by repeatedly updating the base. New publications affect subsequent draws. Already sampled manuscript quality and work already performed are not recomputed. Review quality is still drawn at completion, and rate is still drawn per work turn.

**Parameter semantics.** Alpha is the asymptotic proportional gain; h is the publication count at half that gain. At zero publications the multiplier is 1, at nine it is 1.15, and it approaches 1.30. Alpha must be finite and nonnegative; h finite and positive. Alpha zero disables the effect.

**Actual scope.** The implementation affects agents that enable shared talent sampling. It does not automatically migrate all heuristic/RL populations, retrain policies, or update their forecast models. Both writing and reviewing receive the benefit, although publishing is what earns experience.

**Open research choice.** This amplifies talent rather than explicitly helping low-talent agents catch up. At equal publication counts, base talent ratios are preserved and absolute gaps grow. Different publication counts add unequal experience gains. A common-ceiling catch-up model remains an alternative for later experiments.

### D8. Calibration and Experience Experiment

**Calibration baseline.** Use the same four-cohort fixed strategy, 80 agents, 1,000 steps, seed 11, zero initial papers, and 50 work units per paper, with experience absent. At step 500 the pooled publication median is 8.5; at step 1,000 it is 17.5. Slow agents publish 10-11 papers and fast agents 24-25.

**Choosing h.** Round the midpoint median 8.5 to 9 to place half of the maximum gain near the middle of this experiment. Because the cohort distribution is split by speed, the pooled median is a balancing anchor rather than a typical observed agent.

**Choosing alpha.** Select an approximately 20% end-of-run location increase at the pooled anchor: `alpha = 0.20 * (17.5 + 9) / 17.5 = 0.30286`, rounded to 0.30. The target of 20% is a modeling choice. A no-experience baseline does not identify an optimal alpha or an empirical learning law.

**Observed experience-enabled comparison.**

| Cohort | Baseline papers/agent | Experience papers/agent | Baseline capital/agent | Experience capital/agent |
| --- | ---: | ---: | ---: | ---: |
| Low quality / Slow | 10.10 | 11.25 | 3,190.93 | 3,746.91 |
| Low quality / Fast | 24.30 | 28.00 | 8,218.11 | 10,897.93 |
| High quality / Slow | 10.15 | 11.30 | 10,161.45 | 12,758.30 |
| High quality / Fast | 24.35 | 28.05 | 25,262.29 | 35,949.80 |

Total publications increase from 1,378 to 1,572 (+14.1%). Final group-mean multipliers range from approximately 1.165 to 1.229. Mean listing-to-claim wait decreases from 0.703 to 0.613 steps. Completed good/bad reviews are 1,360/0 without experience and 1,563/0 with experience.

**Interpretation limits.** These results include the feedback from faster work to more publications to more experience. They do not isolate quality and rate effects, establish robustness across seeds, or show that agents independently choose honesty. Review demand is tied to publishing by the strategy. A shared seed does not guarantee matched per-agent draws once actions diverge. Different run lengths, manuscript effort, and learned strategies may require different calibration.

**Version boundary.** The archived comparison is the pre-citation-network experiment recorded in `b50d8d3`. The later merge includes a teammate's citation-network work; that is an integration dependency, not a mechanism authored in this design. Replaying archived figures against changed mechanics may fail the intended equality checks.

**Sources.** [Experience implementation](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/commit/b50d8d3); [calibration rationale](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/blob/b50d8d3/experiments/experience_calibration/seed_11_steps_1000/interpretation.md); [comparison](https://github.com/Liberata-Academic-Publishing/Agent-Simulation-Environment/blob/b50d8d3/experiments/experience_calibration/experience_a030_h9_seed11/comparison.md).

## Alternatives Considered

The table distinguishes prior behavior and discussed alternatives from approaches that were actually tested.

| Decision | Selected approach | Alternative | Reason / remaining question |
| --- | --- | --- | --- |
| Planning horizon | Default 200 | Prior default 30 | Include delayed writing returns; no horizon-only optimum established |
| Review valuation | Value divided by expected duration | Prior total-value comparison | Compare time-normalized opportunities; duration remains an approximation |
| Initial supply | Smaller stock with scheduled release | Prior larger bootstrap stock | Reduce initial-stock dominance; verify entry-point consistency |
| Ability dimensions | Two talents shared across writing/review | Four separate abilities | Keep initial comparisons interpretable; separate domains could be studied later |
| Strategy | Prescribed behavior for initial experiments | Learned or reward-based choices | Isolate mechanisms; cannot answer strategic equilibrium questions |
| Positive outputs | Floored Gaussian | Log-normal or truncated normal | Retain implemented sampling; alternatives are untested |
| Experience growth | Bounded proportional gain | Unbounded linear growth | Limit long-run amplification; no linear comparator run |
| Distribution change | Shift locations; fixed sigma | Scale whole draws or alter sigma | Separate expected ability growth from absolute noise scale |
| Talent inequality | Proportional amplification | Move toward a common ceiling | Follow current agreement; catch-up question remains open |
| Publication count | Derive from authored world papers | Maintain an independent counter | Avoid synchronization bugs; introduces repeated scanning |

## Dependencies

Core execution relies on the local Python runtime and repository modules, without a live external service. NumPy is needed for relevant numerical/RL paths, and Matplotlib for plotting; missing plotting support affects figures rather than defining the underlying mechanics. Git metadata is required by the calibration script's provenance capture. GitHub and Notion support sharing and review, not simulation execution.

Reproduction depends on source revision, effective configuration, seed, and strategy. Archived experiments predate the citation-network merge. Coordination with the team lead is needed for PR review and release; citation integration and the shared diagnostics pipeline require coordination with their maintainers. Names and owners beyond the supplied template remain to be confirmed.

## Migrations

There is no production database migration or service downtime. Shared talent sampling is opt-in; old agent populations are not silently converted. Alpha zero preserves no-experience sampling for enabled agents, but does not undo unrelated later changes such as citation integration.

Do not overwrite archived baselines with outputs from current defaults. Use fresh output directories and preserve configuration and source provenance. Existing policies should not be presented as validated for changed economics or talent inputs without retraining/evaluation. Moving general agents onto shared talents and aligning their forecast estimates are separate follow-up tasks.

## Technical Debt

| Issue | Risk / follow-up |
| --- | --- |
| General history groups agents by class | Four talent cohorts can be collapsed; integrate explicit cohort labels |
| Separate local diagnostic script | Outputs may not be reproducible from committed files alone; consolidate or commit deliberately |
| Repeated world-paper scan for experience | Cost increases with papers and draws; profile before introducing cached counters |
| Legacy forecast estimates omit full talent/experience dynamics | Learned/heuristic comparisons may use inconsistent expectations; address before strategic claims |
| Initial listing helpers differ | Check scheduled release through each entry point |
| Rolling metrics and compound action records | Verify diagnostics against raw events before relying on them for causal claims |
| Known legacy test failures | Exact accepted list is in `AGENTS.md`; new failures are not exempt |
| Separate citation integration failure | Track independently from the accepted legacy list |
| Historical docs and changed defaults | Clearly date baseline statements; rerunning with alpha 0.30 is not the no-experience baseline |

---

# Quality

## Testing

### D1-D5: Incentives, lifecycle, and diagnostics

The September change includes regression coverage for decision/lifecycle and diagnostic behavior. This document does not claim all tests passed or attribute an observed behavior change to any one parameter. Follow-up checks should separately verify horizon-sensitive writing valuation, duration-normalized claims, continuous review commitment, staggered release through simulation/training entry points, and one unit of time per agent turn. Compare rolling metrics to direct calculations from raw logs.

### D6: Talent mechanics

Six recorded talent tests cover cohort construction, sampling, input validation, review quality effects, and continuous/discrete integration with share conservation. The five-seed baseline compares output at a fixed strategy. Plot bands are standard deviations across seeds, not confidence intervals.

### D7-D8: Experience and comparison

The recorded experience tests check authored counts including seeded papers, exclusion of review ownership, correct sampling locations and unchanged sigmas, both publication paths, preservation of already sampled manuscript quality and base talents, alpha-zero behavior, and invalid parameters. Together with talent tests, eight targeted tests passed before the citation merge. The recorder checks publication events against authored papers at snapshots. The market replay verifies saved group summaries before adding waiting-time metrics.

### Known test state

The historical 81-test legacy suite has nine failures also reproduced with experience disabled; the project owner accepted those specific failures as non-blocking in `AGENTS.md`. After the citation merge, the recorded combined experience/talent/citation run passed 24 of 25 tests. The remaining `CitationEnvironmentIntegrationTest.test_history_reports_citation_metrics` also failed with experience disabled and is separate from the accepted nine. Its release disposition needs explicit review; do not report the full suite as green.

### Reproduction

Use an appropriate isolated checkout of the documented experiment revision, not the current merged source, when reproducing archived numerical results. No checkout, simulation, or test execution was performed while drafting this document.

```bash
# Focused mechanism checks at the chosen source revision
python -m unittest test_experience test_talents -v

# Experience comparison at b50d8d3; output directory must be new
python calibrate_experience_baseline.py --output experiments/experience_calibration/reproduction_experience

# At the compatible pre-citation experiment revision; writes replay outputs
python measure_experience_market.py
```

For the original five-seed no-experience talent experiment, use its September 23 revision and `python run_talent_comparison.py --steps 1000 --seeds 1,2,3,4,5 --output experiments/talent_reproduction`. At a later revision, alpha must be explicitly disabled and all other effective conditions matched; merely reusing the old command does not recreate the old model.

---

# Project Management

## Work Estimates

Completed work is reported as milestones, not invented estimates of time spent. Future work is divided into approximate one-week planning units; these are proposed scopes, not committed dates or delivery promises.

| Phase | Deliverable | Status / schedule |
| --- | --- | --- |
| D1-D5 | Parameter, valuation, lifecycle, diagnostic updates | September 15 attempts; committed September 21 |
| D6 | Shared talents, cohort runner, five-seed baseline | Implemented September 23 |
| D7-D8 | Experience, calibration, replay and documentation | Implemented September 28; review/merge status to confirm |
| Review week | Review design and PR; resolve or disposition integration failure; verify listing/metric concerns | Proposed one-week scope; owner/start TBD |
| Experiment week | Multi-seed experience comparisons and alpha/h sensitivity, including alpha zero | Proposed one-week scope; depends on agreed experiment conditions |
| Strategy week | Compare writing-only, good-faith and bad-faith fixed strategies; define subsequent RL evaluation | Proposed one-week scope; depends on validated comparisons |
| Reporting week | Consolidate cohort diagnostics and update reproducibility documentation | Proposed one-week scope; may overlap with experiments |

## Documentation Plan

This document is the project-level design record, preserving separate modules rather than merging all changes into an experience feature note. Each module records its intended mechanism, implementation boundary, evidence, and open questions.

Keep `config.py` as the tunable source of truth; `README.md` for dated results and reproduction; `TALENT_MODEL.md` for sampling semantics; `AGENTS.md` for operational guidance and the accepted legacy failure list; and `experiments/experience_calibration/` for calibration and comparison artifacts. Align historical statements about constant talents with the newer distinction between fixed base talents and changing effective locations.

Before publishing this draft, confirm author/reviewer names, the PR URL and status, and whether additional earlier personal work is missing from the available Git record. No approval state should be inferred from this document. The author must approve the draft before it is written to Notion.

## Launch Plan

This is a research simulation change, with no established live production-platform rollout. Visible outputs are repository code, saved experiments, figures, and documentation. Updating a public gallery is a separate publication action; no gallery deployment is implied here.

The proposed sequence is author approval of this document, team review of the PR and known test state, resolution or explicit disposition of new integration issues, merge when the reviewer approves, and publication of clearly versioned experiment documentation. PR URL, merge decision, and release date remain unverified/TBD. Expanding experience to all agent types or claiming strategic effects requires further implementation and evaluation beyond the current opt-in fixed-strategy result.
