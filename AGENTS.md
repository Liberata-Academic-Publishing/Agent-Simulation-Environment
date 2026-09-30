# Liberata Peer-Review Simulation

Agent brief for this academic research project. This is the source of truth for
project context; keep it concise and current. `README.md` is the human-facing doc.

## What this is

An agent-based simulation of the Liberata academic publishing platform. Each
day, every agent chooses how to spend its turn between two activities:

1. **Writing papers** (advancing its own research), and
2. **Peer review** (participating in a review marketplace for academic capital).

The research question is **good-faith vs bad-faith peer review**: the sim
presents both choices and observes the emergent agentic behavior. Reviewer share
does not depend on the good/bad-faith classification; the classification only
affects the reviewed paper's future accrual rate.

## Core mechanics

`config.py` is the single source of truth for all tunable parameters. Edit the
`SimConfig`/`TrainConfig` dataclass defaults to change behavior everywhere; pass
CLI flags for one-off overrides.

- **Writing effort**: each `write_paper` adds a `writing_effort_delta`; once
  cumulative progress reaches the current paper's effort target, a paper
  publishes and progress resets. `paper_effort_mode` can preserve fixed targets
  or sample the 50-150 timestep band discussed in sync.
- **Review paradigms**: each environment run is either `continuous` or
  `discrete`, configured by `review_paradigm`; paradigms are not mixed inside one
  run.
- **Continuous review effort**: agents choose time spent by continuing or
  finishing reviews. The environment classifies completed reviews as bad faith
  below `good_faith_review_threshold` and good faith at or above it.
- **Discrete review effort**: agents choose `bad_faith` or `good_faith` when
  claiming a review. By default `T_B = 1`, `T_G = 5 * T_B`, and discrete
  manuscript work uses `T_M = 200 * T_B`.
- **Review reward curve**: by default `review_effort_curve = "sigmoid"` models
  `E = F(T)` with low impact for short reviews, a rise through the good-faith
  region, and saturation for long reviews. Set it to `"log"` to use the older
  logarithmic curve, or `"jump"` to test the high-effort threshold experiment.
- **Review-share economics**: a review grants ownership share on the reviewed
  paper. Fair-market offers use
  `ε/(1+ε) × (F−A₀)/F × reviewer_surplus_share` (incremental surplus split;
  default 50/50), estimated from reviewer epsilon history and forecast horizon,
  then adjusted by quality/scarcity/adaptive multipliers and clamped to
  `[min_offer_share, author share]` (defaults 0--100%). Author adaptive pricing is controlled only by
  `pricing_policy` (`static_fair_market` | `adaptive_multiplier`); scarcity
  pricing is disabled automatically when adaptive pricing is active.
- **Results display**: `History.to_dict()` exports `agent_group_summary`, and
  both `run_simulation.py` and the static `docs/` gallery compare heuristic,
  random, probabilistic, RL, and low-talent RL agent outcomes.
- **RL settings** live in `SimConfig` too (`rl_backend`, `rl_epsilon`,
  `rl_gamma`, reward weights). Continuous tabular/linear RL uses `train_rl.py`;
  discrete 3-action RL (write / bad claim / good claim) uses
  `train_discrete_rl.py` and `DiscreteQLearningAgent`. Low-talent RL agents
  (`LowTalentQLearningAgent`) load a separate policy trained with
  `train_rl.py --low-talent` (saved to `policies/policy_<backend>_low_talent`).
- **DQN settings** are separate (`num_dqn_agents`, `dqn_*` hyperparameters);
  train with `train_dqn.py`, policies saved to `policies/policy_dqn.pkl`.

## Architecture map

- `config.py` - `SimConfig` (`SIM`) and `TrainConfig` (`TRAIN`) dataclasses;
  config-first with CLI overrides. `default_policy_path()` resolves policy files.
- `Agent.py` - abstract `Agent` base + the action protocol: `write_paper`,
  `peer_review`, `finish_review_write_paper`, `finish_review_peer_review`.
  `ActionRecord` describes one turn. `Agent.all_papers` is a shared class list.
- `HeuristicAgent.py`, `QLearningAgent.py`, `DiscreteQLearningAgent.py`, `DQNAgent.py`, `RandomAgent.py` - agent variants,
  including random controls and discrete-only probability agents.
- `Paper.py` - paper economics, reviews, and accrual; defines
  `MIN_REVIEW_EFFORT_THRESHOLD`, `REVIEW_EFFORT_PER_DAY`.
- `Environment.py` - the world / turn loop (`agentact`, `nextstep`).
- `History.py` - run logging and metrics (e.g. `gini`).
- `run_simulation.py` - main entry point (run a sim, print summary, optionally
  archive to the `docs/` gallery).
- `train_rl.py` - continuous self-play RL training + greedy evaluation.
- `train_discrete_rl.py` - discrete 3-action RL; saves `policies/policy_<backend>_discrete.pkl`.
- `train_dqn.py` - DQN self-play training; auto-saves to `policies/policy_dqn.pkl`.
- `visualize.py` - charts and summary figures for saved runs.
- `docs/` - static GitHub Pages gallery of saved runs (`docs/data/<run_id>/`).

## Commands

```bash
# Run a simulation (prompts to archive afterward)
python run_simulation.py
python run_simulation.py --no-archive            # don't save
python run_simulation.py --name "my run"         # save non-interactively
python run_simulation.py --review-paradigm discrete --random-agents 5 --no-archive
python run_simulation.py --rl-agents 20 --rl-backend tabular
python run_simulation.py --seeds 1,2,3,4,5 --timesteps 10000 --heuristic-agents 50 --rl-agents 50 --name "discrete sweep"

# Train continuous RL (6-action merged phase; auto-saves to policies/)
python train_rl.py
python train_rl.py --backend linear --episodes 300
python train_rl.py --load policies/policy_tabular.pkl --episodes 0   # eval only
python train_rl.py --low-talent --no-archive                       # low-talent policy

# Train discrete RL (3-action: write / bad claim / good claim)
python train_discrete_rl.py
python train_discrete_rl.py --backend linear --episodes 300
python train_discrete_rl.py --load policies/policy_tabular_discrete.pkl --episodes 0

# Train the DQN agent (auto-saves to policies/policy_dqn.pkl)
python train_dqn.py
python train_dqn.py --num-dqn 20 --episodes 300
python run_simulation.py --dqn-agents 10 --rl-agents 0 --no-archive

# Tests (stdlib unittest)
python test_simulation.py
python -m unittest test_simulation
```

## Experiment workflow

Standard loop for parameter experiments (continuous paradigm, full agent
comparison):

1. Edit defaults in `config.py` (`SimConfig` / `TrainConfig`).
2. Train RL: `python train_rl.py --no-archive`
3. Run simulation with full comparison and archive: `python run_simulation.py ... --name "<title>"`
4. Check `runs/` charts and the archived `docs/data/<run_id>/` path.

**What we're looking for:** emergent good-faith vs bad-faith peer review under
the configured economics. Key outputs: agent-group comparison (heuristic vs
random vs probabilistic vs RL), choice breakdown, review behavior, and capital
inequality (Gini). Produced by `run_simulation.py` + `visualize.py`; archived
via `export_run.py`.

### Step 1 — Change parameters

Edit `config.py` defaults. CLI flags are for one-off overrides only. Common
experiment knobs:

- `review_bump_duration`, `review_bump_decay_rate`, `review_bump_decay_cap_timesteps`
- `paper_effort_mode`, `paper_effort_min`, `paper_effort_max`
- `continuous_publishing`, `continuous_paper_timesteps`
- Market economics toggles (`use_scarcity_pricing`, etc.)

`review_paradigm` must match the training script. Default workflow uses
**continuous** → `train_rl.py` (not `train_discrete_rl.py`).

### Step 2 — Train continuous RL

```bash
python train_rl.py --no-archive
```

Defaults (`TrainConfig`): 200 episodes, 1000 timesteps/episode, 10 RL agents,
tabular backend. Policy auto-saves to `policies/policy_tabular.pkl` (or
`policy_linear.npy` for linear). Training logs/charts go to `runs/`. Use
`--name "training: <experiment>"` to archive training metrics to the gallery.

`run_simulation.py` auto-loads the saved policy when `rl_autoload_policy=True`
(default) and `review_paradigm=continuous`. Low-talent RL policies auto-load
from `policies/policy_<backend>_low_talent` when `num_low_talent_rl_agents > 0`
and `rl_low_talent_autoload_policy=True` (default).

Train a low-talent baseline:

```bash
python train_rl.py --low-talent --no-archive
```

### Step 3 — Run simulation (full comparison)

```bash
python run_simulation.py \
  --heuristic-agents 20 \
  --rl-agents 20 \
  --low-talent-rl-agents 20 \
  --random-agents 20 \
  --probabilistic-agents 20 \
  --name "<descriptive experiment name>"
```

Full-comparison default: **20 agents per group** (80 total). Always pass
`--name` for non-interactive runs so results archive without a prompt.

**Local outputs** (overwritten each run in `runs/`): `history.csv`,
`history.json`, and charts (`summary.png`, `agent_group_comparison.png`,
`choice_breakdown.png`, `review_behavior.png`, etc.).

**Gallery archive** (`--name` provided): `docs/data/<run_id>/` (slim JSON +
chart PNGs for GitHub Pages); `local_data/<run_id>/history.json` (full lossless
history, gitignored).

### Step 4 — Verify results

1. Terminal summary: agent-group comparison + good/bad-faith review counts.
2. `runs/summary.png` and `runs/agent_group_comparison.png`.
3. Gallery path: `Archived run to docs/data/<run_id>/`.

Optional publish: `git add docs/data && git commit -m 'Add run: <name>' && git push`

### "Run it" shorthand

When the user says **"run it"** or **"run the experiment"**:

1. Read current `config.py` defaults to know active parameters.
2. Run `python train_rl.py --no-archive`.
3. Run `python run_simulation.py` with full-comparison agent counts and
   `--name "<name>"` — ask for the name if not provided.
4. Report: policy path, `runs/` chart paths, archived `run_id`, and a one-line
   summary (good/bad-faith ratio, RL vs heuristic mean capital).

If parameters were just changed, confirm `config.py` state before running.

### Prerequisites

- Python 3; stdlib for sim logic.
- `matplotlib` for charts: `python -m pip install matplotlib`
- Full pipeline (train + 2000-timestep full comparison) may take several minutes.

### Adaptive surplus sweep

Overnight-friendly script that trains RL, runs a simulation, logs author/reviewer
surplus to a live CSV, and adapts accrual-bump parameters each step:

```bash
python3 sweep_surplus.py
python3 sweep_surplus.py 2>&1 | tee runs/sweep_surplus.log
```

### Example (decay bump experiment)

After setting in `config.py`:

```python
review_bump_duration = "decay"
review_bump_decay_rate = 0.05
```

```bash
python train_rl.py --no-archive
python run_simulation.py \
  --heuristic-agents 20 --rl-agents 20 \
  --random-agents 20 --probabilistic-agents 20 \
  --name "decay bump k=0.05 full comparison"
```

## Conventions

- **Stdlib-only where possible** (`config.py` uses only `dataclasses`); matplotlib
  is optional and guarded behind an import check in tests.
- **Config-first**: change defaults in `config.py`; use CLI flags for one-offs.
- **Determinism**: runs use a fixed `seed` (default 7) for reproducibility.
- Frozen dataclasses for config; type hints use `from __future__ import annotations`.

## Accepted legacy test failures (2026-09-28)

The project owner explicitly requested that these nine existing test_simulation
failures be ignored in future work. Do not investigate or block commits/pushes
on these known failures unless requested. This exemption does not cover new
failures. The same nine reproduce with publication experience disabled.

- ContinuousMergedPhaseTest.test_reviewer_claim_switch_finalizes_old_review
- EconomicsTest.test_review_bump_rises_and_saturates_with_sigmoid
- HeuristicPolicyTest.test_work_phase_continues_when_marginal_effort_dominates
- MarketEconomicsTest.test_environment_scarcity_multiplier_when_crowded
- MarketplaceLifecycleTest.test_min_effort_review_earns_reward
- ReviewParadigmTest.test_continuous_mode_classifies_finished_reviews_by_threshold
- ReviewParadigmTest.test_discrete_bad_faith_review_finishes_after_one_timestep
- ReviewerStateTest.test_grabbing_new_paper_finalizes_active_review
- ReviewerStateTest.test_peer_review_history_updates_on_completion
