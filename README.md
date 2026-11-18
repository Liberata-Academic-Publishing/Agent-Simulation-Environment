# Liberata Peer-Review Simulation

Academic research project: an agent-based simulation of incentive structures, market dynamics, and quality accrual on the Liberata academic publishing platform. Agents allocate each timestep between advancing their own research and participating in a single-review peer-review marketplace.

## Single-review marketplace

Each paper has a `quality` sampled from a Gaussian centered on its author's intrinsic talent, known to the author before they start writing. Quality sets the paper's base accrual rate and the accrual bump a review can earn. A paper is listed on the market one timestep after it is written, and it can be reviewed exactly once: the first agent to claim it takes it off the market permanently.

While a paper is listed, its author offers each potential reviewer a distinct share price (`Paper.price_table`). The default base offer splits incremental review surplus fairly: `ε/(1+ε) × (F−A₀)/F × reviewer_surplus_share` (default 50/50), using each reviewer's epsilon history and the same forecast horizon agents use for claim decisions. A higher-quality paper (relative to the market) offers a smaller share; scarcity and adaptive author multipliers can adjust offers further. The price table refreshes every timestep because it depends on which papers are currently on the market.

`peer_review_history` remains a public per-agent reputation metric: the mean share-weighted accrual rate on papers the agent has reviewed (i.e. reviewer share times each paper's accrual rate, averaged over completed reviews). `peer_review_epsilon_history` separately tracks the average proportional accrual improvement caused by that reviewer, and is the metric used for fair-market pricing.

## Timestep structure

Every timestep runs in two phases over a freshly shuffled agent order:

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

The minimum reward threshold is one timestep, so a one-timestep review earns the
smallest quality-scaled accrual bump. The default review reward curve is a
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

`train_rl.py` trains a Q-learning agent against heuristic opponents. Note that the action space and feature vector changed with the single-review marketplace overhaul, so any policy saved before that change (in `policies/`) is incompatible and must be retrained. `run_simulation.py` defaults to heuristic agents; pass `--rl-agents N` to include RL agents.


### Experimental AC/price reviewer marketplace

An opt-in, bounded-search bidding mechanism lets reviewers quote ownership
shares and authors select by existing academic capital divided by the quote.
It supports fixed heuristic/random strategies; legacy remains the default.
See [mechanism and evaluation](design_docs/ac_review_market.md) for parameters,
assumptions, run commands and the paired comparison script. No RL training is
needed for the pilot.
