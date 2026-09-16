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