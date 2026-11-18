# Simulation Quickstart

This guide covers the integrated fixed-strategy simulation. It uses four talent cohorts, publication experience, and citations. It does not train or load an RL policy.

## Run

```powershell
cd C:\Agent-Simulation-Environment
python run_simulation.py --fixed-strategy --timesteps 1000 --seed 11 --name "Fixed strategy - talent experience citation"
```

| Argument | Meaning |
|---|---|
| `--fixed-strategy` | Enables fixed-strategy cohorts, experience, and citations. |
| `--timesteps 1000` | Runs 1,000 steps; each agent acts once per step. |
| `--seed 11` | Reproducible random seed. |
| `--agents-per-group 20` | Agents per cohort; default 20, 80 total. |
| `--name "..."` | Archives the run in the visualization gallery. |
| `--no-archive` | Writes local outputs only. |

## Fixed-strategy settings

| Area | Setting |
|---|---|
| Cohorts | Low quality/slow rate, low quality/fast rate, high quality/slow rate, high quality/fast rate. |
| Base means | Quality: 0.6/1.4; rate: 0.6/1.4. |
| Gaussian spread | Quality and rate standard deviations are 0.20. Experience does not change them. |
| Experience | `M(n) = 1 + 0.30 * n / (n + 9)`, where `n` is the agent's published-paper count. |
| Experience effect | Multiplies both quality and rate means; base talent is not mutated or compounded. |
| Initial state | No initial papers and no initial capital. |
| Writing | Each paper requires 50 units of effort, not necessarily 50 steps. |
| Strategy | After publishing, seek one good-faith review; write when no review is available. |
| Review | Continuous mode; good faith at 5 effort; minimum reward/share threshold is 3 effort. |
| Marketplace | New papers list on the next step; claiming removes them from the market; one review per paper. |
| Pricing | Adaptive pricing targets a one-step wait; this is not a forced delay. |
| Citations | Enabled; each new paper cites up to 20 distinct earlier papers and distributes 1 AC across them. |
| RL | Not loaded, trained, or updated in this mode. |

At nine published papers the experience multiplier is 1.15; it approaches 1.30 over time.

## Outputs and checks

| Output or metric | What to inspect |
|---|---|
| `runs/history.json`, `runs/history.csv` | Latest time series; overwritten by the next run. |
| `runs/agent_group_comparison.png` | Capital and review comparison across cohorts. |
| `runs/review_behavior.png` | Good/bad review and paper counts. |
| `runs/marketplace_activity.png` | Listings, claims, active reviews, and backlog. |
| `runs/fixed_strategy_metrics.json` | Per-agent talent, experience, publications, citations, capital, and market waits. |
| `docs/data/<run_id>/` | Archived gallery data and charts. |
| `experience_multiplier > 1` | Experience is active. |
| `total_citations > 0` | The citation network is active. |
| `total_agent_ac ~= total_citation_ac` | Citation AC is conserved when initial capital is zero. |
| `market.mean_wait` | Mean listing-to-claim wait. |
| `market.good_reviews` / `bad_reviews` | Fixed strategy normally produces good reviews only; this is prescribed, not learned. |

Verified seed-11 run: 1,577 papers, 31,191 citations, 1,576 AC, mean wait about 0.231 steps, and 1,562 good / 0 bad reviews. Exact values can change with code or configuration.

Focused checks:

```powershell
python -m unittest test_fixed_strategy test_experience test_talents -v
```

## Changing settings

| Goal | How |
|---|---|
| Change steps, seed, or cohort size | Use the command-line arguments above. |
| Change talent, experience, citation, or review mechanics | Edit `config.py`. |
| Train the shared-talent RL variant later | `python train_rl.py --shared-talents --num-rl 80 --horizon 30 --no-archive` |
| Update the online gallery | Push `docs/data`, merge into `main`, and wait for GitHub Pages to deploy `/docs`. |

Gallery: <https://liberata-academic-publishing.github.io/Agent-Simulation-Environment/>
