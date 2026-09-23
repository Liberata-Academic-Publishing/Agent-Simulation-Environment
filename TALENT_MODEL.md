# Two shared talents

`talent_agents.build_talent_cohorts()` constructs four groups of 20 agents
(80 total). Quality and rate talents independently take the low/high values
in `config.py`, currently 0.6 and 1.4. All groups use the same fixed strategy:
write a paper, then attempt one good-faith review, then write again. When no
review is available they continue writing. These are prescribed behaviors,
not learned strategies or evidence of an equilibrium.

Writing and reviewing share the agent's two talent parameters:

- Paper quality: one draw from `max(min_paper_quality, Normal(quality_talent,
  quality_sigma))` when a manuscript starts; reused on publication.
- Review quality: one independent draw from that same quality distribution
  when a review completes. This is a multiplier of the existing effort-based
  review effect, not a replacement for the reviewed paper's quality.
- Progress: a fresh draw from `max(talent_min_rate, Normal(rate_talent,
  talent_rate_sigma))` on each working turn, multiplied by the existing base
  writing/review progress. A larger rate therefore reduces completion time.

These are floored Gaussians (with probability mass at the floor), not
truncated-normal distributions. Their location parameters need not equal the
post-floor means. No claim of empirical calibration is made.

`Paper.paper_quality` stores actual quality; `quality` is a compatibility
property referring to the same value. For enabled agents, a completed eligible
review adds `old_quality * epsilon` to quality, where epsilon is the existing
effort-based proportional improvement times the sampled review quality.
The existing AC bump uses the same epsilon; locked ownership shares do not
depend on the quality draw. Quality improvement persists even when the legacy
AC bump decays. Continuous good/bad classification still uses effective work,
not elapsed clock time; discrete mode retains the declared review kind.

Existing agents retain legacy behavior unless `configure_talents(quality,
rate)` is called before work starts. TalentAgent enables it automatically.
The standard simulation CLI does not automatically replace its population.
Existing RL policies and economic forecast heuristics are not retrained or
recalibrated for these distributions. The citation model is unchanged.

Minimal use:

```python
import random
from config import SIM
from Environment import Environment
from talent_agents import build_talent_cohorts

random.seed(SIM.seed)
groups = build_talent_cohorts()
agents = [agent for group in groups.values() for agent in group]
env = Environment(agents=agents, papers=[],
                  continuous_publishing="threshold",
                  use_merit_market_clearing=False)
env.run(1000)
```

Validation: `python -m unittest test_talents -v`.
