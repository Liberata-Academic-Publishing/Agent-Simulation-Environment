"""Collect a two-talent run using the current experience configuration."""
from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import mean, median
import subprocess

from config import SIM
from run_talent_comparison import Actions, run


class PublicationHistory(Actions):
    def __init__(self):
        super().__init__()
        self.publications = Counter()
        self.publication_times = {}
        self.snapshots = {}
        self.agents = []

    def record_action(self, env, agent, record):
        super().record_action(env, agent, record)
        if record.published:
            self.publications[agent] += 1
            self.publication_times.setdefault(agent.name, []).append(env.timestep)

    def record_step(self, env):
        self.agents = env.agents
        if env.timestep in (250, 500, 750, 1000):
            counts = Counter(p.author for p in env.papers)
            assert all(counts[a] == self.publications[a] for a in env.agents)
            self.snapshots[env.timestep] = {
                a.name: counts[a] for a in env.agents
            }
            print(f"Step {env.timestep}: {len(env.papers)} publications", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    history = PublicationHistory()
    rows = run(seed=11, steps=1000, actions=history)
    agent_rows = [dict(name=a.name, quality_talent=a.quality_talent,
                       rate_talent=a.rate_talent,
                       publications=history.publications[a],
                       experience_multiplier=a.experience_multiplier,
                       effective_quality_talent=a.effective_quality_talent,
                       effective_rate_talent=a.effective_rate_talent,
                       publication_times=history.publication_times.get(a.name, []))
                  for a in history.agents]
    files = ("config.py", "Agent.py", "Paper.py", "Environment.py",
             "talent_agents.py", "run_talent_comparison.py",
             "calibrate_experience_baseline.py")
    result = dict(
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        working_tree=subprocess.check_output(["git", "status", "--short"], text=True),
        source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        config=asdict(SIM),
        overrides=dict(seed=11, steps=1000, initial_papers=0,
                       continuous_publishing="threshold", paper_effort_mode="fixed",
                       review_paradigm="continuous", use_merit_market_clearing=False,
                       forecast_horizon_timesteps=30),
        experience_enabled=SIM.experience_alpha > 0, groups=rows, agents=agent_rows,
        snapshots=history.snapshots,
    )
    (output / "baseline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in rows:
        group_counts = [a["publications"] for a in agent_rows
                        if a["name"].rsplit("_", 1)[0] == row["group"]]
        print(row["group"], dict(mean=mean(group_counts), median=median(group_counts),
                                 minimum=min(group_counts), maximum=max(group_counts)))
    for step, snapshot in history.snapshots.items():
        print("Snapshot", step, "median", median(snapshot.values()))
    print(f"Saved {output / 'baseline.json'}")


if __name__ == "__main__":
    main()
