"""Reproducible fixed-strategy comparison; writes only to the supplied output dir."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import mean, stdev

from Agent import Agent
from Environment import Environment
from config import SIM
from talent_agents import build_talent_cohorts


class Actions:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.initial_quality = defaultdict(list)

    def record_action(self, env, agent, record):
        self.counts[agent][record.kind] += 1
        if record.published:
            self.initial_quality[agent].append(env.papers[-1].paper_quality)

    def record_step(self, env):
        pass


def run(seed, steps):
    random.seed(seed)
    groups = build_talent_cohorts()
    agents = [a for group in groups.values() for a in group]
    actions = Actions()
    env = Environment(agents=agents, papers=[], history=actions,
                      continuous_publishing="threshold", paper_effort_mode="fixed",
                      review_paradigm="continuous", use_merit_market_clearing=False)
    env.run(steps)
    rows = []
    for label, group in groups.items():
        papers = [p for p in env.papers if p.author in group]
        records = [r for p in env.papers for r in p.review_records if r['reviewer'] in group]
        counts = sum((actions.counts[a] for a in group), Counter())
        writing = counts['write_paper'] + counts['review_finished_write']
        reviewing = counts['review_started'] + counts['review_continued']
        assert writing + reviewing == len(group) * steps
        assert all(abs(sum(p.share_distribution.values()) - 1) < 1e-8 for p in papers)
        rows.append(dict(seed=seed, group=label,
                         papers_per_agent=len(papers)/len(group),
                         reviews_per_agent=len(records)/len(group),
                         writing_percent=100*writing/(len(group)*steps),
                         review_percent=100*reviewing/(len(group)*steps),
                         initial_quality=mean(q for a in group for q in actions.initial_quality[a]),
                         review_quality=mean(r['review_quality'] for r in records),
                         review_delta=mean(r['review_quality_delta'] for r in records),
                         capital=mean(a.academic_capital for a in group),
                         review_capital=mean(sum(p.current_ac*p.share_distribution.get(a, 0)
                                                for p in env.papers if p.author is not a) for a in group),
                         good_reviews=sum(r['review_kind']=='good_faith' for r in records),
                         bad_reviews=sum(r['review_kind']=='bad_faith' for r in records)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', default='1,2,3,4,5')
    parser.add_argument('--output', default='experiments/two_talent_comparison')
    args = parser.parse_args()
    rows = []
    for seed in map(int, args.seeds.split(',')):
        rows.extend(run(seed, args.steps))
        print(f'Finished seed {seed}', flush=True)
    groups = list(dict.fromkeys(r['group'] for r in rows))
    metrics = [key for key in rows[0] if key not in ('seed', 'group')]
    summary = {}
    for group in groups:
        subset = [r for r in rows if r['group']==group]
        summary[group] = {key: {'mean': mean(r[key] for r in subset),
                                'seed_sd': stdev(r[key] for r in subset) if len(subset)>1 else 0}
                          for key in metrics}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(dict(config=asdict(SIM),
        overrides=dict(steps=args.steps, seeds=args.seeds, initial_papers=0,
                       continuous_publishing='threshold', paper_effort_mode='fixed',
                       review_paradigm='continuous', use_merit_market_clearing=False),
        per_seed=rows, summary=summary), indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    labels = ['Low Q\nSlow', 'Low Q\nFast', 'High Q\nSlow', 'High Q\nFast']
    colors = ['#94a3b8', '#38bdf8', '#a78bfa', '#10b981']
    for ax, key, title in zip(axes.flat,
        ['papers_per_agent','reviews_per_agent','writing_percent','capital'],
        ['Papers per agent', 'Completed reviews per agent', 'Time spent writing (%)', 'Final AC per agent']):
        ax.bar(labels, [summary[g][key]['mean'] for g in groups],
               yerr=[summary[g][key]['seed_sd'] for g in groups], color=colors, capsize=4)
        ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
        if key=='writing_percent':
            ax.set_ylim(0, 100)
    fig.suptitle(f'Two talents, same fixed strategy | 20 agents/group | {args.steps} turns\n'
                 f'{len(rows)//4} seeds; error bars = SD of seed-level group means')
    fig.tight_layout(rect=(0,0,1,0.93))
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
