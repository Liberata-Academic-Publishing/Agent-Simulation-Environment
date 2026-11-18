"""Reproducible fixed-strategy comparison; writes only to the supplied output dir."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
from statistics import mean, median, stdev


def mean_or_zero(values):
    values = list(values)
    return mean(values) if values else 0.0

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


def run(seed, steps, actions=None, *, count_per_group=SIM.talent_agents_per_group):
    if steps <= 0:
        raise ValueError('steps must be positive')
    random.seed(seed)
    groups = build_talent_cohorts(count_per_group)
    agents = [a for group in groups.values() for a in group]
    actions = Actions() if actions is None else actions
    env = Environment(agents=agents, papers=[], history=actions,
                      continuous_publishing="threshold", paper_effort_mode="fixed",
                      review_paradigm="continuous", use_merit_market_clearing=False)
    env.run(steps)
    actions.environment = env
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
                         initial_quality=mean_or_zero(q for a in group for q in actions.initial_quality[a]),
                         review_quality=mean_or_zero(r['review_quality'] for r in records),
                         review_delta=mean_or_zero(r['review_quality_delta'] for r in records),
                         capital=mean(a.academic_capital for a in group),
                         review_capital=mean(sum(p.current_ac*p.share_distribution.get(a, 0)
                                                for p in env.papers if p.author is not a) for a in group),
                         good_reviews=sum(r['review_kind']=='good_faith' for r in records),
                         bad_reviews=sum(r['review_kind']=='bad_faith' for r in records)))
    return rows


def mechanism_results(env):
    agents = []
    for agent in env.agents:
        papers = [p for p in env.papers if p.author is agent]
        agents.append(dict(name=agent.name, group=agent.name.rsplit('_', 1)[0],
            quality_talent=agent.quality_talent, rate_talent=agent.rate_talent,
            publications=agent.publication_count,
            experience_multiplier=agent.experience_multiplier,
            effective_quality_talent=agent.effective_quality_talent,
            effective_rate_talent=agent.effective_rate_talent,
            citations=sum(p.citation_count for p in papers),
            capital=agent.academic_capital))
    claimed = [p for p in env.papers if p.claimed_timestep is not None]
    waits = [p.claimed_timestep - p.listed_timestep for p in claimed]
    pending = [p for p in env.papers if p.review_available]
    records = [r for p in env.papers for r in p.review_records]
    good = sum(r['review_kind'] == 'good_faith' for r in records)
    bad = sum(r['review_kind'] == 'bad_faith' for r in records)
    return dict(agents=agents, market=dict(
        claimed_papers=len(claimed), mean_wait=mean(waits) if waits else None,
        median_wait=median(waits) if waits else None,
        max_wait=max(waits) if waits else None,
        unclaimed_listed=len(pending),
        unclaimed_ages=[env.timestep-p.listed_timestep for p in pending],
        good_reviews=good, bad_reviews=bad,
        bad_to_good_ratio=bad/good if good else None),
        total_papers=len(env.papers),
        total_citations=sum(p.citation_count for p in env.papers),
        total_citation_ac=sum(p.citation_accrued for p in env.papers),
        total_agent_ac=sum(a.academic_capital for a in env.agents))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', default='1,2,3,4,5')
    parser.add_argument('--agents-per-group', type=int, default=SIM.talent_agents_per_group)
    parser.add_argument('--output', default='experiments/fixed_strategy_integrated')
    args = parser.parse_args()
    rows = []
    details = []
    for seed in map(int, args.seeds.split(',')):
        actions = Actions()
        rows.extend(run(seed, args.steps, actions, count_per_group=args.agents_per_group))
        details.append(dict(seed=seed, **mechanism_results(actions.environment)))
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
        strategy='publish_then_good_faith_review', training_performed=False,
        mechanisms=dict(shared_talents=True, experience_alpha=SIM.experience_alpha,
                        experience_h=SIM.experience_h, citations=SIM.citations_enabled),
        overrides=dict(steps=args.steps, seeds=args.seeds, initial_papers=0,
                       agents_per_group=args.agents_per_group, forecast_horizon_timesteps=30,
                       continuous_publishing='threshold', paper_effort_mode='fixed',
                       review_paradigm='continuous', use_merit_market_clearing=False),
        per_seed=rows, summary=summary, mechanism_results=details), indent=2), encoding='utf-8')
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
    fig.suptitle(f'Two talents + experience + citations | {args.agents_per_group} agents/group | {args.steps} turns\n'
                 f'{len(rows)//4} seeds; error bars = SD of seed-level group means')
    fig.tight_layout(rect=(0,0,1,0.93))
    fig.savefig(output/'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    for detail in details:
        print(json.dumps({k: v for k, v in detail.items() if k != 'agents'}, indent=2))


if __name__ == '__main__':
    main()
