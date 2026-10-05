"""Four diagnostic plots, with raw time-series CSVs, for the fixed baseline."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_talent_comparison import Actions, run

OUT = Path('experiments/two_talent_comparison/diagnostics')
GROUPS = ['quality_low_rate_low', 'quality_low_rate_high',
          'quality_high_rate_low', 'quality_high_rate_high']
LABELS = ['Low quality / Slow', 'Low quality / Fast',
          'High quality / Slow', 'High quality / Fast']
COLORS = ['#64748b', '#0284c7', '#8b5cf6', '#059669']
SEEDS = [1, 2, 3, 4, 5]
STEPS = 1000


class Diagnostics(Actions):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        self.daily = Counter()
        self.group_rows = []
        self.market_rows = []

    def record_action(self, env, agent, record):
        super().record_action(env, agent, record)
        group = agent.name.rsplit('_', 1)[0]
        work = 'writing' if record.kind in ('write_paper', 'review_finished_write') else 'reviewing'
        self.daily[group, work] += 1
        if record.published:
            self.daily['published'] += 1
        if record.kind == 'review_started':
            self.daily['claimed'] += 1
        if record.kind == 'review_finished_write':
            self.daily[group, record.review_kind] += 1

    def record_step(self, env):
        for group in GROUPS:
            agents = [a for a in env.agents if a.name.rsplit('_', 1)[0] == group]
            self.group_rows.append(dict(seed=self.seed, timestep=env.timestep, group=group,
                mean_ac=sum(a.academic_capital for a in agents)/len(agents),
                writing_turns=self.daily[group, 'writing'],
                review_turns=self.daily[group, 'reviewing'],
                good_reviews=self.daily[group, 'good_faith'],
                bad_reviews=self.daily[group, 'bad_faith']))
        self.market_rows.append(dict(seed=self.seed, timestep=env.timestep,
            published=self.daily['published'], claimed=self.daily['claimed'],
            listed_backlog=sum(p.review_available for p in env.papers),
            active_reviews=sum(p.review_in_progress_by is not None for p in env.papers),
            awaiting_listing=sum(not p.market_listed and not p.review_claimed and not p.reviewed for p in env.papers),
            total_papers=len(env.papers)))
        self.daily.clear()


def save_csv(name, rows):
    with (OUT/name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, name, note):
    fig.text(0.08, 0.025, note, fontsize=9, color='#475569')
    fig.savefig(OUT/name, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    group_rows, market_rows, results = [], [], []
    for seed in SEEDS:
        recorder = Diagnostics(seed)
        rows = run(seed, STEPS, actions=recorder)
        results.extend(rows)
        assert all(r['writing_turns'] + r['review_turns'] == 20 for r in recorder.group_rows)
        for row in rows:
            selected = [r for r in recorder.group_rows if r['group'] == row['group']]
            assert sum(r['good_reviews'] for r in selected) == row['good_reviews']
            assert sum(r['bad_reviews'] for r in selected) == row['bad_reviews']
        group_rows.extend(recorder.group_rows)
        market_rows.extend(recorder.market_rows)
        print(f'Collected seed {seed}', flush=True)
    baseline = json.loads((OUT.parent/'results.json').read_text(encoding='utf-8'))
    assert results == baseline['per_seed'], 'Instrumented run differs from saved baseline'
    save_csv('agent_group_timeseries.csv', group_rows)
    save_csv('market_timeseries.csv', market_rows)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.grid': True, 'grid.alpha': 0.15})
    t = np.arange(1, STEPS+1)
    def group_series(group, key):
        return np.array([[r[key] for r in group_rows if r['seed']==s and r['group']==group] for s in SEEDS])
    def market_series(key):
        return np.array([[r[key] for r in market_rows if r['seed']==s] for s in SEEDS])
    def band(ax, x, data, label, color):
        avg, sd = data.mean(axis=0), data.std(axis=0, ddof=1)
        ax.plot(x, avg, label=label, color=color, linewidth=2.2)
        ax.fill_between(x, np.maximum(0, avg-sd), avg+sd, color=color, alpha=0.14)

    fig, ax = plt.subplots(figsize=(10, 5.7))
    fig.subplots_adjust(bottom=0.18)
    for g, label, color in zip(GROUPS, LABELS, COLORS):
        data = group_series(g, 'mean_ac')
        band(ax, t, data, f'{label} ({data[:, -1].mean():,.0f})', color)
    ax.set(title='Academic capital over time', xlabel='Simulation timestep', ylabel='Mean AC per agent')
    ax.legend(loc='upper left', frameon=False)
    finish(fig, '01_ac_curves.png', '20 agents/group; 5 seeds. Shading: +/-1 seed SD. AC uses the existing accrual model, not citations.')

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.3), sharex=True)
    fig.subplots_adjust(bottom=0.14, hspace=0.32)
    x = np.arange(50, STEPS+1, 50)
    for key, label, color in [('published','New publications','#0284c7'), ('claimed','New review claims','#d97706')]:
        data = market_series(key).reshape(5, 20, 50).sum(axis=2)
        band(axes[0], x, data, label, color)
    axes[0].set(title='Paper supply and review demand', ylabel='Count per 50-step window')
    axes[0].legend(frameon=False)
    for key, label, color in [('listed_backlog','Listed, unclaimed papers','#dc2626'), ('active_reviews','Reviews in progress','#059669')]:
        band(axes[1], t, market_series(key), label, color)
    axes[1].set(xlabel='Simulation timestep', ylabel='End-of-step count')
    axes[1].legend(frameon=False)
    finish(fig, '02_paper_supply.png', 'Whole market (80 agents); 5-seed means +/-1 SD. Backlog excludes newly published papers awaiting listing.')

    fig, ax = plt.subplots(figsize=(10, 5.7))
    fig.subplots_adjust(bottom=0.2, top=0.86)
    totals = []
    for g in GROUPS:
        good = group_series(g, 'good_reviews').sum()
        bad = group_series(g, 'bad_reviews').sum()
        totals.append((int(good), int(bad)))
    good_pct = [100*a/(a+b) for a,b in totals]
    ax.bar(np.arange(4), good_pct, color='#059669', label='Good faith')
    ax.bar(np.arange(4), [100-v for v in good_pct], bottom=good_pct, color='#dc2626', label='Bad faith')
    for i, (good, bad) in enumerate(totals):
        ax.text(i, 50, f'100% good\n{good:,} good\n{bad} bad', ha='center', va='center', color='white', weight='bold', fontsize=11)
    ax.set_xticks(np.arange(4), [s.replace(' / ', '\n') for s in LABELS])
    ax.set(ylim=(0, 105), ylabel='Share of completed reviews (%)', title='Completed reviews: good faith vs. bad faith')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    finish(fig, '03_review_faith.png', 'Counts pooled across 5 seeds. All agents are programmed to review in good faith; this is NOT a strategy finding.')

    fig, ax = plt.subplots(figsize=(10, 5.7))
    fig.subplots_adjust(bottom=0.2)
    review = np.array([group_series(g,'review_turns').sum(axis=1)/(20*STEPS)*100 for g in GROUPS])
    avg, sd = review.mean(axis=1), review.std(axis=1, ddof=1)
    ax.barh(np.arange(4), 100-avg, color='#0284c7', label='Writing')
    ax.barh(np.arange(4), avg, left=100-avg, color='#f59e0b', label='Reviewing')
    for i,v in enumerate(avg):
        ax.text(45,i,f'{100-v:.1f}%',ha='center',va='center',color='white',weight='bold')
        ax.text(100-v/2,i,f'{v:.1f}%',ha='center',va='center',weight='bold')
    ax.set_yticks(np.arange(4), LABELS)
    ax.invert_yaxis()
    ax.set(xlim=(0,100), xlabel='Share of agent timesteps (%)', title='Time allocation under the same fixed strategy')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.12), ncol=2, frameon=False)
    finish(fig, '04_time_allocation.png', '5-seed means; every timestep is accounted for. Review-share seed SD ranges from %.2f to %.2f percentage points.' % (sd.min(),sd.max()))
    print(json.dumps({'faith_counts':dict(zip(GROUPS,totals)),
        'final_listed_backlog_mean':market_series('listed_backlog')[:,-1].mean(),
        'final_active_reviews_mean':market_series('active_reviews')[:,-1].mean(),
        'final_total_papers_mean':market_series('total_papers')[:,-1].mean(),
        'baseline_exactly_reproduced':True}, indent=2))


if __name__ == '__main__':
    main()
