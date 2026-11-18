"""One reproducible fixed-strategy run using the normal simulation builder."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collections import Counter
from dataclasses import asdict
import json
import math
import statistics

from config import SIM
from History import History, gini
import run_simulation as runner


def main():
    output = Path(__file__).resolve().parent
    history = History()
    env = runner.build_simulation(
        history, num_agents=20, random_agents=20, rl_agents=0,
        low_talent_rl_agents=0, dqn_agents=0, probabilistic_agents=0,
        market_mechanism='sampled_bidding', seed=11,
    )
    initial_papers = len(env.papers)
    for step in range(1000):
        before = len(history.actions)
        env.run_timestep()
        if len(history.actions) - before != len(env.agents):
            raise RuntimeError('incorrect labour turn count')
        busy = [a.active_review_paper for a in env.agents if a.active_review_paper is not None]
        if len(busy) != len(set(busy)):
            raise RuntimeError('duplicate review assignment')
        for agent in env.agents:
            if not math.isfinite(agent.academic_capital) or agent.academic_capital < 0:
                raise RuntimeError('invalid capital')
            if agent.active_review_paper is not None and agent.active_review_paper.review_in_progress_by is not agent:
                raise RuntimeError('inconsistent review state')
        for paper in env.papers:
            shares = list(paper.share_distribution.values())
            if not all(math.isfinite(s) and 0 <= s <= 1 for s in shares) or not math.isclose(sum(shares), 1, abs_tol=1e-9):
                raise RuntimeError('invalid ownership')
            if len(paper.review_records) > 1 or paper.reviewer is paper.author:
                raise RuntimeError('invalid completed review')
        if not math.isclose(sum(a.academic_capital for a in env.agents), sum(p.current_ac for p in env.papers), rel_tol=1e-9, abs_tol=1e-7):
            raise RuntimeError('capital accounting mismatch')
        m = env.review_market.metrics
        if m['candidates'] > SIM.market_candidate_limit * m['requests']:
            raise RuntimeError('search cap exceeded')
        if (step + 1) % 200 == 0:
            print(f'step={step+1}, papers={len(env.papers)}, completed={sum(p.reviewed for p in env.papers)}, pending={sum(p.review_available for p in env.papers)}', flush=True)

    trades = history.market_trades
    allocations = Counter(t['reviewer_id'] for t in trades)
    actions = Counter(kind for _, _, kind, _ in history.actions)
    completed = history.completed_reviews
    good = sum(kind == 'good_faith' for _, _, _, _, kind in completed)
    pending = [p for p in env.papers if p.review_available]
    writing = sum(kind in {'write_paper', 'review_finished_write'} for _, _, kind, _ in history.actions)
    windows = {}
    for label, lo, hi in [('startup_1_100', 1, 100), ('last_200', 801, 1000)]:
        rows = [row for row in history.actions if lo <= row[0] <= hi]
        done = [row for row in completed if lo <= row[0] <= hi]
        windows[label] = dict(
            writing_fraction=sum(row[2] in {'write_paper', 'review_finished_write'} for row in rows)/len(rows),
            completed_reviews=len(done),
            good_faith_fraction=sum(row[4] == 'good_faith' for row in done)/len(done) if done else None,
        )
    summary = dict(
        seed=11, steps=1000, agents=40, initial_papers=initial_papers,
        new_papers=len(env.papers)-initial_papers, final_papers=len(env.papers),
        completed_reviews=len(completed), good_faith_reviews=good,
        bad_faith_reviews=len(completed)-good, good_faith_fraction=good/len(completed),
        writing_fraction=writing/len(history.actions), review_fraction=1-writing/len(history.actions),
        matches=len(trades), pending=len(pending), active_reviews=len(busy),
        mean_request_wait=statistics.mean(t['request_wait'] for t in trades),
        p95_request_wait=sorted(t['request_wait'] for t in trades)[math.ceil(.95*len(trades))-1],
        mean_pending_age=statistics.mean(env.timestep-p.listed_timestep for p in pending) if pending else 0,
        mean_agreed_share=statistics.mean(t['agreed_share'] for t in trades),
        claim_gini=gini(allocations.get(i,0) for i in range(len(env.agents))),
        top_20_percent_order_share=sum(sorted(allocations.values(),reverse=True)[:8])/len(trades),
        capital_gini=gini(a.academic_capital for a in env.agents),
        total_ac=sum(a.academic_capital for a in env.agents),
        candidates_per_request=sum(history.scalars['market_candidates'])/sum(history.scalars['market_requests']),
        declines=sum(history.scalars['market_declined']),
        groups=history.agent_group_summary(), windows=windows, action_counts=dict(actions),
        invariant_checks='passed at all 1000 timesteps',
        parameters={**asdict(SIM), 'market_mechanism':'sampled_bidding', 'num_heuristic_agents':20,
                    'num_random_agents':20, 'num_rl_agents':0, 'num_timesteps':1000},
    )
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    runner.OUTPUT_DIR = str(output)
    runner.print_summary(env, history)
    runner.save_outputs(history)
    print(json.dumps({k:v for k,v in summary.items() if k not in {'parameters','groups'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
