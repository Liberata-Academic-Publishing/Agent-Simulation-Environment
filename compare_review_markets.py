"""Paired fixed-strategy pilot; no RL, archive writes, or training.

legacy: original economics+allocation
legacy_fixed: same explicit asks as sampled, with legacy allocation
sampled: bounded AC/ask selection
sampled_full: all available candidates, diagnoses search loss
sampled_random: same sampled market but random ranking, diagnoses selection effect

Paired seeds mean identical initialization, not identical later random paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics
import time

from Agent import Agent
from config import SIM
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History, gini
from Paper import Paper
from RandomAgent import RandomAgent


def run_one(mode, seed, steps, count, candidate_limit=5, ask_scale=1):
    random.seed(seed)
    Agent.all_papers = []
    agents = []
    for i in range(count):
        cls = HeuristicAgent if i % 2 == 0 else RandomAgent
        a = cls(intrinsic_talent=random.uniform(.6, 1.4), name=f'agent_{i}')
        # Fixed heterogeneous quotes, independent of AC and talent.
        a.review_ask = (.0025, .005, .01)[i % 3] * ask_scale
        agents.append(a)
    # Independent initial corpus, zero AC cold start, identical in each arm.
    papers = [Paper(a, quality=random.uniform(.6, 1.4), accrual_rate=1,
                    current_ac=0, market_listed=True) for a in agents for _ in range(2)]
    for p in papers:
        p.list_on_market(0)
    Agent.all_papers = papers
    initial_count = len(papers)
    config = replace(SIM, seed=seed,
        market_mechanism='sampled_bidding' if mode.startswith('sampled') else 'legacy',
        market_candidate_limit=count if mode == 'sampled_full' else candidate_limit)
    h = History(track_papers=False)
    env = Environment(agents=agents, papers=papers, history=h, market_config=config,
        market_mechanism='sampled_bidding' if mode.startswith('sampled') else 'legacy',
        market_seed=seed)
    if mode == 'sampled_random':
        # Experimental control only: same search/quotes/capacity, random ranking.
        env.review_market.rank_candidates = lambda offers: env.review_market.rng.sample(list(offers), len(offers))
    if mode == 'legacy_fixed':
        def fixed_prices():
            for p in env.papers:
                if p.review_available:
                    p.price_table = {a: a.review_ask for a in agents if a is not p.author}
        env._update_market_prices = fixed_prices
    durations = []
    writing_turns = review_turns = 0
    model_gain = author_net = 0.0
    ac_before_review = {}
    claim_rank = []
    claimed = set()
    # Runtime invariant checks use exceptions, so python -O cannot disable them.
    def check(ok, message):
        if not ok:
            raise RuntimeError(message)
    for _ in range(steps):
        rates = {p: p.accrual_rate for p in env.papers}
        capital = [a.academic_capital for a in agents]
        before = len(h.actions)
        started = time.perf_counter()
        env.run_timestep()
        durations.append(time.perf_counter() - started)
        turns = h.actions[before:]
        # Legacy switches may emit two log entries for one labour turn.
        labour = Counter(label for _, label, kind, _ in turns if kind != 'review_finished_peer_review')
        check(len(labour) == count and all(v == 1 for v in labour.values()), 'labour budget violated')
        for _, _, kind, _ in turns:
            writing_turns += kind in {'write_paper', 'review_finished_write'}
            review_turns += kind in {'review_started', 'review_continued', 'good_faith_review', 'bad_faith_review'}
        busy = [a.active_review_paper for a in agents if a.active_review_paper is not None]
        check(len(busy) == len(set(busy)), 'duplicate paper assignment')
        for a in agents:
            check(math.isfinite(a.academic_capital) and a.academic_capital >= 0, 'invalid capital')
            if a.active_review_paper is not None:
                check(a.active_review_paper.review_in_progress_by is a, 'review state mismatch')
        for p in env.papers:
            check(all(math.isfinite(s) and 0 <= s <= 1 for s in p.share_distribution.values()), 'invalid share')
            check(math.isclose(sum(p.share_distribution.values()), 1, abs_tol=1e-9), 'share conservation')
            check(len(p.review_records) <= 1, 'duplicate review')
            if p.review_claimed and p not in claimed:
                claimed.add(p)
                reviewer = p.review_in_progress_by or p.reviewer
                rid = agents.index(reviewer)
                # Fraction strictly below this reviewer's pre-step capital.
                claim_rank.append(sum(x < capital[rid] for x in capital) / count)
            if p.reviewed:
                ac_before_review.setdefault(p, rates.get(p, p.accrual_rate))
                base = ac_before_review[p]
                gain = max(0, p.accrual_rate - base)
                model_gain += gain
                # Post-completion author flow gain vs same paper with no review.
                author_net += p.share_distribution.get(p.author, 0) * p.accrual_rate - base
        check(math.isclose(sum(a.academic_capital for a in agents),
                           sum(p.current_ac for p in env.papers), rel_tol=1e-9, abs_tol=1e-7), 'capital accounting')
    completed = [p for p in env.papers if p.reviewed]
    waits = [p.time_on_market_timesteps for p in env.papers if p.review_claimed]
    pending = [env.timestep - p.listed_timestep for p in env.papers if p.review_available and p.listed_timestep is not None]
    shares = [p.agreed_review_share for p in env.papers if p.review_claimed]
    allocations = Counter(agents.index(p.reviewer or p.review_in_progress_by) for p in env.papers if p.review_claimed)
    total_claims = sum(allocations.values())
    result = dict(mode=mode, seed=seed, steps=steps, agents=count,
        new_papers=len(env.papers)-initial_count, completed_reviews=len(completed), claims=total_claims,
        writing_fraction=writing_turns/(steps*count), review_fraction=review_turns/(steps*count),
        total_ac=sum(a.academic_capital for a in agents), capital_gini=gini(a.academic_capital for a in agents),
        model_review_gain=model_gain, author_post_review_net_flow=author_net,
        author_net_including_existing_ac=author_net-sum(
            p.agreed_review_share*p.review_records[-1]['current_ac_at_review'] for p in completed),
        good_faith_fraction=sum(p.review_records[-1]['review_kind']=='good_faith' for p in completed)/max(1,len(completed)),
        mean_claim_wait=statistics.mean(waits) if waits else None,
        pending=len(pending), mean_pending_age=statistics.mean(pending) if pending else 0,
        mean_share=statistics.mean(shares) if shares else None,
        claim_gini=gini(allocations.get(i,0) for i in range(count)),
        mean_winner_ac_percentile=statistics.mean(claim_rank) if claim_rank else None,
        milliseconds_per_step=1000*statistics.mean(durations), invariant_checks='passed',
        parameters=asdict(config), reviewer_asks=[a.review_ask for a in agents],
        invariant_steps_checked=steps)
    if env.review_market:
        result['candidates_per_request'] = sum(h.scalars['market_candidates']) / max(1,sum(h.scalars['market_requests']))
        result['quote_metadata'] = h.market_metadata
    return result


def summarize(rows):
    metrics = ['new_papers','completed_reviews','writing_fraction','review_fraction','total_ac',
        'capital_gini','model_review_gain','author_post_review_net_flow','author_net_including_existing_ac','good_faith_fraction',
        'mean_claim_wait','pending','mean_pending_age','mean_share','claim_gini',
        'mean_winner_ac_percentile','milliseconds_per_step']
    modes = sorted({r['mode'] for r in rows})
    means = {m: {key: statistics.mean(r[key] for r in rows if r['mode']==m and r[key] is not None)
                 if any(r['mode']==m and r[key] is not None for r in rows) else None for key in metrics} for m in modes}
    deltas = {}
    for control in ['legacy','legacy_fixed','sampled_random','sampled_full']:
        if control not in modes:
            continue
        pairs = [(r, next(s for s in rows if s['seed']==r['seed'] and s['mode']==control))
                 for r in rows if r['mode']=='sampled']
        deltas[control] = {}
        for key in metrics:
            values = [a[key]-b[key] for a,b in pairs if a[key] is not None and b[key] is not None]
            if values:
                deltas[control][key] = dict(mean=statistics.mean(values), n=len(values),
                    standard_error=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)
    return dict(means=means, paired_sampled_minus_control=deltas)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--agents', type=int, default=16)
    parser.add_argument('--seeds', default='11,22,33,44,55')
    parser.add_argument('--candidate-limit', type=int, default=5)
    parser.add_argument('--ask-scale', type=float, default=1)
    parser.add_argument('--output', default='experiments/ac_review_market')
    args = parser.parse_args()
    if args.steps < 1 or args.agents < 2 or args.candidate_limit < 1 or not 0 < args.ask_scale <= 10:
        parser.error('positive steps/candidate limit, >=2 agents, ask scale in (0,10] required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in map(int,args.seeds.split(',')):
        for mode in ['legacy','legacy_fixed','sampled','sampled_full','sampled_random']:
            row = run_one(mode,seed,args.steps,args.agents,args.candidate_limit,args.ask_scale)
            rows.append(row)
            (out/'results.json').write_text(json.dumps(dict(rows=rows,summary=summarize(rows) if len(rows)%5==0 else None),indent=2),encoding='utf-8')
            print(f"seed={seed} mode={mode}: papers={row['new_papers']} reviews={row['completed_reviews']} checks=passed",flush=True)
    print(json.dumps(summarize(rows)['means'],indent=2))


if __name__ == '__main__':
    main()
