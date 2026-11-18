"""Bounded-search AC/price matching, with one-step quotes and locked contracts.

AC is a public capital snapshot, not a prediction of citation improvement.
No history-based pricing, supply subsidy, or forced labour is performed here.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import random

from ReviewSeller import ReviewSeller


@dataclass(frozen=True)
class ReviewOffer:
    reviewer_id: int
    ac: float
    ask_ticks: int


@dataclass(frozen=True)
class ReviewRequest:
    paper_id: int
    author_id: int
    min_ac: float
    max_ticks: int


@dataclass(frozen=True)
class ReviewTrade:
    timestep: int
    paper_id: int
    author_id: int
    reviewer_id: int
    ac_at_match: float
    agreed_share: float
    score: float
    request_wait: int


class ReviewMarketplace:
    def __init__(self, config, agents, seed):
        self.config = config
        self.agents = list(agents)
        self.agent_ids = {a: i for i, a in enumerate(agents)}
        self.sellers = [ReviewSeller(a) for a in agents]
        self.seed = seed
        self.rng = random.Random(seed)
        self.paper_ids = {}
        self.first_requested = {}
        self.edges = tuple(config.market_ac_bin_edges)
        self._validate()
        self.offers = {}
        self.bins = [[] for _ in range(len(self.edges) + 1)]
        self.positions = {}
        self.trades = []
        self.metrics = {}

    def _validate(self):
        c = self.config
        if not isinstance(c.market_candidate_limit, int) or c.market_candidate_limit < 1:
            raise ValueError("market_candidate_limit must be a positive integer")
        if any(not math.isfinite(x) or x < 0 for x in self.edges):
            raise ValueError("AC bin edges must be finite and nonnegative")
        if any(a >= b for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("AC bin edges must strictly increase")
        if not math.isfinite(c.market_share_tick) or not 0 < c.market_share_tick <= 1:
            raise ValueError("market_share_tick must be in (0, 1]")
        for value in (c.market_default_ask_share, c.market_author_max_share):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("default asks and author budgets must be in (0, 1]")
        if not math.isfinite(c.market_min_reviewer_ac) or c.market_min_reviewer_ac < 0:
            raise ValueError("minimum reviewer AC must be finite and nonnegative")

    def remove_offer(self, reviewer_id):
        bucket, index = self.positions.pop(reviewer_id)
        ids = self.bins[bucket]
        last = ids.pop()
        if index < len(ids):
            ids[index] = last
            self.positions[last] = (bucket, index)
        del self.offers[reviewer_id]

    def sample(self, minimum):
        """Uniform without replacement across eligible buckets; no full flatten."""
        start = bisect_right(self.edges, minimum)
        sizes = [len(ids) for ids in self.bins[start:]]
        total = sum(sizes)
        result = []
        for offset in self.rng.sample(range(total), min(total, self.config.market_candidate_limit)):
            for bucket, size in enumerate(sizes, start):
                if offset < size:
                    result.append(self.offers[self.bins[bucket][offset]])
                    break
                offset -= size
        return result

    def rank_candidates(self, candidates):
        """AC/ask ranking; exposed separately for experimental ablation only."""
        ranked = list(candidates)
        self.rng.shuffle(ranked)
        ranked.sort(key=lambda o: o.ac / o.ask_ticks, reverse=True)
        return ranked

    def clear(self, env):
        """Commit matches BEFORE labour; return agents needing their first effort."""
        c = self.config
        self.metrics = dict(requests=0, offers=0, candidates=0, matches=0,
                            self_rejected=0, below_ac=0, over_budget=0,
                            declined=0, unmatched=0)
        self.offers.clear()
        self.positions.clear()
        self.bins = [[] for _ in range(len(self.edges) + 1)]
        self.trades = []
        # O(N), once per step. Freeze AC before any agent performs work.
        for rid, seller in enumerate(self.sellers):
            seller.owner.current_timestep = env.timestep
            ticks = seller.quote_ticks(c)
            if ticks is None or not seller.owner.offer_review_this_step():
                continue
            ac = float(seller.owner.academic_capital)
            if not math.isfinite(ac) or ac < 0:
                raise ValueError("reviewer AC must be finite and nonnegative")
            offer = ReviewOffer(rid, ac, ticks)
            self.offers[rid] = offer
            bucket = bisect_right(self.edges, ac)
            self.positions[rid] = (bucket, len(self.bins[bucket]))
            self.bins[bucket].append(rid)
        self.metrics['offers'] = len(self.offers)
        requests = []
        for paper in env.papers:
            if not paper.review_available:
                continue
            paper.price_table.clear()  # no legacy prices in this mechanism
            request = paper.author.review_market_request(paper, c)
            if request is None:
                paper.market_listed = False
                paper.scheduled_listing_timestep = env.timestep + 1
                continue
            minimum, budget = map(float, request)
            if not math.isfinite(minimum) or minimum < 0 or not math.isfinite(budget) or not 0 <= budget <= 1:
                raise ValueError("invalid author request")
            budget = min(budget, paper.share_distribution.get(paper.author, 0), paper.max_reviewer_share)
            pid = self.paper_ids.setdefault(paper, len(self.paper_ids))
            self.first_requested.setdefault(paper, env.timestep)
            req = ReviewRequest(pid, self.agent_ids[paper.author], minimum,
                                math.floor(budget / c.market_share_tick + 1e-12))
            requests.append((paper, req))
        self.metrics['requests'] = len(requests)
        self.rng.shuffle(requests)
        matched = set()
        for paper, request in requests:
            sampled = self.sample(request.min_ac)
            self.metrics['candidates'] += len(sampled)
            eligible = []
            for offer in sampled:
                if offer.reviewer_id == request.author_id:
                    self.metrics['self_rejected'] += 1
                elif offer.ac < request.min_ac:
                    self.metrics['below_ac'] += 1
                elif offer.ask_ticks > request.max_ticks:
                    self.metrics['over_budget'] += 1
                else:
                    eligible.append(offer)
            # Randomize equal scores, including the all-zero cold start.
            for offer in self.rank_candidates(eligible):
                reviewer = self.agents[offer.reviewer_id]
                share = offer.ask_ticks * c.market_share_tick
                # A bounded, temporary quote for existing valuation helpers.
                paper.price_table[reviewer] = share
                try:
                    accepts = reviewer.accept_review_offer(paper, share)
                finally:
                    paper.price_table.pop(reviewer, None)
                if not accepts:
                    self.metrics['declined'] += 1
                    continue
                if reviewer.active_review_paper is not None or not paper.can_start_review(reviewer):
                    raise RuntimeError("market state changed during acceptance")
                # All checks precede mutations. claim_review locks an explicit price.
                reviewer.claim_review(paper, agreed_share=share)
                matched.add(reviewer)
                self.remove_offer(offer.reviewer_id)
                self.trades.append(ReviewTrade(env.timestep, request.paper_id,
                    request.author_id, offer.reviewer_id, offer.ac, share, offer.ac / share,
                    env.timestep - self.first_requested[paper]))
                break
        self.metrics['matches'] = len(matched)
        self.metrics['unmatched'] = len(requests) - len(matched)
        waits = [env.timestep - self.first_requested[p] for p, _ in requests if not p.review_claimed]
        self.metrics['pending_mean_wait'] = sum(waits) / len(waits) if waits else 0.0
        return matched
