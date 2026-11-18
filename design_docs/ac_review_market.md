# Sampled reviewer bidding — initial implementation

Base: local main ad5a586. Branch: codex/ac-review-market.

## What is exchanged

Reviewers sell their own review labour for a fraction of the WHOLE paper's
ownership. An ask of .005 transfers .5 percentage points from the author upon
valid completion, under the existing effort/reward rules. Listing and selection
are instantaneous; the actual work uses the existing one-turn labour budget.
ReviewSeller is a policy component of the existing agent, not a new worker.

## Public signal and selection

The only selection signal is the reviewer's existing academic_capital at the
beginning of the timestep. No talent or peer-review history enters ranking or
bucket selection. Internal talent still affects the original research strategies
and production mechanics; changing the public signal does not remove it there.
AC measures accumulated capital, not causal citation contribution or diligence.

Author filters: non-self, AC >= minimum, ask <= budget. Rank the at-most-K
sampled candidates by AC / ask. The seller may reject a particular paper.
Ties are randomized, including all-zero AC cold starts; there is no hidden
positive AC prior or price tiebreak. Winner is paid its own ask. No best-bid
increment, forced undercut, or dynamic quote learning is implemented in v1.

## Parameters in SimConfig

| Parameter | Default | Meaning |
| --- | --- | --- |
| market_mechanism | legacy | opt in with sampled_bidding |
| market_candidate_limit | 5 | candidate examination cap per request per step |
| market_ac_bin_edges | (1,10,100,1000) | raw AC index boundaries; last bucket unbounded |
| market_min_reviewer_ac | 0 | default author eligibility threshold |
| market_share_tick | .0001 | price precision: .01 percentage points |
| market_default_ask_share | .005 | fixed seller ask: .5% |
| market_author_max_share | .05 | fixed author budget: 5% of whole paper |

Existing seed initializes a separate market random generator. Actual budget is
min(author budget, remaining author ownership, existing paper review cap).
Asks round up to integer ticks, budgets down. Invalid/nonfinite values fail.
Fixed AC boundaries are indexing parameters, not predictions of paper uplift.

Agents can override review_market_request(paper, config), review_ask_share(config),
offer_review_this_step(), accept_review_offer(paper, share), and
choose_market_work_action(). Default author request can be withheld via
request_reviews_enabled=False; review_min_ac and review_max_share override the
defaults. review_ask supplies a per-reviewer fixed ask. These are strategy
inputs, not extra accounts. Default authors request review of every eligible
paper; endogenous/learned release policy is outside v1.

Heuristics advertise when idle and accept only when existing quoted review
valuation exceeds their writing alternative. They keep their original effort
and research-plan decisions. Random agents advertise once per step using their
existing claim probability; willingness is not redrawn per invitation. Existing
discrete good/bad choice is preserved. RL/DQN agents fail explicitly in the new
mode: their action spaces and learned policies have not been adapted. Legacy
mode remains available for all existing agents and training.

## One timestep

1. Existing publication scheduler makes papers available, as before.
2. O(N) pass over idle sellers to rebuild one-step quotes and freeze their AC.
3. O(P) pass over available papers; request hook can withhold them. Withheld
   papers are delisted and scheduled for reconsideration next step.
4. Shuffle requests. For each, sample at most K distinct sellers uniformly
   across the union of buckets at/above its AC threshold bucket. Filter exact
   threshold, self and budget after sampling. Failed candidates use up search
   slots; never replenish with an unbounded search.
5. Randomize ties, sort by AC/ask, ask each in order until one accepts.
6. Validate ownership/capacity/availability, lock explicit share via claim_review,
   remove seller from all further matches this step. No share transfers yet.
7. Newly matched agents apply first review effort exactly once. Others follow
   original research/continuation rules without claiming outside the market.
8. Existing accrual and portfolio accounting run unchanged. History records
   market counters, locked trades, request waits, AC snapshots and configuration.

The sparse quote exposed during seller valuation exists only for that candidate
and is removed in a finally block. The old all-reviewer price tables, scarcity
and adaptive pricing and all-pairs merit matching are bypassed in the new mode.
Old author-price feedback is skipped for explicit-price contracts.

## Data structures and complexity

Frozen ReviewOffer / ReviewRequest / ReviewTrade dataclasses; dictionaries for
offers and stable IDs; a list of seller-ID arrays per AC bucket; a position map
for swap-delete removal in O(1). No P-by-N matrix. Sampling uses virtual indices
over the buckets, without flattening all sellers. Search cost per request is
O(B*K + K log K), with B buckets and bounded K. Whole market phase is
O(N + total papers + P*(B*K + K log K)), excluding existing strategy valuations.
These are market-only bounds: portfolio accrual and other simulation logic
still traverse papers. With full candidate sampling, K=N intentionally removes
the bounded-search benefit for diagnostic comparisons.

Public API notes: Paper.start_review and Agent.claim_review accept optional
agreed_share. None retains legacy price lookup and adaptive feedback. Explicit
shares are validated before paper state changes and never silently clipped.
Agent.act_continuous uses market-only work hooks in the new mode, so no agent
can claim an unmatched paper. Discrete work likewise skips legacy claiming.

## Run

```powershell
python -m unittest test_review_market -v
python run_simulation.py --market-mechanism sampled_bidding --heuristic-agents 8 --random-agents 8 --rl-agents 0 --dqn-agents 0 --low-talent-rl-agents 0 --timesteps 300 --no-archive
python compare_review_markets.py --steps 300 --agents 16 --seeds 11,22,33,44,55
```

The comparison uses fixed heterogeneous asks (.25%, .5%, 1%) assigned independently
of initial talent and AC, half heuristic / half random, two zero-AC initial
papers per agent. It writes results.json without archiving or training.
--candidate-limit and --ask-scale support bounded-search/price sensitivity runs.

## How to evaluate

### Correctness is a gate, not an improvement metric

Check finite/nonnegative capital and shares, ownership summing to one, total
portfolio capital equalling paper capital, one current review per agent and
one reviewer per paper, no self-review, locked prices and budget, exact labour
turn counts, and <=K examined candidates. Multi-step tests cover continuous
and discrete modes, zero-AC random ties, withholding, seed replay and history
round trips. The pilot checks accounting/capacity/labour each step and aborts
on violations. Confirm writing, new publications and completed reviews remain
possible; no-trade can be an economic outcome, not necessarily a code defect.

### Performance and behaviour are measured separately

Five paired-seed arms:
* legacy: original pricing and allocation;
* legacy_fixed: same per-seller asks, original allocation/participation;
* sampled: new bounded AC/ask market;
* sampled_full: same new mechanism with every candidate visible;
* sampled_random: same bounded market with randomized candidate ranking, an
  experimental ablation of the AC/ask selection rule (not a production option).

legacy comparison measures the bundled change. legacy_fixed reduces the price
confound but also differs in participation/acceptance semantics, so it does NOT
isolate a pure ranking causal effect. sampled_full diagnoses bounded-search
effects. Identical seeds give identical initial worlds, not identical later
random trajectories. Report per-seed rows, paired mean deltas and standard
errors; a five-seed pilot is preliminary, not a significance claim.

Primary outcomes: new papers, completed reviews, actual writing/review time,
model-generated review uplift, author net benefit, waiting/backlog and capital
inequality. Also report average ask, allocation concentration and pre-step AC
percentiles of winners, so favouring already-rich agents is visible.

model_review_gain integrates the extra accrual on the same completed papers
relative to their pre-review base rate. author_post_review_net_flow deducts
the author's ongoing payment share; author_net_including_existing_ac also
deducts the share of already-accrued AC transferred on completion. These are
within-model, same-paper counterfactuals, NOT total social welfare (foregone
writing is not included), and NOT causal citation estimates. Total AC alone
can rise through publication volume or existing mechanics and is insufficient.

An improvement claim requires useful review benefit/author outcomes and
acceptable publication throughput, time allocation, backlog and inequality.
No scientifically justified universal healthy writing fraction exists here;
agree on acceptable tradeoffs before a confirmatory experiment. No quotas,
subsidies or automatic balance controllers are added to force the result.
Use longer runs, more seeds and candidate/ask sensitivity before claiming
stable supply-demand balance or general welfare improvement. Machine runtime
is descriptive; the pilot's invariant instrumentation is included in timing.

## Known limits

Fixed quotes do not establish equilibrium price discovery. AC can reinforce
wealth concentration and is not review expertise. Homogeneous quotes reduce
ranking to AC. Fixed bins may get crowded; sampling stays bounded but search
quality can suffer. Finite sampling can miss feasible trades; high thresholds
can stall the market. No quality guarantee or reputation punishment exists.
Default authors use fixed budgets and release all eligible papers. None of
these assumptions should be hidden when interpreting experimental outcomes.
