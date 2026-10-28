"""Train and evaluate Q-learning agents on the Liberata market.

Self-play across many short episodes. Each agent has its own Q table: an
update from one agent does not change any other agent's policy, so an agent
cannot learn to review in order to make the others review its papers. Pass
``--shared-policy`` to train the old way, with one table shared by every agent.

ε decays each episode. After training, a greedy (ε=0, no learning) evaluation
reports capital and how often the learned policies start a review.

Training runs under the configured ``review_paradigm`` (``--review-paradigm``):
in the default *continuous* paradigm the RL agent makes one merged decision per
timestep; in *discrete* it uses the two-phase flow. The action space differs
between paradigms, so a policy trained in one is not meaningful in the other.

Each agent keeps two tables. One picks the greedy next action and the other
scores it, so a noisy high cell is not copied back into itself. After the
greedy evaluation, the mean of the two tables belonging to the agent with the
most academic capital is saved to ``policies/policy_<backend>``. The
simulation loads that one policy.

``--train-one`` trains a single agent for longer. The other agents play the
latest saved policy and do not learn. The saved file is that one agent's
table, and the review count is that agent's alone.

Usage:
    python train_rl.py                       # one Q table per agent
    python train_rl.py --shared-policy       # one table shared by every agent
    python train_rl.py --train-one --no-archive
    python train_rl.py --backend linear --episodes 300
    python train_rl.py --load policies/policy_tabular_independent --episodes 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

from Agent import Agent, assign_rate_talents
from config import (
    SIM,
    TRAIN,
    default_low_talent_policy_path,
    default_policy_path,
    list_independent_policy_files,
)
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper
from QLearningAgent import (
    QLearningAgent,
    average_backends,
    describe_progress_policy,
    make_backend,
    write_claim_gaps,
)

OUTPUT_DIR = SIM.output_dir

TRAINING_CHART_DESCRIPTIONS = {
    "training_log.json": "Per-episode training metrics (JSON)",
    "episode_return.png": "Episode return (mean final AC per episode)",
    "avg_peer_review_time.png": "Average peer review time per episode",
}


def open_chart(path: str) -> None:
    """Open a saved chart with the OS default image viewer."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def seed_initial_papers(agents: list[Agent], rng: random.Random) -> None:
    """Seed a small bootstrap supply, released gradually into the market."""
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                quality=agent.intrinsic_talent,
                current_ac=rng.uniform(SIM.init_ac_min, SIM.init_ac_max),
                market_listed=False,
            )
            stagger = max(1, int(SIM.initial_listing_stagger_timesteps))
            paper.scheduled_listing_timestep = 1 + ((index - 1) % stagger)
            paper.title = f"Paper {index}"
            Agent.all_papers.append(paper)


def build_env(
    *,
    backend=None,
    backends=None,
    backends_b=None,
    epsilon: float,
    learning: bool,
    num_rl: int,
    num_heuristic: int,
    horizon: int,
    seed: int,
    review_paradigm: str = SIM.review_paradigm,
    gamma: float = SIM.rl_gamma,
    history: History | None = None,
    intrinsic_talent: float = 1.0,
    shared_talents: bool = False,
    frozen_backends=None,
) -> tuple[Environment, list[QLearningAgent], list[HeuristicAgent]]:
    """Fresh env: RL agents vs. heuristic opponents.

    ``backends`` assigns one Q table per agent, in order. Passing a single
    ``backend`` shares that same table across the RL agents. ``backends_b`` is
    the second table used to score the action the first table picks.
    """
    if backends is None:
        if backend is None:
            raise ValueError("build_env requires backend or backends")
        backends = [backend] * num_rl
    if len(backends) != num_rl:
        raise ValueError(f"expected {num_rl} backends, got {len(backends)}")
    if backends_b is not None and len(backends_b) != num_rl:
        raise ValueError(f"expected {num_rl} second backends, got {len(backends_b)}")
    rng = random.Random(seed)
    Agent.all_papers = []

    rl_agents = [
        QLearningAgent(
            intrinsic_talent=intrinsic_talent,
            forecast_horizon_timesteps=horizon,
            name=f"RL {i}",
            backend=backends[i],
            backend_b=None if backends_b is None else backends_b[i],
            epsilon=epsilon,
            learning=learning,
            gamma=gamma,
        )
        for i in range(num_rl)
    ]
    heuristics = [
        HeuristicAgent(intrinsic_talent=1.0, forecast_horizon_timesteps=horizon,
                       name=f"Heuristic {i}")
        for i in range(num_heuristic)
    ]
    frozen = [
        QLearningAgent(
            intrinsic_talent=intrinsic_talent,
            forecast_horizon_timesteps=horizon,
            name=f"Frozen {i}",
            backend=backend,
            epsilon=0.0,
            learning=False,
            gamma=gamma,
        )
        for i, backend in enumerate(frozen_backends or [])
    ]
    agents: list[Agent] = [*rl_agents, *frozen, *heuristics]
    if SIM.use_rate_talents:
        assign_rate_talents(agents, rng)

    environment_options = {}
    if shared_talents:
        pairs = [(q, r) for q in (SIM.talent_low, SIM.talent_high)
                 for r in (SIM.talent_low, SIM.talent_high)]
        for cohort in (rl_agents, frozen, heuristics):
            for index, agent in enumerate(cohort):
                agent.configure_talents(*pairs[index % len(pairs)])
        environment_options = dict(continuous_publishing='threshold',
                                   paper_effort_mode='fixed',
                                   use_merit_market_clearing=False)
    else:
        seed_initial_papers(agents, rng)
    env = Environment(agents=agents, papers=Agent.all_papers,
                      forecast_horizon_timesteps=horizon,
                      review_paradigm=review_paradigm,
                      history=history, **environment_options)
    return env, rl_agents, heuristics


def mean_review_effort(history: History) -> float:
    """Mean completed peer-review duration for one training episode."""
    efforts = [float(effort) for _, _, _, effort, _ in history.completed_reviews]
    if not efforts:
        return 0.0
    return sum(efforts) / len(efforts)


def mean_capital(agents) -> float:
    if not agents:
        return 0.0
    return sum(a.academic_capital for a in agents) / len(agents)


def mean_write_claim_gap(backends, twins) -> tuple[float, float]:
    """Mean and minimum ``Q(write) - Q(claim)`` across agents and manuscript buckets."""
    gaps: list[float] = []
    for index, backend in enumerate(backends):
        twin = None if twins is None else twins[index]
        gaps.extend(write_claim_gaps(backend, twin))
    if not gaps:
        return 0.0, 0.0
    return sum(gaps) / len(gaps), min(gaps)


def rl_review_starts(history: History, agents) -> tuple[int, int]:
    """Review claims and total decisions for the named RL agents only."""
    names = {agent.name for agent in agents}
    starts = 0
    decisions = 0
    for _, agent_label, kind, _ in history.actions:
        if agent_label not in names:
            continue
        decisions += 1
        if kind == "review_started":
            starts += 1
    return starts, decisions


def build_train_config(args) -> dict[str, Any]:
    """Snapshot of the training-harness settings used for this run."""
    return {
        "backend": args.backend,
        "episodes": args.episodes,
        "timesteps": args.timesteps,
        "num_rl": args.num_rl,
        "num_heuristic": args.num_heuristic,
        "horizon": args.horizon,
        "review_paradigm": args.review_paradigm,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "seed": args.seed,
        "low_talent": args.low_talent,
        "low_talent_value": args.low_talent_value,
        "shared_talents": args.shared_talents,
        "reviewer_surplus_share": SIM.reviewer_surplus_share,
        "rl_manuscript_buckets": SIM.rl_manuscript_buckets,
        "rl_review_buckets": SIM.rl_review_buckets,
        "rl_review_progress_cap": SIM.rl_review_progress_cap,
        "rl_reward_ac_weight": SIM.rl_reward_ac_weight,
        "rl_reward_rank_weight": SIM.rl_reward_rank_weight,
        "rl_reward_accrual_weight": SIM.rl_reward_accrual_weight,
        "rl_gamma": SIM.rl_gamma,
    }


def save_training_outputs(
    training_log: list[dict[str, Any]], *, open_chart_flag: bool = False
) -> str | None:
    """Write ``training_log.json`` and ``episode_return.png`` to ``runs/``."""
    if not training_log:
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(training_log, fh, indent=2)

    episodes = [row["episode"] for row in training_log]
    returns = [row["mean_return"] for row in training_log]
    avg_review_times = [row.get("mean_review_effort", 0.0) for row in training_log]

    png_path: str | None = None
    try:
        import visualize
    except ImportError as exc:
        print(f"\nWrote training log to {log_path}")
        print(f"Skipping training charts ({exc}).")
        print("Install matplotlib with: python -m pip install matplotlib")
        return None

    png_path = os.path.join(OUTPUT_DIR, "episode_return.png")
    visualize.plot_episode_return(episodes, returns, None, png_path)
    visualize.plot_avg_peer_review_time(
        episodes,
        avg_review_times,
        os.path.join(OUTPUT_DIR, "avg_peer_review_time.png"),
    )

    print(f"\nWrote training outputs to the {OUTPUT_DIR}/ folder:")
    for name, description in TRAINING_CHART_DESCRIPTIONS.items():
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            print(f"- {path}  ({description})")

    if open_chart_flag and png_path and os.path.exists(png_path):
        print(f"\nOpening episode return chart: {png_path}")
        open_chart(png_path)
    elif png_path:
        print(f"\nView the episode return chart at: {os.path.abspath(png_path)}")

    return png_path


def archive_training_run(
    training_log: list[dict[str, Any]],
    title: str | None,
    args,
) -> None:
    from export_run import export_training_run

    run_id = export_training_run(
        training_log,
        config=build_train_config(args),
        title=title,
    )
    print(f"\nArchived training run to docs/data/{run_id}/ (visible in the gallery).")
    print("Publish it with: "
          "git add docs/data && git commit -m 'Add training run' && git push")


def prompt_and_archive_training(
    training_log: list[dict[str, Any]],
    args,
) -> None:
    """Ask whether to save this training run and, if so, what to title it."""
    try:
        answer = input("\nSave this training run to the gallery? [y/N]: ").strip().lower()
    except EOFError:
        print("Not archived (no interactive input).")
        return

    if answer not in ("y", "yes"):
        print("Not archived.")
        return

    try:
        name = input("Name this run (leave blank for an auto name): ").strip()
    except EOFError:
        name = ""
    archive_training_run(training_log, name or None, args)


def _single_policy_path(args) -> str:
    """Where the best trained table is written."""
    if args.shared_talents:
        suffix = ".npy" if args.backend == "linear" else ".pkl"
        return os.path.join(
            SIM.policies_dir, f"policy_{args.backend}_shared_talents{suffix}"
        )
    if args.low_talent:
        return default_low_talent_policy_path(args.backend)
    return default_policy_path(args.backend)


def load_frozen_backends(path: str, count: int, kind: str) -> list:
    """One private copy of a saved policy per frozen opponent."""
    if count <= 0:
        return []
    if not os.path.isfile(path):
        raise SystemExit(
            f"No saved policy at {path}. Train a policy first, or pass --frozen-policy."
        )
    backends = []
    for _ in range(count):
        backend = make_backend(kind)
        backend.load(path)
        backends.append(backend)
    return backends


def make_backends(args) -> tuple[list, list]:
    """One pair of Q tables per agent, or one shared pair with ``--shared-policy``.

    The second table scores the action the first one picks, so a noisy high
    cell is not copied back into itself.
    """
    count = 1 if args.shared_policy else args.num_rl
    backends = [make_backend(args.backend, alpha=args.alpha) for _ in range(count)]
    twins = [make_backend(args.backend, alpha=args.alpha) for _ in range(count)]
    if args.load:
        if os.path.isdir(args.load):
            files = list_independent_policy_files(args.load)
            if not files:
                raise SystemExit(f"No agent policies in {args.load}")
            for index, backend in enumerate(backends + twins):
                backend.load(files[index % len(files)])
            print(f"Loaded {len(files)} policies from {args.load}")
        else:
            for backend in backends + twins:
                backend.load(args.load)
            print(f"Loaded policy from {args.load}")
    elif args.episodes == 0:
        print("Warning: --episodes 0 with no --load evaluates an empty policy.")
    if args.shared_policy:
        return backends * args.num_rl, twins * args.num_rl
    return backends, twins


def save_best_policy(backends, twins, args, best_index: int) -> str:
    """Write the mean of the selected agent's two tables. Returns the file path."""
    path = args.save or _single_policy_path(args)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    average_backends(backends[best_index], twins[best_index]).save(path)
    return path


def train(args) -> None:
    backends, twins = make_backends(args)
    frozen_backends = (
        load_frozen_backends(args.frozen_policy, args.frozen_agents, args.backend)
        if args.train_one else []
    )

    training_log: list[dict[str, Any]] = []

    if args.episodes:
        talent_note = f" low_talent={args.low_talent_value}" if args.low_talent else ""
        state_note = (
            f"state={SIM.rl_manuscript_buckets}x{SIM.rl_review_buckets} "
            f"reviewer_surplus_share={SIM.reviewer_surplus_share}"
        )
        if args.train_one:
            print(f"Training one agent: paradigm={args.review_paradigm} "
                  f"backend={args.backend} episodes={args.episodes} "
                  f"timesteps={args.timesteps} "
                  f"frozen={args.frozen_agents} policy={args.frozen_policy}"
                  f"{talent_note} {state_note}")
        else:
            policy_note = "shared" if args.shared_policy else "independent"
            print(f"Training: paradigm={args.review_paradigm} backend={args.backend} "
                  f"policies={policy_note} episodes={args.episodes} "
                  f"timesteps={args.timesteps} "
                  f"rl={args.num_rl} heuristic={args.num_heuristic}"
                  f"{talent_note} {state_note}")
    rl_talent = args.low_talent_value if args.low_talent else 1.0
    for episode in range(args.episodes):
        # Linear ε decay from eps_start to eps_end.
        frac = episode / max(1, args.episodes - 1)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        history = History()
        env, rl_agents, _ = build_env(
            backends=backends, backends_b=twins, epsilon=epsilon, learning=True,
            num_rl=args.num_rl, num_heuristic=args.num_heuristic,
            horizon=args.horizon, seed=args.seed + episode,
            review_paradigm=args.review_paradigm, gamma=args.gamma,
            history=history, intrinsic_talent=rl_talent,
            shared_talents=args.shared_talents,
            frozen_backends=frozen_backends,
        )
        env.run(args.timesteps)
        for agent in rl_agents:
            agent.end_episode()

        mean_return = mean_capital(rl_agents)
        starts, decisions = rl_review_starts(history, rl_agents)
        gap_backends = backends[:1] if args.shared_policy else backends
        gap_twins = twins[:1] if args.shared_policy else twins
        mean_gap, min_gap = mean_write_claim_gap(gap_backends, gap_twins)
        training_log.append({
            "episode": episode,
            "mean_return": mean_return,
            "mean_review_effort": mean_review_effort(history),
            "epsilon": epsilon,
            "review_starts": starts,
            "decisions": decisions,
            "mean_write_minus_claim": mean_gap,
            "min_write_minus_claim": min_gap,
        })

        last = episode == args.episodes - 1
        if episode % max(1, args.episodes // 10) == 0 or last:
            review_note = ""
            if decisions:
                review_note = (
                    f"  reviews={starts}/{decisions} ({starts / decisions:6.2%})"
                )
            print(f"  ep {episode:4d}  eps={epsilon:.3f}  "
                  f"RL mean AC={mean_return:8.2f}{review_note}  "
                  f"write-claim gap mean={mean_gap:+.4f} min={min_gap:+.4f}")

    if training_log:
        save_training_outputs(training_log, open_chart_flag=args.open)

    best_index = evaluate(backends, twins, args, frozen_backends)

    # The saved file is the table the simulation will load for every RL agent.
    # With --train-one that table is the single agent that was trained.
    if args.episodes and not args.no_save:
        save_path = save_best_policy(backends, twins, args, best_index)
        print(f"Saved best policy to {save_path}")

    if training_log:
        if args.no_archive:
            print("\nNot archived to the gallery (--no-archive).")
        elif args.name is not None:
            archive_training_run(training_log, args.name, args)
        else:
            prompt_and_archive_training(training_log, args)


def evaluate(backends, twins, args, frozen_backends=None) -> int:
    """Greedy (ε=0, no learning) evaluation. Returns the best agent's index.

    Best means the highest academic capital on this fresh seed. Ties keep the
    earliest agent. ``--train-one`` scores only the agent that learned.
    """
    rl_talent = args.low_talent_value if args.low_talent else 1.0
    history = History()
    env, rl_agents, heuristics = build_env(
        backends=backends, backends_b=twins, epsilon=0.0, learning=False,
        num_rl=args.num_rl, num_heuristic=args.num_heuristic,
        horizon=args.horizon, seed=args.seed + 10_000,
        review_paradigm=args.review_paradigm,
        history=history, intrinsic_talent=rl_talent,
        shared_talents=args.shared_talents,
        frozen_backends=frozen_backends,
    )
    env.run(args.timesteps)

    rl_ac = mean_capital(rl_agents)
    heur_ac = mean_capital(heuristics)
    starts, decisions = rl_review_starts(history, rl_agents)
    if args.train_one:
        policy_mode = "training agent only"
    elif args.shared_policy:
        policy_mode = "shared policy"
    else:
        policy_mode = "independent policies"
    print(f"\nEvaluation (greedy, {policy_mode}):")
    print(f"  RL mean AC        = {rl_ac:8.2f}")
    if not args.train_one:
        print(f"  Heuristic mean AC = {heur_ac:8.2f}")
        if heur_ac:
            print(f"  RL / Heuristic    = {rl_ac / heur_ac:6.2%}")
    if decisions:
        print(
            f"  RL review starts  = {starts:8d} / {decisions} decisions "
            f"({starts / decisions:6.2%})"
        )
    else:
        print("  RL review starts  =        0")
    best_index = max(
        range(len(rl_agents)),
        key=lambda index: rl_agents[index].academic_capital,
    )
    best = rl_agents[best_index]
    print(f"  Best policy       = {best.name}  AC={best.academic_capital:8.2f}")
    print(describe_progress_policy(backends[best_index], twins[best_index]))
    gaps = write_claim_gaps(backends[best_index], twins[best_index])
    learned_refusal = all(gap > 1e-6 for gap in gaps)
    tied = all(abs(gap) <= 1e-6 for gap in gaps)
    if learned_refusal:
        print("  Learned refusal: Q(write) > Q(claim) in every manuscript bucket.")
    elif tied:
        print("  Q(write) and Q(claim) are tied. Greedy tie-break keeps writing.")
    elif min(gaps) > 0.0:
        print("  Q(write) >= Q(claim) in every manuscript bucket.")
    else:
        print("  Some manuscript buckets still score claiming above writing.")
    return best_index


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the Liberata Q-learning agent."
    )
    p.add_argument("--backend", choices=["tabular", "linear"],
                   default=SIM.rl_backend)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--timesteps", dest="timesteps", type=int, default=TRAIN.timesteps)
    p.add_argument("--num-rl", dest="num_rl", type=int, default=TRAIN.num_rl)
    p.add_argument("--num-heuristic", dest="num_heuristic", type=int,
                   default=TRAIN.num_heuristic)
    p.add_argument("--horizon", type=int, default=SIM.forecast_horizon_timesteps)
    p.add_argument("--review-paradigm", choices=["continuous", "discrete"],
                   default=SIM.review_paradigm)
    p.add_argument("--alpha", type=float, default=None,
                   help="learning rate (defaults: 0.1 tabular / 0.01 linear)")
    p.add_argument("--gamma", type=float, default=SIM.rl_gamma)
    p.add_argument("--eps-start", dest="eps_start", type=float,
                   default=TRAIN.eps_start)
    p.add_argument("--eps-end", dest="eps_end", type=float, default=None)
    p.add_argument("--seed", type=int, default=SIM.seed)
    p.add_argument("--save", default=None,
                   help="where to write the best policy "
                        "(default: policies/policy_<backend>)")
    p.add_argument(
        "--shared-policy",
        dest="shared_policy",
        action="store_true",
        help="Train one Q table shared by every RL agent.",
    )
    p.add_argument(
        "--train-one",
        dest="train_one",
        action="store_true",
        help="Train one agent against frozen copies of the latest saved policy. "
             "Save that agent's table and count only its reviews. "
             "Default length is 1000 episodes, and exploration ends at 0.",
    )
    p.add_argument(
        "--frozen-agents",
        dest="frozen_agents",
        type=int,
        default=9,
        help="How many frozen opponents --train-one adds (default 9).",
    )
    p.add_argument(
        "--frozen-policy",
        dest="frozen_policy",
        default=None,
        help="Policy the frozen opponents play "
             "(default: policies/policy_<backend>).",
    )
    p.add_argument("--no-save", dest="no_save", action="store_true",
                   help="do not persist the trained policy")
    p.add_argument("--load", default=None,
                   help="load a saved policy before training/evaluating")
    p.add_argument(
        "--name",
        default=None,
        help="Save to the gallery with this title and skip the prompt (for scripting).",
    )
    p.add_argument(
        "--no-archive",
        action="store_true",
        help="Skip the gallery prompt and do not save the training run.",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the episode return chart after training.",
    )
    p.add_argument(
        "--low-talent",
        dest="low_talent",
        action="store_true",
        default=TRAIN.low_talent,
        help="Train RL agents at low intrinsic talent (separate policy path).",
    )
    p.add_argument(
        "--low-talent-value",
        dest="low_talent_value",
        type=float,
        default=TRAIN.low_talent_value,
        metavar="X",
        help="Intrinsic talent when --low-talent is set.",
    )
    p.add_argument('--shared-talents', action='store_true',
                   help='Four quality/rate cohorts with experience, no seed papers, fixed writing effort.')
    args = p.parse_args(argv)
    if args.shared_talents and (args.low_talent or args.review_paradigm != 'continuous'):
        p.error('--shared-talents requires continuous mode without --low-talent')
    if args.train_one and args.shared_policy:
        p.error("--train-one trains one agent; do not pass --shared-policy")
    if args.alpha is None:
        args.alpha = 0.1 if args.backend == "tabular" else 0.01
    if args.train_one:
        args.num_rl = 1
        if args.episodes is None:
            args.episodes = 1000
        if args.eps_end is None:
            args.eps_end = 0.0
        if args.frozen_policy is None:
            args.frozen_policy = default_policy_path(args.backend)
        if args.frozen_agents < 1:
            p.error("--train-one needs at least one frozen opponent")
    else:
        if args.episodes is None:
            args.episodes = TRAIN.episodes
        if args.eps_end is None:
            args.eps_end = TRAIN.eps_end
        args.frozen_agents = 0
    return args


if __name__ == "__main__":
    train(parse_args())
