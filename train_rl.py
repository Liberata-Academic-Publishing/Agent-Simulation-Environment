"""Train and evaluate the Q-learning agent on the Liberata market.

Self-play across many short episodes: a pool of ``QLearningAgent``s sharing
one Q backend learns online (reward = Δ academic capital), with ε decayed each
episode. After training we run a greedy (ε=0) evaluation against
``HeuristicAgent``s and report the capital gap.

Training runs under the configured ``review_paradigm`` (``--review-paradigm``):
in the default *continuous* paradigm the RL agent makes one merged decision per
timestep; in *discrete* it uses the two-phase flow. The action space differs
between paradigms, so a policy trained in one is not meaningful in the other.

The trained policy (the Q backend) is auto-saved to ``policies/`` and can be
reloaded with ``--load`` to resume training or to run a frozen policy.

Usage:
    python train_rl.py                       # train + auto-save to policies/
    python train_rl.py --backend linear --episodes 300
    python train_rl.py --review-paradigm discrete
    python train_rl.py --load policies/policy_tabular.pkl --episodes 0   # eval only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

from Agent import Agent
from config import SIM, TRAIN, default_policy_path, default_low_talent_policy_path
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from Paper import Paper
from QLearningAgent import QLearningAgent, make_backend

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
    """Seed starting papers per SimConfig (listed on the market from timestep 1)."""
    index = 0
    for agent in agents:
        for _ in range(SIM.init_papers_per_agent):
            index += 1
            paper = Paper(
                author=agent,
                quality=agent.intrinsic_talent,
                current_ac=rng.uniform(SIM.init_ac_min, SIM.init_ac_max),
                market_listed=True,
            )
            paper.title = f"Paper {index}"
            Agent.all_papers.append(paper)


def build_env(
    *,
    backend,
    epsilon: float,
    learning: bool,
    num_rl: int,
    num_heuristic: int,
    horizon: int,
    seed: int,
    review_paradigm: str = SIM.review_paradigm,
    gamma: float = 0.95,
    history: History | None = None,
    intrinsic_talent: float = 1.0,
) -> tuple[Environment, list[QLearningAgent], list[HeuristicAgent]]:
    """Fresh env: shared-backend RL agents vs. heuristic opponents."""
    rng = random.Random(seed)
    Agent.all_papers = []

    rl_agents = [
        QLearningAgent(
            intrinsic_talent=intrinsic_talent,
            forecast_horizon_timesteps=horizon,
            name=f"RL {i}",
            backend=backend,
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
    agents: list[Agent] = [*rl_agents, *heuristics]

    seed_initial_papers(agents, rng)
    env = Environment(agents=agents, papers=Agent.all_papers,
                      forecast_horizon_timesteps=horizon,
                      review_paradigm=review_paradigm,
                      history=history)
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


def train(args) -> None:
    backend = make_backend(args.backend, alpha=args.alpha)

    if args.load:
        backend.load(args.load)
        print(f"Loaded policy from {args.load}")
    elif args.episodes == 0:
        print("Warning: --episodes 0 with no --load evaluates an empty policy.")

    training_log: list[dict[str, Any]] = []

    if args.episodes:
        talent_note = f" low_talent={args.low_talent_value}" if args.low_talent else ""
        print(f"Training: paradigm={args.review_paradigm} backend={args.backend} "
              f"episodes={args.episodes} timesteps={args.timesteps} "
              f"rl={args.num_rl} heuristic={args.num_heuristic}{talent_note}")
    rl_talent = args.low_talent_value if args.low_talent else 1.0
    for episode in range(args.episodes):
        # Linear ε decay from eps_start to eps_end.
        frac = episode / max(1, args.episodes - 1)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        history = History()
        env, rl_agents, _ = build_env(
            backend=backend, epsilon=epsilon, learning=True,
            num_rl=args.num_rl, num_heuristic=args.num_heuristic,
            horizon=args.horizon, seed=args.seed + episode,
            review_paradigm=args.review_paradigm, gamma=args.gamma,
            history=history, intrinsic_talent=rl_talent,
        )
        env.run(args.timesteps)
        for agent in rl_agents:
            agent.end_episode()

        mean_return = mean_capital(rl_agents)
        training_log.append({
            "episode": episode,
            "mean_return": mean_return,
            "mean_review_effort": mean_review_effort(history),
            "epsilon": epsilon,
        })

        last = episode == args.episodes - 1
        if episode % max(1, args.episodes // 10) == 0 or last:
            print(f"  ep {episode:4d}  eps={epsilon:.3f}  "
                  f"RL mean AC={mean_return:8.2f}")

    if training_log:
        save_training_outputs(training_log, open_chart_flag=args.open)

    # Persist the trained policy unless told not to (nothing new if no training).
    if args.episodes and not args.no_save:
        if args.save:
            save_path = args.save
        elif args.low_talent:
            save_path = default_low_talent_policy_path(args.backend)
        else:
            save_path = default_policy_path(args.backend)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        backend.save(save_path)
        print(f"Saved policy to {save_path}")

    if training_log:
        if args.no_archive:
            print("\nNot archived to the gallery (--no-archive).")
        elif args.name is not None:
            archive_training_run(training_log, args.name, args)
        else:
            prompt_and_archive_training(training_log, args)

    evaluate(backend, args)


def evaluate(backend, args) -> None:
    """Greedy (ε=0, no learning) RL agents vs. heuristics on a fresh seed."""
    rl_talent = args.low_talent_value if args.low_talent else 1.0
    env, rl_agents, heuristics = build_env(
        backend=backend, epsilon=0.0, learning=False,
        num_rl=args.num_rl, num_heuristic=args.num_heuristic,
        horizon=args.horizon, seed=args.seed + 10_000,
        review_paradigm=args.review_paradigm,
        intrinsic_talent=rl_talent,
    )
    env.run(args.timesteps)

    rl_ac = mean_capital(rl_agents)
    heur_ac = mean_capital(heuristics)
    print("\nEvaluation (greedy):")
    print(f"  RL mean AC        = {rl_ac:8.2f}")
    print(f"  Heuristic mean AC = {heur_ac:8.2f}")
    if heur_ac:
        print(f"  RL / Heuristic    = {rl_ac / heur_ac:6.2%}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train the Liberata Q-learning agent."
    )
    p.add_argument("--backend", choices=["tabular", "linear"],
                   default=SIM.rl_backend)
    p.add_argument("--episodes", type=int, default=TRAIN.episodes)
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
    p.add_argument("--eps-end", dest="eps_end", type=float,
                   default=TRAIN.eps_end)
    p.add_argument("--seed", type=int, default=SIM.seed)
    p.add_argument("--save", default=None,
                   help="policy save path (default: policies/policy_<backend>)")
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
    args = p.parse_args(argv)
    if args.alpha is None:
        args.alpha = 0.1 if args.backend == "tabular" else 0.01
    return args


if __name__ == "__main__":
    train(parse_args())
