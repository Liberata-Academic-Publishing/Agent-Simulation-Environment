"""Train and evaluate the DQN agent on the Liberata market.

Mirrors ``train_rl.py`` but uses ``DQNAgent`` with a shared replay-backed MLP.
The network is dueling Double DQN with an n-step return, so a review that pays
nothing does not inherit the maximum next value. Self-play uses ε decay. The
trained network is saved to ``policies/policy_dqn.pkl``.

Usage:
    python train_dqn.py
    python train_dqn.py --episodes 300 --num-dqn 20
    python train_dqn.py --load policies/policy_dqn.pkl --episodes 0   # eval only
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

from Agent import Agent
from config import SIM, TRAIN_DQN, default_dqn_policy_path
from DQNAgent import DQNAgent, make_dqn_backend
from Environment import Environment
from HeuristicAgent import HeuristicAgent
from History import History
from train_rl import (
    mean_capital,
    mean_review_effort,
    prompt_and_archive_training,
    rl_review_starts,
    save_training_outputs,
    seed_initial_papers,
)

OUTPUT_DIR = SIM.output_dir


def build_env(
    *,
    backend,
    epsilon: float,
    learning: bool,
    num_dqn: int,
    num_heuristic: int,
    horizon: int,
    seed: int,
    review_paradigm: str = SIM.review_paradigm,
    gamma: float = SIM.dqn_gamma,
    history: History | None = None,
    n_step: int = SIM.dqn_n_step,
) -> tuple[Environment, list[DQNAgent], list[HeuristicAgent]]:
    """Fresh env: shared-backend DQN agents vs. heuristic opponents."""
    rng = random.Random(seed)
    Agent.all_papers = []

    dqn_agents = [
        DQNAgent(
            intrinsic_talent=1.0,
            forecast_horizon_timesteps=horizon,
            name=f"DQN {i}",
            backend=backend,
            epsilon=epsilon,
            learning=learning,
            gamma=gamma,
            n_step=n_step,
        )
        for i in range(num_dqn)
    ]
    heuristics = [
        HeuristicAgent(
            intrinsic_talent=1.0,
            forecast_horizon_timesteps=horizon,
            name=f"Heuristic {i}",
        )
        for i in range(num_heuristic)
    ]
    agents: list[Agent] = [*dqn_agents, *heuristics]

    seed_initial_papers(agents, rng)
    env = Environment(
        agents=agents,
        papers=Agent.all_papers,
        forecast_horizon_timesteps=horizon,
        review_paradigm=review_paradigm,
        history=history,
    )
    return env, dqn_agents, heuristics


def build_train_config(args) -> dict[str, Any]:
    return {
        "agent_kind": "dqn",
        "episodes": args.episodes,
        "timesteps": args.timesteps,
        "num_dqn": args.num_dqn,
        "num_heuristic": args.num_heuristic,
        "horizon": args.horizon,
        "review_paradigm": args.review_paradigm,
        "lr": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "target_sync": args.target_sync,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "eps_start": args.eps_start,
        "eps_end": args.eps_end,
        "seed": args.seed,
    }


def train(args) -> None:
    backend = make_dqn_backend(
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_sync=args.target_sync,
    )

    if args.load:
        backend.load(args.load)
        print(f"Loaded DQN policy from {args.load}")
    elif args.episodes == 0:
        print("Warning: --episodes 0 with no --load evaluates an empty network.")

    training_log: list[dict[str, Any]] = []

    if args.episodes:
        print(
            f"Training DQN: paradigm={args.review_paradigm} "
            f"episodes={args.episodes} timesteps={args.timesteps} "
            f"dqn={args.num_dqn} heuristic={args.num_heuristic}"
        )
    for episode in range(args.episodes):
        frac = episode / max(1, args.episodes - 1)
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)

        history = History()
        env, dqn_agents, _ = build_env(
            backend=backend,
            epsilon=epsilon,
            learning=True,
            num_dqn=args.num_dqn,
            num_heuristic=args.num_heuristic,
            horizon=args.horizon,
            seed=args.seed + episode,
            review_paradigm=args.review_paradigm,
            gamma=args.gamma,
            history=history,
            n_step=args.n_step,
        )
        env.run(args.timesteps)
        for agent in dqn_agents:
            agent.end_episode()

        mean_return = mean_capital(dqn_agents)
        training_log.append(
            {
                "episode": episode,
                "mean_return": mean_return,
                "mean_review_effort": mean_review_effort(history),
                "epsilon": epsilon,
            }
        )

        last = episode == args.episodes - 1
        if episode % max(1, args.episodes // 10) == 0 or last:
            print(
                f"  ep {episode:4d}  eps={epsilon:.3f}  "
                f"DQN mean AC={mean_return:8.2f}"
            )

    if training_log:
        save_training_outputs(training_log, open_chart_flag=args.open)

    if args.episodes and not args.no_save:
        save_path = args.save or default_dqn_policy_path()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        backend.save(save_path)
        print(f"Saved DQN policy to {save_path}")

    if training_log:
        if args.no_archive:
            print("\nNot archived to the gallery (--no-archive).")
        elif args.name is not None:
            archive_training_run(training_log, args.name, args)
        else:
            prompt_and_archive_training(training_log, args)

    evaluate(backend, args)


def evaluate(backend, args) -> None:
    history = History()
    env, dqn_agents, heuristics = build_env(
        backend=backend,
        epsilon=0.0,
        learning=False,
        num_dqn=args.num_dqn,
        num_heuristic=args.num_heuristic,
        horizon=args.horizon,
        seed=args.seed + 10_000,
        review_paradigm=args.review_paradigm,
        history=history,
        n_step=args.n_step,
    )
    env.run(args.timesteps)

    dqn_ac = mean_capital(dqn_agents)
    heur_ac = mean_capital(heuristics)
    starts, decisions = rl_review_starts(history, dqn_agents)
    print("\nEvaluation (greedy):")
    print(f"  DQN mean AC        = {dqn_ac:8.2f}")
    print(f"  Heuristic mean AC  = {heur_ac:8.2f}")
    if heur_ac:
        print(f"  DQN / Heuristic    = {dqn_ac / heur_ac:6.2%}")
    if decisions:
        print(
            f"  DQN review starts  = {starts:8d} / {decisions} decisions "
            f"({starts / decisions:6.2%})"
        )
    else:
        print("  DQN review starts  =        0")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the Liberata DQN agent.")
    p.add_argument("--episodes", type=int, default=TRAIN_DQN.episodes)
    p.add_argument("--timesteps", dest="timesteps", type=int, default=TRAIN_DQN.timesteps)
    p.add_argument("--num-dqn", dest="num_dqn", type=int, default=TRAIN_DQN.num_dqn)
    p.add_argument(
        "--num-heuristic",
        dest="num_heuristic",
        type=int,
        default=TRAIN_DQN.num_heuristic,
    )
    p.add_argument("--horizon", type=int, default=SIM.forecast_horizon_timesteps)
    p.add_argument(
        "--review-paradigm",
        choices=["continuous", "discrete"],
        default=SIM.review_paradigm,
    )
    p.add_argument("--lr", type=float, default=SIM.dqn_lr)
    p.add_argument("--gamma", type=float, default=SIM.dqn_gamma)
    p.add_argument(
        "--n-step",
        dest="n_step",
        type=int,
        default=SIM.dqn_n_step,
        help="Reward steps in each target. A zero-share review adds nothing.",
    )
    p.add_argument("--batch-size", dest="batch_size", type=int, default=SIM.dqn_batch_size)
    p.add_argument(
        "--target-sync", dest="target_sync", type=int, default=SIM.dqn_target_sync
    )
    p.add_argument("--hidden-size", dest="hidden_size", type=int, default=SIM.dqn_hidden_size)
    p.add_argument(
        "--hidden-layers", dest="hidden_layers", type=int, default=SIM.dqn_hidden_layers
    )
    p.add_argument("--eps-start", dest="eps_start", type=float, default=TRAIN_DQN.eps_start)
    p.add_argument("--eps-end", dest="eps_end", type=float, default=TRAIN_DQN.eps_end)
    p.add_argument("--seed", type=int, default=SIM.seed)
    p.add_argument(
        "--save",
        default=None,
        help="policy save path (default: policies/policy_dqn.pkl)",
    )
    p.add_argument(
        "--no-save",
        dest="no_save",
        action="store_true",
        help="do not persist the trained policy",
    )
    p.add_argument(
        "--load",
        default=None,
        help="load a saved DQN policy before training/evaluating",
    )
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
    return p.parse_args(argv)


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
    print(
        "Publish it with: "
        "git add docs/data && git commit -m 'Add DQN training run' && git push"
    )


if __name__ == "__main__":
    train(parse_args())
