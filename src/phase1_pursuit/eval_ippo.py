"""Evaluate an IPPO checkpoint against the random baseline and plot the results.

    python -m phase1_pursuit.eval_ippo --run-name ippo-baseline --episodes 20

Uses the same seeds (0..N-1) as random_baseline.py. Writes to outputs/<run-name>/:
    eval.json, learning_curve.png, eval_comparison.png
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from phase1_pursuit.env import evaders_captured, make_pursuit
from phase1_pursuit.ippo import REPO_ROOT, load_checkpoint


def run_episode(env, seed: int, policy) -> tuple[float, int]:
    """policy(obs_dict) -> action_dict. Returns (team reward, evaders captured)."""
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while env.agents:
        obs, rew, _, _, _ = env.step(policy(obs))
        total += sum(rew.values())
    return total, evaders_captured(env)


def trained_policy(nets, agent_ids, greedy: bool):
    def act(obs):
        out = {}
        with torch.no_grad():
            for a in obs:
                logits, _ = nets[agent_ids.index(a)](torch.as_tensor(obs[a][None], dtype=torch.float32))
                out[a] = int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())
        return out
    return act


def summarize(results):
    rewards = [r for r, _ in results]
    caps = [c for _, c in results]
    return {"team_reward_mean": statistics.mean(rewards), "team_reward_std": statistics.pstdev(rewards),
            "captures_mean": statistics.mean(caps), "captures_std": statistics.pstdev(caps),
            "per_episode": results}


def plot(out_dir: Path, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(out_dir / "train_log.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["team_return"]]
    steps = np.array([int(r["global_step"]) for r in rows])
    ret = np.array([float(r["team_return"]) for r in rows])
    cap = np.array([float(r["captures"]) for r in rows])
    rnd = results["random"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, y, base, label in [(axes[0], ret, rnd["team_reward_mean"], "Team reward per episode"),
                               (axes[1], cap, rnd["captures_mean"], "Evaders captured per episode (of 30)")]:
        ax.plot(steps, y, color="#2a6fdb", lw=1.6, label="IPPO (8 independent nets)")
        ax.axhline(base, color="#888", ls="--", lw=1.2, label="Random baseline")
        ax.set_xlabel("Environment steps")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=130)
    plt.close(fig)

    names = ["random", "ippo_sampled", "ippo_greedy"]
    labels = ["Random", "IPPO (sampled)", "IPPO (greedy)"]
    colors = ["#999999", "#2a6fdb", "#7aa7f0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [(axes[0], "team_reward", "Team reward per episode"),
                           (axes[1], "captures", "Evaders captured per episode")]:
        means = [results[n][f"{key}_mean"] for n in names]
        stds = [results[n][f"{key}_std"] for n in names]
        ax.bar(labels, means, yerr=stds, color=colors, capsize=4)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "eval_comparison.png", dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ippo-baseline")
    p.add_argument("--checkpoint", default=None, help="defaults to the path in outputs/<run>/run.json")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    out_dir = REPO_ROOT / "outputs" / args.run_name
    ckpt = args.checkpoint or json.loads((out_dir / "run.json").read_text())["checkpoint"]
    nets, meta = load_checkpoint(ckpt)
    agent_ids = meta["agent_ids"]
    torch.manual_seed(0)

    env = make_pursuit()
    policies = {
        "random": lambda obs: {a: env.action_space(a).sample() for a in obs},
        "ippo_sampled": trained_policy(nets, agent_ids, greedy=False),
        "ippo_greedy": trained_policy(nets, agent_ids, greedy=True),
    }
    results = {}
    for name, pol in policies.items():
        env.action_space(agent_ids[0]).seed(0)
        results[name] = summarize([run_episode(env, s, pol) for s in range(args.episodes)])
        r = results[name]
        print(f"{name:13s} team reward {r['team_reward_mean']:8.1f} ± {r['team_reward_std']:6.1f}   "
              f"captures {r['captures_mean']:5.1f} ± {r['captures_std']:4.1f}")
    env.close()

    results["checkpoint"] = str(ckpt)
    results["trained_env_steps"] = meta.get("global_step")
    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    plot(out_dir, results)
    print(f"Wrote {out_dir}/eval.json, learning_curve.png, eval_comparison.png")


if __name__ == "__main__":
    main()
