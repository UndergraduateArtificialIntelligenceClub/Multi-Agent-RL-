"""Independent PPO (IPPO) on Pursuit.

Every pursuer gets its OWN actor-critic network and its OWN optimizer, and is trained only
on its own observations, actions and rewards. The other pursuers are just part of the
environment from its point of view. This is the "independent learners" setup, as opposed to
parameter sharing (one network for all pursuers, e.g. SB3 + SuperSuit) or CTDE (a
centralized critic, e.g. MAPPO).

    python -m phase1_pursuit.ippo --total-steps 2000000 --run-name ippo-baseline

Outputs (both gitignored):
    outputs/<run-name>/train_log.csv     one row per PPO update
    checkpoints/phase1_ippo-<run-name>_<date>.pt
"""
import argparse
import copy
import csv
import datetime
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from phase1_pursuit.env import evaders_captured, make_pursuit

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- network
class AgentNet(nn.Module):
    """Small CNN actor-critic for one pursuer. Input: (B, 7, 7, 3) local observation."""

    def __init__(self, n_actions: int = 5, in_channels: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3), nn.ReLU(),   # 7x7 -> 5x5
            nn.Conv2d(32, 64, 3), nn.ReLU(),            # 5x5 -> 3x3
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 256), nn.ReLU(),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs.permute(0, 3, 1, 2))  # HWC -> CHW
        return self.policy(h), self.value(h).squeeze(-1)


def batched_forward(nets: list[AgentNet]):
    """Run all agents' networks in ONE vectorised call (inference only).

    Each agent still uses its own weights; vmap just evaluates the 8 networks side by side
    instead of launching 8 separate tiny GPU calls. Returns f(obs[A, B, 7, 7, 3]) ->
    (logits[A, B, n_actions], values[A, B]). Rebuild after every optimizer update.
    """
    params, buffers = torch.func.stack_module_state(nets)
    params = {k: v.detach() for k, v in params.items()}
    base = copy.deepcopy(nets[0]).to("meta")

    def one(p, b, x):
        return torch.func.functional_call(base, (p, b), (x,))

    return lambda obs: torch.vmap(one)(params, buffers, obs)


def load_checkpoint(path, device="cpu") -> tuple[list[AgentNet], dict]:
    ckpt = torch.load(path, map_location=device)
    nets = []
    for sd in ckpt["agents"]:
        net = AgentNet().to(device)
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return nets, ckpt


# --------------------------------------------------------------------------- vector env
def _worker(remote, seed: int):
    """One Pursuit env in its own process. Auto-resets when the episode ends."""
    env = make_pursuit()
    agents = env.possible_agents
    ep = 0

    def stack(obs):
        return np.stack([obs.get(a, np.zeros((7, 7, 3), np.float32)) for a in agents]).astype(np.float32)

    obs, _ = env.reset(seed=seed)
    agent_returns = np.zeros(len(agents))
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            remote.send(stack(obs))
        elif cmd == "step":
            obs, rew, term, trunc, _ = env.step({a: int(data[i]) for i, a in enumerate(agents)})
            r = np.array([rew.get(a, 0.0) for a in agents], np.float32)
            agent_returns += r
            terminated = any(term.values())
            truncated = any(trunc.values()) and not terminated
            info = None
            final_obs = None
            if terminated or truncated or not env.agents:
                final_obs = stack(obs)
                info = {
                    "team_return": float(agent_returns.sum()),
                    "agent_returns": agent_returns.tolist(),
                    "captures": evaders_captured(env),
                }
                ep += 1
                obs, _ = env.reset(seed=seed + 100_000 * ep)
                agent_returns = np.zeros(len(agents))
                terminated = terminated or not truncated
            remote.send((stack(obs), r, terminated, truncated, final_obs, info))
        elif cmd == "close":
            env.close()
            remote.close()
            return


class VecPursuit:
    def __init__(self, n: int, seed: int):
        ctx = mp.get_context("fork")
        self.remotes, procs = zip(*[ctx.Pipe() for _ in range(n)])
        self.procs = []
        for i, child in enumerate(procs):
            p = ctx.Process(target=_worker, args=(child, seed * 1000 + i), daemon=True)
            p.start()
            self.procs.append(p)
        self.n = n

    def reset(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("reset", None))
        return np.stack([r.recv() for r in self.remotes])

    def step(self, actions: np.ndarray):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        return zip(*[r.recv() for r in self.remotes])

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)


# --------------------------------------------------------------------------- training
def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    probe = make_pursuit()
    agent_ids = list(probe.possible_agents)
    n_actions = probe.action_space(agent_ids[0]).n
    probe.close()
    A, N, T = len(agent_ids), args.num_envs, args.num_steps
    venv = VecPursuit(N, args.seed)  # fork workers before CUDA is initialised

    nets =[AgentNet(n_actions).to(device) for _ in range(A)]
    opts = [torch.optim.Adam(n.parameters(), lr=args.lr, eps=1e-5) for n in nets]

    out_dir = REPO_ROOT / "outputs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.date.today().isoformat()
    ckpt_path = REPO_ROOT / "checkpoints" / f"phase1_ippo-{args.run_name}_{date}.pt"
    (out_dir / "run.json").write_text(json.dumps({**vars(args), "device": str(device),
                                                  "checkpoint": str(ckpt_path)}, indent=2))

    fields = (["global_step", "update", "episodes", "team_return", "captures"]
              + [f"return_{a}" for a in agent_ids]
              + ["policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "sps"])
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log = csv.DictWriter(log_f, fieldnames=fields)
    log.writeheader()

    # rollout buffers, indexed [t, env, agent]
    obs_buf = torch.zeros((T, N, A, 7, 7, 3), device=device)
    act_buf = torch.zeros((T, N, A), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, N, A), device=device)
    val_buf = torch.zeros((T, N, A), device=device)
    rew_buf = torch.zeros((T, N, A), device=device)
    done_buf = torch.zeros((T, N), device=device)

    next_obs =torch.as_tensor(venv.reset(), device=device)
    next_done = torch.zeros(N, device=device)

    n_updates = args.total_steps // (N * T)
    global_step = 0
    start = time.time()
    print(f"IPPO: {A} independent agents, {N} envs x {T} steps, {n_updates} updates, device={device}")

    try:
        for update in range(1, n_updates + 1):
            frac = 1.0 - (update - 1) / n_updates
            for o in opts:
                o.param_groups[0]["lr"] = frac * args.lr
            episodes = []
            fwd = batched_forward(nets)  # snapshot of the 8 agents' current weights

            # ---- collect a rollout: each agent acts with its own network
            for t in range(T):
                global_step += N
                obs_buf[t] = next_obs
                done_buf[t] = next_done
                with torch.no_grad():
                    logits, v = fwd(next_obs.transpose(0, 1))  # (A, N, ...)
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    act_buf[t] = a.T
                    logp_buf[t] = dist.log_prob(a).T
                    val_buf[t] = v.T
                obs_np, rews, terms, truncs, finals, infos = venv.step(act_buf[t].cpu().numpy())
                rew = torch.as_tensor(np.stack(rews), device=device)
                # time-limit truncation: bootstrap from the value of the final observation
                for e in range(N):
                    if truncs[e]:
                        fo = torch.as_tensor(finals[e], device=device)
                        with torch.no_grad():
                            rew[e] += args.gamma * fwd(fo[:, None])[1][:, 0]
                    if infos[e] is not None:
                        episodes.append(infos[e])
                rew_buf[t] = rew
                next_obs = torch.as_tensor(np.stack(obs_np), device=device)
                next_done = torch.as_tensor(np.logical_or(terms, truncs), dtype=torch.float32, device=device)

            # ---- GAE per agent
            with torch.no_grad():
                next_val = fwd(next_obs.transpose(0, 1))[1].T
                adv = torch.zeros_like(rew_buf)
                last = torch.zeros((N, A), device=device)
                for t in reversed(range(T)):
                    nonterm = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                    nv = next_val if t == T - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + args.gamma * nv * nonterm[:, None] - val_buf[t]
                    last = delta + args.gamma * args.gae_lambda * nonterm[:, None] * last
                    adv[t] = last
                ret = adv + val_buf

            # ---- PPO update, separately for each agent on its own data
            stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}
            B = T * N
            mb = B // args.num_minibatches
            for i, (net, opt) in enumerate(zip(nets, opts)):
                b_obs = obs_buf[:, :, i].reshape(B, 7, 7, 3)
                b_act = act_buf[:, :, i].reshape(B)
                b_logp = logp_buf[:, :, i].reshape(B)
                b_adv = adv[:, :, i].reshape(B)
                b_ret = ret[:, :, i].reshape(B)
                for _ in range(args.update_epochs):
                    perm = torch.randperm(B, device=device)
                    for s in range(0, B, mb):
                        idx = perm[s : s + mb]
                        logits, v = net(b_obs[idx])
                        dist = Categorical(logits=logits)
                        logp = dist.log_prob(b_act[idx])
                        ratio = (logp - b_logp[idx]).exp()
                        a = b_adv[idx]
                        a = (a - a.mean()) / (a.std() + 1e-8)
                        pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                        vl = 0.5 * (v - b_ret[idx]).pow(2).mean()
                        ent = dist.entropy().mean()
                        loss = pg + args.vf_coef * vl - args.ent_coef * ent
                        opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                        opt.step()
                with torch.no_grad():  # stats from the last minibatch of this agent
                    stats["policy_loss"].append(pg.item())
                    stats["value_loss"].append(vl.item())
                    stats["entropy"].append(ent.item())
                    stats["approx_kl"].append(((ratio - 1) - ratio.log()).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())

            sps = int(global_step / (time.time() - start))
            row = {"global_step": global_step, "update": update, "episodes": len(episodes), "sps": sps,
                   **{k: float(np.mean(v)) for k, v in stats.items()}}
            if episodes:
                row["team_return"] = float(np.mean([e["team_return"] for e in episodes]))
                row["captures"] = float(np.mean([e["captures"] for e in episodes]))
                per_agent = np.mean([e["agent_returns"] for e in episodes], axis=0)
                row.update({f"return_{a}": float(r) for a, r in zip(agent_ids, per_agent)})
                print(f"step {global_step:>9,}  team_return {row['team_return']:8.1f}  "
                      f"captures {row['captures']:5.1f}  entropy {row['entropy']:.3f}  sps {sps}", flush=True)
            log.writerow(row)
            log_f.flush()

            if update % 50 == 0 or update == n_updates:
                torch.save({"agents": [n.state_dict() for n in nets], "agent_ids": agent_ids,
                            "config": vars(args), "global_step": global_step}, ckpt_path)
    finally:
        venv.close()
        log_f.close()
    print(f"Done in {time.time() - start:.0f}s. Checkpoint: {ckpt_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="ippo-baseline")
    p.add_argument("--total-steps", type=int, default=2_000_000, help="environment steps (all envs)")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128, help="rollout length per env per update")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    train(p.parse_args())


if __name__ == "__main__":
    main()
