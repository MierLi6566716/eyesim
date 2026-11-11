"""
Closed-loop refraction with an eye that refocuses: baselines and a PPO agent.

  /opt/anaconda3/bin/python scripts/rl_closed_loop.py baselines            # score the hand-made policies
  /opt/anaconda3/bin/python scripts/rl_closed_loop.py train --steps 3000000 # train the agent
  /opt/anaconda3/bin/python scripts/rl_closed_loop.py eval                 # agent vs baselines, same eyes

Environment: eyesim/rl/env.py (sensor = real model-eye readings at 37 cm; eye =
eyesim/rl/accommodation.py). All policies are scored on the same 2000 eyes.

Policies only use what a device would see: past readings (or "no reading"),
the lens power, and time. None of them sees the true refraction.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eyesim.rl.env import REPORT, STEPS, ClosedLoopRefraction  # noqa: E402

OUT = ROOT / "checkpoints" / "rl_closed_loop"
N_EVAL = 2000


def act_toward(dL_wanted):
    """Lens change -> nearest allowed action (largest step not overshooting)."""
    best, err = 3, abs(dL_wanted)
    for i, s in enumerate(STEPS):
        if abs(dL_wanted - s) < err - 1e-9:
            best, err = i, abs(dL_wanted - s)
    return best


class Searcher:
    """Shared start for the rule-based policies: from lens 0, find a lens with a reading.
    Tries 0, then +6 (myopic side of the band), then -6 (hyperopic side)."""
    PLAN = [0.0, 6.0, -6.0, 9.0]

    def __init__(self):
        self.k, self.wait = 0, 0

    def __call__(self, env):
        target = self.PLAN[min(self.k, len(self.PLAN) - 1)]
        if abs(env.L - target) > 1e-6:
            return act_toward(target - env.L)
        if env.readings and env.readings[-1][0] == env.t:
            return None                         # readable: searching is done
        self.wait += 1
        if self.wait > 3:
            self.k, self.wait = self.k + 1, 0
        return 3


class OneShot:
    """Find any reading, take REPORT_FRAMES of them, report."""
    def reset(self):
        self.s, self.n = Searcher(), 0

    def __call__(self, env, obs):
        a = self.s(env)
        if a is not None:
            self.n = 0
            return a
        self.n += 1
        return REPORT if self.n >= 6 else 3


class NullSeek:
    """The glasses design's loop, on the bench: steer eye+lens to a fixed readable point
    (S* = -7.5 D, just past the blind band) with a proportional controller, report when settled."""
    def reset(self):
        self.s, self.ok, self.found = Searcher(), 0, False

    def __call__(self, env, obs):
        if not self.found:
            a = self.s(env)
            if a is not None:
                return a
            self.found = True
        if not env.readings or env.readings[-1][0] != env.t:
            return act_toward(3.0)              # lost the crescent: more plus
        S = env.readings[-1][1] - env.L
        err = S + 7.5
        self.ok = self.ok + 1 if abs(err) < 0.4 else 0
        if self.ok >= 6:
            return REPORT
        return act_toward(0.8 * err)


class FogFlash:
    """Hand-designed with knowledge of the eye model: get a rough reading, fog MILDLY
    (+1.5 D past the rough estimate, which relaxes accommodation but gives no reading at 37 cm)
    for 3 s, then jump to a readable lens and report within 0.2 s, before the eye can react."""
    def reset(self):
        self.s, self.phase, self.n, self.est = Searcher(), 0, 0, None

    def __call__(self, env, obs):
        if self.phase == 0:
            a = self.s(env)
            if a is not None:
                return a
            self.est = env.readings[-1][1]
            self.phase, self.n = 1, 0
        if self.phase == 1:                      # mild fog
            target = self.est + 1.5
            if abs(env.L - target) > 0.13:
                return act_toward(target - env.L)
            self.n += 1
            if self.n >= 90:
                self.phase, self.n = 2, 0
            return 3
        target = self.est + 7.5                  # flash: eye+lens about -7.5, readable
        if abs(env.L - target) > 0.13:
            return act_toward(target - env.L)
        self.n += 1
        return REPORT if self.n >= 6 else 3


class FogFlash2(FogFlash):
    """Stronger hand rule: fog-then-flash twice. The second fog uses the (less accommodated)
    estimate from the first flash, and is shorter."""
    def reset(self):
        super().reset(); self.cycle = 0

    def __call__(self, env, obs):
        a = super().__call__(env, obs)
        if a == REPORT and self.cycle == 0:
            self.cycle = 1
            self.est = env.report() if env.report() is not None else self.est   # flash estimate
            self.phase, self.n = 1, 45                 # second fog: 1.5 s
            return 3
        return a


def oracle_floor(env, n=N_EVAL, seed=123):
    rng = np.random.default_rng(seed)
    errs = []
    for i in range(n):
        env.rng = np.random.default_rng(rng.integers(1 << 31))
        env.reset()
        L = env.M0 + 7.5
        reads = [env._read(env.M0 - L) for _ in range(6)]
        errs.append(abs(np.median([r + L for r in reads]) - env.M0))
    e = np.array(errs)
    return dict(mae=float(e.mean()), median=float(np.median(e)), w025=float((e <= 0.25).mean()),
                w05=float((e <= 0.5).mean()), fail=0.0, secs=0.2)


# ---------------------------------------------------------------- PPO
class Net(nn.Module):
    def __init__(self, obs_dim, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh())
        self.pi = nn.Linear(256, n_act)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class Agent:
    def __init__(self, net):
        self.net = net

    def reset(self):
        pass

    def __call__(self, env, obs):
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs)[None])
        return int(logits.argmax())


def behaviour_clone(net, episodes=4000, epochs=10, seed=7, expert=None):
    """Warm start: imitate a hand-designed rule (default: fog-flash twice), expert actions on its own trajectories."""
    env = ClosedLoopRefraction(seed=seed)
    O, A = [], []
    for _ in range(episodes):
        obs, p, done = env.reset(), (expert or FogFlash2)(), False
        p.reset()
        while not done:
            a = p(env, obs)
            O.append(obs); A.append(a)
            obs, _, done, _ = env.step(a)
    O, A = torch.from_numpy(np.array(O, np.float32)), torch.from_numpy(np.array(A))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(O)); tot = 0
        for s in range(0, len(O), 1024):
            idx = perm[s:s + 1024]
            logits, _ = net(O[idx])
            loss = nn.functional.cross_entropy(logits, A[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        with torch.no_grad():
            acc = (net(O)[0].argmax(1) == A).float().mean().item()
        print(f"behaviour cloning epoch {ep + 1}: loss {tot / len(O):.3f}, matches the rule on {100 * acc:.1f}% of steps", flush=True)


def train(steps, n_env=64, horizon=256, seed=0, bc=True):
    torch.set_num_threads(2)
    envs = [ClosedLoopRefraction(seed=seed + i) for i in range(n_env)]
    obs = np.stack([e.reset() for e in envs])
    net = Net(envs[0].obs_dim, envs[0].n_actions)
    if bc:
        behaviour_clone(net)
        res = run_policy(ClosedLoopRefraction(seed=0), lambda: Agent(net))
        print(f"after cloning: mean |err| {res['mae']:.2f} D, within 0.25 D {100 * res['w025']:.0f}%", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), OUT / "agent_clone.pt")
    import copy
    ref = copy.deepcopy(net).eval() if bc else None          # the cloned expert, frozen
    kl_coef = 0.1 if bc else 0.0
    opt = torch.optim.Adam(net.parameters(), lr=5e-5 if bc else 3e-4)
    it, warm, best_err = 0, (6 if bc else 0), 1e9
    gamma, lam, clip = 0.995, 0.95, 0.2
    ent_coef = 0.003 if bc else 0.01
    done_err, done_t, t0, total = [], [], time.time(), 0
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "train_log.csv", "w")
    log.write("steps,mean_err,median_err,within_0.25,episodes,secs\n")
    while total < steps:
        B = {k: [] for k in ("o", "a", "lp", "r", "d", "v")}
        for _ in range(horizon):
            with torch.no_grad():
                logits, v = net(torch.from_numpy(obs))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            B["o"].append(obs); B["a"].append(a.numpy()); B["lp"].append(dist.log_prob(a).numpy()); B["v"].append(v.numpy())
            rs, ds = [], []
            for i, e in enumerate(envs):
                o2, r, d, info = e.step(int(a[i]))
                if d:
                    done_err.append(info["err"]); done_t.append(info["t"])
                    o2 = e.reset()
                obs[i] = o2; rs.append(r); ds.append(d)
            B["r"].append(np.array(rs, np.float32)); B["d"].append(np.array(ds, np.float32))
        with torch.no_grad():
            _, v_last = net(torch.from_numpy(obs))
        R, D, V = np.array(B["r"]), np.array(B["d"]), np.array(B["v"])
        adv = np.zeros_like(R); last = np.zeros(n_env, np.float32); nv = v_last.numpy()
        for t in reversed(range(horizon)):
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            last = delta + gamma * lam * (1 - D[t]) * last
            adv[t] = last; nv = V[t]
        ret = adv + V
        O = torch.from_numpy(np.concatenate(B["o"])); A = torch.from_numpy(np.concatenate(B["a"]))
        LP = torch.from_numpy(np.concatenate(B["lp"])); ADV = torch.from_numpy(adv.reshape(-1))
        RET = torch.from_numpy(ret.reshape(-1)); ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O)
        for _ in range(4):
            perm = torch.randperm(n)
            for s in range(0, n, 2048):
                idx = perm[s:s + 2048]
                logits, v = net(O[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                pl = -torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * ADV[idx]).mean()
                vl = 0.5 * ((v - RET[idx]) ** 2).mean()
                loss = vl if it < warm else pl + vl - ent_coef * dist.entropy().mean()   # value warm-up first
                if ref is not None and it >= warm:                 # stay near the expert unless it pays to leave
                    with torch.no_grad():
                        lp_ref = torch.log_softmax(ref(O[idx])[0], -1)
                    loss = loss + kl_coef * (lp_ref.exp() * (lp_ref - torch.log_softmax(logits, -1))).sum(-1).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        total += n
        if done_err:
            e = np.array(done_err[-2000:])
            line = f"{total},{e.mean():.3f},{np.median(e):.3f},{(e <= 0.25).mean():.3f},{len(done_err)},{time.time() - t0:.0f}"
            log.write(line + "\n"); log.flush()
            print(f"steps {total:>9}  episodes {len(done_err):>6}  error mean {e.mean():.2f} median {np.median(e):.2f}  "
                  f"within 0.25 D {100 * (e <= 0.25).mean():.0f}%  episode {np.mean(done_t[-2000:]) / 30:.1f}s  {time.time() - t0:.0f}s", flush=True)
        it += 1
        if it % 10 == 0 and it > warm:                      # greedy score on held-out eyes (not the final test set)
            g = run_policy(ClosedLoopRefraction(seed=1), lambda: Agent(net), n=300, seed=999)
            print(f"   greedy check: mean |err| {g['mae']:.3f} D, within 0.25 D {100 * g['w025']:.0f}%, "
                  f"{g['secs']:.1f} s per eye", flush=True)
            log.write(f"# greedy {total} {g['mae']:.4f} {g['w025']:.4f} {g['secs']:.2f}\n"); log.flush()
            if g["mae"] < best_err:
                best_err = g["mae"]
                torch.save(net.state_dict(), OUT / "agent.pt")      # keep the best, not the last
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["baselines", "train", "eval"])
    ap.add_argument("--steps", type=int, default=3_000_000)
    args = ap.parse_args()
    env = ClosedLoopRefraction(seed=0)
    print(f"sensor from real photos: blind band {env.band[0]:+.2f} .. {env.band[1]:+.2f} D, "
          f"readings from {env.S_tab[0]:+.1f} to {env.S_tab[-1]:+.1f} D")
    if args.mode == "train":
        train(args.steps)
        return
    res = {"oracle floor (no accommodation)": oracle_floor(env)}
    for name, pol in (("one-shot", OneShot), ("null-seeking loop", NullSeek), ("fog-then-flash (hand-designed)", FogFlash),
                      ("fog-flash twice (stronger hand rule)", FogFlash2)):
        res[name] = run_policy(env, pol)
    if args.mode == "eval":
        net = Net(env.obs_dim, env.n_actions)
        net.load_state_dict(torch.load(OUT / "agent.pt"))
        res["RL agent (PPO)"] = run_policy(env, lambda: Agent(net))
    for k, v in res.items():
        print(f"{k:34s} mean |err| {v['mae']:.2f} D  median {v['median']:.2f}  within 0.25 D {100 * v['w025']:3.0f}%  "
              f"within 0.5 D {100 * v['w05']:3.0f}%  no answer {100 * v['fail']:2.0f}%  time {v['secs']:.1f} s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.mode}.json").write_text(json.dumps(res, indent=2))


def run_policy(env, make, n=N_EVAL, seed=123):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        env.rng = np.random.default_rng(rng.integers(1 << 31))
        obs = env.reset()
        p = make()
        if hasattr(p, "reset"):
            p.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(p(env, obs))
        out.append((info["err"], info["t"]))
    e = np.array(out)
    return dict(mae=float(e[:, 0].mean()), median=float(np.median(e[:, 0])), w025=float((e[:, 0] <= 0.25).mean()),
                w05=float((e[:, 0] <= 0.5).mean()), fail=float((e[:, 0] >= 5).mean()), secs=float(e[:, 1].mean() / 30))


if __name__ == "__main__":
    main()
