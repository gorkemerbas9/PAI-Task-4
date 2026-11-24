"""Failure / route analysis of a policy on scenes_local (official sampler)."""
import sys, numpy as np, torch
from utils import *
torch.set_num_threads(1)
def analyze(path, n=2000, seed=0):
    m = load_policy(path); m.eval()
    sc = load_scenes("scenes_local.npz")
    env = NavEnv(n, sc, seed=seed); obs = env.reset()
    gen = torch.Generator().manual_seed(seed)
    done = np.zeros(n, bool); succ = np.zeros(n, bool); kind = np.array([""] * n, dtype=object)
    start = env.p.copy(); goal = env.goal.copy(); centers = env.centers.copy()
    side = np.zeros((n, 2))  # signed side of each obstacle the robot passed on
    maxspeed = np.zeros(n)
    while not done.all():
        a = sample_actions(m, torch.as_tensor(obs), generator=gen).numpy()
        for h in range(H):
            p0 = env.p.copy()
            obs, _, te, tr, info = env.step(a[:, h], ~done)
            # crash type
            newfail = info["failure"] & ~done
            wall = (env.p < 0).any(1) | (env.p > ARENA).any(1)
            kind[newfail & wall] = "wall"; kind[newfail & ~wall] = "obstacle"
            kind[tr & ~done] = "timeout"
            succ |= info["success"]
            # route: sign of cross product (goal-start) x (p-center) when passing the obstacle's projection
            d = goal - start
            for j in range(2):
                c = centers[:, j]
                s0 = ((p0 - c) * d).sum(1); s1 = ((env.p - c) * d).sum(1)
                passed = (s0 < 0) & (s1 >= 0) & ~done
                cr = d[:, 0] * (env.p - c)[:, 1] - d[:, 1] * (env.p - c)[:, 0]
                side[passed, j] = np.sign(cr[passed])
            maxspeed = np.maximum(maxspeed, np.linalg.norm(env.v, axis=1) * ~done)
            done |= te | tr
    steps = env.t
    score = np.where(succ, GAMMA ** (steps - 1.0), 0).mean()
    route = np.where((side[:, 0] > 0) & (side[:, 1] > 0), "left", np.where((side[:, 0] < 0) & (side[:, 1] < 0), "right", "gap"))
    print(f"{path}: score {score:.4f} success {succ.mean():.3f} steps {steps[succ].mean():.1f}  "
          f"fail: obstacle {np.mean(kind=='obstacle'):.3f} wall {np.mean(kind=='wall'):.3f} timeout {np.mean(kind=='timeout'):.3f}  "
          f"routes L/R/gap {np.mean(route=='left'):.2f}/{np.mean(route=='right'):.2f}/{np.mean(route=='gap'):.2f}  "
          f"max speed {maxspeed.mean():.2f}")
    # score if no failures (time cost only) and if success had zero cost
    print(f"   score lost to failures ~ {(1-succ.mean())*np.mean(GAMMA**(steps[succ]-1.0)):.4f}")
for p in sys.argv[1:]:
    analyze(p)
