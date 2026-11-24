"""Per-scene route value: run many episodes per scene, group by route, compare mean score vs best-route score."""
import sys, numpy as np, torch
from utils import *
torch.set_num_threads(1)
path = sys.argv[1]; S_ = int(sys.argv[2]) if len(sys.argv) > 2 else 200; R = 24
m = load_policy(path); m.eval()
sc = load_scenes("scenes_local.npz")
sub = {k: np.repeat(v[:S_], R, axis=0) for k, v in sc.items()}
n = S_ * R
env = NavEnv(n, sub, seed=1)
# deterministic scene assignment: reset picks random scenes, so override
for i in range(n):
    env.p[i] = sub["start"][i]; env.goal[i] = sub["goal"][i]; env.centers[i] = sub["centers"][i]; env.radii[i] = sub["radii"][i]
    env.radii_true[i] = sub["radii"][i] + OBST_NOISE * np.abs(env.rng.standard_normal(2)); env.v[i] = 0; env.t[i] = 0
obs = env.obs(); gen = torch.Generator().manual_seed(1)
done = np.zeros(n, bool); succ = np.zeros(n, bool); side = np.zeros((n, 2))
start = env.p.copy(); goal = env.goal.copy(); centers = env.centers.copy(); d = goal - start
while not done.all():
    a = sample_actions(m, torch.as_tensor(obs), generator=gen).numpy()
    for h in range(H):
        p0 = env.p.copy()
        obs, _, te, tr, info = env.step(a[:, h], ~done)
        succ |= info["success"]
        for j in range(2):
            c = centers[:, j]
            passed = (((p0 - c) * d).sum(1) < 0) & (((env.p - c) * d).sum(1) >= 0) & ~done
            cr = d[:, 0] * (env.p - c)[:, 1] - d[:, 1] * (env.p - c)[:, 0]
            side[passed, j] = np.sign(cr[passed])
        done |= te | tr
ret = np.where(succ, GAMMA ** (env.t - 1.0), 0).reshape(S_, R)
route = np.where((side[:, 0] > 0) & (side[:, 1] > 0), 0, np.where((side[:, 0] < 0) & (side[:, 1] < 0), 1, 2)).reshape(S_, R)
mean_actual = ret.mean()
best = []; frac_best = []
for s in range(S_):
    vals = [ret[s][route[s] == r].mean() if (route[s] == r).sum() >= 2 else -1 for r in range(3)]
    b = int(np.argmax(vals)); best.append(vals[b]); frac_best.append((route[s] == b).mean())
print(f"{path}: actual {mean_actual:.4f}  oracle-best-route {np.mean(best):.4f}  frac episodes on best route {np.mean(frac_best):.2f}")
