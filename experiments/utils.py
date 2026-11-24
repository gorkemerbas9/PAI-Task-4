"""Environment, diffusion policy and official sampler of Task 4.

Everything in this file is fixed. Do not change it: changes have no effect on your score,
because the checker evaluates your weights with its own private copy of all of this.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Constants
###############################################################################
ARENA = 10.0  # the arena is [0, ARENA]^2
DT = 0.1  # time step
ACTION_NOISE = 0.05  # standard deviation of the actuator noise
GOAL_RADIUS = 0.5  # an episode succeeds when the robot enters this disc around the goal
V_MAX = 3.0  # speed limit
OBST_NOISE = 0.15  # true radius = observed radius + OBST_NOISE * |N(0, 1)|, drawn per episode and obstacle
T_MAX = 150  # episode cap: truncation during training, failure during evaluation
N_OBSTACLES = 2
GAMMA = 0.99  # discount factor of the score

H = 4  # actions per chunk; the whole chunk is executed before the policy is queried again
K = 10  # denoising steps
ACT_DIM = 2
CHUNK_DIM = H * ACT_DIM
OBS_DIM = 6 + 3 * N_OBSTACLES  # position (2), velocity (2), goal (2), per obstacle: center (2), radius (1)
WIDTH = 128  # width of the denoiser


###############################################################################
# Environment
###############################################################################
def clip_action(u):
    """Projects accelerations (..., 2) onto the unit disk."""
    u = np.asarray(u, dtype=np.float64)
    nrm = np.linalg.norm(u, axis=-1, keepdims=True)
    return u * np.minimum(1.0, 1.0 / np.maximum(nrm, 1e-12))


def _segment_circle_entry(p0, p1, c, r):
    """First t in [0, 1] at which segment p0 -> p1 enters circle (c, r), inf if it does not."""
    d = p1 - p0
    f = p0 - c
    a = np.maximum((d * d).sum(-1), 1e-12)
    b = 2 * (f * d).sum(-1)
    cc = (f * f).sum(-1) - r ** 2
    disc = b * b - 4 * a * cc
    inside = cc <= 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = (-b - sq) / (2 * a)
    hit = (disc >= 0) & (t >= 0) & (t <= 1)
    out = np.where(hit, t, np.inf)
    return np.where(inside, 0.0, out)


def _segment_wall_exit(p0, p1):
    """First t in [0, 1] at which segment p0 -> p1 leaves the arena, inf if it does not."""
    t = np.full(p0.shape[0], np.inf)
    for dim in range(2):
        lo = p1[:, dim] < 0
        hi = p1[:, dim] > ARENA
        denom = p0[:, dim] - p1[:, dim]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lo = np.where(lo, p0[:, dim] / np.where(lo, denom, 1.0), np.inf)
            t_hi = np.where(hi, (ARENA - p0[:, dim]) / np.where(hi, -denom, 1.0), np.inf)
        t = np.minimum(t, np.minimum(t_lo, t_hi))
    return t


def load_scenes(path):
    """Loads a scene file: arrays start (S, 2), goal (S, 2), centers (S, 2, 2) and radii (S, 2)."""
    with np.load(path) as d:
        return {k: d[k] for k in ["start", "goal", "centers", "radii"]}


class NavEnv:
    """Vectorized 2D point-mass navigation.

    State: position p and velocity v. Action: acceleration u with |u| <= 1 (longer actions are
    rescaled). An episode succeeds when the robot enters the goal disc, at any speed, and fails
    when it hits an obstacle or leaves the arena. Obstacles can be larger than observed: the true
    radius is hidden from the policy. After T_MAX steps an episode is truncated.

    The reward is +1 when the goal is reached and 0 otherwise. With a discount factor below 1,
    the return therefore prefers reaching the goal sooner, and it is 0 after a failure.
    Every reset draws a scene uniformly at random from ``scenes`` (see load_scenes).
    """

    def __init__(self, n_envs, scenes, seed=0):
        self.n = n_envs
        self.scenes = scenes
        self.n_scenes = len(scenes["start"])
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((n_envs, 2))
        self.v = np.zeros((n_envs, 2))
        self.goal = np.zeros((n_envs, 2))
        self.centers = np.zeros((n_envs, N_OBSTACLES, 2))
        self.radii = np.zeros((n_envs, N_OBSTACLES))
        self.radii_true = np.zeros((n_envs, N_OBSTACLES))
        self.t = np.zeros(n_envs, dtype=np.int64)

    def reset(self, mask=None):
        """Resets all environments, or those in ``mask``, to new random scenes."""
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        for i in idx:
            s = self.rng.integers(self.n_scenes)
            r = self.scenes["radii"][s]
            self.p[i] = self.scenes["start"][s]
            self.goal[i] = self.scenes["goal"][s]
            self.centers[i] = self.scenes["centers"][s]
            self.radii[i] = r
            self.radii_true[i] = r + OBST_NOISE * np.abs(self.rng.standard_normal(len(r)))
            self.v[i] = 0.0
            self.t[i] = 0
        return self.obs()

    def obs(self):
        obst = np.concatenate([self.centers, self.radii[..., None]], -1).reshape(self.n, -1)
        return np.concatenate([self.p, self.v, self.goal, obst], -1).astype(np.float32)

    def step(self, u, active=None):
        """One step for all environments; inactive ones stay as they are.

        Returns obs, reward, terminated, truncated, info{success, failure}.
        """
        if active is None:
            active = np.ones(self.n, dtype=bool)
        u = clip_action(u)
        noise = ACTION_NOISE * self.rng.standard_normal((self.n, 2))
        u_eff = clip_action(u + noise)
        v1 = self.v + DT * u_eff
        speed = np.linalg.norm(v1, axis=1, keepdims=True)
        v1 = v1 * np.minimum(1.0, V_MAX / np.maximum(speed, 1e-12))
        p1 = self.p + DT * v1

        t_goal = _segment_circle_entry(self.p, p1, self.goal, GOAL_RADIUS)
        t_obs = np.full(self.n, np.inf)
        for j in range(N_OBSTACLES):
            t_obs = np.minimum(t_obs, _segment_circle_entry(self.p, p1, self.centers[:, j], self.radii_true[:, j]))
        t_wall = _segment_wall_exit(self.p, p1)
        t_fail = np.minimum(t_obs, t_wall)
        success = active & np.isfinite(t_goal) & (t_goal <= t_fail)
        failure = active & np.isfinite(t_fail) & ~success

        self.p = np.where(active[:, None], p1, self.p)
        self.v = np.where(active[:, None], v1, self.v)
        self.t = self.t + active
        terminated = success | failure
        truncated = active & ~terminated & (self.t >= T_MAX)
        reward = success.astype(np.float64) * active
        return self.obs(), reward, terminated, truncated, {"success": success, "failure": failure}

    def step_chunk(self, actions, active=None):
        """Executes action chunks (n, H, 2), stopping each environment when it is done.

        Returns obs, rewards (n, H) (zero after done), terminated, truncated,
        n_exec (executed steps per environment), success, failure.
        """
        if active is None:
            active = np.ones(self.n, dtype=bool)
        actions = np.asarray(actions)
        alive = active.copy()
        rewards = np.zeros((self.n, actions.shape[1]))
        term = np.zeros(self.n, dtype=bool)
        trunc = np.zeros(self.n, dtype=bool)
        succ = np.zeros(self.n, dtype=bool)
        fail = np.zeros(self.n, dtype=bool)
        n_exec = np.zeros(self.n, dtype=np.int64)
        for h in range(actions.shape[1]):
            if not alive.any():
                break
            _, r, te, tr, info = self.step(actions[:, h], alive)
            rewards[:, h] = r
            n_exec += alive
            term |= te
            trunc |= tr
            succ |= info["success"]
            fail |= info["failure"]
            alive &= ~(te | tr)
        return self.obs(), rewards, term, trunc, n_exec, succ, fail


class ChunkRunner:
    """Keeps n_envs environments running, with automatic resets. One call to step executes one
    chunk in every environment, so one step of this class is one step t of the outer decision process."""

    def __init__(self, n_envs, scenes, gamma, seed=0):
        self.env = NavEnv(n_envs, scenes, seed=seed)
        self.obs = self.env.reset()
        self.gamma = gamma
        self.ep_len = np.zeros(n_envs, dtype=np.int64)
        self.finished = []  # (score with the official GAMMA, success) of finished episodes
        self.total_steps = 0

    def step(self, actions):
        """Executes chunks (n_envs, H, 2). Returns a dict of arrays (n_envs, ...):

        obs           observation in which the chunk was chosen
        next_obs      observation after the chunk, before any reset
        reward        discounted sum of the rewards during the chunk
        discount      gamma ** (number of steps executed), the discount of what follows the chunk
        terminated    the episode ended in success or failure: do not bootstrap from next_obs
        done          the episode ended for any reason, including truncation: the next chunk
                      belongs to a new episode
        """
        obs0 = self.obs
        nobs, rew, term, trunc, n_exec, succ, _ = self.env.step_chunk(np.asarray(actions))
        self.total_steps += int(n_exec.sum())
        self.ep_len += n_exec
        done = term | trunc
        out = {
            "obs": obs0,
            "next_obs": nobs.copy(),
            "reward": (rew * self.gamma ** np.arange(rew.shape[1])[None]).sum(1),
            "discount": self.gamma ** n_exec,
            "terminated": term,
            "done": done,
        }
        for i in np.flatnonzero(done):
            self.finished.append((GAMMA ** (self.ep_len[i] - 1) if succ[i] else 0.0, bool(succ[i])))
        self.ep_len[done] = 0
        if done.any():
            nobs = self.env.reset(done)
        self.obs = nobs
        return out

    def pop_stats(self):
        """Mean score and success rate of the episodes finished since the last call, or None."""
        if not self.finished:
            return None
        s = np.array(self.finished, dtype=np.float64)
        self.finished = []
        return {"score": float(s[:, 0].mean()), "success": float(s[:, 1].mean()), "episodes": len(s)}


###############################################################################
# Diffusion policy
###############################################################################
def featurize(obs):
    """Normalized features of observations (B, OBS_DIM), with positions relative to the robot."""
    p, v, g = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
    ob = obs[:, 6:].reshape(-1, N_OBSTACLES, 3)
    rel = (ob[..., :2] - p[:, None]) / 5.0
    return torch.cat([(p - 5.0) / 5.0, v / 3.0, (g - p) / 5.0, rel.flatten(1), ob[..., 2] / 2.0], -1)


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc2(F.mish(self.fc1(self.norm(x))))


def timestep_embedding(k, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=k.device) / half)
    args = k.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class Denoiser(nn.Module):
    """Noise-prediction network eps_theta(a^k, k, o) for action chunks. The architecture is fixed."""

    def __init__(self, width=WIDTH, n_blocks=3, temb_dim=32):
        super().__init__()
        self.temb_dim = temb_dim
        self.inp = nn.Linear(OBS_DIM + CHUNK_DIM + temb_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, CHUNK_DIM)

    def forward(self, x, k, obs):
        h = torch.cat([featurize(obs), x, timestep_embedding(k, self.temb_dim)], -1)
        h = self.inp(h)
        for b in self.blocks:
            h = b(h)
        return self.out(F.mish(self.out_norm(h)))


def _cosine_schedule(n_steps, s=0.008):
    x = np.linspace(0, n_steps, n_steps + 1)
    ac = np.cos(((x / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    return np.clip(1 - ac[1:] / ac[:-1], 0, 0.999)


class Schedule:
    """DDPM quantities indexed by k = 0 (least noisy) ... K - 1 (most noisy)."""

    def __init__(self, n_steps=K):
        betas = _cosine_schedule(n_steps)
        alphas = 1.0 - betas
        ab = np.cumprod(alphas)
        ab_prev = np.concatenate([[1.0], ab[:-1]])
        t = lambda a: torch.tensor(a, dtype=torch.float32)
        self.betas = t(betas)
        self.alphas_cumprod = t(ab)
        self.sqrt_ab = t(np.sqrt(ab))
        self.sqrt_1mab = t(np.sqrt(1 - ab))
        self.coef_x0 = t(betas * np.sqrt(ab_prev) / (1 - ab))
        self.coef_xk = t((1 - ab_prev) * np.sqrt(alphas) / (1 - ab))
        self.post_std = t(np.sqrt(betas * (1 - ab_prev) / (1 - ab)))  # post_std[0] == 0


SCHEDULE = Schedule(K)


def diffusion_loss(model, obs, chunks):
    """Standard noise-prediction loss on clean chunks (B, CHUNK_DIM)."""
    B = obs.shape[0]
    k = torch.randint(0, K, (B,), device=obs.device)
    eps = torch.randn_like(chunks)
    s = SCHEDULE
    xk = s.sqrt_ab.to(obs.device)[k, None] * chunks + s.sqrt_1mab.to(obs.device)[k, None] * eps
    return ((model(xk, k, obs) - eps) ** 2).mean()


def step_mean(model, x, k, obs):
    """Mean of denoising step k, the Gaussian that maps the chunk x before the step to the chunk after it.

    x has shape (B, CHUNK_DIM) and k (B,) holds integers from K - 1 (first, most noisy step) to 0 (last step).
    """
    s = SCHEDULE
    dev = x.device
    eps = model(x, k, obs)
    x0 = (x - s.sqrt_1mab.to(dev)[k, None] * eps) / s.sqrt_ab.to(dev)[k, None]
    x0 = x0.clamp(-1.0, 1.0)
    return s.coef_x0.to(dev)[k, None] * x0 + s.coef_xk.to(dev)[k, None] * x


def step_std(k, min_std=0.0):
    """Standard deviation of denoising step k (B,), bounded below by min_std."""
    return SCHEDULE.post_std.to(k.device)[k].clamp_min(min_std)


def gaussian_logprob(x, mean, std):
    """Log-density of x (B, CHUNK_DIM) under N(mean, std^2 I), with std (B,)."""
    return (-0.5 * ((x - mean) / std[:, None]) ** 2 - torch.log(std[:, None])
            - 0.5 * math.log(2 * math.pi)).sum(-1)


@torch.no_grad()
def sample_actions(model, obs, generator=None):
    """Official sampler, used for evaluation: the official noise levels and all K denoising steps.

    Returns chunks (B, H, 2) in [-1, 1].
    """
    B = obs.shape[0]
    dev = obs.device
    x = torch.randn(B, CHUNK_DIM, generator=generator).to(dev)
    for k in reversed(range(K)):
        kk = torch.full((B,), k, dtype=torch.long, device=dev)
        mean = step_mean(model, x, kk, obs)
        x = mean + step_std(kk)[:, None] * torch.randn(B, CHUNK_DIM, generator=generator).to(dev)
    return x.clamp(-1.0, 1.0).reshape(B, H, ACT_DIM)


###############################################################################
# Evaluation
###############################################################################
@torch.no_grad()
def evaluate(model, scenes, n_episodes=2000, seed=0, record=False):
    """Official evaluation with the official sampler: one episode per environment.

    The score of an episode is its discounted return: GAMMA ** (steps to goal - 1) on success and 0 otherwise.
    With record=True the result also holds the positions (T + 1, n_episodes, 2) under "positions",
    the scenes under "scene" and the steps per episode under "steps".
    """
    model.eval()
    env = NavEnv(n_episodes, scenes, seed=seed)
    obs = env.reset()
    scene = {k: getattr(env, k).copy() for k in ["goal", "centers", "radii"]}
    gen = torch.Generator().manual_seed(seed)
    done = np.zeros(n_episodes, dtype=bool)
    success = np.zeros(n_episodes, dtype=bool)
    positions = [env.p.copy()]
    while not done.all():
        a = sample_actions(model, torch.as_tensor(obs), generator=gen).numpy()
        if record:
            for h in range(H):
                obs, _, te, tr, info = env.step(a[:, h], ~done)
                positions.append(env.p.copy())
                success |= info["success"]
                done |= te | tr
        else:
            obs, _, te, tr, _, su, _ = env.step_chunk(a, ~done)
            success |= su
            done |= te | tr
    steps = env.t.copy()
    returns = np.where(success, GAMMA ** (steps - 1.0), 0.0)
    res = {
        "score": float(returns.mean()),
        "success_rate": float(success.mean()),
        "mean_steps_success": float(steps[success].mean()) if success.any() else float("nan"),
    }
    if record:
        res.update(positions=np.stack(positions), scene=scene, steps=steps, success=success)
    return res


###############################################################################
# Weights
###############################################################################
class FrozenEarly(nn.Module):
    """A fine-tuned denoiser for the last ft_steps denoising steps and the frozen pretrained one for the earlier steps.

    The denoising steps k = K - 1, ..., ft_steps use base, the steps k = ft_steps - 1, ..., 0 use ft. Only ft has
    trainable parameters: parameters() and state_dict() are those of ft. save_policy stores ft_steps with the
    weights, and the checker then uses the pretrained network for the early steps in the same way.
    """

    def __init__(self, base, ft, ft_steps):
        super().__init__()
        self.ft = ft
        self.__dict__["base"] = base.eval().requires_grad_(False)  # not a submodule: not trained, not saved
        self.ft_steps = int(ft_steps)

    def forward(self, x, k, obs):
        out = self.ft(x, k, obs)
        early = k >= self.ft_steps
        if early.any():
            with torch.no_grad():
                frozen = self.base(x[early], k[early], obs[early])
            out = out.clone()
            out[early] = frozen
        return out


def save_policy(model, path):
    """Saves a Denoiser, or the fine-tuned part of a FrozenEarly together with its ft_steps."""
    from safetensors.torch import save_file
    metadata = None
    if isinstance(model, FrozenEarly):
        metadata = {"ft_steps": str(model.ft_steps)}
        model = model.ft
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path, metadata=metadata)


def load_policy(path, base_path="base_policy.safetensors"):
    """Loads a Denoiser, or a FrozenEarly if the file stores ft_steps (the early steps then come from base_path)."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    model = Denoiser()
    model.load_state_dict(load_file(path))
    with safe_open(path, "pt") as f:
        metadata = f.metadata() or {}
    if "ft_steps" in metadata:
        return FrozenEarly(load_policy(base_path), model, int(metadata["ft_steps"]))
    return model
