"""Solution template of Task 4: fine-tuning a pretrained diffusion policy with DPPO.

Sections for your implementation are marked with TODO. Please preserve the signature of fine_tune,
since the checker calls it. Apart from that, you are free to change anything in this file, and you may
replace DPPO by any other fine-tuning method, as long as fine_tune returns a policy that utils.save_policy
can save (a Denoiser, or a FrozenEarly).
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from utils import (CHUNK_DIM, K, OBS_DIM, ChunkRunner, FrozenEarly, evaluate, featurize, gaussian_logprob, load_policy,
                   load_scenes, save_policy, step_mean, step_std)

# Set to True to save a PDF that compares your policy with the pretrained one (see the task description).
ENABLE_EXTENDED_ANALYSIS = True


class Config:
    # Tuned settings (see experiments/EXPERIMENTS.md for the full log). Compared with the template defaults:
    # - epochs 5 -> 1: the actor update dominates the run time (rollouts are ~10% of it), so one pass over
    #   fresh data per iteration gives ~3x more iterations in the same 30 minutes and keeps every update
    #   close to the policy that collected the data (fewer crashes than 2 or 5 epochs).
    # - min_std 0.1 -> 0.3: the PPO clip bounds the probability ratio, which for a Gaussian denoising step
    #   bounds the mean shift to roughly clip * sigma. With the official noise levels the last steps have
    #   sigma 0.243, 0.144 and 0, so a small lower bound makes the clip, not the LR, the brake. A larger
    #   bound widens the trust region in action space, explores more, and the training noise teaches
    #   safety margins against the hidden obstacle radius. Beyond ~0.4 the mismatch with the official
    #   sampler starts to cost crashes.
    # - actor_lr 3e-5 -> 1e-4.
    budget_minutes = 30.0  # wall-clock training time, without the periodic evaluations
    gamma = 0.99  # discount factor per environment step for training; the score always uses 0.99
    gae_lambda = 0.95  # GAE parameter, per chunk
    n_envs = 64  # parallel environments
    n_chunks = 16  # chunks per environment per iteration
    epochs = 1  # passes over the rollout per iteration
    minibatch = 2048  # (environment step, denoising step) pairs per actor update
    actor_lr = 1e-4
    critic_lr = 1e-3
    critic_warmup = 5  # iterations that only fit the value function
    clip = 0.05  # PPO clipping parameter
    min_std = 0.3  # lower bound on the noise level sigma_k during training
    ft_steps = K  # only the last ft_steps denoising steps enter the PPO loss
    freeze_early = False  # True: the other steps use the frozen pretrained network (utils.FrozenEarly)
    eval_every_minutes = 2.0
    num_threads = 4  # CPU threads for training; using every core is often slower, also inside Docker
    seed = 0
    # ---- extensions (defaults reproduce the template behaviour) ----
    min_std_end = None  # if set, min_std is annealed linearly (in training time) from min_std to min_std_end
    actor_lr_end = None  # if set, the actor learning rate is annealed linearly to this value
    target_kl = None  # if set, stop the actor epochs of an iteration early once the approx. KL exceeds it
    critic_epochs = None  # epochs of the value function per iteration (default: epochs)
    eval_episodes = 200  # episodes of the periodic monitoring evaluation
    log = None  # optional callable receiving a dict per monitoring evaluation (used by the experiment harness)


class ValueFunction(nn.Module):
    """Value function V_phi(o) on the outer level. The denoising steps do not enter it."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, 1))

    def forward(self, obs):
        return self.net(featurize(obs)).squeeze(-1)


def sample_with_chain(model, obs, min_std, generator=None):
    """Runs the reverse chain and records it.

    Returns chain (B, K + 1, CHUNK_DIM), from pure noise at index 0 to the final chunk at index K,
    and logp (B, K), where logp[:, j] is the log-probability of the transition from chain[:, j]
    to chain[:, j + 1].

    Careful: the task description numbers the denoising steps k = K, ..., 1, whereas the functions
    in utils.py take k = K - 1, ..., 0. Transition j of the chain is k = K - 1 - j in utils.py.
    """
    B = obs.shape[0]
    x = torch.randn(B, CHUNK_DIM, generator=generator)
    chain, logps = [x], []
    for j in range(K):
        k = torch.full((B,), K - 1 - j, dtype=torch.long)
        mean = step_mean(model, x, k, obs)
        std = step_std(k, min_std)
        x = mean + std[:, None] * torch.randn(B, CHUNK_DIM, generator=generator)
        chain.append(x)
        logps.append(gaussian_logprob(x, mean, std))
    return torch.stack(chain, 1), torch.stack(logps, 1)


def compute_gae(rewards, values, next_values, discounts, terminated, dones, lam):
    """Generalized advantage estimation over chunks. All arrays are numpy arrays of shape (T, N).

    next_values holds V(next_obs) before any reset. discounts holds the discount of what follows
    each chunk. A chunk's episode can end by reaching the goal or crashing (terminated) or by the
    time limit (dones, which also covers terminated).

    Returns advantages and regression targets for the value function, both of shape (T, N).
    """
    T = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1])
    for t in reversed(range(T)):
        # Bootstrap from V(next_obs) unless the episode terminated (goal or crash): truncation bootstraps.
        delta = rewards[t] + discounts[t] * (1.0 - terminated[t]) * next_values[t] - values[t]
        # Pass on the following advantage only if the next chunk belongs to the same episode.
        last = delta + discounts[t] * lam * (1.0 - dones[t]) * last
        adv[t] = last
    return adv, adv + values


def ppo_loss(model, obs, x_in, x_out, j, old_logp, adv, clip, min_std):
    """Clipped PPO objective for a minibatch of (environment step, denoising step) pairs.

    x_in, x_out (B, CHUNK_DIM) are the chunks before and after chain transition j (B,), old_logp (B,)
    their log-probabilities during the rollout and adv (B,) the advantage of the environment step.
    Returns the loss, to be minimized, and the probability ratio (B,).
    """
    k = K - 1 - j
    mean = step_mean(model, x_in, k, obs)
    std = step_std(k, min_std)
    logp = gaussian_logprob(x_out, mean, std)
    ratio = torch.exp(logp - old_logp)
    loss = -torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean()
    return loss, ratio


def fine_tune(model, scenes, local_scenes, cfg=Config):
    """Fine-tunes model on scenes and returns it. local_scenes are for monitoring only."""
    torch.set_num_threads(cfg.num_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.freeze_early and cfg.ft_steps < K:
        model = FrozenEarly(copy.deepcopy(model), model, cfg.ft_steps)
    runner = ChunkRunner(cfg.n_envs, scenes, cfg.gamma, seed=cfg.seed + 1)
    critic = ValueFunction()
    opt_actor = torch.optim.Adam(model.parameters(), lr=cfg.actor_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    gen = torch.Generator().manual_seed(cfg.seed + 2)
    chain_steps = np.arange(K - cfg.ft_steps, K)

    t_start = time.time()
    t_eval = 0.0
    last_eval = time.time()
    it = 0
    def progress():
        return min(1.0, (time.time() - t_start - t_eval) / (cfg.budget_minutes * 60))

    while time.time() - t_start - t_eval < cfg.budget_minutes * 60:
        it += 1
        frac = progress()
        min_std = cfg.min_std if cfg.min_std_end is None else cfg.min_std + frac * (cfg.min_std_end - cfg.min_std)
        if cfg.actor_lr_end is not None:
            for g in opt_actor.param_groups:
                g["lr"] = cfg.actor_lr + frac * (cfg.actor_lr_end - cfg.actor_lr)
        # ---------------- rollout ----------------
        T, N = cfg.n_chunks, cfg.n_envs
        obs_b = np.zeros((T, N, OBS_DIM), np.float32)
        nobs_b = np.zeros_like(obs_b)
        chain_b = torch.zeros(T, N, K + 1, CHUNK_DIM)
        logp_b = torch.zeros(T, N, K)
        rew, disc, term, done = (np.zeros((T, N)) for _ in range(4))
        model.eval()
        with torch.no_grad():
            for t in range(T):
                chain, logp = sample_with_chain(model, torch.as_tensor(runner.obs), min_std, gen)
                actions = chain[:, -1].clamp(-1.0, 1.0).reshape(N, -1, 2)
                tr = runner.step(actions.numpy())
                obs_b[t], nobs_b[t] = tr["obs"], tr["next_obs"]
                chain_b[t], logp_b[t] = chain, logp
                rew[t], disc[t], term[t], done[t] = tr["reward"], tr["discount"], tr["terminated"], tr["done"]
            v = critic(torch.as_tensor(obs_b.reshape(T * N, -1))).numpy().reshape(T, N)
            nv = critic(torch.as_tensor(nobs_b.reshape(T * N, -1))).numpy().reshape(T, N)
        adv, ret = compute_gae(rew, v, nv, disc, term, done, cfg.gae_lambda)
        model.train()

        B = T * N
        obs_t = torch.as_tensor(obs_b.reshape(B, -1))
        chain_t = chain_b.reshape(B, K + 1, CHUNK_DIM)
        logp_t = logp_b.reshape(B, K)
        adv_t = torch.as_tensor(adv.reshape(B), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret.reshape(B), dtype=torch.float32)

        # ---------------- value function ----------------
        for _ in range(cfg.critic_epochs or cfg.epochs):
            perm = torch.randperm(B)
            for s in range(0, B, 512):
                idx = perm[s:s + 512]
                loss_c = ((critic(obs_t[idx]) - ret_t[idx]) ** 2).mean()
                opt_critic.zero_grad()
                loss_c.backward()
                opt_critic.step()

        # ---------------- actor ----------------
        if it > cfg.critic_warmup:
            # All pairs of environment step and fine-tuned denoising step.
            pairs_b = np.repeat(np.arange(B), len(chain_steps))
            pairs_j = np.tile(chain_steps, B)
            stop = False
            for _ in range(cfg.epochs):
                if stop:
                    break
                perm = np.random.permutation(len(pairs_b))
                for s in range(0, len(perm), cfg.minibatch):
                    sel = perm[s:s + cfg.minibatch]
                    b = torch.as_tensor(pairs_b[sel])
                    j = torch.as_tensor(pairs_j[sel])
                    loss, ratio = ppo_loss(model, obs_t[b], chain_t[b, j], chain_t[b, j + 1], j, logp_t[b, j],
                                           adv_t[b], cfg.clip, min_std)
                    opt_actor.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_actor.step()
                    if cfg.target_kl is not None:
                        with torch.no_grad():  # KL estimator of the course script (Remark 12.12)
                            approx_kl = (ratio - 1 - torch.log(ratio)).mean().item()
                        if approx_kl > cfg.target_kl:
                            stop = True
                            break

        # ---------------- monitoring ----------------
        if time.time() - last_eval > cfg.eval_every_minutes * 60:
            t0 = time.time()
            stats = runner.pop_stats() or {}
            r = evaluate(model, local_scenes, n_episodes=cfg.eval_episodes, seed=999)
            model.train()
            t_eval += time.time() - t0
            last_eval = time.time()
            if cfg.log is not None:
                cfg.log({"minutes": (time.time() - t_start - t_eval) / 60, "iteration": it,
                         "env_steps": runner.total_steps, "train_score": stats.get("score", float("nan")),
                         "train_success": stats.get("success", float("nan")), "eval_score": r["score"],
                         "eval_success": r["success_rate"], "eval_steps": r["mean_steps_success"],
                         "min_std": min_std})
            print(f"{(time.time() - t_start - t_eval) / 60:5.1f} min  iteration {it}  env steps {runner.total_steps}  "
                  f"training score {stats.get('score', float('nan')):6.3f}  evaluation score {r['score']:6.3f}  "
                  f"success {r['success_rate']:.3f}", flush=True)
    return model


# This block lets you run your training outside of Docker. The checker does not use it.
if __name__ == "__main__":
    policy = fine_tune(load_policy("base_policy.safetensors"), load_scenes("scenes_train.npz"),
                       load_scenes("scenes_local.npz"))
    save_policy(policy, "finetuned_policy.safetensors")
    print(evaluate(policy, load_scenes("scenes_local.npz")))
