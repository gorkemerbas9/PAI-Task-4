"""Experiment harness: python exp.py NAME [key=value ...] -- runs fine_tune with Config overrides,
logs the learning curve, saves the policy and evaluates it with the official evaluator (2000 local episodes)."""
import ast, json, os, sys, time
import numpy as np
import torch
import solution as S
from utils import evaluate, load_policy, load_scenes, save_policy

args = sys.argv[1:]
if "=" in args[0]:  # name missing (old queue.sh): recover it from the log file stdout is redirected to
    name = os.path.basename(os.readlink("/proc/self/fd/1")).rsplit(".", 1)[0]
else:
    name, args = args[0], args[1:]
over = {}
for kv in args:
    k, v = kv.split("=", 1)
    try:
        v = ast.literal_eval(v)
    except Exception:
        pass
    over[k] = v
os.makedirs("runs", exist_ok=True)
curve = []
def log(d):
    curve.append(d)
    print(json.dumps(d), flush=True)
Cfg = type("Cfg", (S.Config,), dict(over, log=staticmethod(log)))
t0 = time.time()
pol = S.fine_tune(load_policy("base_policy.safetensors"), load_scenes("scenes_train.npz"), load_scenes("scenes_local.npz"), Cfg)
train_time = time.time() - t0
save_policy(pol, f"runs/{name}.safetensors")
torch.set_num_threads(over.get("num_threads", S.Config.num_threads))
pol = load_policy(f"runs/{name}.safetensors")  # reload exactly as the checker would
r = evaluate(pol, load_scenes("scenes_local.npz"), n_episodes=2000, seed=0)
res = dict(name=name, overrides=over, final=r, train_wall_min=train_time / 60,
           iterations=curve[-1]["iteration"] if curve else None, env_steps=curve[-1]["env_steps"] if curve else None,
           host=os.uname().nodename, curve=curve)
json.dump(res, open(f"runs/{name}.json", "w"), indent=1)
with open("results.jsonl", "a") as f:
    f.write(json.dumps({k: v for k, v in res.items() if k != "curve"}) + "\n")
print("FINAL", name, json.dumps(r), flush=True)
