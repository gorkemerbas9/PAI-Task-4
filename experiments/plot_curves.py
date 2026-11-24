"""Plots monitoring-score curves (200-episode evals on scenes_local) of selected runs."""
import json, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]; names = sys.argv[2:]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in names:
    r = json.load(open(f"runs/{n}.json")); c = r["curve"]
    m = [0] + [d["minutes"] for d in c]
    ax[0].plot(m, [0.4185] + [d["eval_score"] for d in c], marker=".", label=f"{n} ({r['final']['score']:.3f})")
    ax[1].plot(m, [0.984] + [d["eval_success"] for d in c], marker=".")
ax[0].axhline(0.6, color="k", ls="--", lw=1, label="baseline 0.600")
ax[0].set(xlabel="training minutes", ylabel="monitor score (200 ep)", title="score (legend: final 2000-ep local score)")
ax[1].set(xlabel="training minutes", ylabel="success rate (200 ep)", title="success rate")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out, dpi=130)
