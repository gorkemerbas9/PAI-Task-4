"""Prints a markdown table of all finished runs (results.jsonl), with the local-score curve."""
import json, sys
rows = [json.loads(l) for l in open("results.jsonl")]
filt = sys.argv[1] if len(sys.argv) > 1 else ""
print("| run | overrides | iters | env steps (M) | local score (2000 ep) | success | steps to goal |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    if filt not in r["name"]:
        continue
    o = {k: v for k, v in r["overrides"].items() if k not in ("eval_every_minutes", "num_threads")}
    f = r["final"]
    print(f"| {r['name']} | {' '.join(f'{k}={v}' for k, v in o.items())} | {r['iterations']} | "
          f"{(r['env_steps'] or 0) / 1e6:.1f} | **{f['score']:.4f}** | {f['success_rate']:.3f} | {f['mean_steps_success']:.1f} |")
