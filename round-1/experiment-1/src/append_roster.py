#!/usr/bin/env python3
"""Append the replacement for the GATED huihui-ai/Qwen3-0.6B-abliterated cell."""
import json
from pathlib import Path
p = Path("results/roster_final.json"); d = json.loads(p.read_text())
new = {"repo_id": "mlabonne/Qwen3-0.6B-abliterated", "family": "qwen3",
       "lineage": "qwen3_06", "condition": "abliterated",
       "substitution": ("huihui-ai/Qwen3-0.6B-abliterated resolves via model_info() but "
                        "its weight download returns 403 GATED; replaced with an "
                        "arch- and param-matched abliteration of the same parent by the "
                        "author of the abliteration technique. Repo resolution alone is "
                        "NOT sufficient to establish availability -- the original cell is "
                        "retained in the coverage table as GATED_REPO.")}
if not any(r["repo_id"] == new["repo_id"] for r in d["roster"]):
    d["roster"].append(new)
    d["n_checkpoints"] = len(d["roster"])
    p.write_text(json.dumps(d, indent=2))
    print("appended", new["repo_id"], "-> roster now", d["n_checkpoints"])
else:
    print("already present")
