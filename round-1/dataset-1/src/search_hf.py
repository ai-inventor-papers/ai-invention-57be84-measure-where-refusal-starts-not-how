#!/usr/bin/env python3
"""Run many HuggingFace dataset searches concurrently and dump consolidated results."""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi

QUERIES = [
 "harmful prompts","jailbreak","refusal","safety benchmark","red teaming",
 "llm safety evaluation","toxicity prompts","over-refusal","adversarial prompts","alignment",
 "harmful behaviors","advbench","strongreject","xstest","or-bench",
 "harmbench","harm severity","harmlevelbench","instruction following safety","do not answer",
 "malicious instructions","prompt injection","content moderation","unsafe questions","beavertails",
 "safe rlhf","hh-rlhf","toxicchat","real toxicity prompts","jailbreakbench",
 "forbidden questions","exaggerated safety","pseudo-harmful","false refusal","wildguard",
 "wildjailbreak","aegis safety","salad-bench","sorry-bench","cyberseceval",
 "dangerous capability","bias and safety","unanswerable questions","abstention","hallucination refusal",
 "paraphrase harmful","multi-turn jailbreak","persuasion attack","safety alignment tuning","llm guardrail",
]

def run(q: str) -> dict:
    api = HfApi()
    out = []
    for attempt in range(3):
        try:
            for d in api.list_datasets(search=q, limit=15):
                out.append({"id": d.id, "downloads": getattr(d, "downloads", None),
                            "likes": getattr(d, "likes", None), "tags": (d.tags or [])[:12]})
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                return {"query": q, "error": str(e), "results": []}
    return {"query": q, "results": out}

def main() -> None:
    with ThreadPoolExecutor(max_workers=10) as ex:
        res = list(ex.map(run, QUERIES))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    seen = {}
    for r in res:
        for d in r["results"]:
            seen.setdefault(d["id"], d)
    print(f"queries={len(res)} unique_datasets={len(seen)}")
    for d in sorted(seen.values(), key=lambda x: -(x["downloads"] or 0))[:120]:
        print(f'{d["downloads"]:>9} {d["likes"]:>5}  {d["id"]}')

main()
