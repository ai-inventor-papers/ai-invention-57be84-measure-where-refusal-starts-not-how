#!/usr/bin/env python3
"""Freeze the final roster. FAMILY = weight lineage for the leave-one-family-out
split; LINEAGE = the exact base/instruct/abliterated triple (same d_model), used
only to look up the ORACLE direction. Instruct comes FIRST within each lineage so
its direction is on disk before its siblings are scored."""
import json, sys
from pathlib import Path
from core import jdump, WORKSPACE

R = [
  # (repo_id, family, lineage, condition, note)
  ("Qwen/Qwen3-1.7B", "qwen3", "qwen3_17", "instruct", None),
  ("Qwen/Qwen3-1.7B-Base", "qwen3", "qwen3_17", "base", None),
  ("huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2", "qwen3", "qwen3_17", "abliterated", None),
  ("Qwen/Qwen3-0.6B", "qwen3", "qwen3_06", "instruct", None),
  ("Qwen/Qwen3-0.6B-Base", "qwen3", "qwen3_06", "base", None),
  ("huihui-ai/Qwen3-0.6B-abliterated", "qwen3", "qwen3_06", "abliterated", None),
  ("meta-llama/Llama-3.2-1B-Instruct", "llama32", "llama32_1b", "instruct", None),
  ("meta-llama/Llama-3.2-1B", "llama32", "llama32_1b", "base", None),
  ("carsenk/llama3.2_1b_2025_uncensored_v2", "llama32", "llama32_1b", "abliterated",
   "huihui-ai/Llama-3.2-1B-Instruct-abliterated is 404; substituted an arch- and "
   "param-matched community uncensored fine-tune of the same parent"),
  ("google/gemma-3-1b-it", "gemma3", "gemma3_1b", "instruct", None),
  ("google/gemma-3-1b-pt", "gemma3", "gemma3_1b", "base", None),
  ("DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated", "gemma3", "gemma3_1b",
   "abliterated", "huihui-ai/gemma-3-1b-it-abliterated is 404; substituted a "
   "heretic-method abliteration of the same parent"),
  ("Qwen/Qwen2.5-1.5B-Instruct", "qwen25", "qwen25_15", "instruct", None),
  ("Qwen/Qwen2.5-1.5B", "qwen25", "qwen25_15", "base", None),
  ("Goekdeniz-Guelmez/Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v3", "qwen25",
   "qwen25_15", "abliterated", "huihui-ai variant is 404; substituted Josiefied v3"),
  ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2", "smollm2_17", "instruct", None),
  ("HuggingFaceTB/SmolLM2-1.7B", "smollm2", "smollm2_17", "base", None),
  ("ops-malware/smollm2-1.7b-abliterated", "smollm2", "smollm2_17", "abliterated",
   "huihui-ai variant is 404; substituted the only arch/param-matched abliteration found"),
  # ---- spares, reached only if time allows ----
  ("meta-llama/Llama-3.2-3B-Instruct", "llama32", "llama32_3b", "instruct", None),
  ("huihui-ai/Llama-3.2-3B-Instruct-abliterated", "llama32", "llama32_3b", "abliterated", None),
  ("tiiuae/Falcon3-1B-Instruct", "falcon3", "falcon3_1b", "instruct", None),
  ("tiiuae/Falcon3-1B-Base", "falcon3", "falcon3_1b", "base", None),
]
roster = [{"repo_id": a, "family": b, "lineage": c, "condition": d, "substitution": e}
          for a, b, c, d, e in R]
missing = [{"family": "falcon3", "lineage": "falcon3_1b", "condition": "abliterated",
            "reason": "CONDITION_UNAVAILABLE",
            "detail": "no arch- and param-matched abliterated/uncensored fine-tune of "
                      "Falcon3-1B-Instruct exists on the hub; the family therefore "
                      "enters with two conditions and is excluded from the "
                      "within-condition abliterated analysis"}]
jdump({"roster": roster, "missing_cells": missing,
       "n_checkpoints": len(roster),
       "n_families": len(set(r["family"] for r in roster)),
       "families": sorted(set(r["family"] for r in roster))},
      Path("results/roster_final.json"))
print(json.dumps({"n": len(roster), "families": sorted(set(r['family'] for r in roster))}, indent=2))
