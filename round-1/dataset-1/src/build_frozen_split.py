#!/usr/bin/env python3
"""Draw the held-out family split ONCE, from a single seeded shuffle, and never re-draw it.

Three families go to HELD_OUT and are not loaded at all in iteration 1; the surviving candidate
from the screen is confirmed on them in iteration 2.  The draw is stratified only by the
constraint that the Qwen3 triad -- the feasibility-gate and mechanism-figure family -- must be
in SCREEN, because the gate runs on it before anything else.

If `frozen_split.json` already exists this script REFUSES to overwrite it; that refusal is the
mechanism that makes the split frozen.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
REG = ROOT / "registry" / "checkpoint_registry.json"
OUT = ROOT / "prompt_sets" / "frozen_split.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "split.log", rotation="10 MB", level="DEBUG")

SEED = 20260919
N_HELD_OUT = 3
PINNED_TO_SCREEN = "Qwen3"


@logger.catch(reraise=True)
def main() -> None:
    if OUT.exists():
        logger.warning(f"{OUT} already exists - the split is frozen, refusing to re-draw.")
        logger.info(json.dumps(json.loads(OUT.read_text())["assignment"], indent=1))
        return
    reg = json.loads(REG.read_text())
    fams = sorted({c["family"] for c in reg["checkpoints"]})
    if PINNED_TO_SCREEN not in fams:
        raise ValueError(f"{PINNED_TO_SCREEN} missing from registry families: {fams}")
    # A family can only be held out if it can actually CONFIRM the screen's survivor, which
    # needs all three conditions present and loadable; a family missing a condition cannot carry
    # the within-condition contrast the selection rule is decided on.  So the draw is over the
    # complete-triad families only, and every other family goes to SCREEN by construction.
    complete: list[str] = []
    for f in fams:
        conds = {c["condition"]: c for c in reg["checkpoints"] if c["family"] == f}
        if all(conds.get(k, {}).get("usable_for_forward_pass") for k in
               ("base", "instruct", "abliterated")):
            complete.append(f)
    eligible = sorted(f for f in complete if f != PINNED_TO_SCREEN)
    if len(eligible) < N_HELD_OUT:
        raise ValueError(f"only {len(eligible)} complete-triad families are eligible to hold out")
    rng = random.Random(SEED)
    shuffled = eligible[:]
    rng.shuffle(shuffled)
    held = sorted(shuffled[:N_HELD_OUT])
    screen = sorted([f for f in fams if f not in held])
    payload = {
        "metadata_fold": {
            "seed": SEED, "n_families": len(fams), "n_held_out": N_HELD_OUT,
            "draw": "single seeded random.Random(SEED).shuffle over families, Qwen3 pinned to SCREEN",
            "pinned_to_screen": PINNED_TO_SCREEN,
            "pinned_reason": ("the 3-checkpoint feasibility gate and the layer-wise mechanism "
                              "figure both run on the Qwen3 triad, so it cannot be held out"),
            "rule": "HELD_OUT families are not loaded in iteration 1 under any circumstance",
            "shuffled_order_recorded": shuffled,
            "eligibility_rule": ("only families whose base/instruct/abliterated rows are all "
                                 "usable_for_forward_pass are eligible for HELD_OUT"),
            "complete_triad_families": complete,
        },
        "assignment": {"SCREEN": screen, "HELD_OUT": held},
        "per_family": {f: ("HELD_OUT" if f in held else "SCREEN") for f in fams},
    }
    OUT.write_text(json.dumps(payload, indent=1))
    logger.info(f"SCREEN   ({len(screen)}): {screen}")
    logger.info(f"HELD_OUT ({len(held)}): {held}")


if __name__ == "__main__":
    main()
