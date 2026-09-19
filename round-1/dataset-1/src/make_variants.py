#!/usr/bin/env python3
"""Write mini (3 items) and preview (3 items, strings truncated to 200 chars) variants.

Mirrors the aii-json skill's full/mini/preview convention locally, because this workspace
talks to the Hub directly rather than through the ability server.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


def truncate(obj: Any, n: int = 200) -> Any:
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:n] + f"... [+{len(obj) - n} chars]"
    if isinstance(obj, list):
        return [truncate(x, n) for x in obj]
    if isinstance(obj, dict):
        return {k: truncate(v, n) for k, v in obj.items()}
    return obj


def variants(path: Path, array_key: str, keep: int = 3) -> None:
    data = json.loads(path.read_text())
    mini = dict(data)
    mini[array_key] = data[array_key][:keep]
    stem = path.stem.replace("_full", "")
    (path.parent / f"mini_{stem}.json").write_text(json.dumps(mini, indent=1, ensure_ascii=False))
    (path.parent / f"preview_{stem}.json").write_text(
        json.dumps(truncate(mini), indent=1, ensure_ascii=False))
    logger.info(f"{path.name}: {len(data[array_key])} items -> mini_{stem}.json, preview_{stem}.json")


def registry_csv() -> None:
    reg = json.loads((ROOT / "registry" / "checkpoint_registry.json").read_text())
    rows = []
    for c in reg["checkpoints"]:
        r = {k: v for k, v in c.items() if k != "card_text"}
        r["card_len"] = len(c.get("card_text") or "")
        r["primary_weight_files"] = ";".join(r.get("primary_weight_files") or [])
        rows.append(r)
    cols = sorted({k for r in rows for k in r})
    with (ROOT / "registry" / "checkpoint_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    logger.info(f"registry csv rewritten: {len(rows)} rows x {len(cols)} cols")


def prompt_sets_csv() -> None:
    d = json.loads((ROOT / "prompt_sets" / "prompt_sets_full.json").read_text())
    rows = d["rows"]
    cols = sorted(rows[0].keys())
    with (ROOT / "prompt_sets" / "prompt_sets_full.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    logger.info(f"prompt sets csv written: {len(rows)} rows x {len(cols)} cols")


def main() -> None:
    registry_csv()
    prompt_sets_csv()
    variants(ROOT / "registry" / "checkpoint_registry.json", "checkpoints")
    variants(ROOT / "prompt_sets" / "prompt_sets_full.json", "rows")


if __name__ == "__main__":
    main()
