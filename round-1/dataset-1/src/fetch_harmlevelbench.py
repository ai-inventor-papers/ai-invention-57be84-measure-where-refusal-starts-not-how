#!/usr/bin/env python3
"""Recover HarmLevelBench's 7 x 8 severity ladder from the paper itself.

HarmLevelBench (Belkhiter, Zizzo & Maffeis, arXiv 2411.06835, NeurIPS 2024 SafeGenAI workshop,
CC BY 4.0) is the externally-authored graded severity axis this study was designed around, but it
was never released as a file: there is no GitHub repo, no HF dataset, no Zenodo record, and it does
not appear in the 149-dataset SafetyPrompts.com catalogue.  All 56 items exist only as a typeset
table in the paper.

The paper is CC BY 4.0, so the table is recoverable.  This script PARSES it out of the arXiv HTML
rather than transcribing it by hand, so the extraction is reproducible and auditable: it records
the source URL, the sha256 of the fetched HTML, and asserts the 7 x 8 structure the paper claims
before writing anything.
"""
from __future__ import annotations

import hashlib
import html as html_mod
import json
import re
import sys
from pathlib import Path

import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "temp" / "datasets" / "raw" / "direct_harmlevelbench_table6.json"
URL = "https://arxiv.org/html/2411.06835"

# The paper's category ids are numeric; these are its own topic names (Section 3 / Table 7).
TOPIC_NAMES = {1: "Bomb", 2: "Hack", 3: "Drugs", 4: "Kill",
               5: "Shot", 6: "Steal", 7: "Trigger"}

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "harmlevelbench.log", rotation="10 MB", level="DEBUG")


def cells(tr: str) -> list[str]:
    raw = re.findall(r"<t[dh].*?</t[dh]>", tr, re.S)
    return [c for c in (html_mod.unescape(re.sub(r"<[^>]+>", "", x)).strip() for x in raw) if c]


@logger.catch(reraise=True)
def main() -> None:
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()
    body = resp.content
    digest = hashlib.sha256(body).hexdigest()
    page = body.decode("utf-8", errors="replace")
    logger.info(f"fetched {URL} ({len(body)} bytes, sha256={digest[:16]}...)")

    rows: list[dict[str, object]] = []
    for tbl in re.finditer(r"<table.*?</table>", page, re.S):
        block = tbl.group(0)
        header = None
        found: list[dict[str, object]] = []
        for tr in re.findall(r"<tr.*?</tr>", block, re.S):
            c = cells(tr)
            if not c:
                continue
            if header is None:
                header = [x.lower() for x in c]
                continue
            if len(c) != 4 or not c[0].isdigit():
                continue
            idx, cat, lvl, prompt = c
            if not (cat.isdigit() and lvl.isdigit()):
                continue
            found.append({"index": int(idx), "category_id": int(cat),
                          "topic": TOPIC_NAMES.get(int(cat), f"category_{cat}"),
                          "harm_level": int(lvl), "prompt": prompt})
        if header and "harm level" in " ".join(header) and len(found) == 56:
            rows = found
            break

    if len(rows) != 56:
        raise ValueError(f"expected 56 HarmLevelBench items, parsed {len(rows)} - "
                         "the arXiv HTML layout changed; re-check before trusting this axis")
    topics = sorted({r["topic"] for r in rows})
    if len(topics) != 7:
        raise ValueError(f"expected 7 topics, got {topics}")
    for t in topics:
        levels = sorted(r["harm_level"] for r in rows if r["topic"] == t)
        if levels != list(range(1, 9)):
            raise ValueError(f"topic {t} has levels {levels}, expected 1..8")

    payload = {
        "dataset": "HarmLevelBench (Table 6)",
        "paper": ("Belkhiter, Zizzo & Maffeis, 'HarmLevelBench: Evaluating Harm-Level Compliance "
                  "and the Impact of Quantization on Model Alignment', arXiv:2411.06835, "
                  "NeurIPS 2024 Workshop on Safe Generative AI"),
        "license": "CC BY 4.0 (arXiv licence of 2411.06835)",
        "source_url": URL,
        "source_sha256": digest,
        "extraction": ("parsed programmatically from the arXiv HTML table, not hand-transcribed; "
                       "7 topics x harm levels 1-8 asserted before writing"),
        "caveat": ("never released as a standalone file - no GitHub, no HF dataset, no Zenodo "
                   "record, absent from the SafetyPrompts.com 149-dataset catalogue, 0 citations "
                   "as of 2026-09-19. n=56 is small for any dose-response claim on its own."),
        "template": "fixed 'How to ...?' question frame",
        "n_items": len(rows),
        "topics": topics,
        "items": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    logger.info(f"wrote {OUT}: {len(rows)} items over {len(topics)} topics {topics}")
    for t in topics:
        ex = [r for r in rows if r["topic"] == t]
        lo = min(ex, key=lambda r: r["harm_level"])["prompt"]
        hi = max(ex, key=lambda r: r["harm_level"])["prompt"]
        logger.info(f"  {t:<8} L1: {lo[:56]:<58} L8: {hi[:56]}")


if __name__ == "__main__":
    main()
