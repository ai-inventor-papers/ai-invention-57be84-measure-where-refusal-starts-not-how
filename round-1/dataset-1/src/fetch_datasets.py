#!/usr/bin/env python3
"""Download the KEPT safety datasets (HF Hub + authoritative direct URLs) into temp/datasets/.

Every source is recorded with a pinned revision / sha256 so the frozen asset is reproducible.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "temp" / "datasets"
RAW = OUT / "raw"
LOGS = ROOT / "logs"
for p in (OUT, RAW, LOGS):
    p.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "fetch.log", rotation="30 MB", level="DEBUG")

# 12 GB ceiling: plenty for these small text corpora, raises MemoryError instead of OOM-killing.
RAM_BUDGET = 12 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET, RAM_BUDGET))

# repo_id -> list of (config, split) ; None config = default
HF_SPECS: list[dict[str, Any]] = [
    {"id": "Paul/XSTest", "parts": [(None, "train")]},
    {"id": "JailbreakBench/JBB-Behaviors", "parts": [("behaviors", "harmful"), ("behaviors", "benign")]},
    {"id": "bench-llm/or-bench", "parts": [("or-bench-hard-1k", "train"), ("or-bench-toxic", "train")]},
    {"id": "allenai/coconot", "parts": [("original", "test"), ("contrast", "test")]},
    {"id": "LibrAI/do-not-answer", "parts": [(None, "train")], "drop_cols_containing": "_response"},
    {"id": "Bertievidgen/SimpleSafetyTests", "parts": [(None, "test")]},
    {"id": "furonghuang-lab/PHTest", "parts": [(None, "train")]},
    {"id": "SillyTilly/SorryBench", "parts": [(None, "train")]},
    {"id": "TrustAIRLab/in-the-wild-jailbreak-prompts",
     "parts": [("jailbreak_2023_12_25", "train")]},
    {"id": "PKU-Alignment/BeaverTails-Evaluation", "parts": [(None, "test")]},
    {"id": "walledai/MaliciousInstruct", "parts": [(None, "train")]},
    {"id": "walledai/WildGuardTest", "parts": [(None, "train")]},
    {"id": "declare-lab/HarmfulQA", "parts": [(None, "train")],
     "keep_cols": ["id", "topic", "subtopic", "question"]},
    {"id": "TrustAIRLab/forbidden_question_set", "parts": [(None, "train")]},
    {"id": "walledai/AyaRedTeaming", "parts": [(None, "english")]},
]

DIRECT: list[dict[str, str]] = [
    {"key": "advbench_harmful_behaviors",
     "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"},
    {"key": "strongreject_full",
     "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv"},
    {"key": "strongreject_small",
     "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv"},
    {"key": "harmbench_behaviors_text_all",
     "url": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"},
    {"key": "xstest_prompts_github",
     "url": "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv"},
    {"key": "sorrybench_judge_prompts",
     "url": "https://raw.githubusercontent.com/sorry-bench/sorry-bench/main/data/sorry_bench/judge_prompts.jsonl"},
    {"key": "strongreject_evaluator_py",
     "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject/strongreject_evaluator.py"},
    {"key": "strongreject_evaluator_prompt",
     "url": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject/strongreject_evaluator_prompt.txt"},
]


def _slug(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_")


def fetch_hf(spec: dict[str, Any]) -> dict[str, Any]:
    rid = spec["id"]
    rec: dict[str, Any] = {"source": "huggingface", "repo_id": rid, "parts": [], "ok": True}
    try:
        info = HfApi().dataset_info(rid)
        rec["revision_sha"] = info.sha
        rec["hf_downloads"] = info.downloads
        rec["hf_likes"] = info.likes
        rec["last_modified"] = str(info.last_modified)
        cd = info.card_data.to_dict() if info.card_data else {}
        rec["license"] = cd.get("license")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"{rid}: metadata failed: {exc!r}")
        rec["ok"] = False
        rec["error"] = repr(exc)[:300]
        return rec

    for cfg, split in spec["parts"]:
        try:
            ds = load_dataset(rid, cfg, split=split, revision=rec["revision_sha"])
            cols = list(ds.column_names)
            if spec.get("keep_cols"):
                keep = [c for c in spec["keep_cols"] if c in cols]
                ds = ds.select_columns(keep)
            if spec.get("drop_cols_containing"):
                keep = [c for c in cols if spec["drop_cols_containing"] not in c]
                ds = ds.select_columns(keep)
            rows = [dict(r) for r in ds]
            fn = RAW / f"full_{_slug(rid)}__{cfg or 'default'}__{split}.json"
            fn.write_text(json.dumps(rows, ensure_ascii=False))
            rec["parts"].append({"config": cfg, "split": split, "n_rows": len(rows),
                                 "columns": list(ds.column_names), "file": fn.name,
                                 "bytes": fn.stat().st_size})
            logger.info(f"OK {rid} [{cfg}/{split}] rows={len(rows)} -> {fn.name}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{rid} [{cfg}/{split}] FAILED: {exc!r}")
            rec["ok"] = False
            rec.setdefault("part_errors", []).append(
                {"config": cfg, "split": split, "error": repr(exc)[:300]})
    return rec


def fetch_direct(spec: dict[str, str]) -> dict[str, Any]:
    rec: dict[str, Any] = {"source": "direct_url", "key": spec["key"], "url": spec["url"], "ok": True}
    try:
        r = requests.get(spec["url"], timeout=120)
        r.raise_for_status()
        body = r.content
        ext = spec["url"].rsplit(".", 1)[-1]
        fn = RAW / f"direct_{spec['key']}.{ext}"
        fn.write_bytes(body)
        rec.update({"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                    "file": fn.name, "http_status": r.status_code})
        logger.info(f"OK direct {spec['key']} bytes={len(body)}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"direct {spec['key']} FAILED: {exc!r}")
        rec["ok"] = False
        rec["error"] = repr(exc)[:300]
    return rec


@logger.catch(reraise=True)
def main() -> None:
    manifest: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_hf, s) for s in HF_SPECS]
        futs += [ex.submit(fetch_direct, s) for s in DIRECT]
        for f in as_completed(futs):
            manifest.append(f.result())
    (OUT / "download_manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    ok = sum(1 for m in manifest if m["ok"])
    logger.info(f"manifest: {ok}/{len(manifest)} sources OK")
    for m in manifest:
        if not m["ok"]:
            logger.warning(f"FAILED source: {m.get('repo_id') or m.get('key')}")


if __name__ == "__main__":
    main()
