"""Mechanical HuggingFace Hub metadata lookup for the model roster.

For every candidate repo id, calls HfApi().model_info(repo_id, files_metadata=True)
concurrently and records what happened (resolved/gated/not-found/other error),
plus size, param count, architectures, quantization flag and tags.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <7} | {message}")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_PATH = RESULTS_DIR / "roster_resolution.json"

ROSTER: dict[str, dict[str, str]] = {
    "qwen3_17": {
        "base": "Qwen/Qwen3-1.7B-Base",
        "instruct": "Qwen/Qwen3-1.7B",
        "abliterated": "huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2",
        "abliterated_2": "huihui-ai/Qwen3-1.7B-abliterated",
    },
    "qwen3_06": {
        "base": "Qwen/Qwen3-0.6B-Base",
        "instruct": "Qwen/Qwen3-0.6B",
        "abliterated": "huihui-ai/Qwen3-0.6B-abliterated",
    },
    "llama32": {
        "base": "meta-llama/Llama-3.2-1B",
        "instruct": "meta-llama/Llama-3.2-1B-Instruct",
        "abliterated": "huihui-ai/Llama-3.2-1B-Instruct-abliterated",
        "base_unsloth": "unsloth/Llama-3.2-1B",
        "instruct_unsloth": "unsloth/Llama-3.2-1B-Instruct",
    },
    "gemma3": {
        "base": "google/gemma-3-1b-pt",
        "instruct": "google/gemma-3-1b-it",
        "abliterated": "huihui-ai/gemma-3-1b-it-abliterated",
        "base_unsloth": "unsloth/gemma-3-1b-pt",
        "instruct_unsloth": "unsloth/gemma-3-1b-it",
    },
    "qwen25": {
        "base": "Qwen/Qwen2.5-1.5B",
        "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "abliterated": "huihui-ai/Qwen2.5-1.5B-Instruct-abliterated",
    },
    "smollm2": {
        "base": "HuggingFaceTB/SmolLM2-1.7B",
        "instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "abliterated": "huihui-ai/SmolLM2-1.7B-Instruct-abliterated",
    },
    "falcon3": {
        "base": "tiiuae/Falcon3-1B-Base",
        "instruct": "tiiuae/Falcon3-1B-Instruct",
        "abliterated": "huihui-ai/Falcon3-1B-Instruct-abliterated",
    },
    "llama32_3b": {
        "instruct": "meta-llama/Llama-3.2-3B-Instruct",
        "abliterated": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
    },
}


def build_candidates() -> list[dict[str, str]]:
    candidates = []
    for family, conditions in ROSTER.items():
        for condition_key, repo_id in conditions.items():
            # normalize condition label to one of base/instruct/abliterated
            if condition_key.startswith("base"):
                condition = "base"
            elif condition_key.startswith("instruct"):
                condition = "instruct"
            elif condition_key.startswith("abliterated"):
                condition = "abliterated"
            else:
                condition = condition_key
            candidates.append({"repo_id": repo_id, "family": family, "condition": condition})
    return candidates


def resolve_one(api: HfApi, candidate: dict[str, str]) -> dict:
    repo_id = candidate["repo_id"]
    family = candidate["family"]
    condition = candidate["condition"]
    entry = {
        "repo_id": repo_id,
        "family": family,
        "condition": condition,
        "resolved": False,
        "reason": None,
        "sha": None,
        "total_size_bytes": None,
        "num_params": None,
        "architectures": None,
        "quantization": False,
        "tags": [],
    }
    try:
        logger.info(f"Resolving {repo_id} ...")
        info = api.model_info(repo_id, files_metadata=True)

        entry["resolved"] = True
        entry["reason"] = "OK"
        entry["sha"] = info.sha

        total_size = 0
        have_size = False
        if info.siblings:
            for s in info.siblings:
                if s.rfilename and (s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin")):
                    if s.size is not None:
                        total_size += s.size
                        have_size = True
        entry["total_size_bytes"] = total_size if have_size else None

        if info.safetensors is not None and getattr(info.safetensors, "total", None) is not None:
            entry["num_params"] = info.safetensors.total

        config = info.config or {}
        entry["architectures"] = config.get("architectures") if config else None
        entry["quantization"] = "quantization_config" in json.dumps(config)

        entry["tags"] = list(info.tags) if info.tags else []

        logger.success(f"OK {repo_id}")
    except GatedRepoError as e:
        entry["reason"] = "GATED_REPO"
        logger.warning(f"GATED_REPO {repo_id}: {e}")
    except RepositoryNotFoundError as e:
        entry["reason"] = "NOT_FOUND"
        logger.warning(f"NOT_FOUND {repo_id}: {e}")
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status in (401, 403):
            entry["reason"] = "GATED_REPO"
            logger.warning(f"GATED_REPO (HTTP {status}) {repo_id}: {e}")
        elif status == 404:
            entry["reason"] = "NOT_FOUND"
            logger.warning(f"NOT_FOUND (HTTP {status}) {repo_id}: {e}")
        else:
            entry["reason"] = f"OTHER:{type(e).__name__}"
            logger.error(f"OTHER {repo_id}: {type(e).__name__}: {e}")
    except Exception as e:
        entry["reason"] = f"OTHER:{type(e).__name__}"
        logger.error(f"OTHER {repo_id}: {type(e).__name__}: {e}")
    return entry


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates()
    logger.info(f"Resolving {len(candidates)} candidate repos with 8 workers ...")

    api = HfApi()
    entries: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(resolve_one, api, c): c for c in candidates}
        for fut in as_completed(futures):
            entries.append(fut.result())

    # stable ordering matching input order
    order = {c["repo_id"]: i for i, c in enumerate(candidates)}
    entries.sort(key=lambda e: order[e["repo_id"]])

    output = {
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    with OUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote {OUT_PATH}")

    # Summary table
    print("\nrepo_id | family | condition | resolved | reason | size_GB")
    print("-" * 100)
    for e in entries:
        size_gb = f"{e['total_size_bytes'] / 1e9:.2f}" if e["total_size_bytes"] else "n/a"
        print(f"{e['repo_id']} | {e['family']} | {e['condition']} | {e['resolved']} | {e['reason']} | {size_gb}")

    # families with all of base/instruct/abliterated resolved
    by_family: dict[str, dict[str, bool]] = {}
    for e in entries:
        by_family.setdefault(e["family"], {})
        # OR across duplicate condition entries per family (e.g. unsloth variants)
        by_family[e["family"]][e["condition"]] = by_family[e["family"]].get(e["condition"], False) or e["resolved"]

    print("\nFamilies with base+instruct+abliterated all resolved:")
    for family, conds in by_family.items():
        if conds.get("base") and conds.get("instruct") and conds.get("abliterated"):
            print(f"  {family}")

    print("\nGATED_REPO or NOT_FOUND:")
    for e in entries:
        if e["reason"] in ("GATED_REPO", "NOT_FOUND"):
            print(f"  {e['repo_id']} -> {e['reason']}")


if __name__ == "__main__":
    main()
