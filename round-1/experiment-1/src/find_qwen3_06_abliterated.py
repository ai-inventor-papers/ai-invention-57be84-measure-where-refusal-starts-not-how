"""Search HF Hub for a downloadable Qwen3-0.6B abliterated/uncensored model.

huihui-ai/Qwen3-0.6B-abliterated resolves via model_info() but weight download
is 403 gated. This script finds a replacement whose parent is Qwen/Qwen3-0.6B
and whose weights are ACTUALLY fetchable (not just resolvable via metadata).
"""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

CWD = Path(__file__).resolve().parent
ACCESS_PROBE_DIR = CWD / "access_probe"
RESULTS_PATH = CWD / "results" / "qwen3_06_abliterated_search.json"

QUERIES = [
    "Qwen3-0.6B abliterated",
    "Qwen3-0.6B uncensored",
    "Qwen3 0.6B abliterated",
    "Qwen3-0.6B heretic",
    "Qwen3-0.6B no-refusal",
    "Qwen3-0.6B-Instruct abliterated",
]

KEYWORDS = ["ablitera", "uncensor", "heretic", "norefusal", "no-refusal"]

api = HfApi()


def search_query(q: str):
    try:
        # Installed huggingface_hub version's list_models() has no `direction`
        # kwarg; sort="downloads" already returns results descending (verified
        # empirically against this install), matching the requested downloads-desc order.
        return list(api.list_models(search=q, limit=60, sort="downloads"))
    except Exception as e:
        print(f"[search] query failed: {q!r} -> {type(e).__name__}: {str(e)[:150]}")
        return []


def main():
    # Step 1: concurrent search over queries
    all_models = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(search_query, q): q for q in QUERIES}
        for fut in as_completed(futs):
            q = futs[fut]
            models = fut.result()
            print(f"[search] {q!r} -> {len(models)} results")
            all_models.extend(models)

    # Step 2: unique repo ids whose lowercased id matches keywords
    seen = {}
    for m in all_models:
        rid = m.id
        low = rid.lower()
        if any(k in low for k in KEYWORDS):
            if rid not in seen:
                seen[rid] = m
    print(f"[filter] {len(seen)} unique keyword-matching repo ids")

    # Step 3: model_info filter (~0.6B params, no quantization, Qwen3ForCausalLM arch)
    survivors = []
    for rid in seen:
        try:
            info = api.model_info(rid, files_metadata=True)
        except Exception as e:
            print(f"[info] {rid} -> FAILED {type(e).__name__}: {str(e)[:150]}")
            continue

        try:
            total = info.safetensors.total if info.safetensors else None
        except Exception:
            total = None
        if total is None or not (4e8 <= total <= 8e8):
            print(f"[info] {rid} -> skip (safetensors.total={total})")
            continue

        config = info.config or {}
        if "quantization_config" in json.dumps(config):
            print(f"[info] {rid} -> skip (quantized)")
            continue

        if config.get("architectures") != ["Qwen3ForCausalLM"]:
            print(f"[info] {rid} -> skip (architectures={config.get('architectures')})")
            continue

        safetensor_siblings = [
            s.rfilename for s in (info.siblings or []) if s.rfilename.endswith(".safetensors")
        ]
        if not safetensor_siblings:
            print(f"[info] {rid} -> skip (no .safetensors sibling)")
            continue

        survivors.append(
            {
                "repo_id": rid,
                "downloads": getattr(info, "downloads", None),
                "param_total": total,
                "safetensor_file": safetensor_siblings[0],
                "config_architectures": config.get("architectures"),
            }
        )
        print(f"[info] {rid} -> SURVIVOR (downloads={getattr(info, 'downloads', None)}, params={total})")

    # Step 4: access test on top 5 by downloads
    survivors.sort(key=lambda x: (x["downloads"] or 0), reverse=True)
    probe_targets = survivors[:5]

    ACCESS_PROBE_DIR.mkdir(parents=True, exist_ok=True)

    for cand in survivors:
        cand["probed"] = False
        cand["downloadable"] = None
        cand["probe_error"] = None

    for cand in probe_targets:
        rid = cand["repo_id"]
        cand["probed"] = True
        try:
            hf_hub_download(rid, "config.json", cache_dir=str(ACCESS_PROBE_DIR))
            config_ok = True
        except Exception as e:
            config_ok = False
            cand["probe_error"] = f"config.json: {type(e).__name__}: {str(e)[:150]}"

        weight_ok = False
        if config_ok:
            try:
                hf_hub_download(rid, cand["safetensor_file"], cache_dir=str(ACCESS_PROBE_DIR))
                weight_ok = True
            except Exception as e:
                cand["probe_error"] = f"{cand['safetensor_file']}: {type(e).__name__}: {str(e)[:150]}"

        cand["downloadable"] = bool(config_ok and weight_ok)
        print(f"[probe] {rid} -> downloadable={cand['downloadable']} error={cand['probe_error']}")

    # Step 5: write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(
            {
                "queries": QUERIES,
                "keywords": KEYWORDS,
                "unique_keyword_matches": len(seen),
                "survivors_after_info_filter": len(survivors),
                "probed_top_n": len(probe_targets),
                "candidates": survivors,
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {RESULTS_PATH}")

    # Cleanup: delete access_probe dir so no weights remain on disk
    if ACCESS_PROBE_DIR.exists():
        shutil.rmtree(ACCESS_PROBE_DIR)
        print(f"[cleanup] removed {ACCESS_PROBE_DIR}")


if __name__ == "__main__":
    main()
