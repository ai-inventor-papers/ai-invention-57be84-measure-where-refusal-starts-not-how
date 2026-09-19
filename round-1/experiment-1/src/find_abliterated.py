import json
import os
import time
import concurrent.futures as cf
from huggingface_hub import HfApi
from pathlib import Path

api = HfApi(token=os.environ.get("HF_TOKEN"))

QUERIES = [
    "Llama-3.2-1B abliterated",
    "Llama-3.2-1B uncensored",
    "gemma-3-1b abliterated",
    "gemma-3-1b uncensored",
    "Qwen2.5-1.5B abliterated",
    "Qwen2.5-1.5B uncensored",
    "SmolLM2 abliterated",
    "SmolLM2 uncensored",
    "Falcon3-1B abliterated",
    "Falcon3-1B uncensored",
    "Qwen2.5-0.5B abliterated",
    "Llama-3.2-3B abliterated",
    "abliterated 1B",
    "abliterated 1.5B",
]

PARENTS = {
    "llama32": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma3": "google/gemma-3-1b-it",
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "falcon3": "tiiuae/Falcon3-1B-Instruct",
}


def search_query(q, retries=5):
    for attempt in range(retries):
        try:
            models = api.list_models(search=q, limit=80, sort="downloads")
            return [m.id for m in models]
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(95)
                continue
            print(f"search failed for {q!r}: {e}")
            return []
    return []


def get_model_info(repo_id, retries=5):
    for attempt in range(retries):
        try:
            return repo_id, api.model_info(repo_id, files_metadata=True)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(95)
                continue
            return repo_id, None
    return repo_id, None


def main():
    # 1. concurrent search
    pooled_ids = set()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for ids in ex.map(search_query, QUERIES):
            for rid in ids:
                if "ablitera" in rid.lower() or "uncensor" in rid.lower():
                    pooled_ids.add(rid)

    print(f"Pooled {len(pooled_ids)} unique candidate repo ids")

    # 2. fetch model_info concurrently for pooled candidates
    keepers = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(get_model_info, sorted(pooled_ids)))

    for repo_id, info in results:
        if info is None:
            continue
        try:
            if info.safetensors is None or info.safetensors.total is None:
                continue
            total_params = info.safetensors.total
            if not (3e8 <= total_params <= 4e9):
                continue
            config = info.config or {}
            if "quantization_config" in json.dumps(config):
                continue
            siblings = info.siblings or []
            if not any(s.rfilename.endswith(".safetensors") for s in siblings):
                continue
            architectures = config.get("architectures")
            if not architectures:
                continue
            total_size_bytes = sum(
                (s.size or 0)
                for s in siblings
                if s.rfilename.endswith(".safetensors") or s.rfilename.endswith(".bin")
            )
            keepers.append({
                "repo_id": repo_id,
                "downloads": info.downloads,
                "num_params": total_params,
                "architectures": architectures,
                "total_size_bytes": total_size_bytes,
                "tags": info.tags or [],
            })
        except Exception as e:
            print(f"error processing {repo_id}: {e}")
            continue

    print(f"Kept {len(keepers)} candidates after filtering")

    # 3. fetch parent info
    parents_info = {}
    for fam, parent_id in PARENTS.items():
        try:
            _, pinfo = get_model_info(parent_id)
            if pinfo is None:
                raise RuntimeError("model_info returned None after retries")
            p_arch = pinfo.config.get("architectures") if pinfo.config else None
            p_total = pinfo.safetensors.total if pinfo.safetensors else None
            parents_info[fam] = {
                "repo_id": parent_id,
                "architectures": p_arch,
                "num_params": p_total,
            }
        except Exception as e:
            print(f"failed to fetch parent {parent_id}: {e}")
            parents_info[fam] = {
                "repo_id": parent_id,
                "architectures": None,
                "num_params": None,
            }

    # 4. assign candidates to families
    families = {fam: [] for fam in PARENTS}
    families["unmatched"] = []

    for cand in keepers:
        assigned = None
        for fam, pinfo in parents_info.items():
            p_arch = pinfo["architectures"]
            p_total = pinfo["num_params"]
            if p_arch is None or p_total is None:
                continue
            if cand["architectures"] == p_arch:
                lo, hi = p_total * 0.98, p_total * 1.02
                if lo <= cand["num_params"] <= hi:
                    assigned = fam
                    break
        if assigned:
            families[assigned].append(cand)
        else:
            families["unmatched"].append(cand)

    for fam in families:
        families[fam].sort(key=lambda c: c["downloads"] or 0, reverse=True)

    out = {
        "parents": parents_info,
        "candidates": [c for fam_list in families.values() for c in fam_list],
        "families": families,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "abliterated_search.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    # verify parse
    json.loads(out_path.read_text())
    print("JSON verified parseable")

    for fam in PARENTS:
        print(f"\n=== {fam} (parent {PARENTS[fam]}) ===")
        lst = families[fam][:3]
        if not lst:
            print("NONE")
        else:
            for c in lst:
                print(f"  {c['repo_id']} | downloads={c['downloads']} | params={c['num_params']} | arch={c['architectures']}")

    print(f"\n=== unmatched ({len(families['unmatched'])}) ===")
    for c in families["unmatched"][:10]:
        print(f"  {c['repo_id']} | downloads={c['downloads']} | params={c['num_params']} | arch={c['architectures']}")


if __name__ == "__main__":
    main()
