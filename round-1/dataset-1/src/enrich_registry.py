#!/usr/bin/env python3
"""Add the three fields the experiment actually gates on, which a plain metadata read misses.

1. `gated_blocking` — the repo is gated AND this environment's token cannot read it.  Found by a
   real file-listing attempt, not by trusting the `gated` flag, because `gated="auto"` repos are
   readable with an accepted licence and unreadable without one.  The dataset half of this
   artifact already hit exactly this trap (8 gated dataset mirrors were unusable), so it is
   checked here on the models too.
2. `min_load_bytes` — bytes needed to actually LOAD the model (weights + config + tokenizer),
   as opposed to `total_download_size_bytes`, which sums every file in the repo including
   duplicate quantised copies and is therefore 2-8x too large for disk planning.
3. `usable_for_forward_pass` — false for GGUF-only and pre-quantised repos, which cannot be run
   through `transformers` on this CPU-only box without a conversion step.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from loguru import logger

ROOT = Path(__file__).resolve().parent
REG = ROOT / "registry" / "checkpoint_registry.json"

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "enrich.log", rotation="10 MB", level="DEBUG")

MAX_TRIES = 6


def _retry(fn):  # noqa: ANN001, ANN202
    """Hub calls here are bursty enough to hit the 429 `api` rate limit; honour Retry-After."""
    import time
    last: Exception | None = None
    for attempt in range(MAX_TRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "429" not in msg and "rate limit" not in msg.lower():
                raise
            last = exc
            wait = 30.0 * (attempt + 1)
            m = re.search(r"Retry after (\d+) seconds", msg)
            if m:
                wait = float(m.group(1)) + 5.0
            logger.warning(f"429 rate limit, sleeping {wait:.0f}s (attempt {attempt + 1}/{MAX_TRIES})")
            time.sleep(wait)
    raise RuntimeError(f"rate limited after {MAX_TRIES} attempts") from last


WEIGHT_RE = re.compile(r"\.(safetensors|bin|pt|gguf)$")
AUX_RE = re.compile(r"(config\.json|tokenizer.*\.json|tokenizer\.model|tokenizer_config\.json|"
                    r"special_tokens_map\.json|generation_config\.json|vocab\.json|merges\.txt)$")
SHARD_RE = re.compile(r"model(-\d{5}-of-\d{5})?\.safetensors$")
# Some well-used repos (TinyLlama_v1.1) predate safetensors and ship only a .bin; transformers
# loads those fine, so they must not be scored NOT_LOADABLE.
BIN_RE = re.compile(r"pytorch_model(-\d{5}-of-\d{5})?\.bin$")
QUANT_NAME_RE = re.compile(r"(gguf|awq|gptq|4bit|8bit|openvino|int4|int8)", re.I)


def probe(ck: dict[str, Any]) -> dict[str, Any]:
    out = {"gated_blocking": None, "gated_probe_error": None,
           "min_load_bytes": None, "usable_for_forward_pass": None,
           "primary_weight_files": [], "quantisation_verified": None}
    rid = ck.get("repo_id")
    if not ck.get("available") or not rid:
        out["usable_for_forward_pass"] = False
        return out
    api = HfApi()
    sibs: list[Any] = []
    try:
        info = _retry(lambda: api.model_info(rid, files_metadata=True, revision=ck.get("sha")))
        sibs = list(info.siblings or [])
        # A real read of the file the loader needs first.
        _retry(lambda: api.hf_hub_download(rid, "config.json", revision=ck.get("sha")))
        out["gated_blocking"] = False
    except GatedRepoError as exc:
        out["gated_blocking"] = True
        out["gated_probe_error"] = repr(exc)[:200]
    except (RepositoryNotFoundError, OSError) as exc:
        # config.json absent is normal for GGUF-only repos; that is not a gating failure.
        msg = repr(exc)
        if "config.json" in msg or "EntryNotFound" in msg or "404" in msg:
            out["gated_blocking"] = False
        else:
            out["gated_blocking"] = None
            out["gated_probe_error"] = msg[:200]
    except Exception as exc:  # noqa: BLE001
        out["gated_blocking"] = None
        out["gated_probe_error"] = repr(exc)[:200]

    st = [s for s in sibs if SHARD_RE.search(s.rfilename)]
    if not st:
        st = [s for s in sibs if s.rfilename.endswith(".safetensors")
              and "consolidated" not in s.rfilename
              and not QUANT_NAME_RE.search(s.rfilename)]
    if not st:
        st = [s for s in sibs if BIN_RE.search(s.rfilename)]
    aux = [s for s in sibs if AUX_RE.search(s.rfilename)]
    if st:
        out["min_load_bytes"] = sum((s.size or 0) for s in st) + sum((s.size or 0) for s in aux)
    # The primary weight file itself decides quantisation, not the repo-wide file list: several
    # full-precision repos additionally publish GGUF/OpenVINO copies of themselves, which an
    # extension-only or card-text heuristic misreads as "this checkpoint is quantised".
    primary_quantised = bool(st) and all(QUANT_NAME_RE.search(s.rfilename) for s in st)
    out["primary_weight_files"] = [s.rfilename for s in st][:8]
    gguf_only = not st and any(x.rfilename.endswith(".gguf") for x in sibs)
    out["quantisation_verified"] = (
        "quantised (GGUF only - no transformers-loadable weights in this repo)" if gguf_only
        else "unknown (no primary weight file found)" if not st
        else "quantised" if primary_quantised
        else "none (full precision primary weights)")
    out["usable_for_forward_pass"] = bool(
        st and out["gated_blocking"] is not True and not primary_quantised)
    return out


@logger.catch(reraise=True)
def main() -> None:
    reg = json.loads(REG.read_text())
    cks = reg["checkpoints"]
    with ThreadPoolExecutor(max_workers=3) as ex:
        probes = list(ex.map(probe, cks))
    for ck, p in zip(cks, probes):
        if p.get("gated_probe_error") and ck.get("primary_weight_files"):
            logger.warning(f"probe failed for {ck.get('repo_id')}; keeping previous enrichment")
            continue
        ck.update(p)

    # The write-gain ORACLE arm needs the checkpoint an abliteration was actually derived FROM.
    # `parent_repo_declared` is whatever the card says, which is not always the instruct model:
    # several cards list the family BASE in their `base_model` YAML even though the edit was
    # applied to the instruct checkpoint.  So the same-family instruct row is recorded
    # separately as an INFERRED parent, and the two are never conflated.
    instruct_by_family = {c["family"]: c.get("repo_id") for c in cks
                          if c["condition"] == "instruct" and c.get("available")}
    for c in cks:
        if c["condition"] != "abliterated" or not c.get("available"):
            c["parent_repo_inferred_instruct"] = None
            c["parent_declared_matches_inferred"] = None
            continue
        inf = instruct_by_family.get(c["family"])
        c["parent_repo_inferred_instruct"] = inf
        c["parent_declared_matches_inferred"] = (
            None if not c.get("parent_repo_declared") else c["parent_repo_declared"] == inf)
    reg["metadata"]["n_usable_for_forward_pass"] = sum(
        1 for c in cks if c.get("usable_for_forward_pass"))
    reg["metadata"]["n_gated_blocking"] = sum(1 for c in cks if c.get("gated_blocking"))
    reg["metadata"]["enrichment_note"] = (
        "gated_blocking / min_load_bytes / usable_for_forward_pass added by enrich_registry.py; "
        "gated_blocking is a real config.json fetch, not the `gated` metadata flag")
    REG.write_text(json.dumps(reg, indent=1, ensure_ascii=False))

    fams: dict[str, dict[str, str]] = {}
    for c in cks:
        fams.setdefault(c["family"], {})[c["condition"]] = (
            "-" if not c.get("available") else
            ("USABLE" if c.get("usable_for_forward_pass") else
             ("GATED" if c.get("gated_blocking") else "NOT_LOADABLE")))
    logger.info(f"{'family':<14}{'base':<14}{'instruct':<14}abliterated")
    for f in sorted(fams):
        r = fams[f]
        logger.info(f"{f:<14}{r.get('base','-'):<14}{r.get('instruct','-'):<14}{r.get('abliterated','-')}")
    usable_fams = sorted(f for f, r in fams.items()
                         if all(r.get(c) == "USABLE" for c in ("base", "instruct", "abliterated")))
    logger.info(f"families with all 3 conditions USABLE ({len(usable_fams)}): {usable_fams}")
    gb = sum((c.get("min_load_bytes") or 0) for c in cks if c.get("usable_for_forward_pass")) / 1e9
    logger.info(f"total min_load_bytes over usable checkpoints: {gb:.1f} GB "
                f"(disk cap is 20 GB -> stream download/compute/delete)")


if __name__ == "__main__":
    main()
