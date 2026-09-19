#!/usr/bin/env python3
"""Build a live, HF-Hub-verified checkpoint registry of small instruction-tuned
LLM families x {base, instruct, abliterated} conditions.

Only tiny metadata files (config.json, README.md) are ever downloaded; the
actual model weights are never fetched. All HF API calls run concurrently
(ThreadPoolExecutor) with bounded retries. Output is a single JSON registry
plus a flattened CSV (card_text excluded, replaced by card_len).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent
LOG_DIR = WORKSPACE / "logs"
REGISTRY_DIR = WORKSPACE / "registry"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOG_DIR / "registry.log", rotation="30 MB", level="DEBUG")

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
MAX_WORKERS = 8
MAX_RETRIES = 3
RETRYABLE_EXC = (HfHubHTTPError, ConnectionError, TimeoutError, OSError)

WEIGHT_EXTS = {
    ".safetensors", ".bin", ".gguf", ".onnx", ".h5", ".msgpack", ".pt", ".pth",
}
QUANT_KEYWORDS = {
    "awq": "AWQ",
    "gptq": "GPTQ",
    "bnb-4bit": "bnb-4bit",
    "bnb-8bit": "bnb-8bit",
    "4bit": "bnb-4bit",
    "8bit": "bnb-8bit",
    "int4": "int4",
    "int8": "int8",
}


@dataclass
class CheckpointSpec:
    family: str
    condition: str  # base | instruct | abliterated
    repo_id: str | None
    missing_reason: str | None = None


# ---------------------------------------------------------------------------
# Registry specification — verified against the live HF Hub search index
# before being hard-coded here (see conversation transcript for the search
# calls). repo_id=None rows are conditions that do not exist on the Hub and
# are recorded explicitly rather than silently dropped.
# ---------------------------------------------------------------------------
SPECS: list[CheckpointSpec] = [
    # --- Qwen3 (MANDATORY triad) ---
    CheckpointSpec("Qwen3", "base", "Qwen/Qwen3-1.7B-Base"),
    CheckpointSpec("Qwen3", "instruct", "Qwen/Qwen3-1.7B"),
    CheckpointSpec("Qwen3", "abliterated", "mlabonne/Qwen3-1.7B-abliterated"),
    # --- Qwen2.5 ---
    CheckpointSpec("Qwen2.5", "base", "Qwen/Qwen2.5-1.5B"),
    CheckpointSpec("Qwen2.5", "instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    CheckpointSpec(
        "Qwen2.5", "abliterated",
        "Goekdeniz-Guelmez/Josiefied-Qwen2.5-1.5B-Instruct-abliterated-v1",
    ),
    # --- Llama-3.2 ---
    CheckpointSpec("Llama-3.2", "base", "meta-llama/Llama-3.2-1B"),
    CheckpointSpec("Llama-3.2", "instruct", "meta-llama/Llama-3.2-1B-Instruct"),
    CheckpointSpec(
        "Llama-3.2", "abliterated", "mylesgoose/Llama-3.2-1B-Instruct-abliterated"
    ),
    # --- Gemma-2 ---
    CheckpointSpec("Gemma-2", "base", "google/gemma-2-2b"),
    CheckpointSpec("Gemma-2", "instruct", "google/gemma-2-2b-it"),
    CheckpointSpec("Gemma-2", "abliterated", "IlyaGusev/gemma-2-2b-it-abliterated"),
    # --- Gemma-3 ---
    CheckpointSpec("Gemma-3", "base", "google/gemma-3-1b-pt"),
    CheckpointSpec("Gemma-3", "instruct", "google/gemma-3-1b-it"),
    CheckpointSpec(
        "Gemma-3", "abliterated",
        "DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated",
    ),
    # --- SmolLM2 ---
    CheckpointSpec("SmolLM2", "base", "HuggingFaceTB/SmolLM2-1.7B"),
    CheckpointSpec("SmolLM2", "instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    CheckpointSpec(
        "SmolLM2", "abliterated", "venkycs/SmolLM2-1.7B-Instruct-Abliterated"
    ),
    # --- Falcon3 ---
    CheckpointSpec("Falcon3", "base", "tiiuae/Falcon3-1B-Base"),
    CheckpointSpec("Falcon3", "instruct", "tiiuae/Falcon3-1B-Instruct"),
    CheckpointSpec(
        "Falcon3", "abliterated",
        "mradermacher/Falcon3-1B-Instruct-abliterated-GGUF",
    ),
    # --- LFM2 ---
    CheckpointSpec("LFM2", "base", "LiquidAI/LFM2.5-1.2B-Base"),
    CheckpointSpec("LFM2", "instruct", "LiquidAI/LFM2.5-1.2B-Instruct"),
    CheckpointSpec(
        "LFM2", "abliterated", "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated"
    ),
    # --- Granite-3.x ---
    CheckpointSpec("Granite-3.x", "base", "ibm-granite/granite-3.0-2b-base"),
    CheckpointSpec("Granite-3.x", "instruct", "ibm-granite/granite-3.0-2b-instruct"),
    CheckpointSpec(
        "Granite-3.x", "abliterated",
        "mradermacher/granite-3.1-2b-instruct-abliterated-GGUF",
    ),
    # --- TinyLlama ---
    CheckpointSpec("TinyLlama", "base", "TinyLlama/TinyLlama_v1.1"),
    CheckpointSpec("TinyLlama", "instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    # NOTE: the safetensors-format TinyLlama abliterated derivatives found on
    # the Hub (kofimike/..., philippefunk/..., FaceWest/..., muyo123/...)
    # all ship with NO README.md / model card at all. Since a verbatim card
    # is load-bearing for this registry, we use the one abliterated
    # derivative repo that does carry a card (a GGUF re-export, one hop
    # further from the official chat checkpoint) and record quantisation
    # accordingly.
    CheckpointSpec(
        "TinyLlama", "abliterated",
        "mradermacher/abliterated-TinyLlama-1.1B-GGUF",
    ),
    # --- StableLM-2 --- (no abliterated derivative found on the Hub)
    CheckpointSpec("StableLM-2", "base", "stabilityai/stablelm-2-1_6b"),
    CheckpointSpec("StableLM-2", "instruct", "stabilityai/stablelm-2-zephyr-1_6b"),
    CheckpointSpec(
        "StableLM-2", "abliterated", None,
        missing_reason=(
            "No community abliterated/uncensored/amoral/Josiefied/NoWarning "
            "derivative of stabilityai/stablelm-2-zephyr-1_6b (or -1_6b) was "
            "found via HF Hub model search (queries: 'stablelm-2-zephyr-1_6b "
            "abliterated', 'stablelm-2-1_6b abliterated') as of build time."
        ),
    ),
    # --- OLMo-2 --- (no abliterated derivative found on the Hub)
    CheckpointSpec("OLMo-2", "base", "allenai/OLMo-2-0425-1B"),
    CheckpointSpec("OLMo-2", "instruct", "allenai/OLMo-2-0425-1B-Instruct"),
    CheckpointSpec(
        "OLMo-2", "abliterated", None,
        missing_reason=(
            "No community abliterated/uncensored derivative of "
            "allenai/OLMo-2-0425-1B-Instruct was found via HF Hub model "
            "search (query: 'OLMo-2-0425-1B abliterated') as of build time."
        ),
    ),
    # --- EXAONE-3.5 --- (no released non-instruct base checkpoint)
    CheckpointSpec(
        "EXAONE-3.5", "base", None,
        missing_reason=(
            "LG AI Research has not publicly released a non-instruct "
            "pretrained base checkpoint for EXAONE-3.5-2.4B on the HF Hub; "
            "only LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct (and its AWQ/GGUF "
            "re-exports) are published."
        ),
    ),
    CheckpointSpec("EXAONE-3.5", "instruct", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"),
    CheckpointSpec(
        "EXAONE-3.5", "abliterated",
        "mradermacher/EXAONE-3.5-2.4B-Instruct-abliterated-GGUF",
    ),
]


def _org_from_repo_id(repo_id: str) -> str:
    return repo_id.split("/", 1)[0]


def _retry(fn, *args, what: str, **kwargs):
    """Call fn(*args, **kwargs) with up to MAX_RETRIES attempts and backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError):
            raise  # not transient — do not retry
        except RETRYABLE_EXC as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                f"[{what}] attempt {attempt}/{MAX_RETRIES} failed ({exc!r}); "
                f"retrying in {wait}s"
            )
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _detect_quantisation(filenames: list[str], repo_id: str) -> str:
    """Classify quantisation strictly from the repo's actual weight files
    (filenames), never from README prose — a README routinely *mentions*
    'GGUF'/'AWQ' community re-exports of itself without the repo containing
    any such files, which would otherwise misclassify full-precision repos."""
    exts = {Path(f).suffix.lower() for f in filenames}
    full_precision_exts = {".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack"}
    has_full = bool(exts & full_precision_exts)
    has_gguf = ".gguf" in exts

    if has_full:
        weight_files = " ".join(
            f.lower() for f in filenames if Path(f).suffix.lower() in full_precision_exts
        )
        hay = weight_files + " " + repo_id.lower()
        label = "none (full precision)"
        for kw, kw_label in QUANT_KEYWORDS.items():
            if kw in hay:
                label = kw_label
                break
        if has_gguf:
            label += " (repo also ships separate GGUF copies)"
        return label
    if has_gguf:
        return "GGUF"
    return "unknown"


def _extract_parent_declared(card_data: dict | None) -> str | list[str] | None:
    if not card_data:
        return None
    base_model = card_data.get("base_model")
    return base_model


_HF_URL_REPO_RE = re.compile(
    r"huggingface\.co/([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*?)(?:[)\]\s\"'>]|$)"
)
_PARENT_HINT_KEYWORDS = (
    "abliterat", "uncensor", "version of", "finetuned from", "fine-tuned from",
    "derived from", "based on",
)
_NON_MODEL_PATH_SEGMENTS = (
    "/blog/", "/datasets/", "/spaces/", "/papers/", "/docs/", "/blob/", "/tree/",
)


def _scan_card_body_for_parent(card_text: str, own_repo_id: str) -> tuple[str, str] | None:
    """Fallback for cards with no structured `base_model` YAML field: look for
    a huggingface.co/<org>/<repo> URL on a line that reads like it is naming
    the parent this derivative was built from."""
    own_lower = own_repo_id.lower()
    for line in card_text.splitlines():
        low = line.lower()
        if not any(kw in low for kw in _PARENT_HINT_KEYWORDS):
            continue
        for match in _HF_URL_REPO_RE.finditer(line):
            candidate = match.group(1).rstrip("/.,")
            cand_low = candidate.lower()
            if cand_low == own_lower:
                continue
            if any(seg in f"/{cand_low}/" for seg in (s.strip("/") for s in _NON_MODEL_PATH_SEGMENTS)):
                continue
            if candidate.count("/") != 1:
                continue
            return candidate, line.strip()
    return None


def _quote_for_parent(card_text: str, parent_id: str) -> str | None:
    """Find a sentence/line in the README that literally names the parent
    repo, preferring the canonical `base_model:` YAML declaration (checking
    the line above it too, since YAML list syntax puts the id on its own
    line) over an incidental mention such as a license_link URL."""
    lines = card_text.splitlines()
    short_name = parent_id.split("/", 1)[-1]
    candidates = [parent_id, short_name]

    def _matches(line: str, cand: str) -> bool:
        return bool(line.strip()) and cand.lower() in line.lower()

    for cand in candidates:
        for i, line in enumerate(lines):
            if _matches(line, cand) and lines[max(0, i - 1)].strip().lower().startswith("base_model"):
                return line.strip().lstrip("-# ").strip()
            if line.strip().lower().startswith("base_model:") and _matches(line, cand):
                return line.strip()
    for cand in candidates:
        for line in lines:
            if _matches(line, cand):
                cleaned = line.strip().lstrip("-# ").strip()
                if cleaned:
                    return cleaned
    return None


def fetch_one(api: HfApi, spec: CheckpointSpec) -> dict[str, Any]:
    assert spec.repo_id is not None
    repo_id = spec.repo_id
    logger.info(f"Fetching {repo_id} ({spec.family}/{spec.condition})")

    row: dict[str, Any] = {
        "repo_id": repo_id,
        "family": spec.family,
        "condition": spec.condition,
        "org": _org_from_repo_id(repo_id),
        "available": True,
        "missing_reason": None,
        "availability_ok": False,
        "parameter_count": None,
        "parameter_count_source": None,
        "architectures": None,
        "num_hidden_layers": None,
        "hidden_size": None,
        "file_formats": [],
        "total_download_size_bytes": None,
        "quantisation": None,
        "license": None,
        "downloads": None,
        "likes": None,
        "last_modified": None,
        "sha": None,
        "gated": None,
        "card_text": "",
        "card_sha256": None,
        "card_fetched_utc": None,
        "parent_repo_declared": None,
        "parent_evidence_quote": None,
        "parent_declared_null_reason": (
            None if spec.condition == "abliterated" else "not applicable: condition != abliterated"
        ),
        "fetch_error": None,
    }

    try:
        info = _retry(
            api.model_info,
            repo_id,
            files_metadata=True,
            token=HF_TOKEN,
            what=f"model_info:{repo_id}",
        )
    except Exception as exc:
        logger.error(f"model_info failed permanently for {repo_id}: {exc!r}")
        row["fetch_error"] = f"model_info: {exc!r}"
        row["availability_ok"] = False
        return row

    siblings = info.siblings or []
    filenames = [s.rfilename for s in siblings]
    sizes = [s.size for s in siblings if s.size]
    row["file_formats"] = sorted({Path(f).suffix for f in filenames if Path(f).suffix})
    row["total_download_size_bytes"] = int(sum(sizes)) if sizes else None
    row["license"] = (info.card_data or {}).get("license") if info.card_data else None
    if not row["license"]:
        for tag in info.tags or []:
            if tag.startswith("license:"):
                row["license"] = tag.split(":", 1)[1]
                break
    row["downloads"] = info.downloads
    row["likes"] = info.likes
    row["last_modified"] = str(info.last_modified) if info.last_modified else None
    row["sha"] = info.sha
    row["gated"] = info.gated

    if info.safetensors is not None and info.safetensors.total:
        row["parameter_count"] = int(info.safetensors.total)
        row["parameter_count_source"] = "safetensors_metadata"

    cfg = info.config or {}
    architectures = cfg.get("architectures")
    if architectures:
        row["architectures"] = architectures

    # config.json — tiny, fetch for num_hidden_layers / hidden_size
    if "config.json" in filenames:
        try:
            cfg_path = _retry(
                hf_hub_download,
                repo_id,
                "config.json",
                token=HF_TOKEN,
                what=f"config.json:{repo_id}",
            )
            cfg_json = json.loads(Path(cfg_path).read_text())
            # most HF configs use num_hidden_layers/hidden_size; a few custom
            # architectures (e.g. EXAONE) use num_layers/hidden_size instead.
            row["num_hidden_layers"] = cfg_json.get("num_hidden_layers", cfg_json.get("num_layers"))
            row["hidden_size"] = cfg_json.get("hidden_size", cfg_json.get("n_embd"))
            if not row["architectures"] and cfg_json.get("architectures"):
                row["architectures"] = cfg_json["architectures"]
        except Exception as exc:
            logger.warning(f"config.json fetch/parse failed for {repo_id}: {exc!r}")

    # README.md — verbatim card text, load-bearing
    readme_name = next(
        (f for f in filenames if f.lower() == "readme.md"), "README.md"
    )
    try:
        readme_path = _retry(
            hf_hub_download,
            repo_id,
            readme_name,
            token=HF_TOKEN,
            what=f"README.md:{repo_id}",
        )
        card_text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
        row["card_text"] = card_text
        row["card_sha256"] = hashlib.sha256(card_text.encode("utf-8")).hexdigest()
        row["card_fetched_utc"] = datetime.now(timezone.utc).isoformat()
        row["availability_ok"] = True
    except Exception as exc:
        logger.error(f"README.md fetch failed for {repo_id}: {exc!r}")
        row["fetch_error"] = f"README.md: {exc!r}"
        row["availability_ok"] = False

    row["quantisation"] = _detect_quantisation(filenames, repo_id)

    if spec.condition == "abliterated":
        declared = _extract_parent_declared(info.card_data)
        if isinstance(declared, list):
            declared = declared[0] if declared else None
        if declared:
            row["parent_repo_declared"] = declared
            quote = _quote_for_parent(row["card_text"], declared)
            row["parent_evidence_quote"] = (
                quote if quote else f"YAML front-matter field base_model: {declared}"
            )
            row["parent_declared_null_reason"] = None
        else:
            body_hit = _scan_card_body_for_parent(row["card_text"], repo_id)
            if body_hit:
                candidate, quote_line = body_hit
                row["parent_repo_declared"] = candidate
                row["parent_evidence_quote"] = quote_line
                row["parent_declared_null_reason"] = None
            else:
                row["parent_repo_declared"] = None
                row["parent_evidence_quote"] = None
                row["parent_declared_null_reason"] = (
                    "base_model not present in the repo's card_data (YAML "
                    "front-matter) and no huggingface.co/<org>/<repo> URL "
                    "naming a parent was found near any of "
                    f"{_PARENT_HINT_KEYWORDS} in the README body."
                )

    return row


def missing_row(spec: CheckpointSpec) -> dict[str, Any]:
    return {
        "repo_id": None,
        "family": spec.family,
        "condition": spec.condition,
        "org": None,
        "available": False,
        "missing_reason": spec.missing_reason,
        "availability_ok": False,
        "parameter_count": None,
        "parameter_count_source": None,
        "architectures": None,
        "num_hidden_layers": None,
        "hidden_size": None,
        "file_formats": [],
        "total_download_size_bytes": None,
        "quantisation": None,
        "license": None,
        "downloads": None,
        "likes": None,
        "last_modified": None,
        "sha": None,
        "gated": None,
        "card_text": "",
        "card_sha256": None,
        "card_fetched_utc": None,
        "parent_repo_declared": None,
        "parent_evidence_quote": None,
        "parent_declared_null_reason": (
            spec.missing_reason
            if spec.condition == "abliterated"
            else "not applicable: row unavailable and condition != abliterated"
        ),
        "fetch_error": None,
    }


def _inherit_missing_fields(rows: list[dict[str, Any]]) -> None:
    """For GGUF-only abliterated rows lacking config.json (no param/arch
    metadata of their own), inherit parameter_count / architectures /
    num_hidden_layers / hidden_size from the same family's instruct row —
    abliteration and GGUF re-quantisation do not change these quantities."""
    by_family: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_family.setdefault(r["family"], {})[r["condition"]] = r

    for family, conditions in by_family.items():
        instruct = conditions.get("instruct")
        abl = conditions.get("abliterated")
        if not instruct or not abl or not abl.get("available"):
            continue
        if abl.get("parameter_count") is None and instruct.get("parameter_count") is not None:
            abl["parameter_count"] = instruct["parameter_count"]
            abl["parameter_count_source"] = (
                f"inherited_from_instruct_same_family:{instruct['repo_id']}"
            )
        if not abl.get("architectures") and instruct.get("architectures"):
            abl["architectures"] = instruct["architectures"]
        if abl.get("num_hidden_layers") is None and instruct.get("num_hidden_layers") is not None:
            abl["num_hidden_layers"] = instruct["num_hidden_layers"]
        if abl.get("hidden_size") is None and instruct.get("hidden_size") is not None:
            abl["hidden_size"] = instruct["hidden_size"]


def build_summary_matrix(rows: list[dict[str, Any]]) -> str:
    families = sorted({r["family"] for r in rows})
    conditions = ["base", "instruct", "abliterated"]
    col_w = 14
    header = "family".ljust(20) + "".join(c.ljust(col_w) for c in conditions)
    lines = [header, "-" * len(header)]
    for fam in families:
        cells = []
        for cond in conditions:
            match = next((r for r in rows if r["family"] == fam and r["condition"] == cond), None)
            if match is None:
                cells.append("N/A".ljust(col_w))
            elif not match["available"]:
                cells.append("missing".ljust(col_w))
            elif match["availability_ok"]:
                quant = match.get("quantisation") or ""
                q = "GGUF" if quant.startswith("GGUF") else "ok"
                cells.append(f"available({q})".ljust(col_w))
            else:
                cells.append("FETCH_FAIL".ljust(col_w))
        lines.append(fam.ljust(20) + "".join(cells))
    return "\n".join(lines)


@logger.catch(reraise=True)
def main() -> None:
    api = HfApi(token=HF_TOKEN)
    logger.info(f"huggingface_hub version: {__import__('huggingface_hub').__version__}")
    logger.info(f"Total spec rows: {len(SPECS)}")

    fetchable = [s for s in SPECS if s.repo_id is not None]
    unavailable = [s for s in SPECS if s.repo_id is None]
    logger.info(f"Fetchable (live) rows: {len(fetchable)}; explicit-missing rows: {len(unavailable)}")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, api, spec): spec for spec in fetchable}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                logger.error(f"Unrecoverable failure for {spec.repo_id}: {exc!r}")
                rows.append(
                    {
                        **missing_row(spec),
                        "available": True,
                        "repo_id": spec.repo_id,
                        "org": _org_from_repo_id(spec.repo_id),
                        "missing_reason": None,
                        "fetch_error": repr(exc),
                    }
                )

    for spec in unavailable:
        rows.append(missing_row(spec))

    _inherit_missing_fields(rows)

    # stable ordering: family (spec order), then condition order base/instruct/abliterated
    family_order = list(dict.fromkeys(s.family for s in SPECS))
    cond_rank = {"base": 0, "instruct": 1, "abliterated": 2}
    rows.sort(key=lambda r: (family_order.index(r["family"]), cond_rank[r["condition"]]))

    n_families = len({r["family"] for r in rows})
    n_checkpoints = len(rows)
    n_available = sum(1 for r in rows if r["available"])
    n_availability_ok = sum(1 for r in rows if r["availability_ok"])

    metadata = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "hf_api_version": __import__("huggingface_hub").__version__,
        "n_families": n_families,
        "n_checkpoints": n_checkpoints,
        "n_available": n_available,
        "n_availability_ok": n_availability_ok,
        "seed_note": (
            "Registry rows are a hand-curated, HF-Hub-search-verified list of "
            "small instruction-tuned model families x {base, instruct, "
            "abliterated} conditions; every field on an available row is a "
            "live HfApi().model_info()/config.json/README.md read at "
            "built_utc, not a guess. No model weights were downloaded — only "
            "config.json and README.md (a few KB each) per repo."
        ),
    }

    registry = {"metadata": metadata, "checkpoints": rows}

    out_json = REGISTRY_DIR / "checkpoint_registry.json"
    out_json.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

    csv_fields = [k for k in rows[0].keys() if k != "card_text"] + ["card_len"]
    out_csv = REGISTRY_DIR / "checkpoint_registry.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for r in rows:
            csv_row = {k: v for k, v in r.items() if k != "card_text"}
            csv_row["card_len"] = len(r.get("card_text") or "")
            for k, v in csv_row.items():
                if isinstance(v, (list, dict)):
                    csv_row[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(csv_row)
    logger.info(f"Wrote {out_csv} ({out_csv.stat().st_size / 1024:.1f} KB)")

    logger.info("Family x condition availability matrix:\n" + build_summary_matrix(rows))
    print()
    print(build_summary_matrix(rows))
    print()
    print(
        f"n_families={n_families} n_checkpoints={n_checkpoints} "
        f"n_available={n_available} n_availability_ok={n_availability_ok}"
    )


if __name__ == "__main__":
    main()
