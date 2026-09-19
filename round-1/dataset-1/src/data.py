#!/usr/bin/env python3
"""Standardise the frozen assets into the `exp_sel_data_out` schema.

Input  : the frozen assets already built in this workspace
           prompt_sets/prompt_sets_full.json   (15,316 standardised prompt rows)
           registry/checkpoint_registry.json   (39 checkpoints, verbatim model cards)
           prompt_sets/frozen_split.json       (the once-drawn held-out family split)
Output : full_data_out.json       the ten selected dataset groups, one example per ROW
         selected_datasets.json   the selection decision, with the reason per group

Only the ten selected groups are exported.  Nothing is lost by that: every source row stays in
prompt_sets/prompt_sets_full.json and every checkpoint in registry/checkpoint_registry.json, and
selected_datasets.json records the example count and the reason for each group NOT selected.

One example = one data row.  A prompt is one example; a checkpoint is one example.  Nothing is
aggregated: a 9,450-row corpus becomes 9,450 examples.

`output` is the row's supervision TARGET, and `metadata_target_type` says which kind it is,
because this artifact carries three genuinely different targets and collapsing them would be a
silent lie:
  * `ordinal_severity`   - the severity ladders.  Their target is the rung, not a binary verdict;
                           where refusal *should* start on that rung is the quantity the dose fit
                           estimates, so asserting "refuse" here would pre-judge the experiment.
  * `expected_behaviour` - refuse / comply / abstain, for rows where the correct behaviour is not
                           in dispute (harmful behaviour sets, over-refusal sets, abstention sets).
  * `condition`          - base / instruct / abliterated, for the checkpoint registry, whose
                           `input` is the verbatim model card: that group IS the name-and-card
                           regex baseline's task.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "data.log", rotation="30 MB", level="DEBUG")

PROMPTS = ROOT / "prompt_sets" / "prompt_sets_full.json"
REGISTRY = ROOT / "registry" / "checkpoint_registry.json"
SPLIT = ROOT / "prompt_sets" / "frozen_split.json"

LADDER_STUDIES = {"LADDER_AUTHORED", "LADDER_EXTERNAL", "LADDER_HARMLEVELBENCH"}

# Sub-study -> integer fold id.  The strategy specifies metadata_fold as the tag for "which
# sub-study this row serves", and the schema wants an int, so the name is kept alongside in
# metadata_sub_study rather than being squeezed into the int.
SUB_STUDY_FOLD = {
    "LADDER_HARMLEVELBENCH": 0,
    "LADDER_AUTHORED": 1,
    "LADDER_EXTERNAL": 2,
    "BENIGN_TWIN_PAIRS": 3,
    "BEHAVIOUR_GT": 4,
    "OVERREFUSAL": 5,
    "EPISTEMIC_ABSTENTION": 6,
    "WRAPPERS": 7,
    "WRAPPER_TEMPLATES": 8,
    "FORM_VARIANTS": 9,
    "EXTERNAL_FORM_AXIS": 10,
    "RISK_TAXONOMY": 11,
    "CHECKPOINT_REGISTRY": 12,
}

# label -> the behaviour a correctly-aligned model should show.  Only used where that is not in
# dispute; ladder rows never reach this map.
BEHAVIOUR = {
    "harmful": "refuse",
    "benign": "comply",
    "borderline": "comply",          # over-refusal sets: complying IS the correct behaviour
    "abstention_expected": "abstain",
    "safety_refusal_expected": "refuse",
    "template": "template",          # an attack scaffold, not a prompt to be answered
}

# The best-10 selection, decided against what the screen actually needs.  Every candidate readout
# and every baseline must have its inputs covered; a group is kept when it is the only source, or
# the strongest source, of something the screen cannot run without.
SELECTED: dict[str, str] = {
    "harmlevelbench": (
        "C1 anchored dose axis: the study's intended externally-authored ladder, 7 topics x 8 "
        "harm levels on a fixed 'How to ...?' frame. Nothing else in the artifact grades matched "
        "topics on 8 external rungs."),
    "pipeline_authored": (
        "C1 + C4 + the wrapper battery: the syntax-controlled 7x8 ladder (only the object varies), "
        "the 8 attack wrappers and the 6 meaning-preserving surface forms. The only source where "
        "sentence frame is held fixed across the dose axis."),
    "JailbreakBench/JBB-Behaviors": (
        "the matched benign/harmful pair set with affirmative-prefix targets: supplies both the "
        "within-condition contrast the selection rule is decided on and the prefill string the "
        "C2 write-gain wrapper needs."),
    "Paul/XSTest": (
        "the second, independent matched-pair source (safe/unsafe sharing a focus lexeme) and the "
        "canonical over-refusal set. Pairing on a shared lexeme is a stronger control than pairing "
        "on topic alone."),
    "allenai/coconot": (
        "C3 ratiometric: the only source that supplies epistemic (non-safety) abstention prompts "
        "AND a safety-concern arm from the same corpus and annotation process, plus a contrast "
        "split of look-alike prompts that should be complied with."),
    "SillyTilly/SorryBench": (
        "C4 form sensitivity, externally authored: 21 published prompt styles over the same "
        "instruction. The independent counterpart to this pipeline's own form variants, without "
        "which C4 could only be measured on transforms we wrote ourselves."),
    "strongreject_small": (
        "the primary generation-scored behavioural ground truth, and the source of the verbatim "
        "judge rubric the screen scores completions with."),
    "advbench": (
        "the base item pool the wrappers and form variants are instantiated on, with affirmative "
        "targets; the most widely used harmful-behaviour set, so the GT is comparable to prior work."),
    "harmbench": (
        "second established GT source with semantic categories, so behavioural ground truth is not "
        "sourced from a single benchmark's idiosyncrasies."),
    "bench-llm/or-bench": (
        "the external 3-level ordinal severity axis over 10 shared categories - the only externally "
        "authored grading with enough items per cell to check the ladder's ordering at scale - "
        "plus the large-scale over-refusal pool."),
}

RUNNERS_UP: dict[str, str] = {
    "checkpoint_registry": (
        "shipped as its own asset (registry/checkpoint_registry.json, with the verbatim card "
        "text): it is the B2 name-and-card regex baseline's entire input, but it is a model "
        "registry rather than a prompt corpus, so it is not one of the ten prompt datasets."),
    "furonghuang-lab/PHTest": "ordinal pseudo-harmful levels, but only 2 of its 3 levels are populated",
    "walledai/WildGuardTest": "vanilla-vs-adversarial over-refusal, already covered by XSTest + OR-Bench",
    "LibrAI/do-not-answer": "risk taxonomy without an ordinal axis; GT already covered",
    "TrustAIRLab/in-the-wild-jailbreak-prompts": "real wrapper templates, but the 8 authored wrappers are the pre-registered battery",
    "declare-lab/HarmfulQA": "topic/subtopic GT, redundant with AdvBench + HarmBench",
    "PKU-Alignment/BeaverTails-Evaluation": "category-labelled GT, redundant; CC BY-NC licence",
    "walledai/AyaRedTeaming": "human-written GT, redundant for an English-only dose axis",
    "TrustAIRLab/forbidden_question_set": "policy-grid GT, redundant",
    "walledai/MaliciousInstruct": "100-item GT, redundant",
    "Bertievidgen/SimpleSafetyTests": "100-item GT with a 2-level info-vs-action contrast, too coarse to add an axis",
}


def _clean(value: Any) -> Any:
    """Drop empties so no example carries a metadata key with nothing behind it."""
    return value if value not in (None, "", [], {}) else None


def prompt_example(r: dict[str, Any]) -> dict[str, Any]:
    sub = r["sub_study"]
    if sub in LADDER_STUDIES:
        target, target_type = str(r["severity_level"]), "ordinal_severity"
    else:
        target, target_type = BEHAVIOUR.get(r["label"] or "", "unknown"), "expected_behaviour"

    ex: dict[str, Any] = {
        "input": r["prompt"],
        "output": target,
        "metadata_target_type": target_type,
        "metadata_fold": SUB_STUDY_FOLD[sub],
        "metadata_sub_study": sub,
        "metadata_row_id": r["row_id"],
        "metadata_authored_by": r["authored_by"],
        "metadata_license": r["license"],
    }
    optional = {
        "metadata_source_ref": r["source_ref"],
        "metadata_topic": r["topic"],
        "metadata_category": r["category"],
        "metadata_label": r["label"],
        "metadata_severity_level": r["severity_level"],
        "metadata_severity_scheme": r["severity_scheme"],
        "metadata_dose_t_uniform": r["dose_t_uniform"],
        "metadata_dose_t_compressed_tail": r["dose_t_compressed_tail"],
        "metadata_pair_id": r["pair_id"],
        "metadata_pair_role": r["pair_role"],
        "metadata_base_row_id": r["base_row_id"],
        "metadata_wrapper_id": r["wrapper_id"],
        "metadata_form_id": r["form_id"],
        "metadata_affirmative_target": r["affirmative_target"],
        "metadata_notes": r["notes"],
    }
    for k, v in optional.items():
        cv = _clean(v)
        if cv is not None:
            ex[k] = cv
    if target_type == "ordinal_severity":
        ex["metadata_expected_behaviour"] = (
            "comply" if r["severity_level"] == 0 else "unknown_by_design")
        ex["metadata_ordinal_note"] = (
            "target is the rung, not a verdict: where refusal should begin on this axis is the "
            "quantity the dose fit estimates")
    return ex


def registry_example(c: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    # `input` is the verbatim card plus the repo name: exactly, and only, what the honest
    # name-and-card regex baseline is allowed to see.
    payload = {"repo_id": c["repo_id"], "model_card_text": c["card_text"]}
    ex: dict[str, Any] = {
        "input": json.dumps(payload, ensure_ascii=False),
        "output": c["condition"],
        "metadata_target_type": "condition",
        "metadata_fold": SUB_STUDY_FOLD["CHECKPOINT_REGISTRY"],
        "metadata_sub_study": "CHECKPOINT_REGISTRY",
        "metadata_family": c["family"],
        "metadata_frozen_split": split["per_family"][c["family"]],
        "metadata_repo_id": c["repo_id"],
        "metadata_revision_sha": c["sha"],
        "metadata_org": c["org"],
        "metadata_usable_for_forward_pass": bool(c.get("usable_for_forward_pass")),
        "metadata_quantisation_verified": c.get("quantisation_verified"),
        "metadata_min_load_bytes": c.get("min_load_bytes"),
        "metadata_gated_blocking": c.get("gated_blocking"),
        "metadata_card_sha256": c.get("card_sha256"),
        "metadata_license": c.get("license"),
        "metadata_downloads": c.get("downloads"),
        "metadata_likes": c.get("likes"),
        "metadata_num_hidden_layers": c.get("num_hidden_layers"),
        "metadata_hidden_size": c.get("hidden_size"),
        "metadata_parameter_count": c.get("parameter_count"),
        "metadata_parent_repo_declared": c.get("parent_repo_declared"),
        "metadata_parent_repo_inferred_instruct": c.get("parent_repo_inferred_instruct"),
        "metadata_parent_declared_matches_inferred": c.get("parent_declared_matches_inferred"),
    }
    return {k: v for k, v in ex.items() if v is not None or k in ("input", "output")}


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = json.loads(PROMPTS.read_text())["rows"]
    reg = json.loads(REGISTRY.read_text())
    split = json.loads(SPLIT.read_text())
    logger.info(f"loaded {len(rows)} prompt rows and {len(reg['checkpoints'])} checkpoints")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r["source"], []).append(prompt_example(r))
    grouped["checkpoint_registry"] = [
        registry_example(c, split) for c in reg["checkpoints"] if c.get("available")]

    datasets = [{"dataset": name, "examples": grouped[name]} for name in sorted(grouped)]
    for d in datasets:
        logger.info(f"  {d['dataset']:<46} {len(d['examples']):>6} examples")

    meta = {
        "artifact": "model registry and frozen safety prompt sets",
        "built_from": {
            "prompt_sets": str(PROMPTS.relative_to(ROOT)),
            "registry": str(REGISTRY.relative_to(ROOT)),
            "frozen_split": str(SPLIT.relative_to(ROOT)),
        },
        "n_groups": len(datasets),
        "n_examples": sum(len(d["examples"]) for d in datasets),
        "one_example_is": "one prompt row, or one checkpoint in the registry group",
        "target_types": {
            "ordinal_severity": "severity ladders; output is the rung index as a string",
            "expected_behaviour": "refuse / comply / abstain / template",
            "condition": "base / instruct / abliterated, from the verbatim model card",
        },
        "metadata_fold": {"meaning": "sub-study id", "map": SUB_STUDY_FOLD},
        "frozen_split": split["assignment"],
        "held_out_rule": "HELD_OUT families are not loaded in iteration 1 under any circumstance",
    }
    return datasets, meta


def summarise(datasets: list[dict[str, Any]], label: str) -> None:
    n = sum(len(d["examples"]) for d in datasets)
    tt = Counter(e["metadata_target_type"] for d in datasets for e in d["examples"])
    ss = Counter(e["metadata_sub_study"] for d in datasets for e in d["examples"])
    logger.info(f"{label}: {len(datasets)} groups, {n} examples")
    logger.info(f"   target types: {dict(tt)}")
    logger.info(f"   sub-studies:  {dict(sorted(ss.items()))}")


@logger.catch(reraise=True)
def main() -> None:
    datasets, meta = build()
    missing = [k for k in SELECTED if k not in {d["dataset"] for d in datasets}]
    if missing:
        raise ValueError(f"selected groups absent from the built data: {missing}")
    chosen = [d for d in datasets if d["dataset"] in SELECTED]
    sel_meta = dict(meta)
    sel_meta.update({
        "selection": "the best ten groups for the iteration-1 screen",
        "selection_rule": ("a group is kept when it is the only source, or the strongest source, "
                           "of an input the screen cannot run without; redundant ground-truth "
                           "corpora are dropped rather than stacked"),
        "selected_reasons": SELECTED,
        "not_selected_reasons": RUNNERS_UP,
        "n_groups": len(chosen),
        "n_examples": sum(len(d["examples"]) for d in chosen),
        "not_selected_counts": {d["dataset"]: len(d["examples"]) for d in datasets
                                if d["dataset"] not in SELECTED},
        "every_row_also_lives_in": "prompt_sets/prompt_sets_full.json",
    })
    (ROOT / "full_data_out.json").write_text(
        json.dumps({"metadata": sel_meta, "datasets": chosen}, ensure_ascii=False))
    summarise(chosen, "full_data_out.json")

    (ROOT / "selected_datasets.json").write_text(json.dumps({
        "selected": SELECTED, "not_selected": RUNNERS_UP,
        "selected_counts": {d["dataset"]: len(d["examples"]) for d in chosen},
        "all_counts": {d["dataset"]: len(d["examples"]) for d in datasets},
    }, indent=1, ensure_ascii=False))
    dropped = [d for d in datasets if d["dataset"] not in SELECTED]
    logger.info(f"not exported ({len(dropped)} groups, "
                f"{sum(len(d['examples']) for d in dropped)} examples): "
                f"{[d['dataset'] for d in dropped]}")
    logger.info("wrote full_data_out.json, selected_datasets.json")


if __name__ == "__main__":
    main()
