#!/usr/bin/env python3
"""Assemble the frozen, standardised prompt sets from the downloaded sources.

Every output row uses one schema, so the screen can slice by `sub_study` without per-source
special-casing.  Nothing here computes a derived statistic and no model is run: this is a
freeze, not an analysis.
"""
from __future__ import annotations

import hashlib
import json
import random
import resource
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

import ladder as ladder_mod
import wrappers_and_forms as waf

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "temp" / "datasets" / "raw"
OUT = ROOT / "prompt_sets"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(LOGS / "prompt_sets.log", rotation="30 MB", level="DEBUG")

RAM_BUDGET = 12 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET, RAM_BUDGET))

SEED = 20260919
N_GT_BEHAVIOURS = 60        # harmful behaviours carried into the generation-scored ground truth
N_PARAPHRASE_ITEMS = 60     # harmful items that get the 6 surface-form variants
N_WRAPPED_ITEMS = 40        # harmful items that get the 8 attack wrappers

LICENSES = {
    "Paul/XSTest": "cc-by-4.0",
    "JailbreakBench/JBB-Behaviors": "mit",
    "bench-llm/or-bench": "cc-by-4.0",
    "allenai/coconot": "AI2 ImpACT Low-Risk (https://allenai.org/licenses/impact-lr); card notes constituent parts keep their own licences",
    "LibrAI/do-not-answer": "apache-2.0",
    "Bertievidgen/SimpleSafetyTests": "cc-by-2.0",
    "furonghuang-lab/PHTest": "mit",
    "SillyTilly/SorryBench": "UNDECLARED on the mirror (no licence field, no README); upstream SORRY-Bench is Xie et al. 2406.14598",
    "TrustAIRLab/in-the-wild-jailbreak-prompts": "mit",
    "PKU-Alignment/BeaverTails-Evaluation": "cc-by-nc-4.0",
    "walledai/MaliciousInstruct": "cc-by-sa-4.0",
    "walledai/WildGuardTest": "odc-by",
    "declare-lab/HarmfulQA": "apache-2.0",
    "TrustAIRLab/forbidden_question_set": "mit",
    "walledai/AyaRedTeaming": "apache-2.0",
    "advbench": "MIT (llm-attacks, Zou et al. 2307.15043)",
    "strongreject": "MIT (Souly et al. 2402.10260)",
    "harmbench": "MIT (Mazeika et al. 2402.04249)",
    "pipeline_authored": "CC0 - authored in this artifact, published in full",
    "harmlevelbench": "CC BY 4.0 (arXiv 2411.06835)",
}


def load(name: str) -> list[dict[str, Any]]:
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(f"missing downloaded source: {p}")
    return json.loads(p.read_text())


def read_csv(name: str) -> list[dict[str, str]]:
    import csv
    import io
    p = RAW / name
    if not p.exists():
        raise FileNotFoundError(f"missing downloaded source: {p}")
    return list(csv.DictReader(io.StringIO(p.read_text())))


def rid(*parts: object) -> str:
    return hashlib.sha1("||".join(str(x) for x in parts).encode()).hexdigest()[:16]


def row(**kw: Any) -> dict[str, Any]:
    """One canonical prompt-set row.  All keys always present so the schema is rectangular."""
    base: dict[str, Any] = {
        "row_id": None, "sub_study": None, "source": None, "source_ref": None,
        "license": None, "authored_by": "external", "prompt": None,
        "topic": None, "category": None, "label": None,
        "severity_level": None, "severity_scheme": None, "dose_t_uniform": None,
        "dose_t_compressed_tail": None, "pair_id": None, "pair_role": None,
        "base_row_id": None, "wrapper_id": None, "form_id": None,
        "affirmative_target": None, "notes": None,
    }
    base.update(kw)
    if base["row_id"] is None:
        base["row_id"] = rid(base["sub_study"], base["source"], base["source_ref"],
                             base["prompt"], base["wrapper_id"], base["form_id"])
    return base


# ------------------------------------------------------------------ sub-study builders ----
def build_ladder_authored() -> list[dict[str, Any]]:
    out = []
    for r in ladder_mod.ladder_rows():
        out.append(row(
            sub_study="LADDER_AUTHORED", source="pipeline_authored",
            source_ref=f"{r['topic']}:L{r['severity_level']}",
            license=LICENSES["pipeline_authored"], authored_by="pipeline_template",
            prompt=r["prompt"], topic=r["topic"], category="severity_ladder",
            label="benign" if r["severity_level"] == 0 else "graded_harmful",
            severity_level=r["severity_level"],
            severity_scheme="authored_8_rung_fixed_frame",
            dose_t_uniform=r["dose_t_uniform"],
            dose_t_compressed_tail=r["dose_t_compressed_tail"],
            pair_id=f"ladder::{r['topic']}", pair_role="rung",
            notes=("same syntactic frame across all 8 rungs; only the object varies. "
                   "Interior rungs are pipeline-authored, NOT externally rated."),
        ))
    return out


def build_ladder_harmlevelbench() -> list[dict[str, Any]]:
    """HarmLevelBench: the externally-authored 7 topics x 8 harm levels on a fixed 'How to ...?'
    frame.  Recovered from the paper by fetch_harmlevelbench.py because it was never released as
    a file; see that script and the payload's `caveat` field."""
    src = json.loads((RAW / "direct_harmlevelbench_table6.json").read_text())
    out = []
    for r in src["items"]:
        lvl0 = r["harm_level"] - 1            # re-index 1..8 -> 0..7 to match the other ladders
        out.append(row(
            sub_study="LADDER_HARMLEVELBENCH", source="harmlevelbench",
            source_ref=f'{r["topic"]}:L{r["harm_level"]}', license=LICENSES["harmlevelbench"],
            prompt=r["prompt"], topic=r["topic"], category="severity_ladder",
            label="benign" if lvl0 == 0 else "graded_harmful",
            severity_level=lvl0, severity_scheme="harmlevelbench_8_rung_external",
            dose_t_uniform=ladder_mod.RESPACINGS["uniform"][lvl0],
            dose_t_compressed_tail=ladder_mod.RESPACINGS["compressed_tail"][lvl0],
            pair_id=f'hlb::{r["topic"]}', pair_role="rung",
            notes=("HarmLevelBench Table 6 (Belkhiter et al., arXiv 2411.06835, CC BY 4.0); "
                   "externally authored, parsed from the paper - never released as a file"),
        ))
    return out


def build_ladder_external() -> list[dict[str, Any]]:
    """OR-Bench: the same 10 categories graded at 3 externally-authored levels."""
    out = []
    spec = [("or-bench-80k", 0, "seemingly-toxic-but-safe"),
            ("or-bench-hard-1k", 1, "hard borderline"),
            ("or-bench-toxic", 2, "toxic")]
    rng = random.Random(SEED)
    for cfg, level, desc in spec:
        rows = load(f"full_bench-llm__or-bench__{cfg}__train.json")
        by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r)
        for cat in sorted(by_cat):
            pool = sorted(by_cat[cat], key=lambda r: r["prompt"])
            take = pool if len(pool) <= 30 else rng.sample(pool, 30)
            for r in take:
                out.append(row(
                    sub_study="LADDER_EXTERNAL", source="bench-llm/or-bench",
                    source_ref=f"{cfg}:{cat}", license=LICENSES["bench-llm/or-bench"],
                    prompt=r["prompt"], topic=cat, category=cfg,
                    label={0: "benign", 1: "borderline", 2: "harmful"}[level],
                    severity_level=level, severity_scheme="orbench_3_level_external",
                    dose_t_uniform=level / 2.0,
                    notes=f"OR-Bench {cfg} ({desc}); externally authored ordinal level",
                ))
    return out


def build_benign_twin_pairs() -> list[dict[str, Any]]:
    """Externally-authored matched benign/harmful pairs: JBB (by Index) and XSTest (by focus)."""
    out = []
    harm = load("full_JailbreakBench__JBB-Behaviors__behaviors__harmful.json")
    ben = load("full_JailbreakBench__JBB-Behaviors__behaviors__benign.json")
    ben_by_idx = {r["Index"]: r for r in ben}
    for h in harm:
        b = ben_by_idx.get(h["Index"])
        pid = f"jbb::{h['Index']}"
        out.append(row(sub_study="BENIGN_TWIN_PAIRS", source="JailbreakBench/JBB-Behaviors",
                       source_ref=f"behaviors/harmful:{h['Index']}",
                       license=LICENSES["JailbreakBench/JBB-Behaviors"], prompt=h["Goal"],
                       topic=h["Category"], category=h["Behavior"], label="harmful",
                       pair_id=pid, pair_role="harmful", affirmative_target=h["Target"],
                       notes=f"JBB source={h['Source']}"))
        if b is not None:
            out.append(row(sub_study="BENIGN_TWIN_PAIRS", source="JailbreakBench/JBB-Behaviors",
                           source_ref=f"behaviors/benign:{b['Index']}",
                           license=LICENSES["JailbreakBench/JBB-Behaviors"], prompt=b["Goal"],
                           topic=b["Category"], category=b["Behavior"], label="benign",
                           pair_id=pid, pair_role="benign_twin",
                           affirmative_target=b["Target"],
                           notes="matched benign twin, same JBB Index/Behavior"))
    # XSTest pairs SAFE type T with UNSAFE type `contrast_T` by position within the type: the
    # i-th safe prompt and the i-th contrast prompt share a focus lexeme ("kill a Python process"
    # / "kill a person").  Verified: focus matches 25/25 for seven correspondences and 24/25 for
    # safe_contexts (one near-synonym, bank account fraud / bank fraud).
    #
    # Two safe types share their contrast set with another safe type, so only one of each pair of
    # safe types is item-level matched: contrast_discr matches real_group_nons_discr (25/25) and
    # NOT nons_group_real_discr (0/25); contrast_privacy matches privacy_fictional (25/25) and NOT
    # privacy_public (1/25).  Those two safe types are SET-level contrasts, so they are kept in
    # full with no pair_id rather than silently dropped.
    #
    # (An earlier version keyed on (type, focus), which collides whenever two prompts in a type
    # share a focus word - 22 distinct focus values across 25 homonym prompts - and silently lost
    # 108 of XSTest's 450 rows.  Position is the real key; focus is the verification.)
    PAIR_MAP = [
        ("homonyms", "contrast_homonyms"),
        ("figurative_language", "contrast_figurative_language"),
        ("safe_targets", "contrast_safe_targets"),
        ("safe_contexts", "contrast_safe_contexts"),
        ("definitions", "contrast_definitions"),
        ("historical_events", "contrast_historical_events"),
        ("real_group_nons_discr", "contrast_discr"),
        ("privacy_fictional", "contrast_privacy"),
    ]
    SET_LEVEL_ONLY = {
        "nons_group_real_discr": "contrast_discr",
        "privacy_public": "contrast_privacy",
    }

    xs = load("full_Paul__XSTest__default__train.json")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in xs:
        by_type[r["type"]].append(r)

    def _focus(r: dict[str, Any]) -> str:
        return (r.get("focus") or "").strip().lower()

    def _xs_row(r: dict[str, Any], lab: str, pid: str | None, role: str,
                note: str) -> dict[str, Any]:
        return row(sub_study="BENIGN_TWIN_PAIRS", source="Paul/XSTest",
                   source_ref=f"id:{r['id']}", license=LICENSES["Paul/XSTest"],
                   prompt=r["prompt"], topic=r["type"], category=r.get("note"),
                   label=lab, pair_id=pid, pair_role=role, notes=note)

    emitted: set[int] = set()
    for safe_t, unsafe_t in PAIR_MAP:
        safe_rows, unsafe_rows = by_type[safe_t], by_type[unsafe_t]
        if len(safe_rows) != len(unsafe_rows):
            raise ValueError(f"XSTest {safe_t}/{unsafe_t} differ in length: "
                             f"{len(safe_rows)} vs {len(unsafe_rows)}")
        for k, (sr, ur) in enumerate(zip(safe_rows, unsafe_rows)):
            pid = f"xstest::{safe_t}::{k}"
            matched = _focus(sr) == _focus(ur)
            note = (f"XSTest item-level pair by position in {safe_t}/{unsafe_t}; "
                    f"focus_match={'yes' if matched else 'no (near-synonym)'} "
                    f"({_focus(sr)!r} vs {_focus(ur)!r})")
            out.append(_xs_row(sr, "benign", pid, "benign_twin", note))
            out.append(_xs_row(ur, "harmful", pid, "harmful", note))
            emitted.add(sr["id"])
            emitted.add(ur["id"])
    for safe_t, shared in SET_LEVEL_ONLY.items():
        for r in by_type[safe_t]:
            out.append(_xs_row(
                r, "benign", None, "benign_unpaired",
                f"XSTest SET-level contrast only: {safe_t} shares its contrast type {shared} "
                f"with another safe type, so no item-level twin exists for this prompt"))
            emitted.add(r["id"])
    if len(emitted) != len(xs):
        raise ValueError(f"XSTest: emitted {len(emitted)} of {len(xs)} rows - rows were lost")
    return out


def build_behaviour_gt() -> list[dict[str, Any]]:
    """Harmful behaviours used for the generation-scored behavioural ground truth."""
    out = []
    sr = read_csv("direct_strongreject_small.csv")
    for i, r in enumerate(sr):
        out.append(row(sub_study="BEHAVIOUR_GT", source="strongreject_small",
                       source_ref=f"row:{i}", license=LICENSES["strongreject"],
                       prompt=r["forbidden_prompt"], topic=r.get("category"),
                       category=r.get("source"), label="harmful",
                       notes="StrongREJECT small (Souly et al. 2402.10260) - primary GT set"))
    adv = read_csv("direct_advbench_harmful_behaviors.csv")
    rng = random.Random(SEED + 1)
    for i, r in enumerate(rng.sample(adv, 60)):
        out.append(row(sub_study="BEHAVIOUR_GT", source="advbench", source_ref=f"row:{i}",
                       license=LICENSES["advbench"], prompt=r["goal"], topic="advbench",
                       category=None, label="harmful", affirmative_target=r["target"],
                       notes="AdvBench harmful_behaviors (Zou et al. 2307.15043)"))
    hb = read_csv("direct_harmbench_behaviors_text_all.csv")
    std = [r for r in hb if r.get("FunctionalCategory") == "standard"]
    rng2 = random.Random(SEED + 2)
    for r in rng2.sample(std, 60):
        out.append(row(sub_study="BEHAVIOUR_GT", source="harmbench",
                       source_ref=r["BehaviorID"], license=LICENSES["harmbench"],
                       prompt=r["Behavior"], topic=r["SemanticCategory"],
                       category=r["FunctionalCategory"], label="harmful",
                       notes="HarmBench standard behaviours (Mazeika et al. 2402.04249)"))
    for r in load("full_walledai__MaliciousInstruct__default__train.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="walledai/MaliciousInstruct",
                       source_ref=None, license=LICENSES["walledai/MaliciousInstruct"],
                       prompt=r["prompt"], topic="malicious_instruct", label="harmful",
                       notes="MaliciousInstruct (Huang et al. 2310.06987)"))
    for r in load("full_Bertievidgen__SimpleSafetyTests__default__test.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="Bertievidgen/SimpleSafetyTests",
                       source_ref=r["id"], license=LICENSES["Bertievidgen/SimpleSafetyTests"],
                       prompt=r["prompt"], topic=r["harm_area"], category=r["category"],
                       label="harmful", severity_scheme="sst_information_vs_instruction",
                       notes="SimpleSafetyTests; `category` separates info-seeking from action-instructions"))
    for r in load("full_PKU-Alignment__BeaverTails-Evaluation__default__test.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="PKU-Alignment/BeaverTails-Evaluation",
                       source_ref=str(r["category_id"]),
                       license=LICENSES["PKU-Alignment/BeaverTails-Evaluation"],
                       prompt=r["prompt"], topic=r["category"], label="harmful",
                       notes="BeaverTails-Evaluation (Ji et al. 2307.04657)"))
    for r in load("full_TrustAIRLab__forbidden_question_set__default__train.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="TrustAIRLab/forbidden_question_set",
                       source_ref=f"{r['content_policy_id']}:{r['q_id']}",
                       license=LICENSES["TrustAIRLab/forbidden_question_set"],
                       prompt=r["question"], topic=r["content_policy_name"], label="harmful",
                       notes="Forbidden-question set (Shen et al., CCS 2024)"))
    for r in load("full_declare-lab__HarmfulQA__default__train.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="declare-lab/HarmfulQA",
                       source_ref=str(r["id"]), license=LICENSES["declare-lab/HarmfulQA"],
                       prompt=r["question"], topic=r["topic"], category=r["subtopic"],
                       label="harmful", notes="HarmfulQA (Bhardwaj & Poria 2308.09662)"))
    for r in load("full_walledai__AyaRedTeaming__default__english.json"):
        out.append(row(sub_study="BEHAVIOUR_GT", source="walledai/AyaRedTeaming",
                       source_ref=None, license=LICENSES["walledai/AyaRedTeaming"],
                       prompt=r["prompt"], topic=r["harm_category"],
                       category=r.get("global_or_local"), label="harmful",
                       notes="Aya red-teaming, English split (Aakanksha et al. 2406.18682)"))
    return out


def build_overrefusal() -> list[dict[str, Any]]:
    out = []
    for r in load("full_furonghuang-lab__PHTest__default__train.json"):
        lvl = {"harmless": 0, "controversial": 1, "harmful": 2}.get(r["Harmfulness"])
        out.append(row(sub_study="OVERREFUSAL", source="furonghuang-lab/PHTest",
                       source_ref=str(r["ID"]), license=LICENSES["furonghuang-lab/PHTest"],
                       prompt=r["Request"], topic="pseudo_harmful", category=r["Harmfulness"],
                       label={"harmless": "benign", "controversial": "borderline",
                              "harmful": "harmful"}.get(r["Harmfulness"], "unknown"),
                       severity_level=lvl, severity_scheme="phtest_3_level_external",
                       notes="PHTest pseudo-harmful prompts (An et al. 2409.00598)"))
    for r in load("full_allenai__coconot__contrast__test.json"):
        out.append(row(sub_study="OVERREFUSAL", source="allenai/coconot",
                       source_ref=f"contrast:{r['id']}", license=LICENSES["allenai/coconot"],
                       prompt=r["prompt"], topic=r["category"], category=r["subcategory"],
                       label="benign",
                       notes="CoCoNot CONTRAST split: superficially similar but SHOULD be complied with"))
    for r in load("full_walledai__WildGuardTest__default__train.json"):
        if r["label"] == "unharmful":
            out.append(row(sub_study="OVERREFUSAL", source="walledai/WildGuardTest",
                           source_ref=None, license=LICENSES["walledai/WildGuardTest"],
                           prompt=r["prompt"], topic="wildguard_unharmful",
                           category=f"adversarial={r['adversarial']}", label="benign",
                           notes="WildGuardTest unharmful prompts (Han et al. 2406.18495)"))
    return out


def build_epistemic() -> list[dict[str, Any]]:
    """Non-safety abstention: the housekeeping denominator for the ratiometric candidate."""
    keep = {"Incomplete requests", "Unsupported requests", "Indeterminate requests"}
    out = []
    for r in load("full_allenai__coconot__original__test.json"):
        if r["category"] not in keep:
            continue
        out.append(row(sub_study="EPISTEMIC_ABSTENTION", source="allenai/coconot",
                       source_ref=f"original:{r['id']}", license=LICENSES["allenai/coconot"],
                       prompt=r["prompt"], topic=r["category"], category=r["subcategory"],
                       label="abstention_expected",
                       notes=("CoCoNot non-safety noncompliance (Brahman et al. 2407.12043): "
                              "the model should decline for EPISTEMIC, not safety, reasons")))
    for r in load("full_allenai__coconot__original__test.json"):
        if r["category"] != "Requests with safety concerns":
            continue
        out.append(row(sub_study="EPISTEMIC_ABSTENTION", source="allenai/coconot",
                       source_ref=f"original:{r['id']}", license=LICENSES["allenai/coconot"],
                       prompt=r["prompt"], topic=r["category"], category=r["subcategory"],
                       label="safety_refusal_expected",
                       notes=("CoCoNot safety-concern arm, shipped in the SAME sub-study so the "
                              "ratiometric candidate has a matched safety numerator")))
    return out


def build_wrappers(gt_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(SEED + 3)
    pool = [r for r in gt_rows if r["source"] in ("strongreject_small", "advbench")]
    pool = sorted(pool, key=lambda r: r["row_id"])
    items = rng.sample(pool, min(N_WRAPPED_ITEMS, len(pool)))
    out = []
    for it in items:
        for w in waf.WRAPPERS:
            out.append(row(sub_study="WRAPPERS", source="pipeline_authored",
                           source_ref=f"{w['wrapper_id']}::{it['row_id']}",
                           license=LICENSES["pipeline_authored"], authored_by="pipeline_template",
                           prompt=waf.apply_wrapper(w, it["prompt"], it["affirmative_target"]),
                           topic=it["topic"], category=w["family"], label="harmful",
                           base_row_id=it["row_id"], wrapper_id=w["wrapper_id"],
                           affirmative_target=it["affirmative_target"],
                           notes=f"wrapper ref: {w['reference']}"))
    templates = [row(sub_study="WRAPPER_TEMPLATES", source="pipeline_authored",
                     source_ref=w["wrapper_id"], license=LICENSES["pipeline_authored"],
                     authored_by="pipeline_template", prompt=w["template"],
                     topic="attack_wrapper", category=w["family"], label="template",
                     wrapper_id=w["wrapper_id"], notes=w["reference"]) for w in waf.WRAPPERS]
    return out, templates


def build_forms(gt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED + 4)
    pool = sorted([r for r in gt_rows if r["source"] == "advbench"], key=lambda r: r["row_id"])
    items = rng.sample(pool, min(N_PARAPHRASE_ITEMS, len(pool)))
    out = []
    for it in items:
        for v in waf.FORM_VARIANTS:
            out.append(row(sub_study="FORM_VARIANTS", source="pipeline_authored",
                           source_ref=f"{v['form_id']}::{it['row_id']}",
                           license=LICENSES["pipeline_authored"], authored_by="pipeline_template",
                           prompt=waf.apply_form(v, it["prompt"]), topic=it["topic"],
                           category=v["axis"], label="harmful", base_row_id=it["row_id"],
                           form_id=v["form_id"],
                           notes="meaning-preserving surface-form variant (no lexical substitution)"))
    return out


def build_external_form_axis() -> list[dict[str, Any]]:
    """SORRY-Bench: the SAME 450 instructions rendered in 21 externally-authored styles."""
    rows = load("full_SillyTilly__SorryBench__default__train.json")
    rng = random.Random(SEED + 5)
    qids = sorted({r["question_id"] for r in rows})
    keep_q = set(rng.sample(qids, min(90, len(qids))))
    out = []
    for r in rows:
        if r["question_id"] not in keep_q:
            continue
        out.append(row(sub_study="EXTERNAL_FORM_AXIS", source="SillyTilly/SorryBench",
                       source_ref=f"q{r['question_id']}:{r['prompt_style']}",
                       license=LICENSES["SillyTilly/SorryBench"], prompt=r["turns"][0],
                       topic=f"sorrybench_cat_{r['category']}", category=r["prompt_style"],
                       label="harmful", base_row_id=f"sorrybench::q{r['question_id']}",
                       form_id=r["prompt_style"],
                       notes=("SORRY-Bench (Xie et al. 2406.14598): 21 externally-authored prompt "
                              "styles over the same instruction; the independent counterpart to "
                              "this pipeline's FORM_VARIANTS")))
    return out


def build_inthewild_wrappers() -> list[dict[str, Any]]:
    rows = load("full_TrustAIRLab__in-the-wild-jailbreak-prompts__jailbreak_2023_12_25__train.json")
    rng = random.Random(SEED + 6)
    uniq = {r["prompt"]: r for r in rows}
    keep = rng.sample(sorted(uniq), min(150, len(uniq)))
    return [row(sub_study="WRAPPER_TEMPLATES", source="TrustAIRLab/in-the-wild-jailbreak-prompts",
                source_ref=uniq[p]["source"], license=LICENSES["TrustAIRLab/in-the-wild-jailbreak-prompts"],
                prompt=p, topic="in_the_wild_jailbreak", category=uniq[p]["platform"],
                label="template", notes="real in-the-wild jailbreak template (Shen et al., CCS 2024)")
            for p in keep]


def build_risk_taxonomy() -> list[dict[str, Any]]:
    out = []
    for r in load("full_LibrAI__do-not-answer__default__train.json"):
        out.append(row(sub_study="RISK_TAXONOMY", source="LibrAI/do-not-answer",
                       source_ref=str(r["id"]), license=LICENSES["LibrAI/do-not-answer"],
                       prompt=r["question"], topic=r["risk_area"], category=r["types_of_harm"],
                       label="harmful", severity_scheme="dna_5_risk_areas",
                       notes=f"specific_harms: {r['specific_harms']}"))
    return out


@logger.catch(reraise=True)
def main() -> None:
    gt = build_behaviour_gt()
    wrapped, wrapper_templates = build_wrappers(gt)
    parts: dict[str, list[dict[str, Any]]] = {
        "LADDER_AUTHORED": build_ladder_authored(),
        "LADDER_EXTERNAL": build_ladder_external(),
        "LADDER_HARMLEVELBENCH": build_ladder_harmlevelbench(),
        "BENIGN_TWIN_PAIRS": build_benign_twin_pairs(),
        "BEHAVIOUR_GT": gt,
        "OVERREFUSAL": build_overrefusal(),
        "EPISTEMIC_ABSTENTION": build_epistemic(),
        "WRAPPERS": wrapped,
        "WRAPPER_TEMPLATES": wrapper_templates + build_inthewild_wrappers(),
        "FORM_VARIANTS": build_forms(gt),
        "EXTERNAL_FORM_AXIS": build_external_form_axis(),
        "RISK_TAXONOMY": build_risk_taxonomy(),
    }
    rows: list[dict[str, Any]] = []
    for k in sorted(parts):
        rows.extend(parts[k])
        logger.info(f"{k}: {len(parts[k])} rows")

    # Two rows of the SORRY-Bench mirror carry an empty instruction upstream; an empty prompt
    # would silently become a zero-token forward pass, so they are dropped and counted.
    n_empty = sum(1 for r in rows if not (r["prompt"] or "").strip())
    rows = [r for r in rows if (r["prompt"] or "").strip()]
    seen: set[str] = set()
    dedup = []
    for r in rows:
        if r["row_id"] in seen:
            continue
        seen.add(r["row_id"])
        dedup.append(r)
    logger.info(f"total {len(dedup)} rows "
                f"({len(rows) - len(dedup)} duplicate row_ids, {n_empty} empty prompts dropped)")

    meta = {
        "artifact": "frozen safety prompt sets",
        "seed": SEED,
        "n_rows": len(dedup),
        "n_empty_prompts_dropped": n_empty,
        "sub_study_counts": dict(Counter(r["sub_study"] for r in dedup)),
        "source_counts": dict(Counter(r["source"] for r in dedup)),
        "authored_by_counts": dict(Counter(r["authored_by"] for r in dedup)),
        "label_counts": dict(Counter(r["label"] for r in dedup)),
        "schema": sorted(dedup[0].keys()),
        "judge": {
            "name": "StrongREJECT fine-grained rubric",
            "prompt_file": "temp/datasets/raw/direct_strongreject_evaluator_prompt.txt",
            "evaluator_file": "temp/datasets/raw/direct_strongreject_evaluator.py",
            "reference": "Souly et al., arXiv 2402.10260",
        },
        "severity_axes": {
            "authored_8_rung_fixed_frame": {
                "rungs": ladder_mod.N_RUNGS, "topics": ladder_mod.TOPIC_ORDER,
                "respacings": ladder_mod.RESPACINGS,
                "external": False,
                "caveat": ("interior rungs are pipeline-authored, not externally rated; use the "
                           "OR-Bench axis for any claim that needs an external ordering"),
            },
            "orbench_3_level_external": {
                "levels": 3, "external": True,
                "categories": 10,
                "note": "or-bench-80k -> or-bench-hard-1k -> or-bench-toxic, same 10 categories",
            },
            "harmlevelbench_8_rung_external": {
                "rungs": 8, "topics": 7, "external": True,
                "source": "arXiv 2411.06835 Table 6, CC BY 4.0, parsed by fetch_harmlevelbench.py",
                "note": ("the axis the study was designed around; recovered from the paper because "
                         "it was never released as a file. n=56, so it is an anchor for the "
                         "authored ladder rather than a standalone dose-response corpus"),
            },
            "phtest_3_level_external": {"levels": 2, "external": True,
                                        "note": "PHTest ships harmless + controversial only"},
        },
    }
    (OUT / "prompt_sets_full.json").write_text(
        json.dumps({"metadata": meta, "rows": dedup}, ensure_ascii=False))
    logger.info(f"wrote {OUT / 'prompt_sets_full.json'}")
    for k, v in meta["sub_study_counts"].items():
        logger.info(f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
