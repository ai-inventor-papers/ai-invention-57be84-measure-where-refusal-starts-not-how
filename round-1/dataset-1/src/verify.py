#!/usr/bin/env python3
"""Acceptance checks over both frozen assets.  Exits non-zero if any check fails."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
logger.remove()
logger.add(sys.stdout, level="INFO", format="{level:<7}|{message}")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    logger.info(f"{'PASS' if ok else 'FAIL':<5} {name} {detail}")
    if not ok:
        FAILS.append(name)


def main() -> None:
    reg = json.loads((ROOT / "registry" / "checkpoint_registry.json").read_text())
    cks = reg["checkpoints"]
    fams = {c["family"] for c in cks}
    check("registry: >=10 families", len(fams) >= 10, f"({len(fams)})")
    check("registry: >=30 rows", len(cks) >= 30, f"({len(cks)})")
    triad = {c["condition"]: c for c in cks if c["family"] == "Qwen3"}
    check("registry: Qwen3 triad present and usable",
          all(triad.get(k, {}).get("usable_for_forward_pass") for k in
              ("base", "instruct", "abliterated")),
          str({k: triad.get(k, {}).get("repo_id") for k in triad}))
    avail = [c for c in cks if c.get("available")]
    check("registry: every available row has verbatim card_text",
          all((c.get("card_text") or "").strip() for c in avail),
          f"({len(avail)} available rows)")
    check("registry: every available row pins a commit sha",
          all(c.get("sha") for c in avail))
    abl = [c for c in avail if c["condition"] == "abliterated"]
    check("registry: abliterated rows declare a parent or an explicit null reason",
          all(c.get("parent_repo_declared") or c.get("parent_declared_null_reason") for c in abl),
          f"({len(abl)} abliterated rows)")
    miss = [c for c in cks if not c.get("available")]
    check("registry: missing conditions recorded, not dropped",
          all(c.get("missing_reason") for c in miss), f"({len(miss)} explicit missing rows)")
    n_usable = sum(1 for c in cks if c.get("usable_for_forward_pass"))
    check("registry: >=20 checkpoints usable for a forward pass", n_usable >= 20, f"({n_usable})")

    split = json.loads((ROOT / "prompt_sets" / "frozen_split.json").read_text())
    held = set(split["assignment"]["HELD_OUT"])
    screen = set(split["assignment"]["SCREEN"])
    check("split: 3 held-out families", len(held) == 3, str(sorted(held)))
    check("split: SCREEN and HELD_OUT are disjoint and cover every family",
          not (held & screen) and (held | screen) == fams)
    check("split: Qwen3 is in SCREEN", "Qwen3" in screen)
    check("split: every held-out family has a complete usable triad",
          all(all(c.get("usable_for_forward_pass")
                  for c in cks if c["family"] == f and c["condition"] == k)
              for f in held for k in ("base", "instruct", "abliterated")))

    ps = json.loads((ROOT / "prompt_sets" / "prompt_sets_full.json").read_text())
    rows = ps["rows"]
    check("prompts: >=10000 rows", len(rows) >= 10000, f"({len(rows)})")
    check("prompts: row_ids unique", len({r["row_id"] for r in rows}) == len(rows))
    check("prompts: every row has non-empty prompt text",
          all((r["prompt"] or "").strip() for r in rows))
    schema = set(rows[0].keys())
    check("prompts: rectangular schema", all(set(r.keys()) == schema for r in rows),
          f"({len(schema)} fields)")
    need = {"LADDER_AUTHORED", "LADDER_EXTERNAL", "BENIGN_TWIN_PAIRS", "BEHAVIOUR_GT",
            "OVERREFUSAL", "EPISTEMIC_ABSTENTION", "WRAPPERS", "WRAPPER_TEMPLATES",
            "FORM_VARIANTS", "EXTERNAL_FORM_AXIS", "RISK_TAXONOMY",
            "LADDER_HARMLEVELBENCH"}
    have = {r["sub_study"] for r in rows}
    check("prompts: all 12 sub-studies present", need <= have, str(sorted(need - have)))
    hlb = [r for r in rows if r["sub_study"] == "LADDER_HARMLEVELBENCH"]
    check("prompts: HarmLevelBench is 7 topics x 8 rungs", len(hlb) == 56 and
          len({r["topic"] for r in hlb}) == 7 and
          all(sorted(r["severity_level"] for r in hlb if r["topic"] == t) == list(range(8))
              for t in {r["topic"] for r in hlb}))
    lad = [r for r in rows if r["sub_study"] == "LADDER_AUTHORED"]
    check("prompts: ladder is 7 topics x 8 rungs", len(lad) == 56 and
          len({r["topic"] for r in lad}) == 7 and
          all(Counter(r["topic"] for r in lad)[t] == 8 for t in {r["topic"] for r in lad}))
    check("prompts: every ladder topic has a level-0 benign anchor",
          len({r["topic"] for r in lad if r["severity_level"] == 0}) == 7)
    w = {r["wrapper_id"] for r in rows if r["sub_study"] == "WRAPPERS"}
    check("prompts: 8 attack wrappers instantiated", len(w) == 8, str(len(w)))
    f = {r["form_id"] for r in rows if r["sub_study"] == "FORM_VARIANTS"}
    check("prompts: 7 surface forms (identity + 6 variants)", len(f) == 7, str(len(f)))
    pair_ct = Counter(r["pair_id"] for r in rows
                      if r["sub_study"] == "BENIGN_TWIN_PAIRS" and r["pair_id"])
    check("prompts: >=300 complete matched benign/harmful pairs",
          sum(1 for v in pair_ct.values() if v == 2) >= 300,
          f"({sum(1 for v in pair_ct.values() if v == 2)} complete of {len(pair_ct)})")
    check("prompts: no pair_id has more than 2 members",
          all(v <= 2 for v in pair_ct.values()))
    xs_rows = [r for r in rows if r["source"] == "Paul/XSTest"]
    check("prompts: all 450 XSTest rows carried through", len(xs_rows) == 450,
          f"({len(xs_rows)})")
    check("prompts: XSTest pair roles are well formed",
          {r["pair_role"] for r in xs_rows} == {"benign_twin", "harmful", "benign_unpaired"},
          str(dict(Counter(r["pair_role"] for r in xs_rows))))
    ep = [r for r in rows if r["sub_study"] == "EPISTEMIC_ABSTENTION"]
    check("prompts: epistemic set carries both an abstention and a safety arm",
          {"abstention_expected", "safety_refusal_expected"} <= {r["label"] for r in ep},
          str(dict(Counter(r["label"] for r in ep))))
    check("prompts: pipeline-authored rows are labelled as such",
          all(r["authored_by"] == "pipeline_template"
              for r in rows if r["source"] == "pipeline_authored"))
    check("prompts: every row carries a licence string",
          all(r["license"] for r in rows))

    man = json.loads((ROOT / "temp" / "datasets" / "download_manifest.json").read_text())
    check("downloads: every source succeeded", all(m["ok"] for m in man),
          f"({sum(1 for m in man if m['ok'])}/{len(man)})")
    check("downloads: >=15 distinct datasets kept",
          len([m for m in man if m["ok"]]) >= 15, f"({len(man)})")
    judge = ROOT / "temp" / "datasets" / "raw" / "direct_strongreject_evaluator_prompt.txt"
    check("judge: StrongREJECT rubric captured verbatim",
          judge.exists() and "how convincing" in judge.read_text())

    # ---- the schema-standardised export -------------------------------------------------
    import re
    from jsonschema import Draft7Validator
    schema = json.loads(Path("/ai-inventor/.claude/skills/aii-json/schemas/"
                             "exp_sel_data_out.json").read_text())
    val = Draft7Validator(schema)
    key_re = re.compile(r"^metadata_[a-zA-Z_][a-zA-Z0-9_]*$")
    for fname, want_groups in (("full_data_out.json", 10), ("mini_data_out.json", 10)):
        doc = json.loads((ROOT / fname).read_text())
        errs = list(val.iter_errors(doc))
        check(f"{fname}: validates against exp_sel_data_out", not errs,
              str([e.message[:80] for e in errs[:2]]))
        groups = doc["datasets"]
        check(f"{fname}: {want_groups} dataset groups", len(groups) == want_groups,
              f"({len(groups)})")
        exs = [e for g in groups for e in g["examples"]]
        check(f"{fname}: no empty input or output",
              all(str(e["input"]).strip() and str(e["output"]).strip() for e in exs),
              f"({len(exs)} examples)")
        check(f"{fname}: only input/output/metadata_* keys",
              all(k in ("input", "output") or key_re.match(k) for e in exs for k in e))
        check(f"{fname}: forbidden per-example keys absent",
              not any(k in e for e in exs for k in ("split", "dataset", "context")))
        check(f"{fname}: group names unique",
              len({g["dataset"] for g in groups}) == len(groups))
        ordinal = [e for e in exs if e["metadata_target_type"] == "ordinal_severity"]
        check(f"{fname}: ordinal targets equal their severity level",
              all(e["output"] == str(e["metadata_severity_level"]) for e in ordinal),
              f"({len(ordinal)} ordinal examples)")
        beh = [e for e in exs if e["metadata_target_type"] == "expected_behaviour"]
        check(f"{fname}: behaviour targets are in the closed vocabulary",
              {e["output"] for e in beh} <= {"refuse", "comply", "abstain", "template"},
              str(sorted({e["output"] for e in beh})))
        check(f"{fname}: every example carries a fold and a sub-study",
              all("metadata_fold" in e and "metadata_sub_study" in e for e in exs))
        check(f"{fname}: no example is an aggregate of several rows",
              len(exs) == len({(e.get("metadata_row_id") or e.get("metadata_repo_id"))
                               for e in exs}),
              f"({len(exs)} examples, "
              f"{len({(e.get('metadata_row_id') or e.get('metadata_repo_id')) for e in exs})} ids)")

    check("export: preview_data_out.json exists and parses",
          (ROOT / "preview_data_out.json").exists()
          and bool(json.loads((ROOT / "preview_data_out.json").read_text())["datasets"]))
    sel = json.loads((ROOT / "full_data_out.json").read_text())
    sel_names = {g["dataset"] for g in sel["datasets"]}
    dropped = set(json.loads((ROOT / "selected_datasets.json").read_text())["not_selected"])
    check("selection: every selected group has a written reason",
          sel_names == set(sel["metadata"]["selected_reasons"]))
    check("selection: every dropped group has a written reason",
          dropped == set(sel["metadata"]["not_selected_reasons"]),
          str(sorted(dropped ^ set(sel["metadata"]["not_selected_reasons"]))))
    check("selection: selected and dropped groups do not overlap", not (sel_names & dropped))
    check("selection: every dropped row still lives in the frozen prompt sets",
          set(json.loads((ROOT / "selected_datasets.json").read_text())["all_counts"])
          == sel_names | dropped)
    pin = (ROOT / "pyproject.toml").read_text()
    check("pyproject: every dependency is pinned with ==",
          pin.count("==") >= 40 and ">=" not in pin.split("dependencies = [")[1],
          f"({pin.split('dependencies = [')[1].count('==')} pins)")
    for f in ("full_data_out.json", "mini_data_out.json", "preview_data_out.json"):
        sz = (ROOT / f).stat().st_size
        check(f"size: {f} under the 100MB limit", sz < 100 * 1024**2, f"({sz / 1e6:.2f} MB)")
    sel_exs = [e for g in sel["datasets"] for e in g["examples"]]
    need_studies = {"LADDER_HARMLEVELBENCH", "LADDER_AUTHORED", "LADDER_EXTERNAL",
                    "BENIGN_TWIN_PAIRS", "BEHAVIOUR_GT", "OVERREFUSAL",
                    "EPISTEMIC_ABSTENTION", "WRAPPERS", "FORM_VARIANTS", "EXTERNAL_FORM_AXIS"}
    have_studies = {e["metadata_sub_study"] for e in sel_exs}
    check("selection: the ten still cover every readout the screen needs",
          need_studies <= have_studies, str(sorted(need_studies - have_studies)))

    logger.info("")
    if FAILS:
        logger.error(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        sys.exit(1)
    logger.info("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
