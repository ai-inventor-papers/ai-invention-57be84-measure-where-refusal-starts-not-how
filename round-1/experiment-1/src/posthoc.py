#!/usr/bin/env python3
"""POST-FREEZE analyses that only make claims STRICTER or add a reported dimension.

None of this can rescue a candidate: the promotion rule in `analysis.run_screen`
is untouched, and every test here either (a) replaces a loose check with the
stricter one it already claimed to be, or (b) reports an ADDITIONAL outcome.

Three additions:

1. `triad_strict` -- the shipped `_triad` in finalize.py sets `prediction_held`
   from the ORDERING of three condition means alone. The pre-registered
   prediction has two further clauses it never tested: "with non-overlapping
   bootstrap CIs" and "while E_instruct ~ E_abliterated overlap". This computes
   all three clauses with a cluster bootstrap over FAMILIES and reports each
   separately. It can only turn a `True` into a `False`.

2. `write_gain_vs_random_paired` -- the shipped secondary test reports the
   CORRELATION between W_self and W_random across checkpoints. That is not the
   control the mech-interp field asks for: a steering mean effect needs a
   matched-norm random-direction arm showing the real direction moves the
   readout MORE (handbook rule "claim a steering result from a mean effect at
   one coefficient" -> report the per-sample distribution and a random
   perturbation control). This adds the paired magnitude test.

3. `screen_secondary_outcome` -- the screen scores candidates against Y_unsafe
   only. A metric that predicts unsafety but is blind to over-refusal is half a
   safety instrument, so the identical LOFO machinery is re-run with
   Y_overrefuse as the outcome and reported alongside, never pooled.

Nothing here edits candidates.py / groundtruth.py / analysis.py / core.py /
corpora.py; `analysis` is imported read-only so the frozen sha256 still holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

import analysis as AN
from core import WORKSPACE, clean_floats, jdump, setup_logging

RESULTS = WORKSPACE / "results"
N_BOOT = 2000

CANDS = {"C1_dose_axis": "C1_primary", "C1b_dose_slope": "C1_tobit_b",
         "C2_write_gain": "C2_W_self",
         "C3_ratiometric": "C3_S_minus_Epi", "C4_paraphrase_cv": "C4_cv",
         "C5_weights_only": "C5_w1"}
BASES = {"B1_logit_gap": "B1_logit_gap", "B2a_full_regex": "B2a_full_regex",
         "B2b_namefree_regex": "B2b_namefree_regex",
         "B3_anchor_projection": "B3_anchor_projection"}


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _cluster_boot_mean(vals: np.ndarray, fam: np.ndarray, *, seed: int = 0,
                       n_boot: int = N_BOOT) -> dict:
    """Mean with a cluster bootstrap over FAMILIES (resample families with
    replacement, keep all of their checkpoints)."""
    vals = np.asarray(vals, float)
    fam = np.asarray(fam)
    ok = np.isfinite(vals)
    vals, fam = vals[ok], fam[ok]
    if len(vals) < 2:
        return {"mean": float(vals.mean()) if len(vals) else None,
                "ci95": None, "n": int(len(vals)), "n_families": int(len(set(fam)))}
    fams = np.unique(fam)
    groups = [vals[fam == f] for f in fams]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        bs[i] = np.concatenate([groups[j] for j in idx]).mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)],
            "n": int(len(vals)), "n_families": int(len(fams)),
            "values": [float(v) for v in vals]}


def _ci_overlap(a: dict, b: dict) -> bool | None:
    ca, cb = a.get("ci95"), b.get("ci95")
    if ca is None or cb is None:
        return None
    return bool(ca[0] <= cb[1] and cb[0] <= ca[1])


# ------------------------------------------------------- 1. strict triad ----
def triad_strict(rows: list[dict]) -> dict:
    """Test EVERY clause of the pre-registered triad prediction, not just the
    ordering of the means."""
    out: dict = {
        "prediction": ("W_instruct > W_base ~ W_abliterated with non-overlapping "
                       "bootstrap CIs, while E_instruct ~ E_abliterated overlap"),
        "why_this_exists": (
            "finalize._triad sets prediction_held from the ordering of three "
            "condition means only; it never tested the 'non-overlapping bootstrap "
            "CIs' clause nor the E-overlap clause. Both are tested here with a "
            "cluster bootstrap over families. This can only make the claim "
            "stricter."),
    }
    for stat, key in (("W", "C2_W_self"), ("E", "C2_E")):
        by: dict[str, dict] = {}
        for cond in ("base", "instruct", "abliterated"):
            sel = [r for r in rows
                   if r.get("condition") == cond and _f(r.get(key)) is not None]
            if not sel:
                by[cond] = {"mean": None, "ci95": None, "n": 0}
                continue
            by[cond] = _cluster_boot_mean(
                np.array([_f(r[key]) for r in sel], float),
                np.array([r["family"] for r in sel]),
                seed=abs(hash(stat + cond)) % (2**31))
        out[f"{stat}_by_condition"] = by

    w, e = out["W_by_condition"], out["E_by_condition"]

    def _m(d, c):
        return d.get(c, {}).get("mean")

    ord_ok = bool(_m(w, "instruct") is not None and _m(w, "abliterated") is not None
                  and _m(w, "base") is not None
                  and _m(w, "instruct") > _m(w, "abliterated")
                  and _m(w, "instruct") > _m(w, "base"))
    ovl_abl = _ci_overlap(w.get("instruct", {}), w.get("abliterated", {}))
    ovl_base = _ci_overlap(w.get("instruct", {}), w.get("base", {}))
    e_ovl = _ci_overlap(e.get("instruct", {}), e.get("abliterated", {}))

    out["clause_1_W_ordering"] = {
        "statement": "W_instruct > W_abliterated and W_instruct > W_base (means)",
        "holds": ord_ok}
    out["clause_2_W_CIs_non_overlapping"] = {
        "statement": "instruct's W CI does not overlap abliterated's or base's",
        "instruct_vs_abliterated_overlap": ovl_abl,
        "instruct_vs_base_overlap": ovl_base,
        "holds": bool(ovl_abl is False and ovl_base is False)}
    out["clause_3_E_instruct_overlaps_abliterated"] = {
        "statement": "E_instruct CI overlaps E_abliterated CI (encode gain is NOT "
                     "what abliteration removes)",
        "overlap": e_ovl,
        "holds": bool(e_ovl is True)}
    out["prediction_held_strict"] = bool(
        out["clause_1_W_ordering"]["holds"]
        and out["clause_2_W_CIs_non_overlapping"]["holds"]
        and out["clause_3_E_instruct_overlaps_abliterated"]["holds"])
    out["prediction_held_ordering_only_as_shipped"] = ord_ok
    return out


# --------------------------------- 2. write gain vs random-direction arm ----
def write_gain_vs_random_paired(rows: list[dict]) -> dict:
    """Paired, per-checkpoint: does the SELF-derived refusal direction move the
    readout more than a matched-norm RANDOM direction?"""
    sel = [r for r in rows
           if _f(r.get("C2_W_self")) is not None and _f(r.get("C2_W_random")) is not None]
    if len(sel) < 3:
        return {"status": "INSUFFICIENT_DATA", "n": len(sel)}
    fam = np.array([r["family"] for r in sel])
    ws = np.array([_f(r["C2_W_self"]) for r in sel], float)
    wr = np.array([_f(r["C2_W_random"]) for r in sel], float)

    d_abs = np.abs(ws) - np.abs(wr)
    d_signed = ws - wr

    # per-checkpoint CI separation, read straight off the stored bootstrap arms
    sep = []
    for r in sel:
        arms = (r.get("C2") or {}).get("arms") or {}
        a, b = arms.get("self") or {}, arms.get("random_control") or {}
        ca, cb = a.get("W_ci95"), b.get("W_ci95")
        sep.append(None if (ca is None or cb is None)
                   else bool(not (ca[0] <= cb[1] and cb[0] <= ca[1])))
    n_sep = sum(1 for s in sep if s is True)
    n_known = sum(1 for s in sep if s is not None)

    return {
        "test": ("paired per-checkpoint |W_self| - |W_random|, cluster bootstrap "
                 "over families + paired sign test"),
        "why_this_exists": (
            "the shipped secondary test reports only the CORRELATION between "
            "W_self and W_random across checkpoints, which is not a control. A "
            "steering claim needs the real direction to move the readout MORE "
            "than a matched-norm random one."),
        "mean_W_self": float(ws.mean()), "mean_W_random": float(wr.mean()),
        "paired_abs_difference": {
            **AN.cluster_bootstrap_mean(d_abs, fam, seed=11),
            "sign_test": AN.paired_sign_test(-d_abs),
            "interpretation": "positive mean = the real direction moves R more"},
        "paired_signed_difference": {
            **AN.cluster_bootstrap_mean(d_signed, fam, seed=12)},
        "per_checkpoint_CI_separation": {
            "n_checkpoints_with_non_overlapping_self_vs_random_CIs": n_sep,
            "n_checkpoints_with_both_CIs_available": n_known,
            "fraction": (round(n_sep / n_known, 4) if n_known else None)},
        "per_checkpoint": [
            {"ckpt": r["ckpt"], "condition": r["condition"],
             "W_self": _f(r["C2_W_self"]), "W_random": _f(r["C2_W_random"]),
             "cis_separate": s}
            for r, s in zip(sel, sep)],
    }


# ------------------------------------------ 3. over-refusal as an outcome ----
def screen_secondary_outcome(rows: list[dict]) -> dict:
    """Re-run the IDENTICAL LOFO screen with Y_overrefuse as the outcome."""
    usable = [r for r in rows if _f(r.get("Y_overrefuse")) is not None]
    if len(usable) < 6:
        return {"status": "INSUFFICIENT_DATA", "n": len(usable)}
    scr = AN.run_screen(usable, CANDS, BASES, y_key="Y_overrefuse")
    return {
        "outcome": "Y_overrefuse (fraction of benign-but-alarming prompts refused)",
        "why_this_exists": (
            "the pre-registered screen scores candidates against Y_unsafe only. A "
            "readout that tracks unsafety but is blind to over-refusal is half a "
            "safety instrument. Reported SEPARATELY, never pooled with the primary "
            "screen, and it cannot promote a candidate -- the promotion rule is "
            "defined on Y_unsafe alone and is untouched."),
        "promotion_rule_applies": False,
        "n": len(usable),
        "mean_abs_error_intersection": scr.get("mean_abs_error_intersection"),
        "descriptive_spearman_vs_Y_overrefuse": scr.get("descriptive_spearman_vs_Y"),
        "candidates_would_pass_criteria_1_and_2": {
            c["candidate"]: {
                "vs_B2b_negative_CI_excl_0":
                    c["criteria"]["vs_B2b_negative_CI_excl_0"],
                "vs_B1_negative_CI_excl_0":
                    c["criteria"]["vs_B1_negative_CI_excl_0"]}
            for c in scr.get("candidates", [])},
    }


def build_posthoc(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("Y_unsafe") is not None]
    return {
        "note": ("post-freeze analyses. Each either replaces a loose check with the "
                 "stricter one it already claimed, or adds a reported dimension. "
                 "The promotion rule and every frozen metric file are untouched."),
        "triad_strict": triad_strict(rows),
        "write_gain_vs_random_direction_control": write_gain_vs_random_paired(rows),
        "secondary_outcome_screen_overrefusal": screen_secondary_outcome(rows),
    }


@logger.catch(reraise=True)
def main() -> None:
    setup_logging("posthoc")
    raw = RESULTS / "method_out_raw.json"
    try:
        state = json.loads(raw.read_text())
    except FileNotFoundError:
        logger.error(f"missing {raw}; run method.py first")
        raise
    except json.JSONDecodeError:
        logger.error(f"{raw} is not valid JSON")
        raise
    rows = state.get("per_checkpoint") or []
    logger.info(f"post-hoc over {len(rows)} checkpoint records")
    out = build_posthoc(rows)
    jdump(clean_floats(out), RESULTS / "posthoc.json")

    t = out["triad_strict"]
    logger.info(f"triad: ordering={t['prediction_held_ordering_only_as_shipped']} "
                f"STRICT={t['prediction_held_strict']} "
                f"(W CIs non-overlap={t['clause_2_W_CIs_non_overlapping']['holds']}, "
                f"E overlap={t['clause_3_E_instruct_overlaps_abliterated']['holds']})")
    w = out["write_gain_vs_random_direction_control"]
    if w.get("status") != "INSUFFICIENT_DATA":
        pa = w["paired_abs_difference"]
        logger.info(f"write gain vs random: |W_self|-|W_random| mean={pa['mean']:.4f} "
                    f"CI={pa['ci95']} excl0={pa.get('excludes_zero')} | "
                    f"per-ckpt CI separation "
                    f"{w['per_checkpoint_CI_separation']['fraction']}")
    s = out["secondary_outcome_screen_overrefusal"]
    if s.get("status") != "INSUFFICIENT_DATA":
        logger.info(f"over-refusal screen MAE: "
                    f"{json.dumps({k: (round(v, 4) if v is not None else None) for k, v in (s['mean_abs_error_intersection'] or {}).items()})}")
    logger.info(f"wrote {RESULTS / 'posthoc.json'}")


if __name__ == "__main__":
    main()
