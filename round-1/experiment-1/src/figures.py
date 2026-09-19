#!/usr/bin/env python3
"""Render every figure from results/method_out_raw.json via the house-style
chart generator. Each figure ships as .pdf + .png plus the JSON spec that drew it,
so a figure can never disagree with its numbers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from core import WORKSPACE, clean_floats, jdump, setup_logging

SKILL = Path("/ai-inventor/.claude/skills/aii-data-fig-gen/scripts/chart_gen.py")
PY = str(WORKSPACE / ".venv" / "bin" / "python")
FIGS = WORKSPACE / "figs"
RESULTS = WORKSPACE / "results"

COND_ORDER = ["base", "instruct", "abliterated"]


def render(spec: dict, name: str) -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    sp = FIGS / f"{name}.spec.json"
    jdump(clean_floats(spec), sp)
    out = FIGS / f"{name}.pdf"
    try:
        r = subprocess.run([PY, str(SKILL), "--spec", str(sp), "--out", str(out)],
                           capture_output=True, text=True, timeout=180)
        ok = out.exists()
        if not ok:
            logger.error(f"{name}: {r.stdout[-500:]} {r.stderr[-500:]}")
        else:
            logger.info(f"rendered {name}")
        return {"figure": name, "spec": str(sp.relative_to(WORKSPACE)),
                "pdf": str(out.relative_to(WORKSPACE)) if ok else None,
                "png": str((FIGS / f"{name}.png").relative_to(WORKSPACE))
                if (FIGS / f"{name}.png").exists() else None,
                "ok": ok, "stderr": (r.stderr or "")[-300:] if not ok else None}
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.error(f"{name} render failed: {exc}")
        return {"figure": name, "ok": False, "error": str(exc)[:200]}


def _label(repo: str) -> str:
    return repo.split("/")[-1]


def fig1_layer_condition(mech: dict) -> dict | None:
    per = {k: v for k, v in (mech.get("per_model") or {}).items() if "failed" not in v}
    if len(per) < 2:
        return None
    rows, mat = [], []
    n = min(len(v["sep_profile"]) for v in per.values())
    grid = np.linspace(0, n - 1, 24).astype(int)
    for repo, v in per.items():
        rows.append(_label(repo))
        mat.append([v["sep_profile"][i] for i in grid])
    return render({
        "type": "heatmap", "title": "Harmful-vs-benign separation by layer and condition",
        "xlabel": "Layer (relative depth)", "ylabel": "Checkpoint",
        "cbar_label": "||dmean_L|| / RMS_L  (RMS-normalised)",
        "aspect": "16:9", "row_labels": rows,
        "col_labels": [f"{i}" for i in grid], "matrix": mat,
        "annotate": False,
    }, "fig1_layer_condition_heatmap")


def fig2_depth_profile(mech: dict) -> dict | None:
    per = {k: v for k, v in (mech.get("per_model") or {}).items() if "failed" not in v}
    if not per:
        return None
    series = []
    for repo, v in per.items():
        y = v["sep_profile"]
        series.append({"label": _label(repo), "x": list(range(len(y))), "values": y})
    return render({
        "type": "line", "aspect": "16:9",
        "title": "RMS-normalised separation depth profile (Qwen3-1.7B triad)",
        "xlabel": "Layer index", "ylabel": "||dmean_L|| / RMS_L",
        "series": series}, "fig2_depth_profile")


def fig3_dose_shift(mech: dict) -> dict | None:
    per = {k: v for k, v in (mech.get("per_model") or {}).items()
           if "failed" not in v and v.get("dose_shift")}
    if not per:
        return None
    series = []
    for repo, v in per.items():
        d = v["dose_shift"]
        series.append({"label": _label(repo),
                       "x": [x["rung"] for x in d],
                       "values": [x["argmax_layer_frac"] for x in d]})
    return render({
        "type": "line", "aspect": "16:9",
        "title": "Where the decision sits as harm dose rises",
        "xlabel": "Ladder rung (0 = benign anchor, 11 = harmful anchor)",
        "ylabel": "argmax-separation layer (fraction of depth)",
        "series": series}, "fig3_dose_shift")


def _short_id(name: str) -> str:
    """Drop the descriptive suffix, keeping the short code (e.g. 'C1_dose_axis' -> 'C1').

    Presentation only: these codes are the same ones already used as dict
    keys elsewhere (C1_primary, C2_W_self, ...), so the category stays
    unambiguous while dropping the part that made 18 rows collide.
    """
    return name.split("_")[0]


def fig4_screen_forest(screen: dict) -> dict | None:
    cands = screen.get("candidates") or []
    cats, vals, errs = [], [], []
    for c in cands:
        for b in ("B1_logit_gap", "B2b_namefree_regex", "B3_anchor_projection"):
            d = ((c.get("vs") or {}).get(b) or {}).get("intersection") or {}
            if d.get("mean") is None or d.get("ci95") is None:
                continue
            cats.append(f"{_short_id(c['candidate'])} vs {_short_id(b)}")
            vals.append(d["mean"])
            errs.append(max(abs(d["mean"] - d["ci95"][0]), abs(d["ci95"][1] - d["mean"])))
    if not cats:
        return None
    return render({
        "type": "forest", "null_line": 0.0, "aspect": "4:3", "width_in": 8.5,
        "title": "Paired LOFO transfer error: candidate minus baseline (95% cluster CI)",
        "xlabel": "mean(D) = candidate err. − baseline err. (negative = candidate better)",
        "categories": cats, "series": [{"values": vals, "errors": errs}]},
        "fig4_screen_forest")


def fig5_write_gain(rows: list[dict]) -> dict | None:
    by = {}
    for r in rows:
        w = r.get("C2_W_self")
        if w is None:
            continue
        by.setdefault(r["condition"], []).append(w)
    conds = [c for c in COND_ORDER if len(by.get(c, [])) >= 2]
    if len(conds) < 2:
        return None
    rnd = {}
    for r in rows:
        if r.get("C2_W_random") is not None:
            rnd.setdefault(r["condition"], []).append(r["C2_W_random"])
    return render({
        "type": "bar", "aspect": "4:3",
        "title": "Write gain W = dR/d(alpha) by condition, with random-direction control",
        "xlabel": "Condition", "ylabel": "W (nats per unit of Delta)",
        "categories": conds,
        "series": [
            {"label": "refusal direction (self-derived)",
             "values": [float(np.mean(by[c])) for c in conds],
             "errors": [float(np.std(by[c], ddof=1) / max(1, len(by[c]) ** 0.5))
                        if len(by[c]) > 1 else 0.0 for c in conds]},
            {"label": "random-direction control",
             "values": [float(np.mean(rnd.get(c, [0.0]))) for c in conds],
             "errors": [float(np.std(rnd[c], ddof=1) / max(1, len(rnd[c]) ** 0.5))
                        if len(rnd.get(c, [])) > 1 else 0.0 for c in conds]},
        ]}, "fig5_write_gain")


def fig6_candidate_vs_y(rows: list[dict], key: str, label: str, name: str) -> dict | None:
    """One sub-panel per condition rather than one shared axes.

    All three conditions used to share a single scatter with `fit: true`,
    which draws one regression-equation annotation per series at the same
    fixed axes position — with three series that is a guaranteed collision,
    not a sizing problem `width_in` can fix. Splitting by condition (a
    presentation change only — same points, same groupings) gives each
    series its own axes, so each fit equation has the panel to itself; it
    also lets a condition whose x values are all identical (no slope to fit)
    skip `fit` on just that panel instead of forcing it off for everyone.
    """
    panels = []
    ylabel = "Y_unsafe (higher = less safe)"
    for c in COND_ORDER:
        xs = [r.get(key) for r in rows if r["condition"] == c and r.get(key) is not None
              and r.get("Y_unsafe") is not None]
        ys = [r["Y_unsafe"] for r in rows if r["condition"] == c and r.get(key) is not None
              and r.get("Y_unsafe") is not None]
        if not xs:
            continue
        can_fit = len(xs) >= 2 and len(set(xs)) > 1
        panels.append({
            "type": "scatter", "fit": can_fit,
            "title": c, "xlabel": label, "ylabel": ylabel,
            "series": [{"values": ys, "x": xs}]})
    if not panels:
        return None
    return render({
        "type": "panel", "ncols": len(panels), "width_in": 10.5,
        "title": f"{label} against generation-judged unsafety, by condition",
        "panels": panels}, name)


def fig7_ground_truth(rows: list[dict]) -> dict | None:
    by = {}
    for r in rows:
        if r.get("Y_unsafe") is not None:
            by.setdefault(r["condition"], []).append(r["Y_unsafe"])
    over = {}
    for r in rows:
        if r.get("Y_overrefuse") is not None:
            over.setdefault(r["condition"], []).append(r["Y_overrefuse"])
    conds = [c for c in COND_ORDER if by.get(c)]
    if len(conds) < 2:
        return None
    return render({
        "type": "bar", "aspect": "4:3",
        "title": "Behavioural ground truth separates the three conditions",
        "xlabel": "Condition", "ylabel": "rate",
        "categories": conds,
        "series": [
            {"label": "Y_unsafe (harmful x wrappers)",
             "values": [float(np.mean(by[c])) for c in conds],
             "errors": [float(np.std(by[c], ddof=1) / len(by[c]) ** 0.5)
                        if len(by[c]) > 1 else 0.0 for c in conds]},
            {"label": "Y_overrefuse (benign-but-alarming)",
             "values": [float(np.mean(over.get(c, [0.0]))) for c in conds],
             "errors": [float(np.std(over[c], ddof=1) / len(over[c]) ** 0.5)
                        if len(over.get(c, [])) > 1 else 0.0 for c in conds]},
        ]}, "fig7_ground_truth")


def fig8_template_sd(rows: list[dict]) -> dict | None:
    cats, within, across = [], [], []
    vals_by_tpl = {}
    for r in rows:
        pt = ((r.get("C1") or {}).get("per_template") or {})
        for tpl, v in pt.items():
            if v.get("ed50_median") is not None:
                vals_by_tpl.setdefault(tpl, []).append(v["ed50_median"])
    if len(vals_by_tpl) < 2:
        return None
    sd_within = [r["C1"]["ed50_sd_across_templates"] for r in rows
                 if (r.get("C1") or {}).get("ed50_sd_across_templates") is not None]
    sd_across = [float(np.std(v, ddof=1)) for v in vals_by_tpl.values() if len(v) > 1]
    if not sd_within or not sd_across:
        return None
    return render({
        "type": "bar", "aspect": "4:3",
        "title": "Does the instrument measure the model or the template?",
        "xlabel": "", "ylabel": "SD of ED50",
        "categories": ["SD across TEMPLATES\n(within a model)",
                       "SD across MODELS\n(within a template)"],
        "series": [{"values": [float(np.mean(sd_within)), float(np.mean(sd_across))],
                    "errors": [float(np.std(sd_within, ddof=1) / len(sd_within) ** 0.5)
                               if len(sd_within) > 1 else 0.0,
                               float(np.std(sd_across, ddof=1) / len(sd_across) ** 0.5)
                               if len(sd_across) > 1 else 0.0]}]},
        "fig8_template_vs_model_sd")


def fig9_gate_g2(g2: dict) -> dict | None:
    per = {k: v for k, v in (g2.get("per_model") or {}).items() if "failed" not in v}
    if not per:
        return None
    modes = g2.get("modes", [])
    cats = [_label(k) for k in per]
    series = [{"label": mo,
               "values": [per[k][mo]["spread_mean"] for k in per]} for mo in modes]
    return render({
        "type": "bar", "aspect": "16:9",
        "title": "Gate G2: the thinking block moves the refusal readout",
        "xlabel": "Checkpoint",
        "ylabel": "readout spread (nats)",
        "categories": cats, "series": series}, "fig9_gate_G2_thinking_mode")


def fig10_ladder(g1: dict) -> dict | None:
    per = (g1.get("per_condition") or {})
    series = []
    for repo, v in per.items():
        if "failed" in v:
            continue
        fams = v.get("per_family") or {}
        eds = [x["ed50_interp"] for x in fams.values() if x.get("ed50_interp") is not None]
        if not eds:
            continue
        series.append({"label": _label(repo), "x": list(range(len(eds))),
                       "values": sorted(eds)})
    if not series:
        return None
    return render({
        "type": "line", "aspect": "4:3",
        "title": "Gate G1: per-pair-family ED50, sorted (dose axis location)",
        "xlabel": "Pair-family (sorted by ED50)",
        "ylabel": "ED50 (dose at the p=0.5 refusal crossing)",
        "series": series}, "fig10_gate_G1_ed50")


def build_all(state: dict) -> list[dict]:
    figs = []
    rows = [r for r in state.get("per_checkpoint", []) if r.get("Y_unsafe") is not None]
    mech = state.get("mechanism") or {}
    gates = state.get("gates") or {}
    screen = (state.get("screen") or {})
    for fn, arg in [(fig1_layer_condition, mech), (fig2_depth_profile, mech),
                    (fig3_dose_shift, mech)]:
        r = fn(arg)
        if r:
            figs.append(r)
    for fn, arg in [(fig9_gate_g2, gates.get("G2") or {}),
                    (fig10_ladder, gates.get("G1") or {})]:
        r = fn(arg)
        if r:
            figs.append(r)
    if screen:
        r = fig4_screen_forest(screen)
        if r:
            figs.append(r)
    if rows:
        for r in (fig5_write_gain(rows), fig7_ground_truth(rows), fig8_template_sd(rows)):
            if r:
                figs.append(r)
        for key, lab, nm in [("C1_primary", "C1 anchored dose axis", "fig6a_C1_vs_Y"),
                             ("C2_W_self", "C2 write gain W", "fig6b_C2_vs_Y"),
                             ("C3_S_minus_Epi", "C3 ratiometric S - Epi", "fig6c_C3_vs_Y"),
                             ("C4_cv", "C4 paraphrase CV", "fig6d_C4_vs_Y"),
                             ("C5_w1", "C5 weights-only w1", "fig6e_C5_vs_Y"),
                             ("B1_logit_gap", "B1 raw logit gap", "fig6f_B1_vs_Y")]:
            r = fig6_candidate_vs_y(rows, key, lab, nm)
            if r:
                figs.append(r)
    return figs


if __name__ == "__main__":
    setup_logging("figures")
    st = json.loads((RESULTS / "method_out_raw.json").read_text())
    fs = build_all(st)
    jdump(fs, RESULTS / "figures_index.json")
    logger.info(f"{sum(1 for f in fs if f.get('ok'))}/{len(fs)} figures rendered")
    print(json.dumps([f["figure"] for f in fs if f.get("ok")], indent=2))
