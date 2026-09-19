#!/usr/bin/env python3
"""Sanity baseline: does the frozen prompt set carry the structure the screen needs?

Three cheap checks, all lexical (no model loaded):
 1. harmful-vs-benign separability on UNMATCHED pools (should be easy -> the labels are real);
 2. the same on MATCHED twins, grouped so a pair never straddles the split (should be MUCH
    harder -> the twins really are matched, which is what makes them a control);
 3. the authored severity ladder's length/lexical confound (rung must not be predictable from
    prompt length, or the dose axis measures sentence length).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT = Path(__file__).resolve().parent
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(ROOT / "logs" / "sanity.log", rotation="10 MB", level="DEBUG")


def auc(texts: list[str], y: list[int], groups: list[str]) -> float:
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    n_groups = len(set(groups))
    cv = GroupKFold(n_splits=min(5, n_groups))
    return float(np.mean(cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="roc_auc")))


@logger.catch(reraise=True)
def main() -> None:
    d = json.loads((ROOT / "prompt_sets" / "prompt_sets_full.json").read_text())
    rows = d["rows"]
    out: dict[str, object] = {}

    # 1. unmatched pools
    harm = [r for r in rows if r["sub_study"] == "BEHAVIOUR_GT"]
    ben = [r for r in rows if r["sub_study"] == "OVERREFUSAL" and r["label"] == "benign"]
    n = min(len(harm), len(ben), 2000)
    t = [r["prompt"] for r in harm[:n]] + [r["prompt"] for r in ben[:n]]
    y = [1] * n + [0] * n
    g = [f"u{i}" for i in range(2 * n)]
    out["unmatched_harmful_vs_benign_auc"] = round(auc(t, y, g), 4)

    # 2. matched twins, pair-grouped
    tw = [r for r in rows if r["sub_study"] == "BENIGN_TWIN_PAIRS" and r["label"] in ("harmful", "benign")]
    pair_ct = Counter(r["pair_id"] for r in tw)
    tw = [r for r in tw if pair_ct[r["pair_id"]] == 2]
    out["n_complete_matched_pairs"] = len(tw) // 2
    out["matched_twin_auc_pair_grouped"] = round(
        auc([r["prompt"] for r in tw], [1 if r["label"] == "harmful" else 0 for r in tw],
            [r["pair_id"] for r in tw]), 4)

    # 3. ladder confounds
    lad = [r for r in rows if r["sub_study"] == "LADDER_AUTHORED"]
    lens = [len(r["prompt"]) for r in lad]
    lvl = [r["severity_level"] for r in lad]
    rho, p = spearmanr(lens, lvl)
    out["ladder_len_vs_level_spearman"] = round(float(rho), 4)
    out["ladder_len_vs_level_p"] = round(float(p), 4)
    out["ladder_n_rungs"] = len(lad)
    out["ladder_mean_len"] = round(float(np.mean(lens)), 1)
    # within-topic length spread, the quantity that would confound a per-topic dose fit
    by_topic: dict[str, list[int]] = {}
    for r in lad:
        by_topic.setdefault(r["topic"], []).append(len(r["prompt"]))
    out["ladder_within_topic_len_cv"] = round(
        float(np.mean([np.std(v) / np.mean(v) for v in by_topic.values()])), 4)

    # 3b. the same confound check on HarmLevelBench, the externally-authored 7x8 ladder
    hlb = [r for r in rows if r["sub_study"] == "LADDER_HARMLEVELBENCH"]
    h_lens = [len(r["prompt"]) for r in hlb]
    h_lvl = [r["severity_level"] for r in hlb]
    h_rho, h_p = spearmanr(h_lens, h_lvl)
    out["harmlevelbench_len_vs_level_spearman"] = round(float(h_rho), 4)
    out["harmlevelbench_len_vs_level_p"] = round(float(h_p), 4)
    out["harmlevelbench_n_rungs"] = len(hlb)
    by_t: dict[str, list[int]] = {}
    for r in hlb:
        by_t.setdefault(r["topic"], []).append(len(r["prompt"]))
    out["harmlevelbench_within_topic_len_cv"] = round(
        float(np.mean([np.std(v) / np.mean(v) for v in by_t.values()])), 4)
    # do the two external ladders' own level-1 anchors read as benign to a lexical model?
    anchors = [r for r in hlb if r["severity_level"] == 0]
    tops = [r for r in hlb if r["severity_level"] == 7]
    out["harmlevelbench_L0_vs_L7_auc"] = round(
        auc([r["prompt"] for r in anchors + tops],
            [0] * len(anchors) + [1] * len(tops),
            [r["topic"] for r in anchors + tops]), 4)

    # 4. external OR-Bench axis: level predictable from lexis? (it should be, it is a real axis)
    ob = [r for r in rows if r["sub_study"] == "LADDER_EXTERNAL"]
    ob2 = [r for r in ob if r["severity_level"] in (0, 2)]
    out["orbench_L0_vs_L2_auc"] = round(
        auc([r["prompt"] for r in ob2], [1 if r["severity_level"] == 2 else 0 for r in ob2],
            [r["topic"] for r in ob2]), 4)

    for k, v in out.items():
        logger.info(f"{k:<38} {v}")
    (ROOT / "prompt_sets" / "sanity_baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
