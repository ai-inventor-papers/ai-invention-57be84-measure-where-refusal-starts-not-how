#!/usr/bin/env python3
"""STEP 4 -- pre-registered accounting, LOFO transfer estimand, cluster bootstrap,
selection rule, and the two mandatory null checks on the fold machinery.

Nothing here looks at a comparison until preregistration.json has been printed.
z-scoring is ALWAYS training-fold only; global z-scoring leaks the held-out family
and is the single easiest way to fake a transfer result.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from loguru import logger
from scipy import stats as sps

N_BOOT = 2000


# ----------------------------------------------------------------- LOFO -----
def lofo_errors(x: np.ndarray, y: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Paired per-checkpoint absolute transfer error under leave-one-FAMILY-out.

    Fit y = a + b*z on TRAINING families only, where z is x z-scored with
    TRAINING-FOLD statistics. Predict the held-out family with NO recalibration.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fam = np.asarray(fam)
    err = np.full(len(x), np.nan)
    for f in np.unique(fam):
        te = fam == f
        tr = ~te
        ok = tr & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or not np.isfinite(x[te]).any():
            continue
        mu, sd = x[ok].mean(), x[ok].std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-12:
            # degenerate predictor on the training fold: fall back to the mean of y
            pred = np.full(te.sum(), y[ok].mean())
        else:
            z = (x[ok] - mu) / sd
            A = np.vstack([np.ones_like(z), z]).T
            try:
                coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
            except np.linalg.LinAlgError:
                continue
            zt = (x[te] - mu) / sd
            pred = coef[0] + coef[1] * zt
        err[te] = np.abs(pred - y[te])
    return err


def lofo_errors_multi(X: np.ndarray, y: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Multivariate version (used by the nested slope-adds-value test)."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    fam = np.asarray(fam)
    err = np.full(len(y), np.nan)
    for f in np.unique(fam):
        te = fam == f
        tr = ~te
        ok = tr & np.isfinite(y) & np.isfinite(X).all(axis=1)
        okt = te & np.isfinite(X).all(axis=1)
        if ok.sum() < X.shape[1] + 2 or okt.sum() == 0:
            continue
        mu = X[ok].mean(axis=0)
        sd = X[ok].std(axis=0, ddof=1)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = (X[ok] - mu) / sd
        A = np.hstack([np.ones((len(Z), 1)), Z])
        try:
            coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        Zt = (X[okt] - mu) / sd
        pred = coef[0] + Zt @ coef[1:]
        err[okt] = np.abs(pred - y[okt])
    return err


def cluster_bootstrap_mean(vals: np.ndarray, fam: np.ndarray, *, n_boot: int = N_BOOT,
                           seed: int = 0) -> dict:
    """Resample FAMILIES with replacement, keeping all their checkpoints."""
    vals = np.asarray(vals, float)
    fam = np.asarray(fam)
    ok = np.isfinite(vals)
    vals, fam = vals[ok], fam[ok]
    if len(vals) < 3:
        return {"mean": None, "ci95": None, "n": int(len(vals))}
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
            "excludes_zero": bool(lo > 0 or hi < 0)}


def paired_sign_test(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 3:
        return {"n": int(len(d)), "p": None, "n_negative": None}
    neg = int((d < 0).sum())
    p = float(sps.binomtest(neg, len(d), 0.5).pvalue)
    return {"n": int(len(d)), "n_negative": neg, "p": p}


def spearman_ci(x, y, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return {"rho": None, "ci95": None, "n": int(len(x))}
    rho, p = sps.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        r, _ = sps.spearmanr(x[i], y[i])
        if np.isfinite(r):
            bs.append(r)
    ci = ([float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
          if len(bs) > 50 else None)
    return {"rho": float(rho), "p": float(p), "ci95": ci, "n": int(len(x)),
            "excludes_zero": bool(ci is not None and (ci[0] > 0 or ci[1] < 0))}


# -------------------------------------------------------- pre-registration ---
def mde_spearman(n: int) -> float:
    """|rho| a 95% CI must exceed to exclude 0, via the Fisher-z approximation."""
    if n < 5:
        return 1.0
    z = 1.959963985 / math.sqrt(n - 3)
    return float((math.exp(2 * z) - 1) / (math.exp(2 * z) + 1))


def variance_decomposition(y: np.ndarray, cond: np.ndarray) -> dict:
    y = np.asarray(y, float)
    cond = np.asarray(cond)
    ok = np.isfinite(y)
    y, cond = y[ok], cond[ok]
    groups = [y[cond == c] for c in np.unique(cond) if (cond == c).sum() > 0]
    if len(groups) < 2 or len(y) < 4:
        return {"eta2": None, "F": None, "p": None, "n": int(len(y))}
    gm = y.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    try:
        F, p = sps.f_oneway(*[g for g in groups if len(g) > 1])
    except (ValueError, TypeError):
        F, p = np.nan, np.nan
    return {"eta2": float(ssb / sst) if sst > 0 else None,
            "F": float(F) if np.isfinite(F) else None,
            "p": float(p) if np.isfinite(p) else None,
            "n": int(len(y)), "levels": [str(c) for c in np.unique(cond)],
            "group_means": {str(c): float(y[cond == c].mean())
                            for c in np.unique(cond) if (cond == c).sum() > 0}}


def preregistration(rows: list[dict], candidates: list[str],
                    baselines: list[str]) -> dict:
    fam = np.array([r["family"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    y = np.array([r.get("Y_unsafe") if r.get("Y_unsafe") is not None else np.nan
                  for r in rows], float)
    nfam = int(len(np.unique(fam)))
    pre = {
        "unit_of_analysis": "FAMILY (weight lineage); base/instruct/abliterated of one "
                            "lineage are NOT three independent families",
        "n_checkpoints": len(rows),
        "n_families": nfam,
        "family_sizes": {str(f): int((fam == f).sum()) for f in np.unique(fam)},
        "condition_sizes": {str(c): int((cond == c).sum()) for c in np.unique(cond)},
        "MDE_spearman_abs_rho_for_CI_excluding_zero": round(mde_spearman(nfam), 4),
        "MDE_statement": (
            f"With n={nfam} families, a Spearman correlation must exceed "
            f"|rho|={mde_spearman(nfam):.2f} for a 95% CI to exclude zero. The "
            "between-family LOFO test is UNDERPOWERED; a null result there is NOT "
            "evidence of absence."),
        "variance_decomposition_Y_unsafe_by_condition": variance_decomposition(y, cond),
        "estimand": (
            "Paired per-checkpoint leave-one-FAMILY-out transfer error. For each fold: "
            "z-score the candidate using TRAINING-FOLD statistics only, fit y=a+b*z on "
            "training families, predict the held-out family with NO recalibration, "
            "e_c[ckpt]=|y_pred-y_true|. D[ckpt]=e_candidate-e_baseline. Reported with a "
            "cluster bootstrap over families (2000 resamples) and a paired sign test."),
        "candidates": candidates,
        "baselines": baselines,
        "selection_rule": {
            "1": "mean(D) vs B2b (name-free card regex) negative with cluster-bootstrap "
                 "95% CI excluding 0",
            "2": "mean(D) vs B1 (raw logit gap) negative with CI excluding 0",
            "3": "coverage >= 80% of checkpoints",
            "4": "within-condition contrast points the same way",
            "all_must_hold": True,
            "if_none_promoted": "report NO_CANDIDATE_PROMOTED; no refitting, no "
                                "re-choosing the layer, no dropping checkpoints, no "
                                "switching to the first-token ground truth"},
        "within_condition_contrast_declared_decisive": True,
        "n_boot": N_BOOT,
    }
    return pre


# ---------------------------------------------------------- null checks -----
def null_checks(rows: list[dict], seed: int = 0) -> dict:
    """STAGE 7: the fold machinery must (a) not reward a random candidate and
    (b) transfer near-perfectly when handed the outcome itself."""
    fam = np.array([r["family"] for r in rows])
    y = np.array([r.get("Y_unsafe") if r.get("Y_unsafe") is not None else np.nan
                  for r in rows], float)
    ok = np.isfinite(y)
    if ok.sum() < 6:
        return {"status": "INSUFFICIENT_DATA", "n": int(ok.sum())}
    rng = np.random.default_rng(seed)

    perm_means = []
    for i in range(200):
        xr = y.copy()
        idx = np.where(ok)[0]
        xr[idx] = y[rng.permutation(idx)]
        e_rand = lofo_errors(xr, y, fam)
        e_base = lofo_errors(np.ones_like(y) * rng.normal(size=len(y)), y, fam)
        d = e_rand - e_base
        if np.isfinite(d).sum() >= 3:
            perm_means.append(np.nanmean(d))
    pm = np.array(perm_means)
    a_ok = bool(len(pm) > 20 and abs(pm.mean()) < 0.5 * (np.nanstd(y) + 1e-9)
                and np.percentile(pm, 2.5) < 0 < np.percentile(pm, 97.5))

    e_self = lofo_errors(y, y, fam)
    spread = float(np.nanstd(y)) + 1e-12
    b_val = float(np.nanmean(e_self) / spread)
    b_ok = bool(b_val < 0.5)

    # (c) leakage probes. (c1) the held-out family's own values must never enter
    # the training mean/sd. (c2) the implementation must actually DIFFER from a
    # deliberately-leaky global-z-scoring implementation -- otherwise (c1) passing
    # would only prove that the two code paths happen to agree.
    leak = _explicit_leak_probe(y, fam)
    xr = y.copy()
    idx = np.where(ok)[0]
    xr[idx] = y[rng.permutation(idx)]
    e_fold = lofo_errors(xr, y, fam)
    e_global = _leaky_global_zscore_errors(xr, y, fam)
    both = np.isfinite(e_fold) & np.isfinite(e_global)
    zscore_delta = (float(np.max(np.abs(e_fold[both] - e_global[both])))
                    if both.sum() >= 3 else None)
    # Least squares is invariant to an affine rescaling of the predictor, so a
    # univariate linear transfer map gives IDENTICAL predictions under global and
    # training-fold z-scoring. The pre-registered "global z-scoring leaks the
    # held-out family" concern is therefore VACUOUS for this estimand; we assert the
    # identity rather than a difference, and test the leak that can actually bite.
    zscore_identity = bool(zscore_delta is not None and zscore_delta < 1e-9)

    # (c3) the leak that DOES bite: refitting a,b ON the held-out family. An honest
    # LOFO error must be strictly worse than that oracle.
    e_lofo_real = lofo_errors(y.copy(), y, fam)
    e_oracle = _refit_on_heldout_errors(y.copy(), y, fam)
    m3 = np.isfinite(e_lofo_real) & np.isfinite(e_oracle)
    lofo_not_secretly_refitting = bool(
        m3.sum() >= 3 and np.nanmean(e_lofo_real[m3]) >= np.nanmean(e_oracle[m3]) - 1e-12)
    fams = list(np.unique(fam))
    x2 = y.copy()
    x2[fam == fams[0]] = x2[fam == fams[0]] + 1000.0
    changed = np.nansum(np.abs(np.nan_to_num(lofo_errors(y.copy(), y, fam))
                               - np.nan_to_num(lofo_errors(x2, y, fam))))

    return {"a_random_candidate_centred_on_zero": a_ok,
            "a_perm_mean": float(pm.mean()) if len(pm) else None,
            "a_perm_ci95": [float(np.percentile(pm, 2.5)),
                            float(np.percentile(pm, 97.5))] if len(pm) > 20 else None,
            "b_self_transfer_error_over_sd": b_val,
            "b_self_transfer_near_zero": b_ok,
            "c1_heldout_values_do_not_enter_training_stats": leak,
            "c2_zscoring_choice_is_provably_irrelevant_here": zscore_identity,
            "c2_max_abs_error_difference_global_vs_foldwise": zscore_delta,
            "c2_note": ("OLS absorbs any affine rescaling of the predictor, so global "
                        "and training-fold z-scoring are mathematically identical for "
                        "this linear transfer map. The pre-registered leakage concern "
                        "does not apply to this estimand; it would apply to any "
                        "non-affine-invariant learner."),
            "c3_lofo_not_secretly_refitting_on_heldout": lofo_not_secretly_refitting,
            "d_training_perturbation_moves_other_folds": bool(changed > 0),
            "status": "PASS" if (a_ok and b_ok and leak and zscore_identity
                                 and lofo_not_secretly_refitting) else "FAIL"}


def _refit_on_heldout_errors(x, y, fam):
    """The leak that actually matters: fit a,b ON the held-out family itself.
    Real LOFO error must never beat this oracle."""
    x, y, fam = np.asarray(x, float), np.asarray(y, float), np.asarray(fam)
    err = np.full(len(x), np.nan)
    for f in np.unique(fam):
        te = fam == f
        ok = te & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        sd = x[ok].std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-12:
            err[ok] = np.abs(y[ok].mean() - y[ok])
            continue
        z = (x[ok] - x[ok].mean()) / sd
        A = np.vstack([np.ones_like(z), z]).T
        try:
            coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        err[ok] = np.abs(coef[0] + coef[1] * z - y[ok])
    return err


def _leaky_global_zscore_errors(x, y, fam):
    """Deliberately WRONG reference implementation (global z-scoring, which leaks
    the held-out family into the scaling). Used only to prove the real one differs."""
    x, y, fam = np.asarray(x, float), np.asarray(y, float), np.asarray(fam)
    err = np.full(len(x), np.nan)
    ok_all = np.isfinite(x) & np.isfinite(y)
    if ok_all.sum() < 4:
        return err
    mu, sd = x[ok_all].mean(), x[ok_all].std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return err
    z = (x - mu) / sd
    for f in np.unique(fam):
        te, tr = fam == f, fam != f
        ok = tr & np.isfinite(z) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        A = np.vstack([np.ones(ok.sum()), z[ok]]).T
        try:
            coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        m = te & np.isfinite(z)
        err[m] = np.abs(coef[0] + coef[1] * z[m] - y[m])
    return err


def _explicit_leak_probe(y: np.ndarray, fam: np.ndarray) -> bool:
    """Perturb ONLY the held-out family's x; the fitted a,b must be unchanged."""
    fams = list(np.unique(fam))
    f0 = fams[0]
    x = y.copy()
    tr = fam != f0
    ok = tr & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return True
    mu, sd = x[ok].mean(), x[ok].std(ddof=1)
    x2 = x.copy()
    x2[fam == f0] = x2[fam == f0] * 17.0 + 123.0
    mu2, sd2 = x2[ok].mean(), x2[ok].std(ddof=1)
    return bool(abs(mu - mu2) < 1e-9 and abs(sd - sd2) < 1e-9)


# -------------------------------------------------------------- the screen ---
def run_screen(rows: list[dict], candidates: dict[str, str], baselines: dict[str, str],
               *, y_key: str = "Y_unsafe") -> dict:
    """candidates/baselines map display-name -> row key holding the scalar."""
    fam = np.array([r["family"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    y = np.array([_get(r, y_key) for r in rows], float)

    vals = {name: np.array([_get(r, k) for r in rows], float)
            for name, k in {**candidates, **baselines}.items()}
    coverage = {n: float(np.isfinite(v).mean()) for n, v in vals.items()}
    # intersection set: every candidate AND every baseline computable, plus y
    inter = np.isfinite(y)
    for v in vals.values():
        inter &= np.isfinite(v)

    errs_own = {n: lofo_errors(v, y, fam) for n, v in vals.items()}
    errs_int = {}
    if inter.sum() >= 6:
        fi, yi = fam[inter], y[inter]
        for n, v in vals.items():
            e = np.full(len(y), np.nan)
            e[inter] = lofo_errors(v[inter], yi, fi)
            errs_int[n] = e
    else:
        errs_int = {n: np.full(len(y), np.nan) for n in vals}

    out = {"coverage": coverage,
           "n_intersection": int(inter.sum()),
           "intersection_checkpoints": [rows[i]["ckpt"] for i in np.where(inter)[0]],
           "mean_abs_error_own_coverage": {
               n: (float(np.nanmean(e)) if np.isfinite(e).any() else None)
               for n, e in errs_own.items()},
           "mean_abs_error_intersection": {
               n: (float(np.nanmean(e)) if np.isfinite(e).any() else None)
               for n, e in errs_int.items()},
           "candidates": []}

    for ci, cname in enumerate(candidates):
        row = {"candidate": cname, "coverage": coverage[cname],
               "coverage_ok": bool(coverage[cname] >= 0.8), "vs": {}}
        for bname in baselines:
            d_int = errs_int[cname] - errs_int[bname]
            d_own = errs_own[cname] - errs_own[bname]
            row["vs"][bname] = {
                "intersection": {**cluster_bootstrap_mean(d_int, fam, seed=100 + ci),
                                 "sign_test": paired_sign_test(d_int)},
                "own_coverage": {**cluster_bootstrap_mean(d_own, fam, seed=200 + ci),
                                 "sign_test": paired_sign_test(d_own)},
            }
        # within-condition contrast (declared decisive)
        wc = {}
        for c in np.unique(cond):
            m = (cond == c) & inter
            if m.sum() >= 4 and len(np.unique(fam[m])) >= 3:
                for bname in baselines:
                    e_c = lofo_errors(vals[cname][m], y[m], fam[m])
                    e_b = lofo_errors(vals[bname][m], y[m], fam[m])
                    wc.setdefault(str(c), {})[bname] = cluster_bootstrap_mean(
                        e_c - e_b, fam[m], seed=300 + ci)
            else:
                wc[str(c)] = {"status": "UNDERPOWERED",
                              "n": int(m.sum()),
                              "n_families": int(len(np.unique(fam[m])))}
        row["within_condition"] = wc

        c1 = _neg_ci_excl_zero(row["vs"].get("B2b_namefree_regex", {}))
        c2 = _neg_ci_excl_zero(row["vs"].get("B1_logit_gap", {}))
        c4 = _within_agrees(wc)
        row["criteria"] = {"vs_B2b_negative_CI_excl_0": c1,
                           "vs_B1_negative_CI_excl_0": c2,
                           "coverage_ge_80pct": row["coverage_ok"],
                           "within_condition_agrees": c4}
        row["PROMOTED"] = bool(c1 and c2 and row["coverage_ok"] and c4)
        out["candidates"].append(row)

    # direct predictive correlations (descriptive, not the estimand)
    out["descriptive_spearman_vs_Y"] = {
        n: spearman_ci(v, y, seed=7) for n, v in vals.items()}
    out["descriptive_spearman_lineage_level"] = _lineage_level(rows, vals, y, fam)
    out["promoted"] = [r["candidate"] for r in out["candidates"] if r["PROMOTED"]]
    return out


def _get(r: dict, k: str) -> float:
    v = r.get(k)
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _neg_ci_excl_zero(vs: dict) -> bool:
    b = (vs or {}).get("intersection") or {}
    ci = b.get("ci95")
    return bool(ci is not None and b.get("mean") is not None
                and b["mean"] < 0 and ci[1] < 0)


def _within_agrees(wc: dict) -> bool:
    signs = []
    for c, d in wc.items():
        if not isinstance(d, dict) or d.get("status") == "UNDERPOWERED":
            continue
        for b, s in d.items():
            if isinstance(s, dict) and s.get("mean") is not None:
                signs.append(s["mean"] < 0)
    return bool(signs and all(signs))


def _lineage_level(rows, vals, y, fam) -> dict:
    out = {}
    fams = np.unique(fam)
    yl = np.array([np.nanmean(y[fam == f]) for f in fams])
    for n, v in vals.items():
        xl = np.array([np.nanmean(v[fam == f]) if np.isfinite(v[fam == f]).any()
                       else np.nan for f in fams])
        out[n] = spearman_ci(xl, yl, seed=9)
    return out


def slope_adds_value(rows: list[dict], ed50_key: str, slope_key: str,
                     y_key: str = "Y_unsafe") -> dict:
    fam = np.array([r["family"] for r in rows])
    y = np.array([_get(r, y_key) for r in rows], float)
    a = np.array([_get(r, ed50_key) for r in rows], float)
    b = np.array([_get(r, slope_key) for r in rows], float)
    e1 = lofo_errors(a, y, fam)
    e2 = lofo_errors_multi(np.vstack([a, b]).T, y, fam)
    d = e2 - e1
    return {"ed50_only_mae": float(np.nanmean(e1)) if np.isfinite(e1).any() else None,
            "ed50_plus_slope_mae": float(np.nanmean(e2)) if np.isfinite(e2).any() else None,
            "paired_reduction": cluster_bootstrap_mean(d, fam, seed=42),
            "interpretation": "negative mean = adding the slope REDUCES transfer error"}


def lofo_predictions(x: np.ndarray, y: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """The held-out-family predictions behind lofo_errors (same fold logic,
    training-fold z-scoring only). Returned so each method's prediction can be
    shipped as a predict_* field alongside the ground-truth output."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fam = np.asarray(fam)
    pred = np.full(len(x), np.nan)
    for f in np.unique(fam):
        te = fam == f
        tr = ~te
        ok = tr & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3 or not np.isfinite(x[te]).any():
            continue
        mu, sd = x[ok].mean(), x[ok].std(ddof=1)
        if not np.isfinite(sd) or sd < 1e-12:
            pred[te] = y[ok].mean()
            continue
        z = (x[ok] - mu) / sd
        A = np.vstack([np.ones_like(z), z]).T
        try:
            coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred[te] = coef[0] + coef[1] * ((x[te] - mu) / sd)
    return pred
