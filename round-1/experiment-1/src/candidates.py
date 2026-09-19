#!/usr/bin/env python3
"""Candidates C1-C5 and baselines B1/B3, all computed on one loaded checkpoint.

Every function returns a dict and NEVER raises for a merely-uncomputable cell:
uncomputable cells come back as {"failed": "<CLOSED_SET_REASON>"} and count
against that candidate's coverage in the screen.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import stats as sps
from scipy.optimize import curve_fit

import corpora as C
from core import (COMPLY_CONTS, DEVICE, EPISTEMIC_CONTS, REFUSAL_CONTS, WORKSPACE,
                  Readout, first_token_ids, format_prompt, get_input_embedding_matrix,
                  get_layers, get_unembedding)

INPUTS = WORKSPACE / "inputs"

FAIL = {"CENSORED", "UNIDENTIFIABLE_FIT", "NO_DIRECTION", "TEMPLATE_MISSING",
        "OOM", "DOWNLOAD_FAILED", "GATED_REPO", "ARCH_UNSUPPORTED",
        "SPAN_NOT_CONSTRUCTIBLE", "JUDGE_UNAVAILABLE"}

ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
N_INTERP = 15
CV_GUARD = 1.0


# ----------------------------------------------------------------- helpers ---
def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 10000


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _safe_median(vals) -> float | None:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def _sig2(t, k, ed50):
    return 1.0 / (1.0 + np.exp(-np.clip(k * (t - ed50), -60, 60)))


def interp_crossing(t: np.ndarray, p: np.ndarray, target: float = 0.5) -> float | None:
    """Linear interpolation of the first p=target crossing. Monotone-reparam invariant."""
    for i in range(len(t) - 1):
        a, b = p[i], p[i + 1]
        if (a - target) * (b - target) <= 0 and b != a:
            return float(t[i] + (target - a) * (t[i + 1] - t[i]) / (b - a))
        if a == target:
            return float(t[i])
    if p[-1] == target:
        return float(t[-1])
    return None


def fit_ladder(t: np.ndarray, p: np.ndarray, *, n_boot: int = 200,
               rng_seed: int = 0) -> dict:
    """Two-parameter sigmoid with asymptotes FIXED at 0 and 1, plus the
    pre-registered three-part identifiability test."""
    res = {"n": int(len(t)), "p_min": float(p[0]), "p_max": float(p[-1])}
    rho, pv = sps.spearmanr(t, p)
    res["spearman"] = float(rho) if np.isfinite(rho) else 0.0
    res["spearman_p"] = float(pv) if np.isfinite(pv) else 1.0
    res["ed50_interp"] = interp_crossing(t, p)

    w = 1.0 / (p * (1.0 - p) + 0.01)
    k = ed50 = None
    try:
        popt, _ = curve_fit(_sig2, t, p, p0=[8.0, 0.5], sigma=1.0 / np.sqrt(w),
                            bounds=([0.5, -1.0], [200.0, 3.0]), maxfev=20000)
        k, ed50 = float(popt[0]), float(popt[1])
        pred = _sig2(t, k, ed50)
        ss = float(np.sum((p - pred) ** 2))
        tot = float(np.sum((p - p.mean()) ** 2)) + 1e-12
        res["fit_r2"] = 1.0 - ss / tot
    except (RuntimeError, ValueError, TypeError) as exc:
        res["fit_error"] = type(exc).__name__
        res["fit_r2"] = None
    res["hill_k"] = k
    res["ed50_fit"] = ed50

    rng = np.random.default_rng(rng_seed)
    boots = []
    n = len(t)
    for _ in range(n_boot):
        idx = np.sort(rng.integers(0, n, n))
        tb, pb = t[idx], p[idx]
        keep = np.concatenate(([True], np.diff(tb) > 0))
        if keep.sum() < 4:
            continue
        c = interp_crossing(tb[keep], pb[keep])
        if c is not None:
            boots.append(c)
    if len(boots) >= 20:
        lo, hi = np.percentile(boots, [5, 95])
        res["ed50_ci90"] = [float(lo), float(hi)]
        res["ed50_ci90_width"] = float(hi - lo)
    else:
        res["ed50_ci90"] = None
        res["ed50_ci90_width"] = None

    crosses = bool(p[0] < 0.45 and p[-1] > 0.55)
    monotone = bool(res["spearman"] >= 0.5)
    tight = bool(res["ed50_ci90_width"] is not None and res["ed50_ci90_width"] <= 0.5)
    res["crit_crosses_midpoint"] = crosses
    res["crit_monotone"] = monotone
    res["crit_tight_ci"] = tight
    res["identifiable"] = bool(crosses and monotone and tight)
    if not crosses:
        res["censor_side"] = "LOW" if p[-1] <= 0.55 else ("HIGH" if p[0] >= 0.45 else None)
    return res


def tobit_slope(t: np.ndarray, R: np.ndarray) -> dict:
    """FALLBACK-1 estimand: OLS dose slope in nats and the implied crossing t*,
    censored above 1.5 when R never crosses zero."""
    A = np.vstack([np.ones_like(t), t]).T
    try:
        coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    except np.linalg.LinAlgError:
        return {"b": None, "a": None, "t_star": None, "censored": True}
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss = float(np.sum((R - pred) ** 2))
    tot = float(np.sum((R - R.mean()) ** 2)) + 1e-12
    if abs(b) < 1e-9:
        ts, cens = None, True
    else:
        ts = (0.0 - a) / b
        cens = bool(ts > 1.5 or ts < -0.5)
        if cens:
            ts = float(np.clip(ts, -0.5, 1.5))
    return {"a": a, "b": b, "t_star": ts, "censored": cens,
            "r2": 1.0 - ss / tot}


# ------------------------------------------------------ direction / layer ----
def decision_layer(ro: Readout, harmful: list[str], benign: list[str],
                   template: str = "T1_chat") -> dict:
    """L* = argmax_L ||dmean_L|| / rms_L over harmful-vs-benign, at the last token."""
    ph = [format_prompt(ro.tok, x, template) for x in harmful]
    pb = [format_prompt(ro.tok, x, template) for x in benign]
    Hh = ro.all_layer_hidden(ph)
    Hb = ro.all_layer_hidden(pb)
    nl = Hh.shape[0]
    dmean = Hh.mean(1) - Hb.mean(1)                      # [L+1, d]
    rms = np.sqrt((np.concatenate([Hh, Hb], axis=1) ** 2).mean(axis=(1, 2))) + 1e-9
    norm = np.linalg.norm(dmean, axis=1)
    sep = norm / rms
    # exclude the embedding layer (0) and the final layer from L* selection
    cand = np.arange(1, nl - 1) if nl > 3 else np.arange(nl)
    lstar = int(cand[np.argmax(sep[cand])])
    return {"L_star": lstar, "n_layers": int(nl),
            "delta": float(norm[lstar]), "rms_at_Lstar": float(rms[lstar]),
            "sep_profile": sep.tolist(), "norm_profile": norm.tolist(),
            "rms_profile": rms.tolist(),
            "dmean_all": dmean, "H_harm": Hh, "H_benign": Hb}


# ------------------------------------------------------------------ C1 ------
def ladder_prompts(fam: dict, order: list[int], template: str, tok,
                   third: str | None = None) -> list[str]:
    objs = [fam["objects"][i] for i in order]
    if third is not None:
        objs = objs + [third]
    return [format_prompt(tok, C.LADDER_TEMPLATE.format(OBJ=o), template) for o in objs]


def c1_ladder(ro: Readout, spacing: dict, *, templates=("T1_chat", "T2_raw", "T3_shared"),
              use_third_anchor: bool = False) -> dict:
    """A1 discrete ladder: ED50 (primary), Hill k (secondary), 3 templates,
    uniform vs severity respacing, plus the Tobit fallback estimand."""
    out = {"per_template": {}, "families": {}}
    for tpl in templates:
        per_fam = {}
        for fam in C.PAIR_FAMILIES:
            sp = spacing[fam["name"]]
            order = sp["frozen_order"]
            third = fam["third_anchor"] if use_third_anchor else None
            prompts = ladder_prompts(fam, order, tpl, ro.tok, third)
            R = ro.R(prompts)
            p = sigmoid(R)
            n = len(prompts)
            t_uni = np.array([i / (n - 1) for i in range(n)])
            s = list(sp["spacing_s"])
            if use_third_anchor:
                s = s + [1.5]
            t_sev = np.array(s, dtype=float)
            if use_third_anchor:
                t_uni = np.append(np.array([i / 11.0 for i in range(12)]), 1.5)
            fit_u = fit_ladder(t_uni, p, rng_seed=_seed(fam["name"]))
            fit_s = fit_ladder(t_sev, p, rng_seed=_seed(fam["name"]))
            per_fam[fam["name"]] = {
                "R": R.tolist(), "p": p.tolist(),
                "t_uniform": t_uni.tolist(), "t_severity": t_sev.tolist(),
                "fit_uniform": fit_u, "fit_severity": fit_s,
                "tobit_uniform": tobit_slope(t_uni, R),
                "R_low": float(R[0]), "R_high": float(R[-1]),
            }
        ids = [v["fit_uniform"]["ed50_interp"] for v in per_fam.values()]
        ids = [x for x in ids if x is not None]
        ks = [v["fit_uniform"]["hill_k"] for v in per_fam.values()
              if v["fit_uniform"]["hill_k"] is not None]
        ident = sum(v["fit_uniform"]["identifiable"] for v in per_fam.values())
        out["per_template"][tpl] = {
            "ed50_median": float(np.median(ids)) if ids else None,
            "ed50_mean": float(np.mean(ids)) if ids else None,
            "hill_k_median": float(np.median(ks)) if ks else None,
            "n_identifiable": int(ident),
            "n_ed50_defined": len(ids),
            "tobit_b_median": _safe_median([
                v["tobit_uniform"]["b"] for v in per_fam.values()
                if v["tobit_uniform"]["b"] is not None]),
            "tobit_tstar_median": _safe_median([
                v["tobit_uniform"]["t_star"] for v in per_fam.values()
                if v["tobit_uniform"]["t_star"] is not None]),
            "n_tobit_censored": int(sum(v["tobit_uniform"]["censored"]
                                        for v in per_fam.values())),
        }
        if tpl == "T1_chat":
            out["families"] = per_fam
            ed_u = [v["fit_uniform"]["ed50_interp"] for v in per_fam.values()]
            ed_s = [v["fit_severity"]["ed50_interp"] for v in per_fam.values()]
            both = [(a, b) for a, b in zip(ed_u, ed_s) if a is not None and b is not None]
            out["ed50_respacing_max_abs_diff"] = (
                float(max(abs(a - b) for a, b in both)) if both else None)
            k_u = [v["fit_uniform"]["hill_k"] for v in per_fam.values()]
            k_s = [v["fit_severity"]["hill_k"] for v in per_fam.values()]
            kb = [(a, b) for a, b in zip(k_u, k_s) if a is not None and b is not None]
            if len(kb) >= 4:
                r, _ = sps.spearmanr([a for a, _ in kb], [b for _, b in kb])
                out["slope_respacing_spearman_withinmodel"] = float(r)
            else:
                out["slope_respacing_spearman_withinmodel"] = None
    eds = {tpl: v["ed50_median"] for tpl, v in out["per_template"].items()
           if v["ed50_median"] is not None}
    out["ed50_sd_across_templates"] = (
        float(np.std(list(eds.values()), ddof=1)) if len(eds) >= 2 else None)
    out["ed50"] = out["per_template"].get("T1_chat", {}).get("ed50_median")
    out["hill_k"] = out["per_template"].get("T1_chat", {}).get("hill_k_median")
    out["tobit_b"] = out["per_template"].get("T1_chat", {}).get("tobit_b_median")
    out["tobit_tstar"] = out["per_template"].get("T1_chat", {}).get("tobit_tstar_median")
    out["n_identifiable_T1"] = out["per_template"].get("T1_chat", {}).get("n_identifiable")
    if out["ed50"] is None:
        out["failed"] = "CENSORED"
    return out


# ------------------------------------------------- C1 embedding interpolation ---
def _span_of_difference(tok, ids_a: list[int], ids_b: list[int]) -> tuple[int, int, int]:
    pre = 0
    while pre < min(len(ids_a), len(ids_b)) and ids_a[pre] == ids_b[pre]:
        pre += 1
    suf = 0
    while (suf < min(len(ids_a), len(ids_b)) - pre
           and ids_a[len(ids_a) - 1 - suf] == ids_b[len(ids_b) - 1 - suf]):
        suf += 1
    return pre, len(ids_a) - pre - suf, len(ids_b) - pre - suf


def c1_interp(ro: Readout, spacing: dict, template: str = "T1_chat") -> dict:
    """A2 embedding interpolation along the benign->harmful object span."""
    E = get_input_embedding_matrix(ro.model)
    # ONE float32 normalised copy of the embedding table; rebuilding it inside the
    # t-loop would allocate ~1.2GB per step and OOM the 16GB card.
    Enf = (E / (E.norm(dim=1, keepdim=True) + 1e-9)).float()
    ts = np.linspace(0.0, 1.0, N_INTERP)
    fams, equal_len = {}, 0
    for fam in C.PAIR_FAMILIES:
        order = spacing[fam["name"]]["frozen_order"]
        o_b = fam["objects"][order[0]]
        o_h = fam["objects"][order[-1]]
        pa = format_prompt(ro.tok, C.LADDER_TEMPLATE.format(OBJ=o_b), template)
        pb = format_prompt(ro.tok, C.LADDER_TEMPLATE.format(OBJ=o_h), template)
        ia, ib = ro.prompt_ids(pa), ro.prompt_ids(pb)
        pre, la, lb = _span_of_difference(ro.tok, ia, ib)
        if la <= 0 or lb <= 0:
            fams[fam["name"]] = {"failed": "SPAN_NOT_CONSTRUCTIBLE"}
            continue
        if la == lb:
            equal_len += 1
        L = max(la, lb)
        ea = E[torch.tensor(ia[pre: pre + la], device=E.device)]
        eb = E[torch.tensor(ib[pre: pre + lb], device=E.device)]
        if la < L:
            ea = torch.cat([ea, ea[-1:].repeat(L - la, 1)], 0)
        if lb < L:
            eb = torch.cat([eb, eb[-1:].repeat(L - lb, 1)], 0)
        base_ids = ia[:pre] + ib[pre: pre + lb] + ia[len(ia) - (len(ia) - pre - la):]
        suffix = ia[pre + la:]
        base_ids = ia[:pre] + (ib[pre: pre + lb] + [ib[pre + lb - 1]] * (L - lb))[:L] + suffix
        Rs, diag = [], []
        for t in ts:
            e = (1.0 - t) * ea + t * eb
            en = e / (e.norm(dim=1, keepdim=True) + 1e-9)
            cos = en.float() @ Enf.T
            mx, arg = cos.max(dim=1)
            near_ids = arg.detach().cpu().tolist()
            R = ro.R([""], prompt_id_lists=[base_ids],
                     embed_override=[(pre, e.detach())])[0]
            Rs.append(float(R))
            nll = _span_nll(ro, base_ids, pre, near_ids)
            diag.append({"t": float(t),
                         "max_cos_to_vocab": float(mx.mean().item()),
                         "min_cos_to_vocab": float(mx.min().item()),
                         "nearest_text": ro.tok.decode(near_ids)[:120],
                         "nearest_span_nll_per_token": nll})
            del cos, en, e
        Rarr = np.array(Rs)
        p = sigmoid(Rarr)
        fit = fit_ladder(ts, p, rng_seed=7)
        fams[fam["name"]] = {"R": Rs, "p": p.tolist(), "t": ts.tolist(),
                             "fit": fit, "offmanifold": diag,
                             "span_len_benign": la, "span_len_harmful": lb,
                             "equal_length_span": bool(la == lb)}
        del ea, eb
    del Enf
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    ok = [v for v in fams.values() if "fit" in v]
    eds = [v["fit"]["ed50_interp"] for v in ok if v["fit"]["ed50_interp"] is not None]
    mid = [d["max_cos_to_vocab"] for v in ok for d in v["offmanifold"]
           if 0.3 < d["t"] < 0.7]
    out = {"families": fams,
           "n_equal_length_spans": int(equal_len),
           "n_families_constructible": len(ok),
           "ed50_interp_median": float(np.median(eds)) if eds else None,
           "mid_path_max_cos_mean": float(np.mean(mid)) if mid else None,
           "n_identifiable": int(sum(v["fit"]["identifiable"] for v in ok))}
    if not eds:
        out["failed"] = "CENSORED"
    if len(ok) < 5 or (out["mid_path_max_cos_mean"] or 0) < 0.3:
        out["arm_status"] = "NOT_CONSTRUCTIBLE"
    else:
        out["arm_status"] = "CONSTRUCTED"
    return out


@torch.no_grad()
def _span_nll(ro: Readout, base_ids: list[int], start: int, span: list[int]) -> float:
    ids = list(base_ids)
    ids[start: start + len(span)] = span
    t = torch.tensor([ids], device=DEVICE)
    lg = torch.log_softmax(ro.model(input_ids=t, use_cache=False).logits.float(), -1)
    tot = 0.0
    for k, tid in enumerate(span):
        pos = start + k - 1
        if pos < 0:
            continue
        tot += -float(lg[0, pos, tid].item())
    return tot / max(1, len(span))


# ------------------------------------------------------------------ C2 ------
def c2_write_gain(ro: Readout, dl: dict, *, oracle_dir: np.ndarray | None,
                  oracle_layer: int | None, spacing: dict,
                  template: str = "T1_chat", mode: str = "last_prompt",
                  n_boot: int = 2000) -> dict:
    """Write gain W = dR/dalpha in units of the model's own harmful-benign
    displacement Delta, plus encode gain E and the chain-rule residual.
    Includes a RANDOM-DIRECTION control (handbook: steering needs one)."""
    lstar, delta = dl["L_star"], dl["delta"]
    if not np.isfinite(delta) or delta <= 0:
        return {"failed": "NO_DIRECTION"}
    r_self = dl["dmean_all"][lstar]
    r_self = r_self / (np.linalg.norm(r_self) + 1e-9)

    ben, har = [], []
    for fam in C.PAIR_FAMILIES:
        order = spacing[fam["name"]]["frozen_order"]
        ben.append(format_prompt(ro.tok, C.LADDER_TEMPLATE.format(
            OBJ=fam["objects"][order[0]]), template))
        har.append(format_prompt(ro.tok, C.LADDER_TEMPLATE.format(
            OBJ=fam["objects"][order[-1]]), template))
    prompts = ben + har
    n = len(ben)

    rng = np.random.default_rng(1234)
    r_rand = rng.normal(size=r_self.shape)
    r_rand /= np.linalg.norm(r_rand)

    arms = {"self": (r_self, lstar), "random_control": (r_rand, lstar)}
    if oracle_dir is not None and oracle_layer is not None \
            and oracle_dir.shape == r_self.shape:
        arms["oracle"] = (oracle_dir / (np.linalg.norm(oracle_dir) + 1e-9), oracle_layer)

    res = {"L_star": lstar, "delta": delta, "inject_mode": mode, "arms": {}}
    for arm, (vec, lay) in arms.items():
        curves = {}
        try:
            for a in ALPHAS:
                ro.inject = {"layer": int(lay),
                             "vec": torch.tensor(a * delta * vec, dtype=torch.float32,
                                                 device=DEVICE),
                             "mode": mode}
                curves[a] = ro.R(prompts)
        finally:
            # An injection that outlives this block would silently contaminate every
            # later candidate on this checkpoint.
            ro.inject = None
        M = np.stack([curves[a] for a in ALPHAS], axis=0)       # [n_alpha, n_prompt]
        lin = [i for i, a in enumerate(ALPHAS) if abs(a) <= 1.0]
        x = np.array([ALPHAS[i] for i in lin])
        W_per = []
        for j in range(M.shape[1]):
            y = M[lin, j]
            sl, ic, r_, _, _ = sps.linregress(x, y)
            W_per.append((float(sl), float(r_ ** 2)))
        W = np.array([w for w, _ in W_per])
        R2 = np.array([r for _, r in W_per])
        wb = _cluster_boot_mean(W, n_boot, seed=11)
        res["arms"][arm] = {
            "W_mean": float(W.mean()), "W_median": float(np.median(W)),
            "W_ci95": wb, "W_r2_mean": float(R2.mean()),
            "W_per_prompt": W.tolist(),
            "W_benign_mean": float(W[:n].mean()), "W_harmful_mean": float(W[n:].mean()),
            "R_curve_mean": {str(a): float(curves[a].mean()) for a in ALPHAS},
            "sanity_benign_up_with_positive_alpha":
                bool(curves[2.0][:n].mean() > curves[-2.0][:n].mean()),
            "sanity_harmful_down_with_negative_alpha":
                bool(curves[-2.0][n:].mean() < curves[2.0][n:].mean()),
        }

    # ---- encode gain E: d(proj(h,r)/Delta)/dt along the ladder ----
    Es = []
    for fam in C.PAIR_FAMILIES:
        sp = spacing[fam["name"]]
        order = sp["frozen_order"]
        pr = [format_prompt(ro.tok, C.LADDER_TEMPLATE.format(OBJ=fam["objects"][i]),
                            template) for i in order]
        hid, _ = ro.prompt_forward(pr, layers=[lstar])
        proj = (hid[lstar] @ r_self) / delta
        t = np.array(sp["spacing_s"])
        sl, _, r_, _, _ = sps.linregress(t, proj)
        Es.append((float(sl), float(r_ ** 2)))
    E = np.array([e for e, _ in Es])
    res["E_mean"] = float(E.mean())
    res["E_ci95"] = _cluster_boot_mean(E, n_boot, seed=12)
    res["E_r2_mean"] = float(np.mean([r for _, r in Es]))
    res["E_per_family"] = E.tolist()

    W_self = np.array(res["arms"]["self"]["W_per_prompt"]).mean()
    res["W"] = float(W_self)
    res["EW_product"] = float(res["E_mean"] * W_self)
    return res


def _cluster_boot_mean(vals: np.ndarray, n_boot: int, seed: int) -> list[float] | None:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ------------------------------------------------------------------ C3 ------
def c3_ratiometric(ro: Readout, spacing: dict, template: str = "T1_chat") -> dict:
    ben, har = [], []
    for fam in C.PAIR_FAMILIES:
        order = spacing[fam["name"]]["frozen_order"]
        ben.append(format_prompt(ro.tok, C.LADDER_TEMPLATE.format(
            OBJ=fam["objects"][order[0]]), template))
        har.append(format_prompt(ro.tok, C.LADDER_TEMPLATE.format(
            OBJ=fam["objects"][order[-1]]), template))
    S = float(ro.R(har).mean() - ro.R(ben).mean())

    un = [format_prompt(ro.tok, x, template) for x in C.EPISTEMIC_UNANSWERABLE]
    an = [format_prompt(ro.tok, x, template) for x in C.EPISTEMIC_ANSWERABLE]
    epi_u = ro.R(un, pos_set=EPISTEMIC_CONTS)
    epi_a = ro.R(an, pos_set=EPISTEMIC_CONTS)
    Epi = float(epi_u.mean() - epi_a.mean())
    return {"S": S, "Epi": Epi,
            "S_minus_Epi": S - Epi,                      # PRIMARY (log space)
            "S_over_Epi": S / max(Epi, 0.1),             # readability only
            "epi_unanswerable_mean": float(epi_u.mean()),
            "epi_answerable_mean": float(epi_a.mean())}


# ------------------------------------------------------------------ C4 ------
def paraphrase(behaviour: str, rule: dict) -> str:
    b = behaviour
    if rule.get("lowercase_first") and b:
        b = b[0].lower() + b[1:]
    return rule["prefix"] + b.rstrip(".") + rule["suffix"]


def c4_form_sensitivity(ro: Readout, template: str = "T1_chat") -> dict:
    cvs, per = [], []
    prompts, index = [], []
    for bi, beh in enumerate(C.HARMFUL_BEHAVIOURS):
        for rule in C.PARAPHRASE_RULES:
            prompts.append(format_prompt(ro.tok, paraphrase(beh, rule), template))
            index.append(bi)
    R = ro.R(prompts)
    index = np.array(index)
    for bi in range(len(C.HARMFUL_BEHAVIOURS)):
        r = R[index == bi]
        cv = float(np.std(r, ddof=1) / (abs(np.mean(r)) + CV_GUARD))
        cvs.append(cv)
        per.append({"behaviour_index": bi, "cv": cv, "mean_R": float(r.mean())})
    return {"cv_median": float(np.median(cvs)), "cv_mean": float(np.mean(cvs)),
            "cv_per_behaviour": [p["cv"] for p in per],
            "n_behaviours": len(cvs)}


# ------------------------------------------------------------------ C5 ------
def _token_row_ids(tok, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(int(t[0]))
    return sorted(ids)


def c5_weights_only(model, tok) -> dict:
    """Zero-prompt, zero-forward-pass weight statistics."""
    try:
        U_raw = get_unembedding(model)
    except (AttributeError, RuntimeError):
        return {"failed": "ARCH_UNSUPPORTED"}
    ref_ids = _token_row_ids(tok, C.REFUSAL_TOKEN_WORDS)
    ctl_words = [w for w in C.UNIGRAM_FREQ if w not in
                 {x.lower() for x in C.REFUSAL_TOKEN_WORDS}]
    ctl_ids = _token_row_ids(tok, ctl_words[:200] + C.CONTROL_TOKEN_WORDS)
    if len(ref_ids) < 3 or len(ctl_ids) < 10:
        return {"failed": "ARCH_UNSUPPORTED", "n_ref_rows": len(ref_ids),
                "n_ctl_rows": len(ctl_ids)}
    norms = U_raw.float().norm(dim=1)
    U = U_raw
    # match controls to refusal rows on row-norm decile
    ref_n = norms[torch.tensor(ref_ids, device=U.device)]
    qs = torch.quantile(norms.float(), torch.linspace(0, 1, 11, device=U.device))
    ref_dec = torch.bucketize(ref_n, qs[1:-1])
    ctl_t = torch.tensor(ctl_ids, device=U.device)
    ctl_dec = torch.bucketize(norms[ctl_t], qs[1:-1])
    matched = []
    for d in ref_dec.unique():
        pool = ctl_t[ctl_dec == d].tolist()
        matched.extend(pool[: max(4, int((ref_dec == d).sum().item()) * 4)])
    if len(matched) < 8:
        matched = ctl_ids
    out = {"n_ref_rows": len(ref_ids), "n_ctl_rows": len(matched),
           "vocab": int(U.shape[0]), "d_model": int(U.shape[1])}
    fro = float(torch.linalg.vector_norm(U_raw.float()))
    for tag, scale in (("raw", False), ("fro", True)):
        A = U_raw[torch.tensor(ref_ids, device=U.device)].float()
        B = U_raw[torch.tensor(matched, device=U.device)].float()
        if scale:
            A = A / (fro + 1e-9)
            B = B / (fro + 1e-9)
        out[f"w1_{tag}"] = float(_mean_pair_cos(A) - _mean_pair_cos(B))
        out[f"w2_{tag}"] = float(_tsv_share(A) - _tsv_share(B))
    # w3: depth localisation of mlp.down_proj stable rank
    try:
        layers = get_layers(model)
        srs = []
        for lyr in layers:
            w = None
            for path in ("mlp.down_proj", "mlp.c_proj", "mlp.fc2", "mlp.dense_4h_to_h"):
                obj, ok = lyr, True
                for part in path.split("."):
                    if not hasattr(obj, part):
                        ok = False
                        break
                    obj = getattr(obj, part)
                if ok and hasattr(obj, "weight"):
                    w = obj.weight
                    break
            if w is None:
                raise AttributeError("no down_proj")
            wf = w.detach().float()
            fro2 = float((wf ** 2).sum())
            sv = torch.linalg.matrix_norm(wf, ord=2)
            srs.append(fro2 / (float(sv) ** 2 + 1e-12))
            del wf
        srs = np.array(srs)
        d = np.arange(len(srs))
        co = np.polyfit(d, srs, 3)
        resid = srs - np.polyval(co, d)
        sd = resid.std(ddof=1) + 1e-12
        out["w3_depth_localisation"] = float(np.max(np.abs(resid)) / sd)
        out["w3_argmax_layer"] = int(np.argmax(np.abs(resid)))
        out["stable_rank_profile"] = srs.tolist()
    except (AttributeError, RuntimeError, np.linalg.LinAlgError, ValueError) as exc:
        out["w3_depth_localisation"] = None
        out["w3_error"] = type(exc).__name__
    qc = getattr(model.config, "quantization_config", None)
    out["quantization_config_present"] = qc is not None
    return out


def _mean_pair_cos(A: torch.Tensor) -> float:
    An = A / (A.norm(dim=1, keepdim=True) + 1e-9)
    G = An @ An.T
    n = G.shape[0]
    return float((G.sum() - G.diagonal().sum()) / (n * (n - 1) + 1e-9))


def _tsv_share(A: torch.Tensor) -> float:
    s = torch.linalg.svdvals(A.float())
    return float(s[0] ** 2 / ((s ** 2).sum() + 1e-12))


# ------------------------------------------------------------- B1 / B3 ------
def b1_logit_gap(ro: Readout, template: str = "T1_chat") -> dict:
    rid = first_token_ids(ro.tok, REFUSAL_CONTS)
    cid = first_token_ids(ro.tok, COMPLY_CONTS)
    ov = set(rid) & set(cid)
    rid = [i for i in rid if i not in ov]
    cid = [i for i in cid if i not in ov]
    if not rid or not cid:
        return {"failed": "ARCH_UNSUPPORTED"}
    pr = [format_prompt(ro.tok, b, template) for b in C.HARMFUL_BEHAVIOURS]
    _, ft = ro.prompt_forward(pr, want_ft_gap=True, ft_ids=(rid, cid))
    return {"ft_gap_mean": float(ft.mean()), "ft_gap_median": float(np.median(ft)),
            "n_refusal_first_tokens": len(rid), "n_comply_first_tokens": len(cid),
            "ft_gap_per_behaviour": ft.tolist()}


def b3_anchor_states(ro: Readout, lstar: int, template: str = "T1_chat") -> np.ndarray:
    pr = [format_prompt(ro.tok, a, template) for a in C.ANCHOR_POOL]
    hid, _ = ro.prompt_forward(pr, layers=[lstar])
    A = hid[lstar]
    A = A - A.mean(axis=0, keepdims=True)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    return A.astype(np.float32)
