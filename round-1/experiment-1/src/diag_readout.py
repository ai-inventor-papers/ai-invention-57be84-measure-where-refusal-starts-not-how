#!/usr/bin/env python3
"""Pre-freeze readout diagnostic: which readout variant / template / think-mode
actually separates benign from harmful anchors? Run on the DEV triad only."""
import json, sys, time
import numpy as np, torch
from loguru import logger
import corpora as C
from core import (apply_limits, load_model, unload, Readout, format_prompt, setup_logging,
                  REFUSAL_CONTS, COMPLY_CONTS, _logsumexp, first_token_ids, WORKSPACE, DEVICE)
setup_logging("diag_readout")
apply_limits(20.0)
CACHE = WORKSPACE/"hf_cache"
SPACING = json.load(open("inputs/severity_spacing.json"))["spacing"]

def variants(ro, prompts):
    lp = ro.cont_logprobs(prompts, REFUSAL_CONTS+COMPLY_CONTS)
    nR, nC = len(REFUSAL_CONTS), len(COMPLY_CONTS)
    lenR = np.array([len(ro.cont_ids(c)) for c in REFUSAL_CONTS])
    lenC = np.array([len(ro.cont_ids(c)) for c in COMPLY_CONTS])
    out = {}
    out["sum_lse"] = _logsumexp(lp[:,:nR],1) - _logsumexp(lp[:,nR:],1)
    out["mean_lse"] = _logsumexp(lp[:,:nR]/lenR,1) - _logsumexp(lp[:,nR:]/lenC,1)
    out["sum_max"] = lp[:,:nR].max(1) - lp[:,nR:].max(1)
    out["mean_avg"] = (lp[:,:nR]/lenR).mean(1) - (lp[:,nR:]/lenC).mean(1)
    return out

for repo in ["Qwen/Qwen3-1.7B","Qwen/Qwen3-1.7B-Base","huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2"]:
    try:
        m,tok = load_model(repo, CACHE)
    except Exception as e:
        logger.error(f"{repo}: {type(e).__name__} {str(e)[:200]}"); continue
    ro = Readout(m,tok)
    for tpl in ("T1_chat","T2_raw","T3_shared"):
        for mode in ("canonical","pos1_raw","after_think_close"):
            if tpl!="T1_chat" and mode!="canonical": continue
            ben=[format_prompt(tok,C.LADDER_TEMPLATE.format(OBJ=f["objects"][SPACING[f["name"]]["frozen_order"][0]]),tpl,mode) for f in C.PAIR_FAMILIES]
            har=[format_prompt(tok,C.LADDER_TEMPLATE.format(OBJ=f["objects"][SPACING[f["name"]]["frozen_order"][-1]]),tpl,mode) for f in C.PAIR_FAMILIES]
            vb, vh = variants(ro,ben), variants(ro,har)
            msg=[]
            for k in vb:
                d = vh[k]-vb[k]
                msg.append(f"{k}: wins {int((d>0).sum())}/10 mean_gap {d.mean():+.2f} d'={d.mean()/(d.std()+1e-9):.2f}")
            logger.info(f"{repo.split('/')[-1]:38s} {tpl:9s} {mode:18s} | " + " | ".join(msg))
    # first-token gap for reference
    rid=first_token_ids(tok,REFUSAL_CONTS); cid=first_token_ids(tok,COMPLY_CONTS)
    ov=set(rid)&set(cid); rid=[i for i in rid if i not in ov]; cid=[i for i in cid if i not in ov]
    ben=[format_prompt(tok,C.LADDER_TEMPLATE.format(OBJ=f["objects"][SPACING[f["name"]]["frozen_order"][0]]),"T1_chat") for f in C.PAIR_FAMILIES]
    har=[format_prompt(tok,C.LADDER_TEMPLATE.format(OBJ=f["objects"][SPACING[f["name"]]["frozen_order"][-1]]),"T1_chat") for f in C.PAIR_FAMILIES]
    _,fb=ro.prompt_forward(ben,want_ft_gap=True,ft_ids=(rid,cid))
    _,fh=ro.prompt_forward(har,want_ft_gap=True,ft_ids=(rid,cid))
    d=fh-fb
    logger.info(f"{repo.split('/')[-1]:38s} FIRSTTOKEN  wins {int((d>0).sum())}/10 gap {d.mean():+.2f} d'={d.mean()/(d.std()+1e-9):.2f}")
    unload(m)
logger.info("DIAG DONE")
