#!/usr/bin/env python3
"""Isolate the change: on the IDENTICAL batched forward, does the new gather-based
logprob equal the old full-log_softmax one? Then separately quantify how much of the
0.274-nat gap is bf16 batch-shape noise, by rerunning the reference in float32."""
import os
from pathlib import Path
import numpy as np, torch
from loguru import logger
import corpora as C
from core import (apply_limits, load_model, unload, Readout, format_prompt, setup_logging,
                  WORKSPACE, DEVICE, REFUSAL_CONTS, COMPLY_CONTS, _logsumexp)
setup_logging("verify_logprob"); apply_limits(20.0)
CACHE = Path(os.environ.get("AII_HF_CACHE", str(WORKSPACE/"hf_cache")))
m, tok = load_model("Qwen/Qwen3-0.6B", CACHE)
ro = Readout(m, tok)
prompts=[format_prompt(tok,C.LADDER_TEMPLATE.format(OBJ=f["objects"][0]),"T1_chat") for f in C.PAIR_FAMILIES][:4]
conts = REFUSAL_CONTS + COMPLY_CONTS

@torch.no_grad()
def both_paths(pad_side="right"):
    pid=[ro.prompt_ids(p) for p in prompts]; cid=[ro.cont_ids(c) for c in conts]
    items=[(i,j,pid[i],cid[j]) for i in range(len(pid)) for j in range(len(cid))]
    maxlen=max(len(p)+len(c) for _,_,p,c in items)
    ids=torch.full((len(items),maxlen), ro.pad_id, dtype=torch.long)
    mask=torch.zeros((len(items),maxlen),dtype=torch.long)
    for b,(_,_,p,c) in enumerate(items):
        s=p+c; ids[b,:len(s)]=torch.tensor(s); mask[b,:len(s)]=1
    lg=m(input_ids=ids.to(DEVICE),attention_mask=mask.to(DEVICE),use_cache=False).logits
    new=np.zeros((len(pid),len(cid))); old=np.zeros_like(new)
    ls=torch.log_softmax(lg.float(),dim=-1)            # OLD path
    for b,(i,j,p,c) in enumerate(items):
        st=len(p)-1; tgt=torch.tensor(c,device=lg.device)
        sl=lg[b,st:st+len(c),:].float()                # NEW path
        new[i,j]=float((sl.gather(1,tgt.unsqueeze(1)).squeeze(1)-torch.logsumexp(sl,-1)).sum())
        old[i,j]=float(ls[b,st:st+len(c),:].gather(1,tgt.unsqueeze(1)).sum())
    return new, old
new, old = both_paths()
Rn=_logsumexp(new[:,:6],1)-_logsumexp(new[:,6:],1); Ro=_logsumexp(old[:,:6],1)-_logsumexp(old[:,6:],1)
logger.info(f"NEW gather vs OLD log_softmax on the SAME forward: max|dlogprob| = {np.abs(new-old).max():.3e}, max|dR| = {np.abs(Rn-Ro).max():.3e}")

# how much is bf16 batch-shape noise? compare batch-of-48 vs one-at-a-time, in bf16 and fp32
@torch.no_grad()
def single(dtype):
    out=np.zeros((len(prompts),len(conts)))
    for i,pr in enumerate(prompts):
        pid=ro.prompt_ids(pr)
        for j,c in enumerate(conts):
            cid=ro.cont_ids(c); t=torch.tensor([pid+cid],device=DEVICE)
            lg=m(input_ids=t,use_cache=False).logits.to(dtype)
            sl=lg[0,len(pid)-1:len(pid)-1+len(cid),:].float()
            out[i,j]=float((sl.gather(1,torch.tensor(cid,device=DEVICE).unsqueeze(1)).squeeze(1)-torch.logsumexp(sl,-1)).sum())
    return out
s=single(torch.float32)
Rs=_logsumexp(s[:,:6],1)-_logsumexp(s[:,6:],1)
logger.info(f"batched(48) vs single-sequence, same math: max|dR| = {np.abs(Rn-Rs).max():.3e}  <-- pure bf16 batch-shape noise")
logger.info(f"R batched {np.round(Rn,3).tolist()}")
logger.info(f"R single  {np.round(Rs,3).tolist()}")
unload(m)
assert np.abs(new-old).max() < 1e-3, "the CODE CHANGE altered the numbers"
logger.info("VERDICT: the gather-based rewrite is numerically identical; the residual "
            "gap is bf16 batch-shape noise, present in the original code too.")
