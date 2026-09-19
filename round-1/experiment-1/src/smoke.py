#!/usr/bin/env python3
"""STAGE 0-2 confirmation signals: env, readout sanity, determinism, hook sanity."""
import json, sys, time
import numpy as np, torch
from loguru import logger
import corpora as C
from core import (apply_limits, load_model, unload, Readout, format_prompt,
                  setup_logging, free_disk_gb, WORKSPACE, DEVICE, supports_enable_thinking,
                  _resolve_canonical)
from candidates import decision_layer, sigmoid
import candidates as CAND

setup_logging("smoke")
hw = apply_limits(ram_budget_gb=20.0)
logger.info(f"free disk {free_disk_gb('/'):.1f} GB")
CACHE = WORKSPACE / "hf_cache"

t0 = time.time()
m, tok = load_model("Qwen/Qwen3-0.6B", CACHE)
logger.info(f"loaded in {time.time()-t0:.0f}s; canonical mode = {_resolve_canonical(tok)}")
ro = Readout(m, tok)

# --- STAGE 0: one forward ---
ids = torch.tensor([ro.prompt_ids(format_prompt(tok, "Hello", "T1_chat"))], device=DEVICE)
with torch.no_grad():
    lg = m(input_ids=ids).logits
logger.info(f"STAGE0 logits {tuple(lg.shape)}; free VRAM "
            f"{torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB")

# --- STAGE 1: readout sanity ---
ben = [format_prompt(tok, C.LADDER_TEMPLATE.format(OBJ=f["objects"][0]), "T1_chat") for f in C.PAIR_FAMILIES]
har = [format_prompt(tok, C.LADDER_TEMPLATE.format(OBJ=f["objects"][-1]), "T1_chat") for f in C.PAIR_FAMILIES]
t0=time.time(); Rb = ro.R(ben); Rh = ro.R(har); dt=time.time()-t0
wins = int((Rh > Rb).sum())
logger.info(f"STAGE1 R(harm)>R(benign) in {wins}/10 families ({dt:.1f}s for 20 prompts)")
logger.info(f"  R_benign {np.round(Rb,2).tolist()}")
logger.info(f"  R_harmful {np.round(Rh,2).tolist()}")
logger.info(f"  spread |Rh-Rb| mean = {np.abs(Rh-Rb).mean():.2f} nats")
Rb2 = ro.R(ben)
logger.info(f"STAGE1 determinism max|dR| = {np.abs(Rb-Rb2).max():.2e}")

# --- STAGE 2: hook sanity ---
dl = decision_layer(ro, C.HARMFUL_BEHAVIOURS, C.ANCHOR_POOL[:40])
logger.info(f"STAGE2 L*={dl['L_star']}/{dl['n_layers']} delta={dl['delta']:.2f} "
            f"rms={dl['rms_at_Lstar']:.2f}")
r = dl["dmean_all"][dl["L_star"]]; r = r/np.linalg.norm(r)
R0 = ro.R(ben+har)
ro.inject = {"layer": dl["L_star"], "vec": torch.zeros(len(r), dtype=torch.float32, device=DEVICE), "mode":"last_prompt"}
Rz = ro.R(ben+har); ro.inject=None
logger.info(f"STAGE2 alpha=0 identity max|dR| = {np.abs(R0-Rz).max():.2e}")
for mode in ("last_prompt","all_prompt"):
    out={}
    for a in (-2.,-1.,-0.5,0.,0.5,1.,2.):
        ro.inject={"layer":dl["L_star"],"vec":torch.tensor(a*dl["delta"]*r,dtype=torch.float32,device=DEVICE),"mode":mode}
        out[a]=ro.R(ben+har)
    ro.inject=None
    from scipy import stats as sps
    M=np.stack([out[a] for a in (-1.,-0.5,0.,0.5,1.)],0); x=np.array([-1.,-0.5,0.,0.5,1.])
    sl=[sps.linregress(x,M[:,j]) for j in range(M.shape[1])]
    W=np.array([s.slope for s in sl]); R2=np.array([s.rvalue**2 for s in sl])
    logger.info(f"STAGE2 mode={mode}: W mean {W.mean():.3f} R2 mean {R2.mean():.3f} | "
                f"benign +alpha up: {out[2.][:10].mean()>out[-2.][:10].mean()} | "
                f"harmful -alpha down: {out[-2.][10:].mean()<out[2.][10:].mean()}")
json.dump({"wins":wins,"hw":hw,"L_star":dl["L_star"]}, open("results/smoke.json","w"), indent=2)
unload(m)
logger.info("SMOKE DONE")
