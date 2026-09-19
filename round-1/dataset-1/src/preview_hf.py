#!/usr/bin/env python3
"""Preview candidate HF datasets: configs, splits, size, and sample rows."""
from __future__ import annotations
import json, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi

CANDS = [
 "Paul/XSTest","walledai/MaliciousInstruct","Bertievidgen/SimpleSafetyTests","declare-lab/HarmfulQA",
 "walledai/WildGuardTest","walledai/AyaRedTeaming","SillyTilly/SorryBench","thu-coai/Safety-Prompts",
 "AlignmentResearch/XSTest","huihui-ai/harmbench_behaviors","kelly8tom/advbench_orig",
 "walledai/SimpleSafetyTests","Machlovi/strongreject-dataset","jkazdan/HeX-PHI-usable",
 "AlignmentResearch/SorryBench","Lv111/StrongREJECT","Lv111/harmbench_behaviors_text_all",
]

def info(rid: str) -> dict:
    api = HfApi()
    d = {"id": rid}
    try:
        m = api.dataset_info(rid, files_metadata=True)
        d["downloads"] = m.downloads; d["likes"] = m.likes
        d["last_modified"] = str(m.last_modified); d["sha"] = m.sha
        d["gated"] = bool(m.gated); d["tags"] = (m.tags or [])[:15]
        sz = sum((s.size or 0) for s in (m.siblings or []))
        d["repo_bytes"] = sz
        d["files"] = [s.rfilename for s in (m.siblings or [])][:40]
        cd = m.card_data.to_dict() if m.card_data else {}
        d["license"] = cd.get("license")
        d["configs_cardmeta"] = cd.get("configs")
    except Exception as e:  # noqa: BLE001
        d["error"] = repr(e)[:300]
    return d

def rows(rid: str) -> dict:
    import requests
    out = {"id": rid}
    try:
        r = requests.get("https://datasets-server.huggingface.co/splits",
                         params={"dataset": rid}, timeout=45).json()
        sp = r.get("splits", [])
        out["splits"] = [(s["config"], s["split"], ) for s in sp][:20]
        if sp:
            s0 = sp[0]
            rr = requests.get("https://datasets-server.huggingface.co/first-rows",
                              params={"dataset": rid, "config": s0["config"], "split": s0["split"]},
                              timeout=60).json()
            out["columns"] = [f["name"] for f in rr.get("features", [])]
            out["sample"] = json.loads(json.dumps(rr.get("rows", [])[:2]))[:2]
            out["n_rows_split0"] = rr.get("num_rows_total")
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:300]
    return out

def main() -> None:
    with ThreadPoolExecutor(10) as ex:
        meta = list(ex.map(info, CANDS))
        rws = list(ex.map(rows, CANDS))
    merged = []
    for m, r in zip(meta, rws):
        m.update({k: v for k, v in r.items() if k != "id"})
        merged.append(m)
    open(sys.argv[1], "w").write(json.dumps(merged, indent=1)[:0] or json.dumps(merged, indent=1))
    for m in merged:
        s = json.dumps(m.get("sample", []))[:600]
        print(f'\n### {m["id"]}  dl={m.get("downloads")} likes={m.get("likes")} '
              f'MB={round((m.get("repo_bytes") or 0)/1e6,1)} lic={m.get("license")} gated={m.get("gated")}')
        print(f'   splits={m.get("splits")}')
        print(f'   cols={m.get("columns")} n0={m.get("n_rows_split0")} err={m.get("error","")[:120]}')
        print(f'   sample={s}')
main()
