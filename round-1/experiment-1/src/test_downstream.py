#!/usr/bin/env python3
"""Exercise figures.py + finalize.py against whatever checkpoints exist so far."""
import json, sys
from pathlib import Path
from core import jdump, clean_floats, setup_logging, WORKSPACE
from loguru import logger
setup_logging("test_downstream")
R = WORKSPACE/"results"
st = json.loads((R/"method_out_raw.json").read_text())
rows = [json.loads(p.read_text()) for p in sorted((R/"ckpt").glob("*.json"))]
st["per_checkpoint"] = rows
logger.info(f"{len(rows)} checkpoints available")
import method as M, finalize as F, figures as FG
if len(rows) >= 6:
    st.update(M.build_screen(rows, st["gates"]["G1"]))
jdump(clean_floats(st), R/"method_out_raw.json")
figs = FG.build_all(st)
jdump(figs, R/"figures_index.json")
logger.info(f"figures: {sum(1 for f in figs if f.get('ok'))}/{len(figs)} ok")
for f in figs:
    if not f.get("ok"): logger.warning(f"FAILED {f['figure']}: {f.get('stderr') or f.get('error')}")
out = F.build(st)
jdump(out, WORKSPACE/"method_out.json")
print(json.dumps({"verdict": out["metadata"]["verdict"],
                  "datasets": {d["dataset"]: len(d["examples"]) for d in out["datasets"]}}, indent=2))
