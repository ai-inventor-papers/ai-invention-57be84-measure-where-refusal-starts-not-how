"""Re-check only the passages that previously failed, plus the one that could not be fetched."""
import json, pathlib, subprocess, re
SKILL="/ai-inventor/.claude/skills/aii-web-tools"; PY=f"{SKILL}/../.ability_client_venv/bin/python"
out=json.loads(pathlib.Path("research_out.json").read_text())
TARGETS={16,22,23,30,31,32,34}
def norm(s):
    for a,b in [("’","'"),("‘","'"),("“",'"'),("”",'"'),("–","-"),("—","-"),("−","-"),
                (" "," "),("⁢",""),("⁡",""),(" "," ")]:
        s=s.replace(a,b)
    return re.sub(r"\s+"," ",s).strip().lower()
for s in out["sources"]:
    if s["index"] not in TARGETS or not s["supporting_passages"]: continue
    try:
        r=subprocess.run([PY,f"{SKILL}/scripts/aii_fast_web_fetch.py","fetch","--url",s["url"],"--max-chars","90000"],
                         capture_output=True,text=True,timeout=120)
        body=norm(r.stdout) if r.returncode==0 else ""
    except subprocess.TimeoutExpired:
        body=""
    if not body:
        print(f"[{s['index']:>2}] FETCH_FAILED  {s['url']}",flush=True); continue
    miss=[q["quote"] for q in s["supporting_passages"] if norm(q["quote"]) not in body]
    print(f"[{s['index']:>2}] {len(s['supporting_passages'])-len(miss)}/{len(s['supporting_passages'])} found  {s['url']}",flush=True)
    for m in miss: print(f"       STILL MISSING: {m[:120]!r}",flush=True)
print("done",flush=True)
