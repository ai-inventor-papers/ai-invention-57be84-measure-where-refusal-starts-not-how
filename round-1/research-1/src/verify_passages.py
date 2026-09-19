"""Fetch each source that carries supporting_passages and check the quote occurs verbatim.
Flushes one line per source so partial progress is always readable."""
import json, pathlib, subprocess, re, sys

W = pathlib.Path("/ai-inventor/aii_data/runs/run_hBXKTV59ahlj/3_invention_loop/iter_1/gen_art/gen_art_research_1")
SKILL = "/ai-inventor/.claude/skills/aii-web-tools"
PY = f"{SKILL}/../.ability_client_venv/bin/python"
out = json.loads((W / "research_out.json").read_text())

def norm(s: str) -> str:
    for a, b in [("’","'"),("‘","'"),("“",'"'),("”",'"'),("–","-"),("—","-"),("−","-"),(" "," "),(" "," ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()

def fetch(url: str, offsets) -> str:
    chunks = []
    for off in offsets:
        try:
            r = subprocess.run([PY, f"{SKILL}/scripts/aii_fast_web_fetch.py", "fetch", "--url", url,
                                "--max-chars", "80000", "--char-offset", str(off)],
                               capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            break
        if r.returncode != 0:
            break
        chunks.append(r.stdout)
    return norm("\n".join(chunks))

todo = [s for s in out["sources"] if s.get("supporting_passages")]
print(f"checking {len(todo)} sources with passages", flush=True)
tot = found = 0
for s in todo:
    ps = s["supporting_passages"]
    tot += len(ps)
    body = fetch(s["url"], (0,))
    if not body:
        print(f"[{s['index']:>2}] FETCH_FAILED (0/{len(ps)})  {s['url']}", flush=True)
        continue
    miss = [p["quote"] for p in ps if norm(p["quote"]) not in body]
    found += len(ps) - len(miss)
    print(f"[{s['index']:>2}] {len(ps)-len(miss)}/{len(ps)} found  {s['url']}", flush=True)
    for m in miss:
        print(f"       MISS: {m[:110]!r}", flush=True)
print(f"TOTAL {found}/{tot} passages located verbatim in the fetched text", flush=True)
