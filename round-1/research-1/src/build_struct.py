"""Emit ./.terminal_claude_agent_struct_out.json matching the ResearchArtifact schema."""
import json, pathlib

W = pathlib.Path("/ai-inventor/aii_data/runs/run_hBXKTV59ahlj/3_invention_loop/iter_1/gen_art/gen_art_research_1")
out = json.loads((W / "research_out.json").read_text())

summary = (W / "summary.txt").read_text().strip()

struct = {
 "title": "Prior art and specs for a cheap safety metric",
 "layman_summary": ("Checks what has already been published about measuring an AI model's safety cheaply, then writes down the exact "
                    "recipes, prompts and settings the next experiment needs so it can build instead of search."),
 "summary": summary,
 "out_expected_files": {"output": "research_out.json"},
 "upload_ignore_regexes": ["(^|/)\\.repl_agent\\.ptylog$"],
 "answer": out["answer"],
 "sources": out["sources"],
 "follow_up_questions": out["follow_up_questions"],
}
p = W / ".terminal_claude_agent_struct_out.json"
p.write_text(json.dumps(struct, indent=2, ensure_ascii=False))
print("wrote", p, p.stat().st_size, "bytes")
print("title len", len(struct["title"]), "| layman len", len(struct["layman_summary"]), "| summary len", len(struct["summary"]))
