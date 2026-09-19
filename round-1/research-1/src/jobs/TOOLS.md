# Web tool setup (run this first in every Bash call, or export once per call)

export SKILL_DIR="/ai-inventor/.claude/skills/aii-web-tools"
export PY="$SKILL_DIR/../.ability_client_venv/bin/python"

## search
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "QUERY" --max-results 10
$PY "$SKILL_DIR/scripts/aii_fast_web_search.py" --query "QUERY" --mode scholarly

## fetch page/PDF as markdown
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" fetch --url "URL" --max-chars 12000 [--char-offset N]

## regex grep over full page/PDF text (BEST for exact numbers/quotes in PDFs)
$PY "$SKILL_DIR/scripts/aii_fast_web_fetch.py" grep --url "https://arxiv.org/pdf/2605.09875" --pattern "PATTERN" --max-matches 20 --context-chars 300 [-i]

Notes: pymupdf+html2text are installed and working. arXiv HTML at https://arxiv.org/html/<ID>v1 is often easier to grep than the PDF; try both.
You ALSO have built-in WebSearch and WebFetch tools - use them freely too.
