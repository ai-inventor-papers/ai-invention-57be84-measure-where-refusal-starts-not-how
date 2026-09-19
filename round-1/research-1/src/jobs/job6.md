# Per-family chat-template readout-position spec (Job 6)

Full machine-readable version: `job6.json` (same directory). All claims below are sourced with URLs; every raw `tokenizer_config.json` was fetched directly (`.../raw/main/tokenizer_config.json`) except where noted as a mirror (gated repos returned HTTP 401 on the raw file).

**Global rule:** read logits at index **-1** after
`tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors='pt')`.
Use the `tokenize=True` path directly — never `tokenize=False` + a separate `tokenizer(text)` call with default `add_special_tokens=True`. Per HF's own chat-templating docs: *"When you format text with `apply_chat_template(tokenize=False)`, make sure you set `add_special_tokens=False` if you tokenize later to avoid duplicating these tokens. This isn't an issue if you use `apply_chat_template(tokenize=True)`, which means it's usually the safer option!"* ([source](https://huggingface.co/docs/transformers/main/chat_templating)). This one rule resolves the Llama-3.2 / Gemma-2 / Gemma-3 / InternLM2.5 double-BOS risk uniformly.

## Per-family suffix table

| Family | Instruct repo | Param size | Generation-prompt suffix (exact) | Trailing `\n`? | BOS in template? | Forced prefix? | Rule OK? |
|---|---|---|---|---|---|---|---|
| Qwen2.5 | Qwen/Qwen2.5-1.5B-Instruct | 1.54B | `<\|im_start\|>assistant\n` | yes | no (no BOS at all) | **yes** — default persona system turn if none supplied | OK |
| Qwen3 | Qwen/Qwen3-1.7B | 1.7B | default: `<\|im_start\|>assistant\n` · `enable_thinking=False`: `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | yes | no | no default text (thinking block only) | **WRONG by default** — see below |
| Llama-3.2 | meta-llama/Llama-3.2-1B-Instruct | 1.24B | `<\|start_header_id\|>assistant<\|end_header_id\|>\n\n` | yes | **yes, explicit `{{bos_token}}`** — classic double-BOS risk | **yes** — always emits cutting-knowledge/date header | OK if `tokenize=True` |
| Gemma-2 | google/gemma-2-2b-it | 2.6B | `<start_of_turn>model\n` | yes | **yes, explicit `{{bos_token}}`** | system role **forbidden** (raises exception) | OK if `tokenize=True`; never pass a system message |
| Gemma-3 | google/gemma-3-1b-it | 1.0B | `<start_of_turn>model\n` | yes | **yes, explicit `{{bos_token}}`** | system content silently folded into first user turn (not rejected, not literal) | OK if `tokenize=True` |
| Phi-4-mini | microsoft/Phi-4-mini-instruct | ~3.8B (**>2B, flagged**) | `<\|assistant\|>` | **no** | no | no | OK, but no leading space on first token |
| SmolLM2 | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.71B | `<\|im_start\|>assistant\n` | yes | bos_token string == template's own turn marker (confusable, not doubled) | **yes** — default SmolLM persona | OK |
| SmolLM3 | HuggingFaceTB/SmolLM3-3B | 3.08B (**>2B, flagged**) | default: `<\|im_start\|>assistant\n` · `enable_thinking=False`: `<\|im_start\|>assistant\n<think>\n\n</think>\n` (single trailing `\n`, differs from Qwen3's `\n\n`) | yes | no | **yes** — metadata header + independent `/no_think` soft switch | **WRONG by default** — same failure as Qwen3 |
| OLMo-2 | allenai/OLMo-2-0425-1B-Instruct | 1B | `<\|assistant\|>\n` | yes | explicit `{{bos_token}}` but tokenizer's own `add_bos_token` unset (GPT2-family) → likely single BOS, UNVERIFIED at token-id level | no | OK |
| MiniCPM | openbmb/MiniCPM-2B-sft-bf16 | ~2.4B | **none** — `add_generation_prompt` is a no-op; `<AI>` is baked into every user-turn rendering | no | no (tokenizer-level `add_bos_token=True`, one BOS, no template literal) | N/A | **WRONG** — flag is ignored entirely |
| StableLM-2 | stabilityai/stablelm-2-1_6b-chat | 1.6B | `<\|im_start\|>assistant\n` | yes | no | **yes** — default "helpful assistant" persona | OK |
| InternLM2.5 | internlm/internlm2_5-1_8b-chat | 1.8B | `<\|im_start\|>assistant\n` | yes | **yes, explicit `{{bos_token}}`** | no | OK if `tokenize=True` |
| H2O-Danube3 | h2oai/h2o-danube3-500m-chat | 500M | `<\|answer\|>` | **no** | no (`add_bos_token` explicitly forced False) | system role **forbidden** (raises exception) | OK, but no leading space on first token |

## Triad availability (base / instruct / abliterated) and refusal evidence

| Family | base_repo | abliterated_repo | refuses_at_this_size | recommend_include |
|---|---|---|---|---|
| Qwen2.5 | Qwen/Qwen2.5-1.5B | huihui-ai/Qwen2.5-1.5B-Instruct-abliterated | **TRUE**, Safety=77.6 (Tulu-3 benchmark, [OLMo-2 card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)) | **true** |
| Qwen3 | Qwen/Qwen3-1.7B-Base | mlabonne/Qwen3-1.7B-abliterated | TRUE (indirect — mlabonne's own card describes computing a refusal direction) | **true**, with `enable_thinking=False` |
| Llama-3.2 | meta-llama/Llama-3.2-1B | huihui-ai/Llama-3.2-1B-Instruct-abliterated | TRUE — RLHF safety documented; Llama-3.1-1B proxy scores 87.2 Safety | **true** |
| Gemma-2 | google/gemma-2-2b | IlyaGusev/gemma-2-2b-it-abliterated | TRUE — documented Ethics & Safety red-teaming | **true** (no system messages) |
| Gemma-3 | google/gemma-3-1b-pt | mlabonne/gemma-3-1b-it-abliterated | TRUE, Safety=70.2 (Tulu-3 benchmark) | **true** |
| Phi-4-mini | **null** (none released) | huihui-ai/Phi-4-mini-instruct-abliterated | TRUE — dedicated Safety Eval & Red-Teaming section | true, size caveat |
| SmolLM2 | HuggingFaceTB/SmolLM2-1.7B | huihui-ai/SmolLM2-1.7B-Instruct-abliterated | TRUE, Safety=52.4 (Tulu-3 benchmark) | **true** |
| SmolLM3 | HuggingFaceTB/SmolLM3-3B-Base | richardyoung/SmolLM3-3B-abliterated-obliteratus | **UNKNOWN** — no safety text on card at all | true, mechanism-leg (thinking corroboration) |
| OLMo-2 | allenai/OLMo-2-0425-1B | arzaan789/olmo2-1b-uncensored | TRUE, Safety=87.6, **self-reported on its own card** — best-documented number in the set | **true** |
| MiniCPM | **null** | **null** | UNKNOWN — no safety text found | **false** |
| StableLM-2 | stabilityai/stablelm-2-1_6b | arzaan789/stablelm-2-1.6b-uncensored | **WEAK/self-admitted** — card states model "is also willing to output potentially harmful outputs... when the user requests it" | **false** (mostly censored cells) |
| InternLM2.5 | internlm/internlm2_5-1_8b | **null** at ≤4B (only a 7B sibling exists) | WEAK — generic boilerplate only | **false** |
| H2O-Danube3 | h2oai/h2o-danube3-500m-base | **null** (only Danube-v1 uncensored exists, different model) | UNKNOWN — generic liability disclaimer only | **false** |

## Qwen3 thinking-block — recommendation

1. **`enable_thinking=False` mechanism (confirmed from the actual jinja):** passing `enable_thinking=False` to `apply_chat_template` triggers this exact tail block (verbatim from `Qwen/Qwen3-1.7B/tokenizer_config.json`):
   ```
   {%- if add_generation_prompt %}
       {{- '<|im_start|>assistant\n' }}
       {%- if enable_thinking is defined and enable_thinking is false %}
           {{- '<think>\n\n</think>\n\n' }}
       {%- endif %}
   {%- endif %}
   ```
   Resulting literal suffix: **`<|im_start|>assistant\n<think>\n\n</think>\n\n`** — a confirmed empty `<think></think>` pair, exactly as hypothesized.

2. **`/no_think` soft switch:** documented on the [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) under "Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input." It is appended to the **end of the user turn's text content** (e.g. `"Then, how many r's in blueberries? /no_think"`), works **only when `enable_thinking=True`** (layered on top of, not instead of, the hard switch), and is **not parsed by the jinja at all** — it is a purely trained behavioral convention (confirmed: no `/no_think` string literal anywhere in the template).

3. **Triad consistency:** `mlabonne/Qwen3-1.7B-abliterated`'s chat template was diffed line-by-line against the base `Qwen/Qwen3-1.7B` template. The `add_generation_prompt`/`enable_thinking` tail block is **byte-identical** in both (only cosmetic tool-call variable-guard differences elsewhere). The abliterated checkpoint preserves the template verbatim.

**Recommended rule:** use `enable_thinking=False` uniformly for base(N/A)/instruct/abliterated and read index -1 of `apply_chat_template(..., add_generation_prompt=True, tokenize=True, enable_thinking=False)`. This is the only rule available identically across the triad, produces a fixed cross-comparable suffix, and matches Qwen's own stated intent of aligning non-thinking behavior with Qwen2.5-Instruct.

**Fallback:** generate until `</think>\n\n` is produced (or hit a token-budget backstop) and read the first token of `content` immediately after — but log this explicitly as a **different, variable-position** measurement, never silently merged with the static index--1 rule.

**Safest option (recommended if reviewer scrutiny is a concern):** exclude Qwen3 from the family-level transfer analysis, keep it only for the thinking-block mechanism leg — corroborated independently by **SmolLM3**, which implements the identical `enable_thinking=True`-default / soft-switch pattern in its own separately-hosted `chat_template.jinja` (confirmed: `HuggingFaceTB/SmolLM3-3B/raw/main/chat_template.jinja`), with a subtly different empty-think suffix (`<think>\n\n</think>\n` — single trailing `\n`, not Qwen3's `\n\n`) that must be pre-registered separately, not assumed identical.

## Base-model rule (pre-registered)

Base checkpoints ship **no chat_template**. Rule: `raw_prompt_text + "\n"` (a single newline, identical across families), tokenized with the tokenizer's own default `add_special_tokens=True` (which yields a single family-specific BOS per each family's own `add_bos_token` default, tabulated above — this per-family variation is itself pre-registered, not left implicit). Read index -1, same position convention as instruct. This makes base readings **non-comparable to instruct readings on the template axis** — expected and acceptable since base models are right-censored, but must be stated up front.

## Families recommended to drop (or mechanism-leg-only)

- **MiniCPM** — no abliterated checkpoint at this generation/size, no base_repo found, no refusal evidence, `add_generation_prompt` is a silent no-op in its template.
- **InternLM2.5** — no ≤4B abliterated checkpoint (only a 7B sibling), only generic safety boilerplate.
- **H2O-Danube3** — no Danube3-specific abliterated checkpoint (only an older Danube-v1 variant), no quantified refusal evidence, smallest model in the set (500M).
- **StableLM-2** — Stability AI's own card admits the model largely complies with harmful requests on direct ask; recommend excluding from the refusal-detection leg.

## UNVERIFIED / could not confirm

- Token-id-level confirmation (not just config flags) that OLMo-2 and StableLM-2 emit exactly one BOS under `tokenize=True` — no local `encode()` call was run (no compute available to this job).
- Llama-3.2-1B-Instruct's own exact Tulu-3 Safety benchmark number (only the 3.1-1B proxy, 87.2, and the qualitative RLHF claim on its own card, are directly sourced).
- SmolLM3-3B and MiniCPM-2B-sft-bf16 refusal behavior — no safety/refusal text found on either model card at all (marked UNKNOWN, not FALSE).
- google/gemma-2-2b-it and meta-llama/Llama-3.2-1B-Instruct raw `tokenizer_config.json`: official repos are gated (HTTP 401 without an authenticated token); all quotes for these two come from the widely-used ungated `unsloth/*` re-upload mirrors instead, not diffed byte-for-byte against the gated original (no access to do so).
- google/gemma-3-1b-it: same mirror caveat as above (used `unsloth/gemma-3-1b-it`).
- Exact param counts for Qwen3-1.7B / Llama-3.2-1B / Gemma-3-1b / OLMo-2-1B are the commonly-cited model-card figures, not independently recomputed from `config.json`.
