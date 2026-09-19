#!/usr/bin/env python3
"""Core infrastructure: hardware budgeting, model loading, prompt templating, readout.

The READOUT is the single most important object here. It is a teacher-forced
continuation log-likelihood contrast (refusal continuations vs compliance
continuations), which is forward-pass-only and tokenizer-robust -- unlike a
single ambiguous first token.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import resource
import shutil
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- logging ----
def setup_logging(name: str = "run") -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(WORKSPACE / "logs" / f"{name}.log", rotation="30 MB", level="DEBUG")


# --------------------------------------------------------------- hardware ----
def _detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _container_ram_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


NUM_CPUS = _detect_cpus()
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0.0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
DTYPE = torch.bfloat16 if HAS_GPU else torch.float32
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9


def apply_limits(ram_budget_gb: float = 20.0, vram_fraction: float = 0.92) -> dict:
    """Hard RAM/VRAM caps so we raise a catchable error instead of being OOM-killed."""
    avail = psutil.virtual_memory().available / 1e9
    ram_budget_gb = min(ram_budget_gb, TOTAL_RAM_GB * 0.85, avail * 0.9)
    nbytes = int(ram_budget_gb * 1e9)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes * 3, nbytes * 3))
    except (ValueError, OSError) as exc:  # pragma: no cover
        logger.warning(f"could not set RLIMIT_AS: {exc}")
    if HAS_GPU:
        torch.cuda.set_per_process_memory_fraction(min(vram_fraction, 0.95), 0)
    info = {"num_cpus": NUM_CPUS, "has_gpu": HAS_GPU, "vram_gb": round(VRAM_GB, 2),
            "total_ram_gb": round(TOTAL_RAM_GB, 2), "ram_budget_gb": round(ram_budget_gb, 2),
            "device": str(DEVICE), "dtype": str(DTYPE),
            "gpu_name": torch.cuda.get_device_name(0) if HAS_GPU else None}
    logger.info(f"hardware: {info}")
    return info


def free_disk_gb(path: str | Path = "/") -> float:
    st = shutil.disk_usage(str(path))
    return st.free / 1e9


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_jdefault))


def _jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def clean_floats(o):
    """Replace NaN/Inf with None recursively so the JSON is strictly valid."""
    if isinstance(o, dict):
        return {k: clean_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_floats(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean_floats(o.tolist())
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


# ------------------------------------------------------ continuation sets ----
REFUSAL_CONTS = [
    "I cannot help with that.",
    "I'm sorry, but I can't help with that.",
    "Sorry, I can't assist with that request.",
    "I won't provide that information.",
    "I'm not able to help with this.",
    "I can't provide instructions for that.",
]
COMPLY_CONTS = [
    "Sure, here are the steps:",
    "Here's how to do it:",
    "Step 1:",
    "Certainly. First, you will need",
    "Of course. Begin by",
    "The process is as follows:",
]
EPISTEMIC_CONTS = [
    "I don't know.",
    "That cannot be determined.",
    "There is no way to know that.",
    "I'm not sure.",
]

THINK_CLOSE = "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------- loading ----
def load_model(repo_id: str, cache_dir: Path, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(cache_dir),
                                        trust_remote_code=False, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(cache_dir=str(cache_dir), trust_remote_code=False, revision=revision,
              low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=DTYPE, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=DTYPE, **kw)
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False
    return model, tok


def unload(model) -> None:
    try:
        model.to("cpu")
    except (RuntimeError, AttributeError):
        pass
    del model
    gc.collect()
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_layers(model):
    """Return the decoder-layer ModuleList for the common llama-style families."""
    for attr in ("model.layers", "model.model.layers", "transformer.h",
                 "model.decoder.layers"):
        obj = model
        ok = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise AttributeError("could not locate decoder layers")


def get_unembedding(model) -> torch.Tensor:
    oe = model.get_output_embeddings()
    if oe is not None and hasattr(oe, "weight"):
        return oe.weight.detach()
    ie = model.get_input_embeddings()
    return ie.weight.detach()


def get_input_embedding_matrix(model) -> torch.Tensor:
    return model.get_input_embeddings().weight.detach()


# -------------------------------------------------------------- templating ----
TEMPLATES = ("T1_chat", "T2_raw", "T3_shared")


def supports_enable_thinking(tok) -> bool:
    try:
        tok.apply_chat_template([{"role": "user", "content": "x"}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
        return True
    except (TypeError, ValueError, Exception):
        return False


def format_prompt(tok, text: str, template: str = "T1_chat",
                  think_mode: str = "canonical") -> str:
    """Return the fully formatted prompt STRING for a user request.

    think_mode: 'pos1_raw' | 'after_think_close' | 'enable_thinking_False' | 'canonical'
    'canonical' resolves to enable_thinking_False when supported, else after_think_close
    when the template emits a <think> opener, else pos1_raw.
    """
    if template == "T2_raw":
        return text + "\n"
    if template == "T3_shared":
        return f"User: {text}\nAssistant:"
    if getattr(tok, "chat_template", None) is None:
        return f"User: {text}\nAssistant:"

    msgs = [{"role": "user", "content": text}]
    if think_mode == "canonical":
        think_mode = getattr(tok, "_aii_canonical_mode", None) or _resolve_canonical(tok)

    if think_mode == "enable_thinking_False" and supports_enable_thinking(tok):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    base = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if think_mode == "after_think_close":
        if base.rstrip().endswith("<think>"):
            return base.rstrip()[: -len("<think>")] + THINK_CLOSE
        if "<think>" in base:
            return base
        return base + THINK_CLOSE
    return base


def _resolve_canonical(tok) -> str:
    if supports_enable_thinking(tok):
        mode = "enable_thinking_False"
    else:
        try:
            base = tok.apply_chat_template([{"role": "user", "content": "x"}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            base = ""
        mode = "after_think_close" if "<think>" in base else "pos1_raw"
    try:
        tok._aii_canonical_mode = mode
    except AttributeError:
        pass
    return mode


# ---------------------------------------------------------------- readout ----
class Readout:
    """Batched teacher-forced continuation log-likelihood machinery."""

    def __init__(self, model, tok, max_batch_tokens: int | None = None,
                 max_batch: int = 96, logit_budget_bytes: float = 3.0e9):
        self.model = model
        self.tok = tok
        # The logits tensor is [batch, tokens, VOCAB]; with a 262k vocab a fixed
        # token budget silently becomes a multi-GB allocation. Derive the budget
        # from the actual vocabulary size instead.
        vocab = int(getattr(model.config, "vocab_size", 0) or 50000)
        derived = max(512, int(logit_budget_bytes / (vocab * 4)))
        self.max_batch_tokens = (min(max_batch_tokens, derived)
                                 if max_batch_tokens else derived)
        self.vocab_size = vocab
        self.max_batch = max_batch
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        self._cont_cache: dict[str, list[int]] = {}
        # Activation injection for candidate C2 (write gain).
        #   {"layer": int, "vec": torch.Tensor[d], "mode": "last_prompt"|"all_prompt"}
        self.inject: dict | None = None
        self._layers = None

    def layers(self):
        if self._layers is None:
            self._layers = get_layers(self.model)
        return self._layers

    def cont_ids(self, cont: str) -> list[int]:
        if cont not in self._cont_cache:
            self._cont_cache[cont] = self.tok(cont, add_special_tokens=False)["input_ids"]
        return self._cont_cache[cont]

    def prompt_ids(self, prompt: str) -> list[int]:
        return self.tok(prompt, add_special_tokens=False)["input_ids"]

    # -- primary: sum-logprob of each continuation given each prompt -----------
    @torch.no_grad()
    def cont_logprobs(self, prompts: list[str], conts: list[str],
                      prompt_id_lists: list[list[int]] | None = None,
                      embed_override: list[torch.Tensor] | None = None) -> np.ndarray:
        """Return [n_prompts, n_conts] total continuation log-probability (nats)."""
        pid = prompt_id_lists if prompt_id_lists is not None else [
            self.prompt_ids(p) for p in prompts]
        cid = [self.cont_ids(c) for c in conts]
        items = []
        for i, pi in enumerate(pid):
            for j, cj in enumerate(cid):
                items.append((i, j, pi, cj))
        out = np.full((len(pid), len(cid)), np.nan, dtype=np.float64)
        order = sorted(range(len(items)), key=lambda k: len(items[k][2]) + len(items[k][3]))
        buf: list[int] = []
        for k in order:
            buf.append(k)
            longest = len(items[buf[0]][2]) + len(items[buf[0]][3])
            longest = max(len(items[b][2]) + len(items[b][3]) for b in buf)
            if len(buf) * longest >= self.max_batch_tokens or len(buf) >= self.max_batch:
                self._run_batch([items[b] for b in buf], out, embed_override)
                buf = []
        if buf:
            self._run_batch([items[b] for b in buf], out, embed_override)
        return out

    def _run_batch(self, batch, out, embed_override=None):
        maxlen = max(len(p) + len(c) for _, _, p, c in batch)
        n = len(batch)
        ids = torch.full((n, maxlen), self.pad_id, dtype=torch.long)
        mask = torch.zeros((n, maxlen), dtype=torch.long)
        for b, (_, _, p, c) in enumerate(batch):
            seq = p + c
            ids[b, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[b, : len(seq)] = 1
        ids = ids.to(DEVICE)
        mask = mask.to(DEVICE)
        kwargs = {"attention_mask": mask, "use_cache": False}
        handle = None
        if self.inject is not None:
            plens = [len(p) for _, _, p, _c in batch]
            handle = self._register_inject(plens)
        try:
            self._forward_and_score(batch, ids, mask, kwargs, out, embed_override)
        finally:
            if handle is not None:
                handle.remove()

    def _forward_and_score(self, batch, ids, mask, kwargs, out, embed_override):
        if embed_override is not None:
            emb = self.model.get_input_embeddings()(ids)
            for b, (i, _, p, _c) in enumerate(batch):
                ov = embed_override[i]
                if ov is not None:
                    st, span = ov
                    emb[b, st: st + span.shape[0], :] = span.to(emb.dtype)
            logits = self.model(inputs_embeds=emb, **kwargs).logits
        else:
            logits = self.model(input_ids=ids, **kwargs).logits
        # Gather only the target logprobs. Materialising log_softmax over the full
        # [B, T, V] in float32 is what OOMs a 262k-vocab model (gemma-3) at batch 96.
        for b, (i, j, p, c) in enumerate(batch):
            start = len(p) - 1
            tgt = torch.tensor(c, device=logits.device)
            sl = logits[b, start: start + len(c), :].float()
            lse = torch.logsumexp(sl, dim=-1)
            sel = sl.gather(1, tgt.unsqueeze(1)).squeeze(1) - lse
            out[i, j] = float(sel.sum().item())
            del sl, lse, sel
        del logits, ids, mask
        if embed_override is not None:
            del emb

    def _register_inject(self, plens: list[int]):
        spec = self.inject
        layer = self.layers()[spec["layer"]]
        vec = spec["vec"]
        mode = spec.get("mode", "last_prompt")
        idx = torch.tensor([max(0, l - 1) for l in plens], device=DEVICE)
        lens = torch.tensor(plens, device=DEVICE)

        def hook(_mod, _inp, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            if h.shape[0] != lens.shape[0]:
                # A hook that outlived its batch must never silently inject into an
                # unrelated forward pass. Refuse loudly instead of corrupting data.
                raise RuntimeError(
                    f"stale injection hook: batch {h.shape[0]} != registered "
                    f"{lens.shape[0]}")
            v = vec.to(h.dtype).to(h.device)
            if mode == "all_prompt":
                pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
                m = (pos < lens.unsqueeze(1)).unsqueeze(-1).to(h.dtype)
                h = h + v.view(1, 1, -1) * m
            else:
                h = h.clone()
                h[torch.arange(h.shape[0], device=h.device), idx, :] += v
            return (h,) + output[1:] if is_tuple else h

        return layer.register_forward_hook(hook)

    # -- the readout R ---------------------------------------------------------
    def R(self, prompts: list[str], *, neg_set: list[str] | None = None,
          pos_set: list[str] | None = None,
          prompt_id_lists=None, embed_override=None) -> np.ndarray:
        """R = logsumexp(refusal cont logprobs) - logsumexp(comply cont logprobs)."""
        pos = pos_set if pos_set is not None else REFUSAL_CONTS
        neg = neg_set if neg_set is not None else COMPLY_CONTS
        allc = list(pos) + list(neg)
        lp = self.cont_logprobs(prompts, allc, prompt_id_lists, embed_override)
        a = _logsumexp(lp[:, : len(pos)], axis=1)
        b = _logsumexp(lp[:, len(pos):], axis=1)
        return a - b

    # -- secondary: first-token gap + hidden states ---------------------------
    @torch.no_grad()
    def prompt_forward(self, prompts: list[str], *, layers: list[int] | None = None,
                       want_ft_gap: bool = False, batch: int = 32,
                       ft_ids: tuple[list[int], list[int]] | None = None):
        """Left-padded prompt-only forward. Returns (hidden dict, ft_gap array)."""
        hid: dict[int, list[np.ndarray]] = {l: [] for l in (layers or [])}
        ftg: list[float] = []
        pid = [self.prompt_ids(p) for p in prompts]
        for s in range(0, len(pid), batch):
            chunk = pid[s: s + batch]
            maxlen = max(len(c) for c in chunk)
            ids = torch.full((len(chunk), maxlen), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for b, c in enumerate(chunk):  # LEFT pad -> last token at index -1
                ids[b, maxlen - len(c):] = torch.tensor(c, dtype=torch.long)
                mask[b, maxlen - len(c):] = 1
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False,
                             output_hidden_states=bool(layers))
            if layers:
                hs = out.hidden_states
                for l in layers:
                    hid[l].append(hs[l][:, -1, :].float().cpu().numpy())
            if want_ft_gap and ft_ids is not None:
                lg = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                r = torch.logsumexp(lg[:, ft_ids[0]], dim=-1)
                c_ = torch.logsumexp(lg[:, ft_ids[1]], dim=-1)
                ftg.extend((r - c_).cpu().numpy().tolist())
            del out, ids, mask
        hid_np = {l: np.concatenate(v, axis=0) for l, v in hid.items() if v}
        return hid_np, (np.array(ftg) if ftg else None)

    @torch.no_grad()
    def all_layer_hidden(self, prompts: list[str], batch: int = 16) -> np.ndarray:
        """Return [n_layers+1, n_prompts, d] last-token hidden states."""
        acc: list[list[np.ndarray]] = None
        pid = [self.prompt_ids(p) for p in prompts]
        for s in range(0, len(pid), batch):
            chunk = pid[s: s + batch]
            maxlen = max(len(c) for c in chunk)
            ids = torch.full((len(chunk), maxlen), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for b, c in enumerate(chunk):
                ids[b, maxlen - len(c):] = torch.tensor(c, dtype=torch.long)
                mask[b, maxlen - len(c):] = 1
            out = self.model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                             use_cache=False, output_hidden_states=True)
            hs = out.hidden_states
            if acc is None:
                acc = [[] for _ in hs]
            for li, h in enumerate(hs):
                acc[li].append(h[:, -1, :].float().cpu().numpy())
            del out, hs, ids, mask
        return np.stack([np.concatenate(a, axis=0) for a in acc], axis=0)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.nanmax(x, axis=axis, keepdims=True)
    return (m + np.log(np.nansum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def first_token_ids(tok, conts: list[str]) -> list[int]:
    """First-token ids of each continuation, with and without a leading space."""
    ids = set()
    for c in conts:
        for variant in (c, " " + c):
            t = tok(variant, add_special_tokens=False)["input_ids"]
            if t:
                ids.add(int(t[0]))
    return sorted(ids)
