#!/usr/bin/env python3
"""Async OpenRouter client with on-disk response caching and a hard spend cap."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import aiohttp
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent
CACHE_DIR = WORKSPACE / "cache" / "or"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# gemini-2.5-flash-lite published rates (USD per 1M tokens).
PRICE_IN = 0.10
PRICE_OUT = 0.40


class Budget:
    def __init__(self, hard_stop_usd: float = 8.0, warn_usd: float = 6.0):
        self.spent = 0.0
        self.hard_stop = hard_stop_usd
        self.warn = warn_usd
        self.calls = 0
        self.cached = 0
        self.stopped = False
        self.tok_in = 0
        self.tok_out = 0

    def add(self, tin: int, tout: int) -> None:
        self.tok_in += tin
        self.tok_out += tout
        self.spent += tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
        self.calls += 1
        if self.spent >= self.hard_stop:
            self.stopped = True

    def as_dict(self) -> dict:
        return {"spent_usd": round(self.spent, 5), "live_calls": self.calls,
                "cache_hits": self.cached, "tokens_in": self.tok_in,
                "tokens_out": self.tok_out, "hard_stopped": self.stopped,
                "hard_stop_usd": self.hard_stop}


class ORClient:
    def __init__(self, model: str = "google/gemini-2.5-flash-lite",
                 concurrency: int = 24, budget: Budget | None = None):
        self.model = model
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.sem = asyncio.Semaphore(concurrency)
        self.budget = budget or Budget()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / key[:2] / f"{key}.json"

    def _cache_key(self, system: str, user: str, seed: int) -> str:
        blob = json.dumps([self.model, system, user, seed], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    async def call(self, session: aiohttp.ClientSession, system: str, user: str,
                   *, seed: int = 0, max_tokens: int = 200,
                   temperature: float = 0.0) -> str | None:
        key = self._cache_key(system, user, seed)
        cp = self._cache_path(key)
        if cp.exists():
            try:
                self.budget.cached += 1
                return json.loads(cp.read_text())["text"]
            except (json.JSONDecodeError, KeyError):
                pass
        if self.budget.stopped:
            return None
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json"}
        async with self.sem:
            for attempt in range(5):
                if self.budget.stopped:
                    return None
                try:
                    async with session.post(API_URL, json=payload, headers=headers,
                                            timeout=aiohttp.ClientTimeout(total=120)) as r:
                        if r.status in (429, 500, 502, 503, 529):
                            await asyncio.sleep(2 ** attempt + 0.5)
                            continue
                        if r.status != 200:
                            body = (await r.text())[:300]
                            logger.warning(f"OR http {r.status}: {body}")
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        data = await r.json()
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(f"OR net error {type(exc).__name__}; retry {attempt}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                try:
                    text = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    logger.warning(f"OR malformed: {str(data)[:250]}")
                    return None
                usage = data.get("usage") or {}
                self.budget.add(int(usage.get("prompt_tokens", 0) or 0),
                                int(usage.get("completion_tokens", 0) or 0))
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps({"text": text, "ts": time.time()}))
                return text
        return None


def parse_json_block(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1:
        return None
    try:
        return json.loads(t[a: b + 1])
    except json.JSONDecodeError:
        return None
