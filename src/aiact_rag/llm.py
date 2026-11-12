"""Thin wrapper around the Anthropic Messages API.

* Structured outputs (``output_config.format``) for every call that must return JSON.
* Server-side refusal fallback (beta) on by default, switchable with ``LLM_FALLBACKS=false``.
* An on-disk SQLite cache keyed by the full request, so evaluation re-runs are free and reproducible
  (disable with ``LLM_CACHE=false``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from .config import Settings

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False

    def json(self) -> Any:
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as e:
            raise LLMError(f"model did not return valid JSON: {self.text[:200]!r}") from e


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM llm WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()


class LLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self._client = client
        self._cache = _Cache(settings.cache_dir / "llm.sqlite") if settings.llm_cache else None
        self._usage_lock = threading.Lock()
        self.usage_by_model: dict[str, list[int]] = {}  # model -> [input, output] tokens, uncached calls only

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            kwargs: dict[str, Any] = {"max_retries": 4, "timeout": 180}
            if self.settings.anthropic_api_key:
                kwargs["api_key"] = self.settings.anthropic_api_key
            if self.settings.anthropic_base_url:
                kwargs["base_url"] = self.settings.anthropic_base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str,
        effort: str,
        max_tokens: int = 8000,
        schema: dict | None = None,
    ) -> LLMResponse:
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if self._cache and (hit := self._cache.get(key)):
            return LLMResponse(text=hit["text"], model=hit["model"], usage=hit.get("usage", {}), cached=True)

        try:
            if self.settings.llm_fallbacks:
                msg = self.client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                msg = self.client.messages.create(**request)
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"could not reach the Anthropic API: {e}") from e
        except TypeError as e:  # raised by the SDK before sending when no credentials resolve
            if "authentication" not in str(e):
                raise
            raise LLMError("no Anthropic credentials configured: set ANTHROPIC_API_KEY") from e

        if msg.stop_reason == "refusal":
            category = getattr(getattr(msg, "stop_details", None), "category", None)
            raise LLMRefusal(f"model declined the request (category={category})")
        if msg.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        with self._usage_lock:
            tot = self.usage_by_model.setdefault(model, [0, 0])
            tot[0] += usage["input_tokens"]
            tot[1] += usage["output_tokens"]
        if self._cache:
            self._cache.put(key, {"text": text, "model": msg.model, "usage": usage})
        return LLMResponse(text=text, model=msg.model, usage=usage)
