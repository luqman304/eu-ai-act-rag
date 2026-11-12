"""Download the regulation from EUR-Lex and write the parsed corpus to JSONL."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import httpx

from ..config import Settings
from .parse import parse_html, save_units

log = logging.getLogger(__name__)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "aiact-rag/1.0 (+https://github.com/)", "Accept-Language": "en"}
    with httpx.Client(follow_redirects=True, timeout=60, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info(
        "downloaded %s (%d bytes, sha256=%s)",
        url,
        len(resp.content),
        hashlib.sha256(resp.content).hexdigest(),
    )
    return dest


def build_corpus(settings: Settings, html_path: Path | None = None) -> int:
    raw = html_path or settings.data_dir / "raw" / "aiact.html"
    if not raw.exists():
        download(settings.corpus_url, raw)
    units = parse_html(raw.read_text(encoding="utf-8"), include_recitals=True)
    save_units(units, settings.corpus_path)
    return len(units)
