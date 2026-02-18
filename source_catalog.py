from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import yaml


_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogChangeSummary:
    added: int = 0
    removed: int = 0
    updated: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_catalog(text: str | None) -> dict[str, Any]:
    if not text or not str(text).strip():
        return {"version": 1, "updated_at": _now_iso(), "sources": []}
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        return {"version": 1, "updated_at": _now_iso(), "sources": []}
    version = loaded.get("version", 1)
    sources = loaded.get("sources", [])
    if not isinstance(sources, list):
        sources = []
    return {
        "version": int(version) if isinstance(version, int) else 1,
        "updated_at": str(loaded.get("updated_at") or _now_iso()),
        "sources": sources,
    }


def dump_catalog(catalog: dict[str, Any]) -> str:
    catalog = dict(catalog or {})
    catalog.setdefault("version", 1)
    catalog["updated_at"] = str(catalog.get("updated_at") or _now_iso())
    sources = catalog.get("sources")
    if not isinstance(sources, list):
        catalog["sources"] = []

    return yaml.safe_dump(
        catalog,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    ).strip() + "\n"


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    return url


def _is_valid_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    return bool(_URL_RE.match(url.strip()))


def apply_patch_ops(
    catalog: dict[str, Any],
    ops: dict[str, Any],
    *,
    max_sources: int = 2000,
) -> tuple[dict[str, Any], CatalogChangeSummary]:
    """
    Applies structured patch ops to the source catalog.

    Expected ops format:
      {
        "add": [{"url": "...", "title": "...", "tags": [...], "notes": "..."}],
        "remove": [{"url": "..."}],
        "update": [{"url": "...", "fields": {"title": "...", "tags": [...]}}]
      }
    """
    base = load_catalog(dump_catalog(catalog))
    original_updated_at = base.get("updated_at")
    sources_raw = base.get("sources", [])
    sources: list[dict[str, Any]] = [
        s for s in sources_raw if isinstance(s, dict) and _is_valid_url(str(s.get("url", "")))
    ]

    by_url: dict[str, dict[str, Any]] = {
        _normalize_url(str(s.get("url"))): dict(s) for s in sources
    }

    summary = CatalogChangeSummary()

    # Removes
    removes = ops.get("remove", [])
    if isinstance(removes, list):
        removed = 0
        for item in removes:
            if not isinstance(item, dict):
                continue
            url = _normalize_url(str(item.get("url") or ""))
            if not _is_valid_url(url):
                continue
            if url in by_url:
                by_url.pop(url, None)
                removed += 1
        summary = CatalogChangeSummary(
            added=summary.added, removed=removed, updated=summary.updated
        )

    # Adds
    adds = ops.get("add", [])
    if isinstance(adds, list):
        added = 0
        for item in adds:
            if not isinstance(item, dict):
                continue
            url = _normalize_url(str(item.get("url") or ""))
            if not _is_valid_url(url):
                continue
            if url in by_url:
                continue
            entry: dict[str, Any] = {"url": url}
            title = item.get("title")
            if isinstance(title, str) and title.strip():
                entry["title"] = title.strip()
            tags = item.get("tags")
            if isinstance(tags, list):
                entry["tags"] = [str(t).strip() for t in tags if str(t).strip()]
            notes = item.get("notes")
            if isinstance(notes, str) and notes.strip():
                entry["notes"] = notes.strip()
            entry["added_at"] = str(item.get("added_at") or _now_iso())
            entry["last_seen_at"] = str(item.get("last_seen_at") or entry["added_at"])
            by_url[url] = entry
            added += 1
            if len(by_url) >= max_sources:
                break
        summary = CatalogChangeSummary(
            added=added, removed=summary.removed, updated=summary.updated
        )

    # Updates
    updates = ops.get("update", [])
    if isinstance(updates, list):
        updated = 0
        for item in updates:
            if not isinstance(item, dict):
                continue
            url = _normalize_url(str(item.get("url") or ""))
            if not _is_valid_url(url):
                continue
            existing = by_url.get(url)
            if not isinstance(existing, dict):
                continue
            fields = item.get("fields")
            if not isinstance(fields, dict):
                continue
            changed = False
            if "title" in fields and isinstance(fields["title"], str) and fields["title"].strip():
                existing["title"] = fields["title"].strip()
                changed = True
            if "notes" in fields and isinstance(fields["notes"], str) and fields["notes"].strip():
                existing["notes"] = fields["notes"].strip()
                changed = True
            if "tags" in fields and isinstance(fields["tags"], list):
                existing["tags"] = [
                    str(t).strip() for t in fields["tags"] if str(t).strip()
                ]
                changed = True
            if changed:
                existing["last_seen_at"] = str(fields.get("last_seen_at") or _now_iso())
                by_url[url] = existing
                updated += 1
        summary = CatalogChangeSummary(
            added=summary.added, removed=summary.removed, updated=updated
        )

    new_sources = list(by_url.values())
    new_sources.sort(key=lambda s: str(s.get("url", "")))
    base["sources"] = new_sources
    if summary.added or summary.removed or summary.updated:
        base["updated_at"] = _now_iso()
    else:
        base["updated_at"] = str(original_updated_at or base.get("updated_at") or _now_iso())
    return base, summary


_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "will",
    "would",
    "should",
    "could",
    "into",
    "over",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "also",
    "than",
    "then",
    "there",
    "their",
    "they",
    "them",
    "you",
    "your",
    "our",
    "ours",
    "we",
    "us",
    "its",
    "it's",
    "in",
    "on",
    "to",
    "of",
    "a",
    "an",
    "as",
    "at",
    "by",
    "or",
    "not",
    "is",
    "it",
}


def _url_domain(url: str) -> str:
    try:
        parsed = urlparse((url or "").strip())
    except Exception:
        return ""
    host = (parsed.hostname or "").strip().lower()
    return host


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in tokens if t and t not in _DEFAULT_STOPWORDS}


def suggest_sources_for_question(
    catalog: dict[str, Any],
    *,
    query_text: str,
    max_items: int = 15,
) -> list[dict[str, Any]]:
    """
    Deterministically select a small set of relevant catalog entries for a question.

    Scoring is intentionally simple: overlap between question tokens and entry text (title/notes/tags/domain).
    """
    max_items = max(0, int(max_items))
    if max_items <= 0:
        return []

    sources = catalog.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return []

    query_tokens = _tokenize(query_text)

    scored: list[tuple[int, str, dict[str, Any]]] = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not _is_valid_url(url):
            continue
        title = str(entry.get("title") or "").strip()
        notes = str(entry.get("notes") or "").strip()
        tags = entry.get("tags")
        tag_text = " ".join([str(t) for t in tags if isinstance(t, str)]) if isinstance(tags, list) else ""
        domain = _url_domain(url)

        entry_text = " ".join([title, notes, tag_text, domain]).strip()
        if not entry_text:
            score = 0
        else:
            entry_tokens = _tokenize(entry_text)
            score = len(query_tokens & entry_tokens) if query_tokens else 0

        scored.append((int(score), url, dict(entry)))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [e for _, _, e in scored[:max_items]]


def render_sources_markdown(
    sources: list[dict[str, Any]],
    *,
    max_chars: int = 2500,
) -> tuple[str, list[str]]:
    """
    Render a compact markdown-ish block for LLM prompts.

    Returns (text, urls_included).
    """
    max_chars = max(0, int(max_chars))
    if max_chars <= 0 or not sources:
        return "", []

    lines: list[str] = []
    urls: list[str] = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not _is_valid_url(url):
            continue
        title = str(entry.get("title") or "").strip()
        notes = str(entry.get("notes") or "").strip()
        tags = entry.get("tags")
        tag_list = [str(t).strip().lower() for t in tags if str(t).strip()] if isinstance(tags, list) else []
        tag_part = f" [{', '.join(tag_list)}]" if tag_list else ""
        title_part = f"{title}{tag_part}" if title else f"{url}{tag_part}"
        line = f"- {title_part}: {url}"
        if notes:
            line += f" — {notes}"
        lines.append(line)
        urls.append(url)
        if len("\n".join(lines)) >= max_chars:
            break

    text = "\n".join(lines).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[TRUNCATED]"
    return text, urls
