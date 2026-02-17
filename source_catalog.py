from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
