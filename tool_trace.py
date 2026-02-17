import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    urls: list[str] = []
    for raw in _URL_RE.findall(text):
        url = (raw or "").strip().rstrip(").,;:!?]}")
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _append_unique(
    existing: list[str], new_items: Iterable[str], *, max_items: int
) -> list[str]:
    max_items = max(0, int(max_items))
    seen = set(existing)
    for item in new_items:
        if len(existing) >= max_items:
            break
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        existing.append(value)
    return existing


def ensure_tool_trace_base(
    trace: dict[str, Any] | None, *, question_url: str | None = None
) -> dict[str, Any]:
    base: dict[str, Any] = trace if isinstance(trace, dict) else {}
    base.setdefault("version", 1)
    if question_url:
        base.setdefault("question_url", question_url)
    return base


def record_urls(
    trace: dict[str, Any],
    *,
    bucket: str,
    urls: Iterable[str],
    max_urls: int,
) -> None:
    if not isinstance(trace, dict):
        return
    if not bucket:
        return
    existing = trace.get(bucket)
    if not isinstance(existing, list):
        existing = []
    trace[bucket] = _append_unique(
        [str(x) for x in existing if isinstance(x, str)],
        [str(u) for u in urls if isinstance(u, str)],
        max_items=max_urls,
    )


def record_value(trace: dict[str, Any], *, key: str, value: Any) -> None:
    if not isinstance(trace, dict) or not key:
        return
    trace[key] = value


def render_tool_trace_markdown(
    trace: dict[str, Any],
    *,
    max_chars: int = 8000,
) -> str:
    if not isinstance(trace, dict) or not trace:
        return ""

    payload = dict(trace)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n"
        payload = {
            "version": payload.get("version", 1),
            "generated_at": payload.get("generated_at"),
            "truncated": True,
            "tool_trace_preview": text,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    return "\n".join(
        [
            "# TOOL_TRACE",
            "```json",
            text,
            "```",
        ]
    ).strip()


def extract_tool_trace_json(comment_text: str) -> dict[str, Any] | None:
    if not comment_text:
        return None
    marker = "# TOOL_TRACE"
    idx = comment_text.find(marker)
    if idx < 0:
        return None
    fence = "```json"
    start = comment_text.find(fence, idx)
    if start < 0:
        return None
    start = comment_text.find("\n", start)
    if start < 0:
        return None
    end = comment_text.find("```", start + 1)
    if end < 0:
        return None
    raw = comment_text[start:end].strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        logger.debug("Failed to parse TOOL_TRACE JSON", exc_info=True)
        return None
    return parsed if isinstance(parsed, dict) else None

