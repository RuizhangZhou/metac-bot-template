from __future__ import annotations

from typing import Any


_REDACTED = "<redacted>"


def is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_")
    if not normalized:
        return False

    if normalized in {
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "secret",
        "password",
        "authorization",
        "proxy_authorization",
    }:
        return True

    if normalized.endswith(
        (
            "_api_key",
            "_apikey",
            "_token",
            "_access_token",
            "_refresh_token",
            "_client_secret",
            "_secret",
            "_password",
        )
    ):
        return True

    return False


def redact_secrets(value: Any, *, redaction: str = _REDACTED, max_depth: int = 12) -> Any:
    """
    Recursively redacts likely secret values (api keys, tokens, passwords) from
    nested dict/list structures. Intended for safe logging / metadata displays.
    """
    if max_depth <= 0:
        return value

    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for k, v in value.items():
            if is_sensitive_key(k):
                redacted[k] = redaction
            else:
                redacted[k] = redact_secrets(v, redaction=redaction, max_depth=max_depth - 1)
        return redacted

    if isinstance(value, list):
        return [redact_secrets(v, redaction=redaction, max_depth=max_depth - 1) for v in value]

    if isinstance(value, tuple):
        return tuple(redact_secrets(v, redaction=redaction, max_depth=max_depth - 1) for v in value)

    return value

