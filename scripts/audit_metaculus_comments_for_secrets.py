from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metaculus_comment_fetcher import fetch_my_comments  # noqa: E402


def _metaculus_headers(token: str) -> dict[str, str]:
    token = (token or "").strip()
    if not token:
        raise RuntimeError("METACULUS_TOKEN is missing.")
    return {
        "Authorization": f"Token {token}",
        "Accept-Language": "en",
        "User-Agent": "metac-bot-template comment audit (local)",
    }


def _fetch_my_user_id(*, token: str, timeout_seconds: int = 30) -> int:
    url = "https://www.metaculus.com/api/users/me"
    resp = requests.get(url, headers=_metaculus_headers(token), timeout=timeout_seconds)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /users/me payload type: {type(payload)}")
    user_id = payload.get("id")
    if not isinstance(user_id, int):
        raise RuntimeError(f"Unexpected /users/me id type: {type(user_id)}")
    return user_id


def _find_secret_markers(text: str) -> list[str]:
    lowered = text.lower()
    markers = [
        "api_key",
        "access_token",
        "refresh_token",
        "client_secret",
        "authorization",
        "bearer ",
        "chat.kiconnect.nrw",
    ]
    return [m for m in markers if m in lowered]


def _comment_question_url(on_post: object) -> str:
    if isinstance(on_post, int):
        return f"https://www.metaculus.com/questions/{on_post}/"
    return ""


def _dedupe_by_id(comments: list[dict]) -> list[dict]:
    deduped: dict[int, dict] = {}
    for item in comments:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if isinstance(cid, int) and cid not in deduped:
            deduped[cid] = item
    return list(deduped.values())


def main(argv: list[str]) -> int:
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Audit your Metaculus comments for leaked secrets (prints IDs only; never prints comment text)."
        )
    )
    parser.add_argument(
        "--scope",
        choices=["private", "public", "all"],
        default="all",
        help="Which comments to scan (default: all).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=1000,
        help="Maximum comments to fetch per scope (default: 1000).",
    )
    args = parser.parse_args(argv)

    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        print("Error: METACULUS_TOKEN is missing.", file=sys.stderr)
        return 2

    try:
        my_user_id = _fetch_my_user_id(token=token)
    except Exception as e:
        print(f"Error: failed to fetch /api/users/me: {e}", file=sys.stderr)
        return 2

    all_comments: list[dict[str, Any]] = []
    max_items = max(1, int(args.max_items))

    try:
        if args.scope in {"private", "all"}:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=my_user_id,
                    max_items=max_items,
                    include_private=True,
                )
            )
        if args.scope in {"public", "all"}:
            all_comments.extend(
                fetch_my_comments(
                    token=token,
                    author_id=my_user_id,
                    max_items=max_items,
                    include_private=False,
                )
            )
    except Exception as e:
        print(f"Error: failed to fetch comments: {e}", file=sys.stderr)
        return 2

    comments = _dedupe_by_id(all_comments)
    findings: list[tuple[dict, list[str]]] = []
    for comment in comments:
        text = comment.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        markers = _find_secret_markers(text)
        if markers:
            findings.append((comment, markers))

    findings.sort(key=lambda item: (item[0].get("created_at") or ""), reverse=True)

    total = len(comments)
    print(f"Scanned {total} comments ({args.scope}). Potential secret-leak matches: {len(findings)}")
    for comment, markers in findings:
        cid = comment.get("id")
        on_post = comment.get("on_post")
        is_private = comment.get("is_private")
        created_at = comment.get("created_at")
        url = _comment_question_url(on_post)
        markers_str = ",".join(markers)
        print(
            f"- comment_id={cid} post_id={on_post} is_private={is_private} created_at={created_at} markers={markers_str} {url}"
        )

    # Non-zero exit code if anything matched (useful for CI).
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
