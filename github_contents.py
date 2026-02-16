from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


def _first_nonempty(*values: str | None) -> str | None:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_github_token() -> str | None:
    return _first_nonempty(
        os.getenv("GH_TOKEN_MetaculusBot"),
        os.getenv("GH_TOKEN"),
        os.getenv("GITHUB_TOKEN"),
    )


def get_github_repo() -> str | None:
    return _first_nonempty(
        os.getenv("BOT_GITHUB_REPO"),
        os.getenv("GITHUB_REPOSITORY"),
        "RuizhangZhou/metaculus-bot",
    )


@dataclass(frozen=True)
class GithubFile:
    path: str
    sha: str
    content: str


class GithubContentsClient:
    def __init__(self, *, token: str, repo: str) -> None:
        self._token = (token or "").strip()
        self._repo = (repo or "").strip()
        if not self._token:
            raise ValueError("Missing GitHub token")
        if not self._repo or "/" not in self._repo:
            raise ValueError(f"Invalid GitHub repo: {self._repo!r}")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "metaculus-bot",
            }
        )

    def _url(self, path: str) -> str:
        return f"https://api.github.com/repos/{self._repo}{path}"

    def get_branch_sha(self, branch: str) -> str:
        branch = (branch or "").strip()
        if not branch:
            raise ValueError("branch is required")
        encoded = quote(branch, safe="")
        resp = self._session.get(self._url(f"/branches/{encoded}"), timeout=30)
        if resp.status_code == 404:
            raise FileNotFoundError(f"Branch not found: {branch}")
        resp.raise_for_status()
        data = resp.json()
        sha = (
            data.get("commit", {})
            .get("sha")
        )
        if not isinstance(sha, str) or not sha:
            raise RuntimeError(f"Failed to read SHA for branch {branch}")
        return sha

    def ensure_branch(self, *, branch: str, from_branch: str = "main") -> None:
        branch = (branch or "").strip()
        if not branch:
            raise ValueError("branch is required")
        try:
            self.get_branch_sha(branch)
            return
        except FileNotFoundError:
            pass

        base_sha = self.get_branch_sha(from_branch)
        payload = {"ref": f"refs/heads/{branch}", "sha": base_sha}
        resp = self._session.post(self._url("/git/refs"), json=payload, timeout=30)
        if resp.status_code in {200, 201}:
            return
        # Branch may have been created concurrently.
        if resp.status_code == 422:
            return
        resp.raise_for_status()

    def get_file(self, *, path: str, branch: str) -> GithubFile | None:
        path = (path or "").lstrip("/")
        if not path:
            raise ValueError("path is required")
        branch = (branch or "").strip()
        if not branch:
            raise ValueError("branch is required")
        encoded_path = quote(path)
        resp = self._session.get(
            self._url(f"/contents/{encoded_path}"),
            params={"ref": branch},
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return None
        if data.get("type") != "file":
            return None
        sha = data.get("sha")
        content_b64 = data.get("content")
        if not isinstance(sha, str) or not isinstance(content_b64, str):
            return None
        raw = base64.b64decode(content_b64.encode("utf-8")).decode("utf-8", errors="replace")
        return GithubFile(path=path, sha=sha, content=raw)

    def upsert_file(
        self,
        *,
        path: str,
        branch: str,
        message: str,
        content: str,
        committer_name: str | None = None,
        committer_email: str | None = None,
    ) -> str:
        path = (path or "").lstrip("/")
        if not path:
            raise ValueError("path is required")
        branch = (branch or "").strip()
        if not branch:
            raise ValueError("branch is required")
        message = (message or "").strip() or f"Update {path}"
        file_obj = self.get_file(path=path, branch=branch)
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if file_obj is not None:
            payload["sha"] = file_obj.sha
        if committer_name and committer_email:
            payload["committer"] = {"name": committer_name, "email": committer_email}

        encoded_path = quote(path)
        resp = self._session.put(
            self._url(f"/contents/{encoded_path}"),
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        commit_sha = (
            data.get("commit", {})
            .get("sha")
        )
        return str(commit_sha) if commit_sha else ""

