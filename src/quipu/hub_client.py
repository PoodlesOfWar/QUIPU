"""hub_client — QUIPU's door into Hub Holon (HubCore): a stdlib client that holds no secret.

It mints a **Requester** token (HubCore's self-service accountability tier: a
subject, a tenancy and an expiry) and uses it to do exactly three things:

* read the v-ADMIN tool grants (``GET /api/v1/vadmin/tools``) so the MCP server can
  tell whether a human Admin has switched the voice on;
* ask v-ADMIN to narrate (``POST /api/v1/vadmin/voice``), which HubCore refuses unless
  that grant exists;
* queue a task (``POST /api/v1/tasks``) for an authorised ``hub.*`` proposal.

It never asks for Reviewer or Admin and has no operator credential, so it cannot
accept a result, promote a release or turn a tool on: those stay with the humans HubCore
authenticates.  HubCore's own role policy (V10-SEC-001) is the enforcement; this client
cannot do more than a Requester token can.

Configuration (unset => the integration is off and every caller degrades):
    QUIPU_HUBCORE_URL       e.g. http://loadopoly-hubcore:8000 (plain http only to
                            loopback, private and container hosts; https otherwise)
    QUIPU_HUBCORE_ORG_ID    the org this node is registered under (HubCore refuses
                            unregistered orgs; onboarding is an Admin decision)
    QUIPU_HUBCORE_REPO_ID   default "quipu"
    QUIPU_HUBCORE_USER_ID   the token subject, default "quipu-bridge"
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

from .x_process_studio import _plain_http_ok

URL_ENV = "QUIPU_HUBCORE_URL"
ORG_ENV = "QUIPU_HUBCORE_ORG_ID"
REPO_ENV = "QUIPU_HUBCORE_REPO_ID"
USER_ENV = "QUIPU_HUBCORE_USER_ID"

_TIMEOUT_S = 10.0
_TOKEN_MARGIN_S = 30.0
_STATE_TTL_S = 5.0


class HubCoreError(RuntimeError):
    """Base class."""


class HubCoreUnavailable(HubCoreError):
    """Not configured, unreachable, or a server error: retryable, and never a decision."""


class HubCoreRefused(HubCoreError):
    """HubCore answered 4xx: the request was understood and refused (policy, validation)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HubCore {status}: {message}")
        self.status = status


class HubCoreClient:
    def __init__(self, base_url: str | None = None, org_id: str | None = None,
                 repo_id: str | None = None, user_id: str | None = None,
                 opener: Callable[..., Any] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.base = (base_url if base_url is not None else os.environ.get(URL_ENV, "")).strip().rstrip("/")
        self.org_id = (org_id if org_id is not None else os.environ.get(ORG_ENV, "")).strip()
        self.repo_id = (repo_id or os.environ.get(REPO_ENV) or "quipu").strip()
        self.user_id = (user_id or os.environ.get(USER_ENV) or "quipu-bridge").strip()
        self._open = opener or urllib.request.urlopen
        self._clock = clock
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._token_exp = 0.0
        self._state: tuple[float, Any] = (0.0, None)

    def configured(self) -> bool:
        return bool(self.base and self.org_id)

    # -- transport ------------------------------------------------------------
    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None,
                 token: Optional[str] = None) -> dict[str, Any]:
        if not self.configured():
            raise HubCoreUnavailable(f"HubCore is not configured ({URL_ENV} and {ORG_ENV})")
        parsed = urlparse(self.base)
        if parsed.scheme not in ("http", "https"):
            raise HubCoreUnavailable(f"unsupported scheme {parsed.scheme!r}")
        if parsed.scheme == "http" and not _plain_http_ok(parsed.hostname or ""):
            raise HubCoreUnavailable("plain http to a public address refused; use https")
        data = json.dumps(dict(body)).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with self._open(req, timeout=_TIMEOUT_S) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                parsed_body = json.loads(exc.read().decode("utf-8") or "{}")
                detail = str(parsed_body.get("error") or parsed_body.get("message") or parsed_body)[:300]
            except Exception:
                detail = str(getattr(exc, "reason", ""))[:300]
            if 400 <= exc.code < 500:
                raise HubCoreRefused(exc.code, detail) from exc
            raise HubCoreUnavailable(f"HubCore {exc.code}: {detail}") from exc
        except Exception as exc:
            raise HubCoreUnavailable(f"HubCore transport: {exc}") from exc
        try:
            out = json.loads(raw.decode("utf-8") or "{}")
        except ValueError as exc:
            raise HubCoreUnavailable("HubCore answered something that is not JSON") from exc
        return out if isinstance(out, dict) else {"value": out}

    def _bearer(self, force: bool = False) -> str:
        with self._lock:
            if not force and self._token and self._clock() < self._token_exp - _TOKEN_MARGIN_S:
                return self._token
        out = self._request("POST", "/api/v1/session", {
            "orgId": self.org_id, "repoId": self.repo_id, "userId": self.user_id,
            "role": "Requester",                     # never a privileged role; there is no operator credential here
        })
        token = out.get("token")
        if not token:
            raise HubCoreUnavailable("HubCore issued no token")
        with self._lock:
            self._token = str(token)
            self._token_exp = float(out.get("expiresAt") or (self._clock() + 600))
        return self._token

    def _authed(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            return self._request(method, path, body, self._bearer())
        except HubCoreRefused as exc:
            if exc.status != 401:
                raise
        return self._request(method, path, body, self._bearer(force=True))   # expired mid-flight: once

    # -- what QUIPU asks of the hub -------------------------------------------
    def vadmin_tools(self, *, fresh: bool = False) -> dict[str, Any]:
        """v-ADMIN's tool grants, cached for a few seconds (the MCP lists tools often)."""
        at, cached = self._state
        if not fresh and cached is not None and self._clock() - at < _STATE_TTL_S:
            return cached
        out = self._authed("GET", "/api/v1/vadmin/tools")
        self._state = (self._clock(), out)
        return out

    def voice_effective(self) -> tuple[bool, str]:
        """(is the voice on right now, as which persona).  Fails closed: any doubt is "off"."""
        try:
            tools = self.vadmin_tools()
            v = (tools.get("tools") or {}).get("voice") or {}
            return bool(v.get("effective")), str(v.get("persona") or "")
        except HubCoreError:
            return False, ""

    def speak(self, kind: str, facts: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._authed("POST", "/api/v1/vadmin/voice", {"kind": kind, "facts": dict(facts or {})})

    def create_task(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._authed("POST", "/api/v1/tasks", {
            "kind": kind, "orgId": self.org_id, "repoId": self.repo_id, "payload": dict(payload)})
