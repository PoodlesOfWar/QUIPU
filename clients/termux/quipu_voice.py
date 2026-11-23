#!/usr/bin/env python3
"""quipu_voice -- speak a note on a phone; it becomes a *held* proposal for a human to authorise.

    record (Termux:API)  ->  speech-to-text service on the Hideout  ->  you confirm the text  ->  held proposal

What a spoken note can do, by design: become one ``hub.body_directive`` at severity ``info`` for owner role
``Anyone``, held as PROPOSED.  The words never choose the tool, the role or the severity, and a voice is
never an authorisation: a human decides in QUIPU (HubCore Part 0.3.10, V10-SEC-010).  The transcript is
untrusted text: it is shown to you and only staged if you confirm (``y``, or ``--yes`` for a script).

Config (env, or ~/.config/quipu/stt.env + the MCP config from quipu_mcp.py / edge_setup.sh):
    QUIPU_STT_URL            https://<host>/transcribe  (plain http only to private hosts)
    QUIPU_STT_TOKEN_FILE     file holding the bearer token (or QUIPU_STT_TOKEN); never printed

    python quipu_voice.py doctor
    python quipu_voice.py listen [--seconds 10] [--yes] [--dry-run]
    python quipu_voice.py listen --file note.m4a        # an existing clip (no microphone needed)
Exit codes: 0 ok, 1 failure, 3 setup/refusal.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quipu_mcp as qm                                                    # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_SETUP = 0, 1, 3
STT_CONFIG = Path.home() / ".config" / "quipu" / "stt.env"
MAX_CLIP_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 60
RECORD_FLAGS = ("-f", "-l", "-e")                       # termux-microphone-record flags this script uses
MAX_WORDS_TITLE = 140


class VoiceError(RuntimeError):
    pass


def resolve_stt(env: Mapping[str, str], file_env: Mapping[str, str]) -> tuple[str, str]:
    url = (env.get("QUIPU_STT_URL") or file_env.get("QUIPU_STT_URL") or "").strip()
    if not url:
        raise VoiceError("QUIPU_STT_URL is not set (https://<host>/transcribe)")
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise VoiceError("QUIPU_STT_URL must be an http(s) URL")
    if u.scheme == "http" and not qm.plain_http_ok(u.hostname):
        raise VoiceError("refusing plain http to a public address; use https")
    token = (env.get("QUIPU_STT_TOKEN") or "").strip()
    tf = (env.get("QUIPU_STT_TOKEN_FILE") or file_env.get("QUIPU_STT_TOKEN_FILE") or "").strip()
    if not token and tf:
        try:
            token = Path(tf).expanduser().read_text().strip()
        except OSError as exc:
            raise VoiceError(f"cannot read QUIPU_STT_TOKEN_FILE: {exc.strerror}") from None
    if not token:
        raise VoiceError("no STT token: set QUIPU_STT_TOKEN_FILE (or QUIPU_STT_TOKEN)")
    return url, token


def record_clip(seconds: int, out: Path, run=subprocess.run, which=shutil.which) -> None:
    """Record with Termux:API.  It refuses unless this install's termux-microphone-record lists the flags used:
    they are checked, not assumed."""
    exe = which("termux-microphone-record")
    if not exe:
        raise VoiceError("termux-microphone-record not found: install the Termux:API app and "
                         "`pkg install termux-api`, and grant Termux:API the Microphone permission "
                         "(or pass --file with a clip you already have)")
    helptext = run([exe, "-h"], capture_output=True, text=True, timeout=15)
    seen = (helptext.stdout or "") + (helptext.stderr or "")
    missing = [f for f in RECORD_FLAGS if f not in seen]
    if missing:
        raise VoiceError(f"this termux-microphone-record does not list {', '.join(missing)}; not guessing")
    seconds = max(1, min(int(seconds), MAX_SECONDS))
    r = run([exe, "-f", str(out), "-l", str(seconds), "-e", "aac"], capture_output=True, text=True,
            timeout=seconds + 30)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise VoiceError("recording failed (is the Microphone permission granted to Termux:API?)")


def transcribe(url: str, token: str, clip: Path, opener=None, timeout: float = 120.0) -> dict[str, Any]:
    size = clip.stat().st_size
    if size == 0:
        raise VoiceError("the clip is empty")
    if size > MAX_CLIP_BYTES:
        raise VoiceError(f"clip is over {MAX_CLIP_BYTES} bytes; record less")
    req = urllib.request.Request(url, clip.read_bytes(), {
        "Content-Type": "audio/aac", "Authorization": "Bearer " + token}, method="POST")
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as r:
            import json
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise VoiceError(f"the speech service answered {exc.code}") from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise VoiceError(f"cannot reach the speech service: {getattr(exc, 'reason', exc)}") from None


def directive_for(text: str) -> dict[str, Any]:
    """The only thing a spoken note can become.  The words are data in title/do_this; role and severity are fixed."""
    one = " ".join(text.split())
    return {"title": ("Spoken note: " + one)[:MAX_WORDS_TITLE], "do_this": one[:600],
            "why_it_matters": "Recorded by voice on an edge device and transcribed automatically; "
                              "it may contain recognition errors.",
            "owner_role": "Anyone", "severity": "info"}


def stage(client: qm.McpEdge, text: str, model: str, device: str) -> dict[str, Any]:
    res = client.call("process_studio_propose", {
        "tool": "hub.body_directive", "arguments": directive_for(text),
        "reason": "spoken note (edge voice, held for a human)",
        "meta": {"edge_device": device[:64], "spoken": True, "stt_model": str(model)[:40]}})
    if res.get("isError"):
        raise VoiceError(qm.tool_text(res))
    return res


def _confirm(text: str, assume_yes: bool, stdin=None, out=None) -> bool:
    stdin = stdin or sys.stdin                  # resolved at call time, not at import
    out = out or sys.stdout
    print(f'heard: "{text}"', file=out)
    if assume_yes:
        return True
    if not stdin.isatty():
        raise VoiceError("not a terminal: pass --yes to stage without asking")
    return input("stage this as a held note? [y/N] ").strip().lower() in ("y", "yes")


def main(argv: Optional[list] = None, env: Optional[Mapping[str, str]] = None, opener=None,
         mcp_opener=None, run=subprocess.run, which=shutil.which) -> int:
    env = os.environ if env is None else env
    ap = argparse.ArgumentParser(prog="quipu_voice", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    ls = sub.add_parser("listen")
    ls.add_argument("--seconds", type=int, default=10)
    ls.add_argument("--file", default=None)
    ls.add_argument("--yes", action="store_true", help="stage without asking (for scripts)")
    ls.add_argument("--dry-run", action="store_true", help="transcribe and show, stage nothing")
    ls.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    try:
        stt_url, stt_token = resolve_stt(env, qm.load_env_file(STT_CONFIG))
        if a.cmd == "doctor":
            lines, ok = [], True
            try:
                with (opener or urllib.request.urlopen)(urllib.request.Request(
                        stt_url.rsplit("/", 1)[0] + "/health"), timeout=10) as r:
                    lines.append(f"ok    speech service reachable ({r.status})")
            except Exception as exc:                                  # noqa: BLE001
                ok = False
                lines.append(f"FAIL  speech service: {getattr(exc, 'reason', exc)}")
            lines.append("ok    microphone tool present" if which("termux-microphone-record")
                         else "WARN  termux-microphone-record not found (use --file)")
            try:
                murl, mtoken = qm.resolve_config(env, qm.load_env_file())
                mlines, mok = qm.doctor(qm.McpEdge(murl, mtoken, opener=mcp_opener), opener=mcp_opener)
                lines += ["MCP: " + ln for ln in mlines]
                ok = ok and mok
            except qm.EdgeError as exc:
                ok = False
                lines.append(f"FAIL  MCP config: {exc}")
            print("\n".join(lines))
            return EXIT_OK if ok else EXIT_FAIL
        # listen
        murl, mtoken = qm.resolve_config(env, qm.load_env_file())
        with tempfile.TemporaryDirectory(prefix="quipu-voice-") as tmp:
            if a.file:
                clip = Path(a.file).expanduser()
                if not clip.is_file():
                    raise VoiceError(f"no such file: {clip}")
            else:
                clip = Path(tmp) / "clip.aac"
                record_clip(a.seconds, clip, run=run, which=which)
            reply = transcribe(stt_url, stt_token, clip, opener=opener)
        text = " ".join(str(reply.get("text") or "").split())
        if not reply.get("ok") or not text:
            raise VoiceError("nothing was heard (empty transcript)")
        if a.dry_run:
            print(f'heard: "{text}"  (dry run: nothing staged)')
            return EXIT_OK
        if not _confirm(text, a.yes):
            print("not staged.")
            return EXIT_OK
        res = stage(qm.McpEdge(murl, mtoken, opener=mcp_opener), text, str(reply.get("model") or ""),
                    a.device or socket.gethostname())
        print(qm.tool_text(res))
        return EXIT_OK
    except (VoiceError, qm.EdgeError) as exc:
        print(f"quipu_voice: {exc}", file=sys.stderr)
        return EXIT_SETUP if "not set" in str(exc) or "no STT token" in str(exc) or "refusing" in str(exc) \
            or "not found" in str(exc) or "not a terminal" in str(exc) or "no such file" in str(exc) else EXIT_FAIL


if __name__ == "__main__":                                                  # pragma: no cover
    sys.exit(main())
